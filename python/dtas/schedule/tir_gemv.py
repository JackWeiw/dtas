from tvm import tir

from .tir_base import TIRSchedulerBase
from ..common.analisys import *
from ..common.utils import (
    get_max_factor,
    save_to_file,
    get_root_block,
    get_reduction_blocks,
)
from ..common.config import Config


def save_ir(filename, sch):
    with open(filename, "w") as f:
        f.write(sch.mod.script())


"""
llm中 GEMV出现在decode过程
普遍为static shape, 在计算Q, K, V时为static shape
在Attention计算Q*K时 为 1*head_dim * head_dim*n (n为已有上文量)
在计算softmax后*V时 head_dim *n * n*head_dim
后面的MLP部分为static shape
"""


class TIRGEMVScheduler(TIRSchedulerBase):
    def _index_map_func(self, *index_list):
        j = index_list[-1]
        return index_list[:-1] + (j // 64, j % 64)

    def apply_config(self, config: Config) -> Optional[tir.Schedule]:
        ## 考虑到LLM中 GEMV的V长度都较大 > 2560,
        sch, gemv_config = self.func_info.create_schedule(), config
        root_block = get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)
        reduction_blocks = get_reduction_blocks(sch, blocks)
        gemv = reduction_blocks[0]
        epilogue = (
            blocks[1]
            if len(blocks) == 2 and self._is_injective_epilogue(sch, blocks[1])
            else None
        )

        batch, s, r, c = sch.get_loops(gemv)

        # rfactor: reduce to tx * vec_c
        # 第一阶段每个thread_block计算TR * TILE_R 的reduce，需要tx * vec_c * len_VECTOR 的 register
        s = sch.fuse(batch, s)
        r = sch.fuse(r, c)
        bx, ts, tile_s = sch.split(
            s,
            factors=[None, gemv_config.TS, gemv_config.TILE_S],
            preserve_unit_iters=True,
        )
        r, tr, tile_r_vec_n, vec_c = sch.split(
            r,
            factors=[
                None,
                gemv_config.TR,
                gemv_config.TILE_R // gemv_config.VEC_C,
                gemv_config.VEC_C,
            ],
            preserve_unit_iters=True,
        )
        sch.reorder(r, tile_r_vec_n, tr, vec_c)
        tr_vec_c = sch.fuse(tr, vec_c)
        rf = sch.rfactor(tr_vec_c, 0)

        # 第二阶段每个thread_block计算TR 的reduce 需要tx * len_VECTOR 的 register
        # rfactor: reduce to tx
        bx, ts, tile_s, tr_vec_c = sch.get_loops(block=gemv)
        tr, vec_c = sch.split(
            tr_vec_c, factors=[gemv_config.TR, None], preserve_unit_iters=True
        )
        rf2 = sch.rfactor(tr, 0)

        # bind, vectorize compute of rf
        bx, ts, tile_s, r, tile_r_vec_n, tr_vec_c = sch.get_loops(block=rf)
        tr, vec_c = sch.split(
            tr_vec_c, factors=[gemv_config.TR, None], preserve_unit_iters=True
        )
        sch.reorder(bx, ts, tr, r, tile_s, tile_r_vec_n, vec_c)
        sch.bind(bx, "blockIdx.x")
        sch.bind(ts, gemv_config.TAG_S)
        sch.bind(tr, gemv_config.TAG_R)
        sch.vectorize(vec_c)

        # vectorize load A matrix
        # (TODO) this is now actually problematic since the number of loops is dependent on the
        # number of dimensions of A_q
        Aq_local = sch.cache_read(rf, read_buffer_index=1, storage_scope="local")
        sch.compute_at(Aq_local, r, preserve_unit_loops=True)
        s_local, r_local = sch.get_loops(block=Aq_local)[-2:]
        # 从这里推出VEC_LOAD_S < TILE_S 不然会在 load A_q的时候出现不需要的谓词
        s_local, vec_load = sch.split(
            s_local, factors=[None, gemv_config.VEC_LOAD_S], preserve_unit_iters=True
        )
        sch.reorder(s_local, r_local, vec_load)  # either s_local or r_local should be 1
        sch.vectorize(vec_load)

        if gemv_config.LOAD_V_SHARED:
            ## TODO 卫涛 当SMEM无法完全容纳V时，需要分阶段计算
            """
            COORPERATIVE FETCH V
            """
            V_shared = sch.cache_read(rf, read_buffer_index=0, storage_scope="shared")
            sch.compute_at(V_shared, tr, preserve_unit_loops=True)
            sch.transform_layout(V_shared, ("write", 0), self._index_map_func)
            sch.storage_align(V_shared, 0, -2, 16, 8)
            # load vector into shared memory, shape should be the whole vector
            l = sch.get_loops(block=V_shared)[-1]
            # loop: tir.For = sch.get(l)
            # if isinstance(loop.extent, tir.IntImm):
            #     # avoid introducing predicates when vector length is too large
            #     vec_length = max(
            #         min(
            #             get_max_factor(
            #                 (int)(loop.extent),
            #                 [gemv_config.TS * gemv_config.TR * 1, gemv_config.TS * gemv_config.TR * 2, gemv_config.TS * gemv_config.TR * 4, gemv_config.TS * gemv_config.TR * 8],
            #             )
            #             // gemv_config.TS
            #             // gemv_config.TR,
            #             gemv_config.LOAD_V_VEC,
            #         ),
            #         1,
            #     )
            # else:
            #     vec_length = gemv_config.LOAD_V_VEC

            if gemv_config.TAG_R == "threadIdx.x":
                _, ty, tx, vec = sch.split(
                    l,
                    factors=[
                        None,
                        gemv_config.TS,
                        gemv_config.TR,
                        gemv_config.LOAD_V_VEC,
                    ],
                    preserve_unit_iters=True,
                )
            else:
                _, ty, tx, vec = sch.split(
                    l,
                    factors=[
                        None,
                        gemv_config.TR,
                        gemv_config.TS,
                        gemv_config.LOAD_V_VEC,
                    ],
                    preserve_unit_iters=True,
                )
            sch.bind(ty, "threadIdx.y")
            sch.bind(tx, "threadIdx.x")
            sch.vectorize(vec)

        # reduce tile_s * tr * vec to tile_s * tr ,split-k 第一轮算完，开始算第二轮
        sch.reverse_compute_at(rf2, loop=bx, preserve_unit_loops=True)
        tr, vec_c, *ts_tile_s = sch.get_loops(block=rf2)[1:]
        ts_tile_s = sch.fuse(*ts_tile_s)
        ts, tile_s = sch.split(
            ts_tile_s, factors=[gemv_config.TS, None], preserve_unit_iters=True
        )
        tile_s, vec_s = sch.split(
            tile_s,
            factors=[None, get_max_factor(gemv_config.TILE_S, [1, 2, 4, 8])],
            preserve_unit_iters=True,
        )
        sch.reorder(ts, tr, tile_s, vec_s, vec_c)
        sch.bind(ts, gemv_config.TAG_S)
        sch.bind(tr, gemv_config.TAG_R)
        sch.vectorize(vec_s)

        # reduce tile_s * tr to tile_s，最后一轮all thread reduce
        sch.reverse_compute_at(gemv, loop=bx, preserve_unit_loops=True)
        tr, *ts_tile_s = sch.get_loops(block=gemv)[1:]
        ts_tile_s = sch.fuse(*ts_tile_s)
        ts, tile_s = sch.split(
            ts_tile_s, factors=[gemv_config.TS, None], preserve_unit_iters=True
        )
        sch.reorder(tile_s, ts, tr)
        sch.bind(ts, gemv_config.TAG_S)
        sch.bind(tr, gemv_config.TAG_R)

        sch.decompose_reduction(rf, loop=sch.get_loops(block=rf)[3])
        sch.decompose_reduction(rf2, loop=sch.get_loops(block=rf2)[-1])

        sch.set_scope(rf, buffer_index=0, storage_scope="local")
        sch.set_scope(rf2, buffer_index=0, storage_scope="local")

        unroll_factor = gemv_config.UNROLL

        sch.annotate(
            block_or_loop=sch.get_loops(rf)[3],
            ann_key="pragma_auto_unroll_max_step",
            ann_val=unroll_factor,
        )
        sch.annotate(
            block_or_loop=sch.get_loops(rf)[3],
            ann_key="pragma_unroll_explicit",
            ann_val=1,
        )

        sch.annotate(
            block_or_loop=sch.get_loops(rf2)[3],
            ann_key="pragma_auto_unroll_max_step",
            ann_val=unroll_factor,
        )
        sch.annotate(
            block_or_loop=sch.get_loops(rf2)[3],
            ann_key="pragma_unroll_explicit",
            ann_val=1,
        )

        if gemv_config.LOAD_V_SHARED:
            sch.annotate(
                block_or_loop=sch.get_loops(V_shared)[-4],
                ann_key="pragma_unroll_explicit",
                ann_val=unroll_factor,
            )
            sch.annotate(
                block_or_loop=sch.get_loops(V_shared)[-4],
                ann_key="pragma_vectorize",
                ann_val=1,
            )

        # Schedule epilogue
        if epilogue is not None:
            if self._is_broadcast_epilogue(sch, gemv, epilogue):
                sch.reverse_compute_at(epilogue, bx)
                sch.set_scope(gemv, 0, "shared")
                _, _, *s = sch.get_loops(epilogue)  # pylint: disable=invalid-name
                _, tx = sch.split(sch.fuse(*s), factors=[None, gemv_config.TS])
                sch.bind(tx, "threadIdx.x")
            else:
                sch.reverse_compute_at(epilogue, bx, preserve_unit_loops=True)
                ts_tile_s = sch.fuse(*sch.get_loops(epilogue)[1:])
                ts_tile_s = sch.get_loops(epilogue)[-1]
                ts, tile_s = sch.split(
                    ts_tile_s, factors=[gemv_config.TS, None], preserve_unit_iters=True
                )
                sch.bind(ts, gemv_config.TAG_S)
                sch.set_scope(gemv, 0, "local")
        # pylint: enable=invalid-name
        # save_to_file(
        #     "/home/weitao/XIAG8XX/profile/testIR/GEMV/m1n2560k10240/IR/5wt.py", sch
        # )
        return sch
