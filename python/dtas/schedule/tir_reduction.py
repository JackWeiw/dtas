"""
for softmax, layernorm

"""

from typing import Optional
import numpy as np

from tvm import tir
from tvm.tir.schedule import Schedule

from .tir_base import TIRSchedulerBase
from ..common.analisys import *
from ..common.utils import (
    save_to_file,
    get_root_block,
    is_broadcast_epilogue,
    auto_inline_producers,
)
from ..common.config import ReductionConfig


def save_ir(filename, sch):
    with open(filename, "w") as f:
        f.write(sch.mod.script())


class TIRReductionScheduler(TIRSchedulerBase):
    def _index_map_func(self, *index_list):
        j = index_list[-1]
        return index_list[:-1] + (j // 64, j % 64)

    def apply_config(self, config: ReductionConfig) -> Optional[Schedule]:
        # TODO tvm 的 simplify 有待改进, 
        ## 考虑到LLM中 GEMV的V长度都较大 > 2560,
        # 一个block处理一行数据
        # 对于general reduction softmax epilogue > 1
        # reduction epilogue <= 1
        sch = self.func_info.create_schedule()
        root_block = get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)
        if self.func_info.general_red:
            loops = sch.get_loops(blocks[-1])
            # sch.pad_einsum(blocks[0], [1, 1, 1024])
            bx = sch.fuse(*loops[: self.func_info.num_leading_s])
            r_loop, tx = sch.split(loops[-1], [None, config.len_tx])
            sch.reorder(tx, r_loop)
            sch.bind(bx, "blockIdx.x")
            sch.bind(tx, "threadIdx.x")
            sch.annotate(
                r_loop,
                ann_key="pragma_auto_unroll_max_step",
                ann_val=config.unroll_depth,
            )
            sch.annotate(r_loop, ann_key="pragma_unroll_explicit", ann_val=1)
            for block in reversed(blocks[:-1]):
                for i, _ in enumerate(sch.get(block).writes):
                    sch.set_scope(block, buffer_index=i, storage_scope="shared")
                sch.compute_at(block, bx, preserve_unit_loops=True)
                r_loop = sch.fuse(
                    *sch.get_loops(block)[-self.func_info.num_trailing_r :]
                )
                r_loop, tx = sch.split(r_loop, [None, config.len_tx])
                sch.reorder(tx, r_loop)
                sch.bind(tx, "threadIdx.x")
                sch.annotate(
                    r_loop,
                    ann_key="pragma_auto_unroll_max_step",
                    ann_val=config.unroll_depth,
                )
                sch.annotate(r_loop, ann_key="pragma_unroll_explicit", ann_val=1)
            if config.temp_storage in ["shared.dyn", "local"]:
                SS = sch.cache_read(blocks[0], 0, config.temp_storage)
                # if config.temp_storage == "shared.dyn":
                    # sch.transform_layout(SS, ("write", 0), self._index_map_func,pad_value = 0.0)
                #     sch.storage_align(SS, 0, -2, 16, 8)
                sch.compute_at(SS, bx, preserve_unit_loops=True)
                auto_inline_producers(sch, SS)
                r_loop = sch.get_loops(SS)[-1]
                r_loop, tx, vec = sch.split(
                    r_loop,
                    [None, config.len_tx, config.vector_size],
                )
                sch.bind(tx, "threadIdx.x")
                sch.vectorize(vec)
            # save_to_file(
            #     f"/home/weitao/XIAG8XX/profile/testIR/layernorm/ir/{config.len_tx}_{config.temp_storage}_in{config.vector_size}.py",
            #     sch,
            # )
            # print(sch.mod)
            return sch
        else:
            if self.func_info.is_inner_reduction:
                block = blocks[0]
                _, r, _ = sch.get_loops(block)
                _, tx = sch.split(r, factors=[None, config.len_tx])
                # Schedule the RF block
                rf = sch.rfactor(tx, 0)
                bx, r, tx, _ = sch.get_loops(rf)
                sch.reorder(bx, tx, r)
                sch.bind(bx, "blockIdx.x")
                sch.bind(tx, "threadIdx.x")
                sch.annotate(tx, ann_key="pragma_auto_unroll_max_step", ann_val=256)
                sch.annotate(tx, ann_key="pragma_unroll_explicit", ann_val=1)
                sch.set_scope(rf, 0, "local")
                sch.decompose_reduction(rf, r)
                # Schedule the write back block
                sch.reverse_compute_at(block, bx, preserve_unit_loops=True)
                _, tx, *s = sch.get_loops(block)

                if self.func_info.c_factor:
                    assert len(s) == len(self.func_info.loop_order)
                    new_order_s = [
                        s[self.func_info.loop_order[i]] for i in range(len(s))
                    ]
                    sch.reorder(*new_order_s)
                    new_order_s[self.func_info.s_split_index], c = sch.split(
                        new_order_s[self.func_info.s_split_index],
                        factors=[None, self.func_info.c_factor],
                    )
                    sch.reorder(*new_order_s, c)
                    s = sch.fuse(*new_order_s)
                    sch.reorder(s, tx, c)
                else:
                    s = sch.fuse(*s)
                    sch.reorder(s, tx)
                sch.bind(tx, "threadIdx.x")
                # Schedule epilogue
                if len(blocks) > 1:
                    epilogue = blocks[1]
                    sch.reverse_compute_at(epilogue, bx)
                    if is_broadcast_epilogue(sch, block, epilogue):
                        sch.set_scope(block, 0, "shared")
                        _, *s = sch.get_loops(epilogue)  # pylint: disable=invalid-name
                        _, tx = sch.split(
                            sch.fuse(*s), factors=[None, config.len_tx]
                        )
                        sch.bind(tx, "threadIdx.x")
                    else:
                        sch.set_scope(block, 0, "local")
                return sch
            return None
