from typing import Optional

from tvm.arith import Analyzer
from tvm.tir.schedule import Schedule

from .tir_base import TIRSchedulerBase
from ..common.utils import (
    auto_inline_consumer_chain,
    auto_inline_producers,
    save_to_file,
    get_root_block,
    get_reduction_blocks,
)
from ..intrin.wmma_intrin import get_wmma_intrin_group_diy
from ..common.config import WMMAConfig
from ..logging import get_log_level, debug_info
# with open(filename, "w") as f:
#     f.write(sch.mod.script())


class TIRWMMAScheduler(TIRSchedulerBase):
    def apply_config(self, config: WMMAConfig) -> Optional[Schedule]:
        sch= self.func_info.create_schedule()
        root_block = get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)
        reduction_blocks = get_reduction_blocks(sch, blocks)
        main_block = reduction_blocks[0]
        # block_stmt = sch.get(main_block)
        # main_block = self.func_info.main_block

        y_kernel_size = config.j * config.micro_shape_y * config.wmma_m
        x_kernel_size = config.i * config.micro_shape_x * config.wmma_n
        k_kernel_size = config.micro_shape_k * config.wmma_k
        # k_kernel_size = config.micro_shape_k * config.wmma_k
        sch.pad_einsum(
            main_block,
            [1, x_kernel_size, y_kernel_size, k_kernel_size],
        )
        batch, i, j, k = sch.get_loops(main_block)

        block = main_block
        # inner loops for tensor core computation
        i, i_inner = sch.split(i, factors=[None, config.wmma_m])
        j, j_inner = sch.split(j, factors=[None, config.wmma_n])
        k, k_inner = sch.split(k, factors=[None, config.wmma_k])
        sch.reorder(i, j, k, i_inner, j_inner, k_inner)
        # for TensorCore
        block_inner = block
        # 外面一层
        block_outer = sch.blockize(i_inner)
        i0, i1, i2, i3 = sch.split(
            i,
            factors=[
                None,
                1,
                config.i,
                config.micro_shape_x,
            ],
        )
        j0, j1, j2, j3 = sch.split(
            j,
            factors=[
                1,
                None,
                config.j,
                config.micro_shape_y,
            ],
        )

        k0, k1 = sch.split(k, [None, config.micro_shape_k])

        sch.annotate(k0, "software_pipeline_stage", [0, 0, 0, 0, 0, 1, 1])
        sch.annotate(k0, "software_pipeline_order", [0, 3, 1, 4, 5, 2, 6])
        sch.annotate(k1, "software_pipeline_stage", [0, 0, 1])
        sch.annotate(k1, "software_pipeline_order", [0, 1, 2])
            # elif (
            #     config.use_async_copy
            #     and not config.manifest_shared_memory_local_stage
            # ):
            #     sch.annotate(k0, "software_pipeline_stage", [0, 0, 1, 1, 1])
            #     sch.annotate(k0, "software_pipeline_order", [0, 1, 2, 3, 4])
            #     sch.annotate(k0, "software_pipeline_async_stages", [0])
            #     # sch.annotate(k0, "software_pipeline_stage", [0, 0, 1])
            #     # sch.annotate(k0, "software_pipeline_order", [0, 1, 2])
            #     # sch.annotate(k0, "software_pipeline_stage", [0, 0, 1, 1, 2])
            #     # sch.annotate(k0, "software_pipeline_order", [0, 1, 2, 3, 4])
            #     sch.annotate(k1, "software_pipeline_stage", [0, 0, 1])
            #     sch.annotate(k1, "software_pipeline_order", [0, 1, 2])
            # if config.use_async_copy:

            ## inner wmma software pipeline

        ## n/15/16,2560/8/16,10240/12/16,4,5,2,6,3,2
        sch.reorder(i0, j0, i1, j1, j2, i2, k0, k1, i3, j3)

        """
        tile_m=wmma_m*micro_shape_x*i
        tile_n=wmma_n*micro_shape_y*j
        tile_k=wmma_k*micro_shape_k*unroll_k
        """

        block_idx = sch.fuse(i0, j0)
        block_idy = sch.fuse(i1, j1)
        thread_idy = sch.fuse(j2, i2)
        # sch.unroll(k1)
        ## launch statistic
        analyzer = Analyzer()
        grid_x = analyzer.simplify(sch.get(block_idx).extent)
        grid_y = analyzer.simplify(sch.get(block_idy).extent)
        grid_z = analyzer.simplify(sch.get(batch).extent)
        config.grid_size = [grid_z, grid_y, grid_x]
        block_size_y = analyzer.simplify(sch.get(thread_idy).extent)
        config.block_size = [1, block_size_y, config.warp_size]

        sch.bind(batch, "blockIdx.z")
        sch.bind(block_idx, "blockIdx.x")
        sch.bind(block_idy, "blockIdx.y")
        sch.bind(thread_idy, "threadIdx.y")

        AS = sch.cache_read(block_outer, 0, "shared.dyn")
        BS = sch.cache_read(block_outer, 1, "shared.dyn")

        sch.compute_at(AS, k0, preserve_unit_loops=True)
        sch.compute_at(BS, k0, preserve_unit_loops=True)
        # 如果访问的stride是4byte的奇数倍就bank-8conflict free
        # stride即为dtype*vector_size eg.fp16 vec4 stride=2*4是4的2倍，就会有bank-conflict
        self.cooperative_fetch(
            sch,
            AS,
            config,
            2,
            is_a=True,
            is_transpose=self.func_info.t_a
        )
        self.cooperative_fetch(
            sch,
            BS,
            config,
            2,
            is_a=False,
            is_transpose=self.func_info.t_b
        )
        auto_inline_producers(sch, AS)
        auto_inline_producers(sch, BS)

        # create read cache to load matrix from shared memory to wmma fragments
        A_mat = sch.cache_read(block_outer, 0, "wmma.matrix_a")
        B_mat = sch.cache_read(block_outer, 1, "wmma.matrix_b")

        sch.compute_at(A_mat, k1)
        sch.compute_at(B_mat, k1)

        wmma_shape = (
            str(config.wmma_m)
            + "x"
            + str(config.wmma_n)
            + "x"
            + str(config.wmma_k)
        )
        trans_a = False
        trans_b = True
        ## TODO 如果后面没有add，那么就不需要store至SMEM,直接store至GMEM
        intrin_group = get_wmma_intrin_group_diy(
            wmma_shape,
            "shared.dyn",
            "shared.dyn",
            self.func_info.in_dtype,
            self.func_info.out_dtype,
            trans_a,
            trans_b,
        )
        # Tensorization by hardware intrinsics
        try:
            # Schedule for  wmma_matrix_a load
            i, j = sch.get_loops(A_mat)[-2:]
            i0, i1 = sch.split(i, factors=[None, config.wmma_m])
            j0, j1 = sch.split(j, factors=[None, config.wmma_k])
            sch.reorder(i0, j0, i1, j1)
            sch.unroll(i0)
            sch.unroll(j0)
            # save_to_file("/home/weitao/XIAG8XX/profile/IR/before_loada.py",sch)
            sch.tensorize(i1, intrin_group["load_a"])
            # Schedule for  wmma_matrix_b load
            i, j = sch.get_loops(B_mat)[-2:]
            i0, i1 = sch.split(i, factors=[None, config.wmma_n])
            j0, j1 = sch.split(j, factors=[None, config.wmma_k])
            sch.reorder(i0, j0, i1, j1)
            sch.unroll(i0)
            sch.unroll(j0)
            sch.tensorize(i1, intrin_group["load_b"])
        except:  # pylint: disable=bare-except
            if get_log_level()>=2: debug_info("failed to tensorize load")
            return None

        # create write cache to store matrix from wmma fragments to shared memory and global memory
        ## TODO set storage align for write,主要是为了后续S2G使用ST128指令，减少指令数目，一定程度上降延迟
        accumulator_shared_to_global = sch.cache_write(block_outer, 0, "shared.dyn")
        sch.storage_align(
            accumulator_shared_to_global,
            0,
            -2,
            config.wmma_n,
            config.out_pad,
        )
        store = sch.cache_write(block_outer, 0, "wmma.accumulator")
        ##等ktile循环完后计算才结束
        sch.reverse_compute_at(store, thread_idy)
        sch.reverse_compute_at(accumulator_shared_to_global, thread_idy)

        def _tensorize_init_store_compute():
            # split the store loop to match hardware intrinsic pattern
            i, j = sch.get_loops(store)[-2:]
            i0, i1 = sch.split(i, factors=[None, config.wmma_m])
            j0, j1 = sch.split(j, factors=[None, config.wmma_n])
            sch.reorder(i0, j0, i1, j1)
            block_init_c = sch.decompose_reduction(block_outer, k0)
            block_init_c_inner = sch.get_child_blocks(block_init_c)[0]
            sch.tensorize(sch.get_loops(block_init_c_inner)[-2], intrin_group["init"])
            sch.tensorize(sch.get_loops(store)[-2], intrin_group["store"])
            sch.tensorize(sch.get_loops(block_inner)[-3], intrin_group["compute"])

        try:
            _tensorize_init_store_compute()
        except:  # pylint: disable=bare-except
            if get_log_level()>=1: debug_info("failed to tensorize compute")
            return None

        auto_inline_consumer_chain(sch, accumulator_shared_to_global)
        fused = sch.fuse(*sch.get_loops(accumulator_shared_to_global)[-2:])
        ax, f1, f2 = sch.split(
            fused, factors=[None, config.warp_size, config.out_vec_len]
        )
        sch.bind(f1, "threadIdx.x")
        sch.vectorize(f2)
        return sch
