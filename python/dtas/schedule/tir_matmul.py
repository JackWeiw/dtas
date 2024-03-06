from typing import Dict, Optional
from tvm import tir
from tvm.arith import Analyzer
from tvm.tir.schedule import Schedule


from ..common.config import GEMMConfig
from .tir_base import TIRSchedulerBase
from ..common.analisys import *
from ..common.utils import (
    auto_inline_consumer_chain,
    auto_inline_producers,
    get_root_block,
    save_to_file,
    get_reduction_blocks,
)
from ..logging import get_log_level, debug_info


def save_ir(filename, sch):
    with open(filename, "w") as f:
        f.write(sch.mod.script())


class TIRMatmulScheduler(TIRSchedulerBase):
    def apply_config(self, config: GEMMConfig) -> Optional[Schedule]:
        sch = self.func_info.create_schedule()
        """ see https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md#threadblock-rasterization
            block-level-tile -> thread-level-tile -> register-level-tile 
        """
        # we assume GEMM like func always get first block as reduction
        save_to_file(f"/home/weitao/XIAG8XX/profile/testIR/NOWMMA/m_n2560_k10240/ir/b.py", sch)
        root_block = get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)
        reduction_blocks = get_reduction_blocks(sch, blocks)
        main_block = reduction_blocks[0]
        y_kernel_size = (
            config.vthread_y * config.block_y * config.micro_shape_y
        )
        x_kernel_size = (
            config.vthread_x * config.block_x * config.micro_shape_x
        )
        k_kernel_size = config.micro_shape_k
        if config.inner_x:
            sch.pad_einsum(
                main_block,
                [1, y_kernel_size, x_kernel_size, k_kernel_size],
            )
            batch, y, x, k = sch.get_loops(main_block)
        else:
            sch.pad_einsum(
                main_block,
                [1, x_kernel_size, y_kernel_size, k_kernel_size],
            )
            batch, x, y, k = sch.get_loops(main_block)
            
        bx, vx, tx, xi = sch.split(
            x, [None, config.vthread_x, config.block_x, config.micro_shape_x]
        )

        by, vy, ty, yi = sch.split(
            y, [None, config.vthread_y, config.block_y, config.micro_shape_y]
        )

        ko, ki = sch.split(k, factors=[None, config.micro_shape_k])

        # if config.use_software_pipeline:
        #     sch.annotate(ko, "software_pipeline_order", [0, 1, 2, 3])
        #     sch.annotate(ko, "software_pipeline_stage", [0, 0, 1, 1])
        # sch.annotate(ki, "software_pipeline_order", [0, 1])
        # sch.annotate(ki, "software_pipeline_stage", [0, 1])

        sch.reorder(by, bx, vy, vx, ty, tx, ko, ki, yi, xi)
        by = sch.fuse(batch, by)

        analyzer = Analyzer()
        grid_x = analyzer.simplify(sch.get(bx).extent)
        grid_y = analyzer.simplify(sch.get(by).extent)
        config.grid_size = [1, grid_y, grid_x]
        config.block_size = [1, config.block_y, config.block_x]

        sch.bind(bx, "blockIdx.x")
        sch.bind(by, "blockIdx.y")
        sch.bind(vy, "vthread.y")
        sch.bind(vx, "vthread.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

        inner_loop = config.micro_shape_x if config.inner_x else config.micro_shape_y
        
        if inner_loop % config.vector_size == 0 and config.vector_size > 1:
            _, v = sch.split(xi, [None, config.vector_size])
            sch.vectorize(v)

        if config.unroll_depth > 0:
            sch.annotate(tx, ann_key="pragma_auto_unroll_max_step", ann_val=config.unroll_depth)
            sch.annotate(tx, ann_key="pragma_unroll_explicit", ann_val=1)
        # save_ir("/home/weitao/XIAG8XX/shot/IR/before_reg.py",sch)
        ## 需要进一步探讨write to register file时 vectorize,unroll情况

        reg_tile = sch.cache_write(main_block, 0, "local")
        sch.reverse_compute_at(reg_tile, tx, preserve_unit_loops=True)
        
        if config.micro_shape_x % config.vector_size == 0:
            left, v = sch.split(
                sch.get_loops(reg_tile)[-1], [None, config.vector_size]
            )
            sch.vectorize(v)
            # if config.unroll_depth > 0:
            #     sch.unroll(left)

        AS = sch.cache_read(main_block, 0, "shared")
        BS = sch.cache_read(main_block, 1, "shared")
        sch.compute_at(AS, ko, preserve_unit_loops=True)
        sch.compute_at(BS, ko, preserve_unit_loops=True)

        A_local = sch.cache_read(main_block, 0, "local")
        sch.compute_at(A_local, ki, preserve_unit_loops=True)
        B_local = sch.cache_read(main_block, 1, "local")
        sch.compute_at(B_local, ki, preserve_unit_loops=True)


        self.cooperative_fetch(sch, AS, config, 2, 8, config.vector_size, config.vector_size)
        self.cooperative_fetch(sch, BS, config, 2, 8, config.vector_size, config.vector_size)
        
        auto_inline_producers(sch, AS)
        auto_inline_producers(sch, BS)
        
        auto_inline_consumer_chain(sch, reg_tile)
        
        sch.decompose_reduction(main_block, ko)
        
        save_to_file(f"/home/weitao/XIAG8XX/profile/testIR/NOWMMA/m_n2560_k10240/ir/b{config.block_x}_{config.block_y}_t{config.micro_shape_x}_{config.micro_shape_y}_k{config.micro_shape_k}_v{config.vector_size}.py", sch)
        return sch
