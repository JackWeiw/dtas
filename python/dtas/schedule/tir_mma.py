
from typing import Optional, Literal

from tvm.arith import Analyzer
from tvm.tir.schedule import Schedule

from .tir_base import TIRSchedulerBase
from ..common.utils import (
    auto_inline_consumer_chain,
    auto_inline_producers,
    save_to_file,
    get_root_block,
    get_reduction_blocks,
    get_dequantize_block,
)
from ..common.config import MMAConfig
from ..logging import get_log_level, debug_info
from ..intrin.mma_intrin import *

class TIRMMAScheduler(TIRSchedulerBase):
    def apply_config(self, config: MMAConfig) -> Optional[Schedule]:
        
        sch= self.func_info.create_schedule()
        root_block = get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)
        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None
        dequantize_block = get_dequantize_block(sch, blocks)
        main_block = reduction_blocks[0]        
        output_blocks = [sch.get(block) for block in sch.get_output_blocks(root_block)]
        cache_write_required = True if (sch.get(main_block) not in output_blocks or len(self.func_info.dynamic_args)>1) else False

        in_dtype = self.func_info.in_dtype
        out_dtype = self.func_info.out_dtype 
        is_transpose_a = self.func_info.t_a
        is_transpose_b = self.func_info.t_b

        # Tensorization by hardware intrinsics
        from tvm.tir.tensor_intrin.cuda import (  # pylint: disable=import-outside-toplevel
            get_mma_intrin_group,
            shared_16x16_to_ldmatrix_32x8_layout,
        )

        # tile size
        block_m  = config.thread_z * config.micro_block_cnt_in_warp_m * config.micro_size_m
        block_n = config.thread_y * config.micro_block_cnt_in_warp_n * config.micro_size_n
        block_k = config.micro_block_cnt_in_warp_k * config.micro_size_k
        
        # thread size
        # thread_x == warp_size
        # Step 1. already have normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]/B[S, K, J]
        batch, i, j, k = sch.get_loops(main_block)
        # Step 2. Padding for dynamic shape kernels
        sch.pad_einsum(
            main_block,
            [
                1,
                block_m,
                block_n,
                block_k,
            ],
        )
        # sch.pad_einsum(
        #     main_block,
        #     [
        #         1,
        #         config.swizzle_factor_for_l2_m * block_m,
        #         config.swizzle_factor_for_l2_n * block_n,
        #         block_k,
        #     ],
        # )
        # Step 3. Reorder loops for tiling
        # Step 3.1 inner loops for tensor core computation
        i, i_inner = sch.split(i, factors=[None, config.micro_size_m])
        j, j_inner = sch.split(j, factors=[None, config.micro_size_n])
        k, k_inner = sch.split(k, factors=[None, config.micro_size_k])

        sch.reorder(i, j, k, i_inner, j_inner, k_inner)
        block_inner = main_block
        block_outer = sch.blockize(i_inner)

        # Step 3.2 outer loops for tiling
        # split factors for i, j, and k
        i_factors = [None, 1, config.thread_z, config.micro_block_cnt_in_warp_m]
        j_factors = [1, None, config.thread_y, config.micro_block_cnt_in_warp_n]
        # i_factors = [None, config.swizzle_factor_for_l2_m, config.thread_z, config.micro_block_cnt_in_warp_m]
        # j_factors = [None, config.swizzle_factor_for_l2_n, config.thread_y, config.micro_block_cnt_in_warp_n]
        k_factors = [None, config.micro_block_cnt_in_warp_k]

        i0, i1, i2, i3 = sch.split(i, factors=i_factors)
        j0, j1, j2, j3 = sch.split(j, factors=j_factors)
        k0, k1 = sch.split(k, factors=k_factors)

        sch.reorder(i0, j0, i1, j1, i2, j2, k0, k1, i3, j3)


        block_idy = sch.fuse(i0, j0)
        block_idx = sch.fuse(i1, j1)
        thread_idz = i2
        thread_idy = j2
        sch.bind(batch, "blockIdx.z")
        sch.bind(block_idy, "blockIdx.y")
        sch.bind(block_idx, "blockIdx.x")
        
        sch.bind(thread_idz, "threadIdx.z")
        sch.bind(thread_idy, "threadIdx.y")
        
        ## launch statistic
        analyzer = Analyzer()
        grid_z = analyzer.simplify(sch.get(batch).extent.value)
        grid_x = analyzer.simplify(sch.get(block_idx).extent.value)
        grid_y = analyzer.simplify(sch.get(block_idy).extent)
        config.grid_size = [grid_z, grid_y, grid_x]
        config.block_size = [config.thread_z, config.thread_y, config.warp_size]  
              
        # cache to shared memory
        shared_a = sch.cache_read(block_outer, 0, "shared.dyn")
        shared_b = sch.cache_read(block_outer, 1, "shared.dyn")
        sch.compute_at(shared_a, k0)
        sch.compute_at(shared_b, k0)
        self.cooperative_fetch(sch, shared_a, config, 2, True, is_transpose_a)
        self.cooperative_fetch(sch, shared_b, config, 2, False, is_transpose_b)
        
        # cache to register
        def fetch_to_register(block_outer, read_buffer_idx, is_a, is_transpose):
            block_read_reg = sch.cache_read(block_outer, read_buffer_idx, "warp")
            sch.compute_at(block_read_reg, k1)
            # bind_loops
            micro_size_spatial = config.micro_size_m if is_a else config.micro_size_n
            micro_size_1, micro_size_2 = (
                (micro_size_spatial, config.micro_size_k)
                if not is_transpose
                else (config.micro_size_k, micro_size_spatial)
            )
            v00, v01 = sch.split(sch.get_loops(block_read_reg)[-2], [None, micro_size_1])
            v10, v11 = sch.split(sch.get_loops(block_read_reg)[-1], [None, micro_size_2])
            sch.reorder(v00, v10, v01, v11)
            # reorder read axis to match the layout of ldmatrix
            sch.transform_layout(
                block_read_reg,
                ("write", 0),
                lambda v0, v1, v2: (
                    v0,
                    v1 // micro_size_1,
                    v2 // micro_size_2,
                    *shared_16x16_to_ldmatrix_32x8_layout(v1 % micro_size_1, v2 % micro_size_2),
                ),
            )
            # register swizzling
            mma_read_block = sch.blockize(sch.get_loops(block_read_reg)[-2])
            sch.annotate(mma_read_block, ann_key="permuted_layout", ann_val=1)
            return block_read_reg
            

        block_read_reg_a = fetch_to_register(block_outer, 0, True, is_transpose_a)
        block_read_reg_b = fetch_to_register(block_outer, 1, False, is_transpose_b)

        shared_scope = "shared.dyn"
        intrin_group = get_mma_intrin_group(
            load_scope= shared_scope,
            store_scope=shared_scope if cache_write_required else "global",
            in_dtype=str(in_dtype),
            out_dtype=str(out_dtype),
            trans_a=is_transpose_a,
            trans_b=is_transpose_b,
            not_use_mma_store_intrinic=False,
        )
        debug_info(intrin_group)
        # Write to register, and then smem
        # create write cache to store matrix from mma fragments to shared memory and global memory
        if cache_write_required:
            accumulator_shared_to_global = sch.cache_write(block_outer, 0, shared_scope)

        block_write_reg = sch.cache_write(block_outer, 0, "warp")
        sch.reverse_compute_at(block_write_reg, thread_idy)
        # split the block_write_reg loop to match hardware intrinsic pattern
        i, j = sch.get_loops(block_write_reg)[-2:]
        i0, i1 = sch.split(i, factors=[None, config.micro_size_m])
        j0, j1= sch.split(j, factors=[None, config.micro_size_n])
        sch.reorder(i0, j0, i1, j1)           
        # reorder write axis to match the layout of ldmatrix
        sch.transform_layout(
            block_write_reg,
            ("read", 0),
            lambda v0, v1, v2: (
                v0,
                v1 // config.micro_size_m,
                v2 // config.micro_size_n,
                #TODO shared_16x16_to_mma_32x8_layout
                *shared_16x16_to_ldmatrix_32x8_layout(v1 % config.micro_size_m, v2 % config.micro_size_n),
            ),
        )
        # # reorder write axis to match the layout of ldmatrix
        # sch.transform_layout(
        #     block_write_reg,
        #     ("write", 0),
        #     lambda v0, v1, v2: (
        #         v0,
        #         v1 // config.micro_size_m,
        #         v2 // config.micro_size_n,
        #         #TODO shared_16x16_to_mma_32x8_layout
        #         *shared_16x16_to_ldmatrix_32x8_layout(v1 % config.micro_size_m, v2 % config.micro_size_n),
        #     ),
        # )
        # sch.bind(i0, "threadIdx.z")
        # sch.bind(j0, "threadIdx.y")ss
        if cache_write_required:
            auto_inline_consumer_chain(sch, accumulator_shared_to_global)
            sch.reverse_compute_at(
                accumulator_shared_to_global, thread_idy, preserve_unit_loops=True
            )
            fused = sch.fuse(*sch.get_loops(accumulator_shared_to_global)[-2:])
            f0, f1, f2 = sch.split(
                fused, factors=[None, config.warp_size, config.out_vec_len]
            )
            sch.bind(f1, "threadIdx.x")
            sch.vectorize(f2)
            sch.unroll(f0)
            sch.annotate(f0, "pragma_unroll_explicit", False)
        else:
            auto_inline_consumer_chain(sch, block_write_reg)
            
      
        # Step 5. Schedule tensor core computation
        block_init_c = sch.decompose_reduction(block_outer, k0)
        block_init_c_inner = sch.get_child_blocks(block_init_c)[0]
        # debug_info(sch.get(block_init_c_inner).body)
        def tensorize_load():
            sch.tensorize(sch.get_loops(block_read_reg_a)[-2], intrin_group["load_a"])
            sch.tensorize(sch.get_loops(block_read_reg_b)[-2], intrin_group["load_b"])
            
        def tensorize_init_store_compute():
            # debug_info(sch.get(sch.get_loops(block_init_c_inner)[-2]))
            sch.tensorize(sch.get_loops(block_write_reg)[-2], intrin_group["store"])
            sch.tensorize(sch.get_loops(block_init_c_inner)[-2], intrin_group["init"])
            sch.tensorize(sch.get_loops(block_inner)[-3], intrin_group["compute"])
            
        try:
            tensorize_load()
        except:
            if get_log_level()>=1: debug_info("failed to tensorize load")
        try:
            tensorize_init_store_compute()
        except Exception as error:
            print("发生错误:", error)
            if get_log_level()>=1: debug_info("failed to tensorize init, store, compute")
                
        # Step 6. Async pipeline
        sch.annotate(k0, ann_key="software_pipeline_stage", ann_val=[0, 0, 3])
        sch.annotate(k0, ann_key="software_pipeline_order", ann_val=[0, 1, 2])
        sch.annotate(k0, ann_key="software_pipeline_async_stages", ann_val=[0])

        return sch

 