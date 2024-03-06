
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

class TIRMMAScheduler(TIRSchedulerBase):
    def apply_config(self, config: MMAConfig) -> Optional[Schedule]:
        
        sch= self.func_info.create_schedule()
        root_block = get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)
        reduction_blocks = get_reduction_blocks(sch, blocks)
        main_block = reduction_blocks[0]
        
        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        dequantize_block = get_dequantize_block(sch, blocks)

        main_block = reduction_blocks[0]

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
                config.swizzle_factor_for_l2_m * block_m,
                config.swizzle_factor_for_l2_n * block_n,
                block_k,
            ],
        )
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
        i_factors = [config.swizzle_factor_for_l2_m, None, config.thread_z, config.micro_block_cnt_in_warp_m]
        j_factors = [config.swizzle_factor_for_l2_n, None, config.thread_y, config.micro_block_cnt_in_warp_n]
        k_factors = [None, config.micro_block_cnt_in_warp_k]

        i0, i1, i2, i3 = sch.split(i, factors=i_factors)
        j0, j1, j2, j3 = sch.split(j, factors=j_factors)
        k0, k1 = sch.split(k, factors=k_factors)

        sch.reorder(i0, j0, i1, j1, i2, j2, k0, k1, i3, j3)

        block_axis = sch.fuse(batch, i0, j0, i1, j1)
        sch.bind(block_axis, "blockIdx.x")
        sch.bind(i2, "threadIdx.z")
        sch.bind(j2, "threadIdx.y")
        
        # cache to shared memory
        shared_a = sch.cache_read(block_outer, 0, "shared.dyn")
        shared_b = sch.cache_read(block_outer, 1, "shared.dyn")
        sch.compute_at(shared_a, k0)
        sch.compute_at(shared_b, k0)
        self.cooperative_fetch(sch, shared_a, config, -2, is_transpose_a)
        self.cooperative_fetch(sch, shared_b, config, -2, is_transpose_b)
        
        # cache to register
        reg_a = sch.cache_read(block_outer, 0, "warp")
        reg_b = sch.cache_read(block_outer, 1, "warp")
        sch.compute_at(reg_a, k1)
        sch.compute_at(reg_b, k1)
        
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

        output_blocks = [sch.get(block) for block in sch.get_output_blocks(root_block)]
        cache_write_required = True if (sch.get(main_block) not in output_blocks or len(self.func_info.dynamic_args)>1) else False
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
        # Write to register, and then smem
        def store_output(block_outer, write_buffer_idx):
            #Write to register
            block_write_reg = sch.cache_write(block_outer, write_buffer_idx, "warp")
            # 1) Write to shared memory
            if cache_write_required:
                block_write_smem = sch.cache_write(block_outer, write_buffer_idx, "shared.dyn")
                sch.reverse_compute_at(block_write_smem, block_axis)
                auto_inline_consumer_chain(sch, block_write_smem)
                # bind loops
                fused = sch.fuse(*sch.get_loops(block_write_smem)[-2:])
                f0, f1, f2 = sch.split(fused, [None, config.warp_size, config.out_vec_len])
                sch.bind(f1, "threadIdx.x")
                sch.vectorize(f2)
            else:
                auto_inline_consumer_chain(sch, block_write_reg)
            # 2)
            # bind loops
            v0, v1, v2 = sch.get_loops(block_write_reg)[-3:]
            v11, v12, v13 = sch.split(v1, factors=[config.thread_z, None, config.micro_size_m])
            v21, v22, v23 = sch.split(v2, factors=[config.thread_y, None, config.micro_size_n])
            sch.reorder(v11, v21, v12, v22, v13, v23)
            sch.bind(v11, "threadIdx.z")
            sch.bind(v21, "threadIdx.y")

            # reorder write axis to match the layout of ldmatrix
            sch.transform_layout(
                block_write_reg,
                ("read", 0),
                lambda v0, v1, v2: (
                    v0,
                    v1 // config.micro_size_m,
                    v2 // config.micro_size_n,
                    *shared_16x16_to_mma_32x8_layout(v1 % config.micro_size_m, v2 % config.micro_size_n),
                ),
            )
            return  block_write_reg

        block_write_reg = store_output(block_outer, 0)

        # Step 5. Schedule tensor core computation
        block_init = sch.decompose_reduction(block_outer, k0)
        block_init_inner = sch.get_child_blocks(block_init)[0]

        def tensorize_load():
            sch.tensorize(sch.get_loops(block_read_reg_a)[-2], intrin_group["load_a"])
            sch.tensorize(sch.get_loops(block_read_reg_b)[-2], intrin_group["load_b"])
            
        def tensorize_init_store_compute():
            sch.tensorize(sch.get_loops(block_init_inner)[-2], intrin_group["init"])
            sch.tensorize(sch.get_loops(block_inner)[-3], intrin_group["compute"])
            sch.tensorize(sch.get_loops(block_write_reg)[-2], intrin_group["store"])
        
        try:
            tensorize_load()
        except:
            if get_log_level()>=1: debug_info("failed to tensorize load")
            
        try:
            tensorize_init_store_compute()
        except:
            if get_log_level()>=1: debug_info("failed to tensorize init, store, compute")
                
        # Step 6. Async pipeline
        sch.annotate(k0, ann_key="software_pipeline_stage", ann_val=[0, 0, 3])
        sch.annotate(k0, ann_key="software_pipeline_order", ann_val=[0, 1, 2])
        sch.annotate(k0, ann_key="software_pipeline_async_stages", ann_val=[0])

        # Step 7. Handle dequantize block
        # Now we just add a dummy kernel to compute dequantize
        # if dequantize_block is not None:
        #     auto_inline_producers(sch, dequantize_block)
        #     loops = sch.get_loops(dequantize_block)
        #     loop = sch.fuse(*loops)
        #     v0, v1, v2, v3 = sch.split(loop, [None, 128, 2, 4])
        #     sch.bind(v0, "blockIdx.x")
        #     sch.bind(v1, "threadIdx.x")
        #     sch.unroll(v2)
        #     sch.vectorize(v3)
        save_to_file("/home/weitao/XIAG8XX/profile/mma/mma.py", sch)
        return sch

 