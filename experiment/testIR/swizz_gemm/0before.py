# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(p_lv9: T.handle, lv3: T.Buffer((T.int64(2560), T.int64(10240)), "float16"), lv4: T.Buffer((T.int64(2560),), "float16"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv9 = T.match_buffer(p_lv9, (T.int64(1), n, T.int64(10240)), "float16")
        var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        for blockIdx_z in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for blockIdx_x in T.thread_binding((n + T.int64(127)) // T.int64(128), thread="blockIdx.x"):
                for blockIdx_y in T.thread_binding(T.int64(20), thread="blockIdx.y"):
                    for threadIdx_x in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                        for threadIdx_y in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                            with T.block(""):
                                T.reads(lv9[T.int64(0), blockIdx_x * T.int64(128):blockIdx_x * T.int64(128) + T.int64(128), T.int64(0):T.int64(10240)], lv3[blockIdx_y * T.int64(128):blockIdx_y * T.int64(128) + T.int64(128), T.int64(0):T.int64(10240)], lv4[blockIdx_y * T.int64(128) + threadIdx_y // T.int64(2) * T.int64(64):blockIdx_y * T.int64(128) + threadIdx_y // T.int64(2) * T.int64(64) + T.int64(64)])
                                T.writes(var_T_add_intermediate[T.int64(0), blockIdx_x * T.int64(128) + threadIdx_y % T.int64(2) * T.int64(64):blockIdx_x * T.int64(128) + threadIdx_y % T.int64(2) * T.int64(64) + T.int64(64), blockIdx_y * T.int64(128) + threadIdx_y // T.int64(2) * T.int64(64):blockIdx_y * T.int64(128) + threadIdx_y // T.int64(2) * T.int64(64) + T.int64(64)])
                                var_NT_matmul_intermediate_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(128)), "float16", scope="shared.dyn")
                                var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(64)), "float16", scope="wmma.accumulator")
                                for ax1_0_3_init, ax2_0_3_init in T.grid(T.int64(4), T.int64(4)):
                                    with T.block("NT_matmul_o_init"):
                                        T.reads()
                                        T.writes(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), ax1_0_3_init * T.int64(16):ax1_0_3_init * T.int64(16) + T.int64(16), ax2_0_3_init * T.int64(16):ax2_0_3_init * T.int64(16) + T.int64(16)])
                                        with T.block("NT_matmul_init_o"):
                                            T.reads()
                                            T.writes(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), ax1_0_3_init * T.int64(16):ax1_0_3_init * T.int64(16) + T.int64(16), ax2_0_3_init * T.int64(16):ax2_0_3_init * T.int64(16) + T.int64(16)])
                                            T.tvm_fill_fragment(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator.data, 16, 16, 16, ax1_0_3_init * T.int64(4) + ax2_0_3_init, T.float32(0))
                                for ax3_0_0 in T.serial(T.int64(160), annotations={"software_pipeline_order": [0, 3, 1, 4, 5, 2, 6], "software_pipeline_stage": [0, 0, 0, 0, 0, 1, 1]}):
                                    with T.block(""):
                                        T.reads(lv9[T.int64(0), blockIdx_x * T.int64(128):blockIdx_x * T.int64(128) + T.int64(128), ax3_0_0 * T.int64(64):ax3_0_0 * T.int64(64) + T.int64(64)], lv3[blockIdx_y * T.int64(128):blockIdx_y * T.int64(128) + T.int64(128), ax3_0_0 * T.int64(64):ax3_0_0 * T.int64(64) + T.int64(64)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), T.int64(0):T.int64(64), T.int64(0):T.int64(64)])
                                        T.writes(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), T.int64(0):T.int64(64), T.int64(0):T.int64(64)])
                                        lv9_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(64)), "float16", scope="shared.dyn")
                                        lv3_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(64)), "float16", scope="shared.dyn")
                                        lv9_reindex_pad_shared_dyn_local = T.alloc_buffer((T.int64(1), T.int64(8), T.int64(8)), "float16", scope="local")
                                        lv3_reindex_shared_dyn_local = T.alloc_buffer((T.int64(1), T.int64(8), T.int64(8)), "float16", scope="local")
                                        for ax0_1, ax1_ax2_fused_0_0_0 in T.grid(T.int64(1), T.int64(8)):
                                            for ax1_ax2_fused_1 in T.vectorized(T.int64(8), annotations={"check_vector_load": 1, "remove_vector_condition": 1}):
                                                with T.block(""):
                                                    T.reads(lv9[T.int64(0), blockIdx_x * T.int64(128) + ax1_ax2_fused_0_0_0 * T.int64(16) + threadIdx_y * T.int64(4) + threadIdx_x // T.int64(8), ax3_0_0 * T.int64(64) + threadIdx_x % T.int64(8) * T.int64(8) + ax1_ax2_fused_1])
                                                    T.writes(lv9_reindex_pad_shared_dyn_local[T.int64(0), ax1_ax2_fused_0_0_0, ax1_ax2_fused_1])
                                                    lv9_reindex_pad_shared_dyn_local[T.int64(0), ax1_ax2_fused_0_0_0, ax1_ax2_fused_1] = T.if_then_else(blockIdx_x * T.int64(128) + ax1_ax2_fused_0_0_0 * T.int64(16) + threadIdx_y * T.int64(4) + threadIdx_x // T.int64(8) < n, lv9[T.int64(0), blockIdx_x * T.int64(128) + ax1_ax2_fused_0_0_0 * T.int64(16) + threadIdx_y * T.int64(4) + threadIdx_x // T.int64(8), ax3_0_0 * T.int64(64) + threadIdx_x % T.int64(8) * T.int64(8) + ax1_ax2_fused_1], T.float16(0))
                                        for ax0_1, ax1_ax2_fused_0_0_0 in T.grid(T.int64(1), T.int64(8)):
                                            for ax1_ax2_fused_1 in T.vectorized(T.int64(8), annotations={"check_vector_load": 1, "remove_vector_condition": 1}):
                                                with T.block("lv9_reindex_pad_shared.dyn"):
                                                    T.reads(lv9[T.int64(0), blockIdx_x * T.int64(128) + ax1_ax2_fused_0_0_0 * T.int64(16) + threadIdx_y * T.int64(4) + threadIdx_x // T.int64(8), ax3_0_0 * T.int64(64) + threadIdx_x % T.int64(8) * T.int64(8) + ax1_ax2_fused_1])
                                                    T.writes(lv9_reindex_pad_shared_dyn[T.int64(0), ax1_ax2_fused_0_0_0 * T.int64(16) + threadIdx_y * T.int64(4) + threadIdx_x // T.int64(8), threadIdx_x % T.int64(8) * T.int64(8) + ax1_ax2_fused_1])
                                                    T.block_attr({"double_buffer_scope": 0, "permuted_layout": 1})
                                                    lv9_reindex_pad_shared_dyn[T.int64(0), ax1_ax2_fused_0_0_0 * T.int64(16) + threadIdx_y * T.int64(4) + threadIdx_x // T.int64(8), threadIdx_x % T.int64(8) * T.int64(8) + ax1_ax2_fused_1] = lv9_reindex_pad_shared_dyn_local[T.int64(0), ax1_ax2_fused_0_0_0, ax1_ax2_fused_1]
                                        for ax0_1, ax1_ax2_fused_0_0_0 in T.grid(T.int64(1), T.int64(8)):
                                            for ax1_ax2_fused_1 in T.vectorized(T.int64(8), annotations={"check_vector_load": 1, "remove_vector_condition": 1}):
                                                with T.block(""):
                                                    T.reads(lv3[blockIdx_y * T.int64(128) + ax1_ax2_fused_0_0_0 * T.int64(16) + threadIdx_y * T.int64(4) + threadIdx_x // T.int64(8), ax3_0_0 * T.int64(64) + threadIdx_x % T.int64(8) * T.int64(8) + ax1_ax2_fused_1])
                                                    T.writes(lv3_reindex_shared_dyn_local[T.int64(0), ax1_ax2_fused_0_0_0, ax1_ax2_fused_1])
                                                    lv3_reindex_shared_dyn_local[T.int64(0), ax1_ax2_fused_0_0_0, ax1_ax2_fused_1] = lv3[blockIdx_y * T.int64(128) + ax1_ax2_fused_0_0_0 * T.int64(16) + threadIdx_y * T.int64(4) + threadIdx_x // T.int64(8), ax3_0_0 * T.int64(64) + threadIdx_x % T.int64(8) * T.int64(8) + ax1_ax2_fused_1]
                                        for ax0_1, ax1_ax2_fused_0_0_0 in T.grid(T.int64(1), T.int64(8)):
                                            for ax1_ax2_fused_1 in T.vectorized(T.int64(8), annotations={"check_vector_load": 1, "remove_vector_condition": 1}):
                                                with T.block("lv3_reindex_shared.dyn"):
                                                    T.reads(lv3[blockIdx_y * T.int64(128) + ax1_ax2_fused_0_0_0 * T.int64(16) + threadIdx_y * T.int64(4) + threadIdx_x // T.int64(8), ax3_0_0 * T.int64(64) + threadIdx_x % T.int64(8) * T.int64(8) + ax1_ax2_fused_1])
                                                    T.writes(lv3_reindex_shared_dyn[T.int64(0), ax1_ax2_fused_0_0_0 * T.int64(16) + threadIdx_y * T.int64(4) + threadIdx_x // T.int64(8), threadIdx_x % T.int64(8) * T.int64(8) + ax1_ax2_fused_1])
                                                    T.block_attr({"double_buffer_scope": 0, "permuted_layout": 1})
                                                    lv3_reindex_shared_dyn[T.int64(0), ax1_ax2_fused_0_0_0 * T.int64(16) + threadIdx_y * T.int64(4) + threadIdx_x // T.int64(8), threadIdx_x % T.int64(8) * T.int64(8) + ax1_ax2_fused_1] = lv3_reindex_shared_dyn_local[T.int64(0), ax1_ax2_fused_0_0_0, ax1_ax2_fused_1]
                                        for ax3_0_1 in T.serial(T.int64(4), annotations={"software_pipeline_order": [0, 1, 2], "software_pipeline_stage": [0, 0, 1]}):
                                            with T.block(""):
                                                T.reads(lv9_reindex_pad_shared_dyn[T.int64(0), threadIdx_y % T.int64(2) * T.int64(64):threadIdx_y % T.int64(2) * T.int64(64) + T.int64(64), ax3_0_1 * T.int64(16):ax3_0_1 * T.int64(16) + T.int64(16)], lv3_reindex_shared_dyn[T.int64(0), threadIdx_y // T.int64(2) * T.int64(64):threadIdx_y // T.int64(2) * T.int64(64) + T.int64(64), ax3_0_1 * T.int64(16):ax3_0_1 * T.int64(16) + T.int64(16)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), T.int64(0):T.int64(64), T.int64(0):T.int64(64)])
                                                T.writes(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), T.int64(0):T.int64(64), T.int64(0):T.int64(64)])
                                                lv9_reindex_pad_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(16)), "float16", scope="wmma.matrix_a")
                                                lv3_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(16)), "float16", scope="wmma.matrix_b")
                                                for ax0_0 in T.unroll(T.int64(4)):
                                                    for ax1_0 in T.unroll(T.int64(1)):
                                                        with T.block("lv9_reindex_pad_shared.dyn_wmma.matrix_a_o"):
                                                            T.reads(lv9_reindex_pad_shared_dyn[T.int64(0), threadIdx_y % T.int64(2) * T.int64(64) + ax0_0 * T.int64(16):threadIdx_y % T.int64(2) * T.int64(64) + ax0_0 * T.int64(16) + T.int64(16), ax3_0_1 * T.int64(16):ax3_0_1 * T.int64(16) + T.int64(16)])
                                                            T.writes(lv9_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), ax0_0 * T.int64(16):ax0_0 * T.int64(16) + T.int64(16), T.int64(0):T.int64(16)])
                                                            T.tvm_load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a.data, 16, 16, 16, ax0_0, T.tvm_access_ptr(T.type_annotation("float16"), lv9_reindex_pad_shared_dyn.data, threadIdx_y % T.int64(2) * T.int64(4096) + ax0_0 * T.int64(1024) + ax3_0_1 * T.int64(16), T.int64(1024), 1), T.int64(64), "row_major")
                                                for ax0_0 in T.unroll(T.int64(4)):
                                                    for ax1_0 in T.unroll(T.int64(1)):
                                                        with T.block("lv3_reindex_shared.dyn_wmma.matrix_b_o"):
                                                            T.reads(lv3_reindex_shared_dyn[T.int64(0), threadIdx_y // T.int64(2) * T.int64(64) + ax0_0 * T.int64(16):threadIdx_y // T.int64(2) * T.int64(64) + ax0_0 * T.int64(16) + T.int64(16), ax3_0_1 * T.int64(16):ax3_0_1 * T.int64(16) + T.int64(16)])
                                                            T.writes(lv3_reindex_shared_dyn_wmma_matrix_b[T.int64(0), ax0_0 * T.int64(16):ax0_0 * T.int64(16) + T.int64(16), T.int64(0):T.int64(16)])
                                                            T.tvm_load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b.data, 16, 16, 16, ax0_0, T.tvm_access_ptr(T.type_annotation("float16"), lv3_reindex_shared_dyn.data, threadIdx_y // T.int64(2) * T.int64(4096) + ax0_0 * T.int64(1024) + ax3_0_1 * T.int64(16), T.int64(1024), 1), T.int64(64), "col_major")
                                                for ax1_0_3, ax2_0_3 in T.grid(T.int64(4), T.int64(4)):
                                                    with T.block("NT_matmul_o_update"):
                                                        T.reads(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), ax1_0_3 * T.int64(16):ax1_0_3 * T.int64(16) + T.int64(16), ax2_0_3 * T.int64(16):ax2_0_3 * T.int64(16) + T.int64(16)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), ax1_0_3 * T.int64(16):ax1_0_3 * T.int64(16) + T.int64(16), T.int64(0):T.int64(16)], lv3_reindex_shared_dyn_wmma_matrix_b[T.int64(0), ax2_0_3 * T.int64(16):ax2_0_3 * T.int64(16) + T.int64(16), T.int64(0):T.int64(16)])
                                                        T.writes(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), ax1_0_3 * T.int64(16):ax1_0_3 * T.int64(16) + T.int64(16), ax2_0_3 * T.int64(16):ax2_0_3 * T.int64(16) + T.int64(16)])
                                                        with T.block("NT_matmul_o"):
                                                            T.reads(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), ax1_0_3 * T.int64(16):ax1_0_3 * T.int64(16) + T.int64(16), ax2_0_3 * T.int64(16):ax2_0_3 * T.int64(16) + T.int64(16)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), ax1_0_3 * T.int64(16):ax1_0_3 * T.int64(16) + T.int64(16), T.int64(0):T.int64(16)], lv3_reindex_shared_dyn_wmma_matrix_b[T.int64(0), ax2_0_3 * T.int64(16):ax2_0_3 * T.int64(16) + T.int64(16), T.int64(0):T.int64(16)])
                                                            T.writes(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), ax1_0_3 * T.int64(16):ax1_0_3 * T.int64(16) + T.int64(16), ax2_0_3 * T.int64(16):ax2_0_3 * T.int64(16) + T.int64(16)])
                                                            T.tvm_mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator.data, ax1_0_3 * T.int64(4) + ax2_0_3, lv9_reindex_pad_shared_dyn_wmma_matrix_a.data, ax1_0_3, lv3_reindex_shared_dyn_wmma_matrix_b.data, ax2_0_3, var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator.data, ax1_0_3 * T.int64(4) + ax2_0_3)
                                for ax0_0, ax1_0 in T.grid(T.int64(4), T.int64(4)):
                                    with T.block("var_NT_matmul_intermediate_reindex_pad_shared.dyn_wmma.accumulator_o"):
                                        T.reads(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), ax0_0 * T.int64(16):ax0_0 * T.int64(16) + T.int64(16), ax1_0 * T.int64(16):ax1_0 * T.int64(16) + T.int64(16)])
                                        T.writes(var_NT_matmul_intermediate_reindex_pad_shared_dyn[T.int64(0), threadIdx_y % T.int64(2) * T.int64(64) + ax0_0 * T.int64(16):threadIdx_y % T.int64(2) * T.int64(64) + ax0_0 * T.int64(16) + T.int64(16), threadIdx_y // T.int64(2) * T.int64(64) + ax1_0 * T.int64(16):threadIdx_y // T.int64(2) * T.int64(64) + ax1_0 * T.int64(16) + T.int64(16)])
                                        T.tvm_store_matrix_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator.data, 16, 16, 16, ax0_0 * T.int64(4) + ax1_0, T.tvm_access_ptr(T.type_annotation("float16"), var_NT_matmul_intermediate_reindex_pad_shared_dyn.data, threadIdx_y % T.int64(2) * T.int64(8192) + ax0_0 * T.int64(2048) + threadIdx_y // T.int64(2) * T.int64(64) + ax1_0 * T.int64(16), T.int64(2048), 2), T.int64(128), "row_major")
                                for ax0_ax1_fused_0 in range(T.int64(16)):
                                    for ax0_ax1_fused_2 in T.vectorized(T.int64(8)):
                                        with T.block("var_NT_matmul_intermediate_reindex_pad_shared.dyn"):
                                            T.reads(var_NT_matmul_intermediate_reindex_pad_shared_dyn[T.int64(0), threadIdx_y % T.int64(2) * T.int64(64) + ax0_ax1_fused_0 * T.int64(4) + threadIdx_x // T.int64(8), threadIdx_y // T.int64(2) * T.int64(64) + threadIdx_x % T.int64(8) * T.int64(8) + ax0_ax1_fused_2], lv4[blockIdx_y * T.int64(128) + threadIdx_y // T.int64(2) * T.int64(64) + threadIdx_x % T.int64(8) * T.int64(8) + ax0_ax1_fused_2])
                                            T.writes(var_T_add_intermediate[T.int64(0), blockIdx_x * T.int64(128) + threadIdx_y % T.int64(2) * T.int64(64) + ax0_ax1_fused_0 * T.int64(4) + threadIdx_x // T.int64(8), blockIdx_y * T.int64(128) + threadIdx_y // T.int64(2) * T.int64(64) + threadIdx_x % T.int64(8) * T.int64(8) + ax0_ax1_fused_2])
                                            T.block_attr({"buffer_dim_align": [[0, 1, 16, 4]]})
                                            if blockIdx_x * T.int64(128) + threadIdx_y % T.int64(2) * T.int64(64) + ax0_ax1_fused_0 * T.int64(4) + threadIdx_x // T.int64(8) < n:
                                                var_T_add_intermediate[T.int64(0), blockIdx_x * T.int64(128) + threadIdx_y % T.int64(2) * T.int64(64) + ax0_ax1_fused_0 * T.int64(4) + threadIdx_x // T.int64(8), blockIdx_y * T.int64(128) + threadIdx_y // T.int64(2) * T.int64(64) + threadIdx_x % T.int64(8) * T.int64(8) + ax0_ax1_fused_2] = var_NT_matmul_intermediate_reindex_pad_shared_dyn[T.int64(0), threadIdx_y % T.int64(2) * T.int64(64) + ax0_ax1_fused_0 * T.int64(4) + threadIdx_x // T.int64(8), threadIdx_y // T.int64(2) * T.int64(64) + threadIdx_x % T.int64(8) * T.int64(8) + ax0_ax1_fused_2] + lv4[blockIdx_y * T.int64(128) + threadIdx_y // T.int64(2) * T.int64(64) + threadIdx_x % T.int64(8) * T.int64(8) + ax0_ax1_fused_2]