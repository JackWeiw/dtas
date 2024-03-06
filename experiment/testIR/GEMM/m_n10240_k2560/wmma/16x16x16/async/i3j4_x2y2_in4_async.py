from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
dtype="float16"
@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv55: T.handle, lv11: T.Buffer((T.int64(10240), T.int64(2560)), "float16"), lv12: T.Buffer((T.int64(10240),), "float16"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv55 = T.match_buffer(p_lv55, (T.int64(1), n, T.int64(2560)), "float16")
        var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(10240)), "float16")
        # with T.block("root"):
        lv55_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(1), (n + T.int64(95)) // T.int64(96) * T.int64(96), T.int64(2560)), "float16", scope="shared.dyn")
        lv11_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(10240), T.int64(2560)), "float16", scope="shared.dyn")
        lv55_reindex_pad_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(1), (n + T.int64(95)) // T.int64(96) * T.int64(96), T.int64(2560)), "float16", scope="wmma.matrix_a")
        lv11_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(1), T.int64(10240), T.int64(2560)), "float16", scope="wmma.matrix_b")
        var_NT_matmul_intermediate_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(1), (n + T.int64(95)) // T.int64(96) * T.int64(96), T.int64(10240)), "float16", scope="shared.dyn")
        var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(1), (n + T.int64(95)) // T.int64(96) * T.int64(96), T.int64(10240)), "float16", scope="wmma.accumulator")
        for ax0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for ax1_0_0_ax2_0_0_fused in T.thread_binding((n + T.int64(95)) // T.int64(96), thread="blockIdx.x"):
                for ax1_0_1_ax2_0_1_fused in T.thread_binding(T.int64(80), thread="blockIdx.y"):
                    for ax2_0_2_ax1_0_2_fused in T.thread_binding(T.int64(12), thread="threadIdx.y"):
                        for ax1_0_3_init, ax2_0_3_init in T.grid(T.int64(2), T.int64(2)):
                            with T.block("NT_matmul_o_init"):
                                v0_o = T.axis.spatial(T.int64(1), ax0)
                                v1_o = T.axis.spatial((n + T.int64(95)) // T.int64(96) * T.int64(6), ax1_0_0_ax2_0_0_fused * T.int64(6) + ax2_0_2_ax1_0_2_fused % T.int64(3) * T.int64(2) + ax1_0_3_init)
                                v2_o = T.axis.spatial(T.int64(640), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(3) * T.int64(2) + ax2_0_3_init)
                                T.reads()
                                T.writes(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                with T.block("NT_matmul_init_o"):
                                    v1_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v2_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads()
                                    T.writes(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    C = T.match_buffer(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_fill_fragment(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.float32(0))
                        for ax3_0_0 in T.serial(T.int64(40), annotations={"software_pipeline_async_stages": [0], "software_pipeline_order": [0, 1, 3, 2, 4], "software_pipeline_stage": [0, 0, 1, 2, 2]}):
                            for ax0_1, ax1_ax2_fused_0_0_0 in T.grid(T.int64(1), T.int64(4)):
                                for ax1_ax2_fused_0_0_1 in T.thread_binding(T.int64(12), thread="threadIdx.y"):
                                    for ax1_ax2_fused_0_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        for ax1_ax2_fused_1 in T.vectorized(T.int64(4), annotations={"check_vector_load": 1, "remove_vector_condition": 1}):
                                            with T.block("lv55_reindex_pad_shared.dyn"):
                                                v0 = T.axis.spatial(T.int64(1), ax0_1)
                                                v1 = T.axis.spatial((n + T.int64(95)) // T.int64(96) * T.int64(96), ax1_0_0_ax2_0_0_fused * T.int64(96) + (ax1_ax2_fused_0_0_0 * T.int64(1536) + ax1_ax2_fused_0_0_1 * T.int64(128) + ax1_ax2_fused_0_1 * T.int64(4) + ax1_ax2_fused_1) // T.int64(64))
                                                v2 = T.axis.spatial(T.int64(2560), ax3_0_0 * T.int64(64) + (ax1_ax2_fused_0_0_0 * T.int64(1536) + ax1_ax2_fused_0_0_1 * T.int64(128) + ax1_ax2_fused_0_1 * T.int64(4) + ax1_ax2_fused_1) % T.int64(64))
                                                T.reads(lv55[T.int64(0), v1, v2])
                                                T.writes(lv55_reindex_pad_shared_dyn[v0, v1, v2])
                                                T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]]})
                                                lv55_reindex_pad_shared_dyn[v0, v1, v2] = T.if_then_else(v1 < n, lv55[T.int64(0), v1, v2], T.float16(0))
                            for ax0_1, ax1_ax2_fused_0_0_0 in T.grid(T.int64(1), T.int64(6)):
                                for ax1_ax2_fused_0_0_1 in T.thread_binding(T.int64(12), thread="threadIdx.y"):
                                    for ax1_ax2_fused_0_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        for ax1_ax2_fused_1 in T.vectorized(T.int64(4), annotations={"check_vector_load": 1, "remove_vector_condition": 1}):
                                            with T.block("lv11_reindex_shared.dyn"):
                                                v0 = T.axis.spatial(T.int64(1), ax0_1)
                                                v1 = T.axis.spatial(T.int64(10240), ax1_0_1_ax2_0_1_fused * T.int64(128) + ((ax1_ax2_fused_0_0_0 * T.int64(12) + ax1_ax2_fused_0_0_1) * T.int64(128) + ax1_ax2_fused_0_1 * T.int64(4) + ax1_ax2_fused_1) // T.int64(64))
                                                v2 = T.axis.spatial(T.int64(2560), ax3_0_0 * T.int64(64) + ((ax1_ax2_fused_0_0_0 * T.int64(12) + ax1_ax2_fused_0_0_1) * T.int64(128) + ax1_ax2_fused_0_1 * T.int64(4) + ax1_ax2_fused_1) % T.int64(64))
                                                T.where(ax1_ax2_fused_0_0_0 * T.int64(12) + ax1_ax2_fused_0_0_1 < T.int64(64))
                                                T.reads(lv11[v1, v2])
                                                T.writes(lv11_reindex_shared_dyn[v0, v1, v2])
                                                T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]]})
                                                lv11_reindex_shared_dyn[v0, v1, v2] = lv11[v1, v2]
                            for ax3_0_1 in T.serial(T.int64(4), annotations={"software_pipeline_order": [0, 1, 2], "software_pipeline_stage": [0, 0, 1]}):
                                for ax0_0 in T.unroll(T.int64(2)):
                                    for ax1_0 in T.unroll(T.int64(1)):
                                        with T.block("lv55_reindex_pad_shared.dyn_wmma.matrix_a_o"):
                                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1_o = T.axis.spatial(T.int64(6) * ((n + T.int64(95)) // T.int64(96)), ax1_0_0_ax2_0_0_fused * T.int64(6) + ax2_0_2_ax1_0_2_fused % T.int64(3) * T.int64(2) + ax0_0)
                                            v2_o = T.axis.spatial(T.int64(160), ax3_0_0 * T.int64(4) + ax3_0_1 + ax1_0)
                                            T.reads(lv55_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            T.writes(lv55_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A = T.match_buffer(lv55_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                            C = T.match_buffer(lv55_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                            T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(16), 1), A.strides[0], "row_major")
                                for ax0_0 in T.unroll(T.int64(2)):
                                    for ax1_0 in T.unroll(T.int64(1)):
                                        with T.block("lv11_reindex_shared.dyn_wmma.matrix_b_o"):
                                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1_o = T.axis.spatial(T.int64(640), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(3) * T.int64(2) + ax0_0)
                                            v2_o = T.axis.spatial(T.int64(160), ax3_0_0 * T.int64(4) + ax3_0_1 + ax1_0)
                                            T.reads(lv11_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            T.writes(lv11_reindex_shared_dyn_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A = T.match_buffer(lv11_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                            C = T.match_buffer(lv11_reindex_shared_dyn_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                            T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(16), 1), A.strides[0], "col_major")
                                for ax1_0_3, ax2_0_3 in T.grid(T.int64(2), T.int64(2)):
                                    with T.block("NT_matmul_o_update"):
                                        v0_o = T.axis.spatial(T.int64(1), ax0)
                                        v1_o = T.axis.spatial((n + T.int64(95)) // T.int64(96) * T.int64(6), ax1_0_0_ax2_0_0_fused * T.int64(6) + ax2_0_2_ax1_0_2_fused % T.int64(3) * T.int64(2) + ax1_0_3)
                                        v2_o = T.axis.spatial(T.int64(640), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(3) * T.int64(2) + ax2_0_3)
                                        v3_o = T.axis.reduce(T.int64(160), ax3_0_0 * T.int64(4) + ax3_0_1)
                                        T.reads(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], lv55_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], lv11_reindex_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                        T.writes(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                        with T.block("NT_matmul_o"):
                                            v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v3_i_o = T.axis.reduce(T.int64(1), T.int64(0))
                                            T.reads(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], lv55_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], lv11_reindex_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                            T.writes(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A = T.match_buffer(lv55_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                            B = T.match_buffer(lv11_reindex_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                            C = T.match_buffer(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                            T.tvm_mma_sync(C.data, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), A.data, A.elem_offset // A.strides[0] // T.int64(16) * (A.strides[0] // T.int64(16)) + A.elem_offset % A.strides[0] // T.int64(16), B.data, B.elem_offset // B.strides[0] // T.int64(16) * (B.strides[0] // T.int64(16)) + B.elem_offset % B.strides[0] // T.int64(16), C.data, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16))
                        for ax0_0, ax1_0 in T.grid(T.int64(2), T.int64(2)):
                            with T.block("var_NT_matmul_intermediate_reindex_pad_shared.dyn_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v1_o = T.axis.spatial(T.int64(6) * ((n + T.int64(95)) // T.int64(96)), ax1_0_0_ax2_0_0_fused * T.int64(6) + ax2_0_2_ax1_0_2_fused % T.int64(3) * T.int64(2) + ax0_0)
                                v2_o = T.axis.spatial(T.int64(640), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(3) * T.int64(2) + ax1_0)
                                T.reads(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                T.writes(var_NT_matmul_intermediate_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                A = T.match_buffer(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=16)
                                C = T.match_buffer(var_NT_matmul_intermediate_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=16)
                                T.tvm_store_matrix_sync(A.data, 16, 16, 16, A.elem_offset // A.strides[0] // T.int64(16) * (A.strides[0] // T.int64(16)) + A.elem_offset % A.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), C.data, C.elem_offset, C.strides[0] * T.int64(16), 2), C.strides[0], "row_major")
                        for ax0_ax1_fused_0 in range(T.int64(8)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("var_NT_matmul_intermediate_reindex_pad_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial((n + T.int64(95)) // T.int64(96) * T.int64(96), ax1_0_0_ax2_0_0_fused * T.int64(96) + ax2_0_2_ax1_0_2_fused % T.int64(3) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) // T.int64(32))
                                        v2 = T.axis.spatial(T.int64(10240), ax1_0_1_ax2_0_1_fused * T.int64(128) + ax2_0_2_ax1_0_2_fused // T.int64(3) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) % T.int64(32))
                                        T.reads(var_NT_matmul_intermediate_reindex_pad_shared_dyn[v0, v1, v2], lv12[v2])
                                        T.writes(var_T_add_intermediate[T.int64(0), v1, v2])
                                        T.block_attr({"buffer_dim_align": [[0, 1, 16, 4]]})
                                        if v1 < n:
                                            var_T_add_intermediate[T.int64(0), v1, v2] = var_NT_matmul_intermediate_reindex_pad_shared_dyn[v0, v1, v2] + lv12[v2]
                                            
    @R.function
    def WT_test(A: R.Tensor((1, "n", 2560), dtype=dtype), w_q: R.Tensor((10240, 2560), dtype=dtype), bias_q: R.Tensor((10240,), dtype=dtype)) -> R.Tensor((1, "n", 10240), dtype=dtype):
        n=T.int64()
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.main, (A, w_q, bias_q), out_sinfo=R.Tensor((1, n, 10240), dtype=dtype))
            gv: R.Tensor((1, n, 10240), dtype=dtype) = lv
            R.output(gv)
        return gv

mod=Module

import tvm
from tvm import dlight as dl
from tvm import tir
import numpy as np
import tvm.relax as rx
# mod=rx.get_pipeline()(mod)
target = tvm.target.Target("nvidia/geforce-rtx-3090")
dtype="float16"
dev=tvm.cuda(7)

# mod=rx.transform.AttachGlobalSymbol()(mod)    
# lib=tvm.build(mod,target=target)
with tvm.transform.PassContext(disabled_pass=["tir.AnnotateEntryFunc"],config={"tir.use_async_copy": True,}):
    exe=rx.build(mod, target=target)
    vm = rx.VirtualMachine(exe, dev,profile=True)

hidden_size=2560
result={}
import json
# m=[128,300,512,768,1024,2048,3584,4096]
for i in range(4096,4097):    
    x = tvm.nd.array(np.random.uniform(0, 1, (1, i, hidden_size)).astype(dtype), dev)
    weight = tvm.nd.array(np.random.uniform(0, 1,(4*hidden_size, hidden_size)).astype(dtype), dev)
    bias = tvm.nd.array(np.random.uniform(0, 1,(4*hidden_size,)).astype(dtype), dev)
    s=vm.module["profile"]("WT_test",x, weight, bias)
    dic=json.loads(s)
    for j in range(len(dic["calls"])):
        if dic["calls"][j]["Name"]["string"]=="main":
           result[i]=(dic["calls"][j]["Duration (us)"]["microseconds"])
    # print(vm.profile("MyT",a))
    del x,weight,bias

def save_to_json(profile, filename):
    import json
    with open(filename, "w") as f:
        f.write(json.dumps(profile,indent=4))
        
save_to_json(result,"/home/weitao/XIAG8XX/profile/testIR/m_n10240_k2560/data/16x16x16/i3j4_x2y2_in4_async.json")  