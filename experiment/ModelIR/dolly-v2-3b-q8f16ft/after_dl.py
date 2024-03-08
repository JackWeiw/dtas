from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def cast(A: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), compute: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0_fused_0 in T.thread_binding(T.int64(3), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(2560), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(2560))
                    T.reads(A[T.int64(0), T.int64(0), v0])
                    T.writes(compute[T.int64(0), T.int64(0), v0])
                    compute[T.int64(0), T.int64(0), v0] = T.Cast("float32", A[T.int64(0), T.int64(0), v0])

    @T.prim_func(private=True)
    def cast4(A: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), compute: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0_fused_0 in T.thread_binding(T.int64(3), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(2560), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(2560))
                    T.reads(A[T.int64(0), T.int64(0), v0])
                    T.writes(compute[T.int64(0), T.int64(0), v0])
                    compute[T.int64(0), T.int64(0), v0] = A[T.int64(0), T.int64(0), v0]

    @T.prim_func(private=True)
    def cast5(var_A: T.handle, var_compute: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)), "float16")
        compute = T.match_buffer(var_compute, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding((n * T.int64(2560) + T.int64(1023)) // T.int64(1024), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(2560))
                    v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(2560))
                    T.where(ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 < n * T.int64(2560))
                    T.reads(A[T.int64(0), v0, v1])
                    T.writes(compute[T.int64(0), v0, v1])
                    compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])

    @T.prim_func(private=True)
    def divide(A: T.Buffer((T.int64(1), T.int64(1), T.int64(50280)), "float32"), B: T.Buffer((), "float32"), T_divide: T.Buffer((T.int64(1), T.int64(1), T.int64(50280)), "float32")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0_fused_0 in T.thread_binding(T.int64(50), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_divide"):
                    v0 = T.axis.spatial(T.int64(50280), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(50280))
                    T.reads(A[T.int64(0), T.int64(0), v0], B[()])
                    T.writes(T_divide[T.int64(0), T.int64(0), v0])
                    T_divide[T.int64(0), T.int64(0), v0] = A[T.int64(0), T.int64(0), v0] / B[()]

    @T.prim_func(private=True)
    def extend_te(var_A: T.handle, var_concat_te: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(1), n, n), "float16")
        m = T.int64()
        concat_te = T.match_buffer(var_concat_te, (T.int64(1), T.int64(1), n, m), "float16")
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding((n * m + T.int64(1023)) // T.int64(1024), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("concat_te"):
                    v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % (m * n) // m)
                    v1 = T.axis.spatial(m, (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % m)
                    T.where(ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 < n * m)
                    T.reads(A[T.int64(0), T.int64(0), v0, v1 + (n - m)])
                    T.writes(concat_te[T.int64(0), T.int64(0), v0, v1])
                    concat_te[T.int64(0), T.int64(0), v0, v1] = T.if_then_else(v1 < m - n, T.float16(65504), A[T.int64(0), T.int64(0), v0, v1 + n - m])

    @T.prim_func(private=True)
    def full(var_T_full: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        m = T.int64()
        T_full = T.match_buffer(var_T_full, (T.int64(1), T.int64(1), T.int64(1), m), "float16")
        # with T.block("root"):
        for ax0_fused_0 in T.thread_binding((m + T.int64(1023)) // T.int64(1024), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_full"):
                    v0 = T.axis.spatial(m, ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < m)
                    T.reads()
                    T.writes(T_full[T.int64(0), T.int64(0), T.int64(0), v0])
                    T_full[T.int64(0), T.int64(0), T.int64(0), v0] = T.float16(65504)

    @T.prim_func(private=True)
    def fused_NT_matmul2_divide2_maximum1_minimum1_cast7(p_lv30: T.handle, p_lv31: T.handle, p_lv5: T.handle, p_output0: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        lv30 = T.match_buffer(p_lv30, (T.int64(1), T.int64(32), n, T.int64(80)), "float16")
        m = T.int64()
        lv31 = T.match_buffer(p_lv31, (T.int64(1), T.int64(32), m, T.int64(80)), "float16")
        lv5 = T.match_buffer(p_lv5, (T.int64(1), T.int64(1), n, m), "float16")
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m))
        # with T.block("root"):
        lv30_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(32), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(128)), "float16", scope="shared.dyn")
        lv31_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(32), (m + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(128)), "float16", scope="shared.dyn")
        lv30_reindex_pad_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(32), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(128)), "float16", scope="wmma.matrix_a")
        lv31_reindex_pad_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(32), (m + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(128)), "float16", scope="wmma.matrix_b")
        NT_matmul_intermediate_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(32), (n + T.int64(127)) // T.int64(128) * T.int64(128), (m + T.int64(127)) // T.int64(128) * T.int64(128)), "float16", scope="shared.dyn")
        NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(32), (n + T.int64(127)) // T.int64(128) * T.int64(128), (m + T.int64(127)) // T.int64(128) * T.int64(128)), "float16", scope="wmma.accumulator")
        for ax0 in T.thread_binding(T.int64(32), thread="blockIdx.z"):
            for ax1_0_0_ax2_0_0_fused in T.thread_binding((n + T.int64(127)) // T.int64(128), thread="blockIdx.x"):
                for ax1_0_1_ax2_0_1_fused in T.thread_binding((m + T.int64(127)) // T.int64(128), thread="blockIdx.y"):
                    for ax2_0_2_ax1_0_2_fused in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                        for ax1_0_3_init, ax2_0_3_init in T.grid(T.int64(2), T.int64(2)):
                            with T.block("NT_matmul_o_init"):
                                v0_o = T.axis.spatial(T.int64(32), ax0)
                                v1_o = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(8), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax1_0_3_init)
                                v2_o = T.axis.spatial((m + T.int64(127)) // T.int64(128) * T.int64(8), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax2_0_3_init)
                                T.reads()
                                T.writes(NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                with T.block("NT_matmul_init_o"):
                                    v1_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v2_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads()
                                    T.writes(NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    C = T.match_buffer(NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_fill_fragment(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.float32(0))
                        for ax3_0_0 in T.serial(T.int64(2), annotations={"software_pipeline_order": [0, 3, 1, 4, 5, 2, 6], "software_pipeline_stage": [0, 0, 0, 0, 0, 1, 1]}):
                            for ax0_ax1_fused_0 in range(T.int64(4)):
                                for ax0_ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                                            with T.block("lv30_reindex_pad_shared.dyn"):
                                                v0 = T.axis.spatial(T.int64(32), ax0)
                                                v1 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), ax1_0_0_ax2_0_0_fused * T.int64(128) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(64))
                                                v2 = T.axis.spatial(T.int64(128), ax3_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(64))
                                                T.reads(lv30[T.int64(0), v0, v1, v2])
                                                T.writes(lv30_reindex_pad_shared_dyn[v0, v1, v2])
                                                T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "double_buffer_scope": 0, "tir.manifest_shared_memory_local_stage": 1})
                                                lv30_reindex_pad_shared_dyn[v0, v1, v2] = T.if_then_else(v1 < n and v2 < T.int64(80), lv30[T.int64(0), v0, v1, v2], T.float16(0))
                            for ax0_ax1_fused_0 in range(T.int64(4)):
                                for ax0_ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                                            with T.block("lv31_reindex_pad_shared.dyn"):
                                                v0 = T.axis.spatial(T.int64(32), ax0)
                                                v1 = T.axis.spatial((m + T.int64(127)) // T.int64(128) * T.int64(128), ax1_0_1_ax2_0_1_fused * T.int64(128) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(64))
                                                v2 = T.axis.spatial(T.int64(128), ax3_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(64))
                                                T.reads(lv31[T.int64(0), v0, v1, v2])
                                                T.writes(lv31_reindex_pad_shared_dyn[v0, v1, v2])
                                                T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "double_buffer_scope": 0, "tir.manifest_shared_memory_local_stage": 1})
                                                lv31_reindex_pad_shared_dyn[v0, v1, v2] = T.if_then_else(v1 < m and v2 < T.int64(80), lv31[T.int64(0), v0, v1, v2], T.float16(0))
                            for ax3_0_1 in T.serial(T.int64(4), annotations={"software_pipeline_order": [0, 1, 2], "software_pipeline_stage": [0, 0, 1]}):
                                for ax0_0 in T.unroll(T.int64(2)):
                                    for ax1_0 in T.unroll(T.int64(1)):
                                        with T.block("lv30_reindex_pad_shared.dyn_wmma.matrix_a_o"):
                                            v0_o = T.axis.spatial(T.int64(32), ax0)
                                            v1_o = T.axis.spatial(T.int64(8) * ((n + T.int64(127)) // T.int64(128)), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax0_0)
                                            v2_o = T.axis.spatial(T.int64(8), ax3_0_0 * T.int64(4) + ax3_0_1 + ax1_0)
                                            T.reads(lv30_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            T.writes(lv30_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A = T.match_buffer(lv30_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                            C = T.match_buffer(lv30_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                            T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(16), 1), A.strides[0], "row_major")
                                for ax0_0 in T.unroll(T.int64(2)):
                                    for ax1_0 in T.unroll(T.int64(1)):
                                        with T.block("lv31_reindex_pad_shared.dyn_wmma.matrix_b_o"):
                                            v0_o = T.axis.spatial(T.int64(32), ax0)
                                            v1_o = T.axis.spatial(T.int64(8) * ((m + T.int64(127)) // T.int64(128)), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax0_0)
                                            v2_o = T.axis.spatial(T.int64(8), ax3_0_0 * T.int64(4) + ax3_0_1 + ax1_0)
                                            T.reads(lv31_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            T.writes(lv31_reindex_pad_shared_dyn_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A = T.match_buffer(lv31_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                            C = T.match_buffer(lv31_reindex_pad_shared_dyn_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                            T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(16), 1), A.strides[0], "col_major")
                                for ax1_0_3, ax2_0_3 in T.grid(T.int64(2), T.int64(2)):
                                    with T.block("NT_matmul_o_update"):
                                        v0_o = T.axis.spatial(T.int64(32), ax0)
                                        v1_o = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(8), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax1_0_3)
                                        v2_o = T.axis.spatial((m + T.int64(127)) // T.int64(128) * T.int64(8), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax2_0_3)
                                        v3_o = T.axis.reduce(T.int64(8), ax3_0_0 * T.int64(4) + ax3_0_1)
                                        T.reads(NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], lv30_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], lv31_reindex_pad_shared_dyn_wmma_matrix_b[v0_o, v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                        T.writes(NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                        with T.block("NT_matmul_o"):
                                            v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v3_i_o = T.axis.reduce(T.int64(1), T.int64(0))
                                            T.reads(NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], lv30_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], lv31_reindex_pad_shared_dyn_wmma_matrix_b[v0_o, v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                            T.writes(NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A = T.match_buffer(lv30_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                            B = T.match_buffer(lv31_reindex_pad_shared_dyn_wmma_matrix_b[v0_o, v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                            C = T.match_buffer(NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                            T.tvm_mma_sync(C.data, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), A.data, A.elem_offset // A.strides[0] // T.int64(16) * (A.strides[0] // T.int64(16)) + A.elem_offset % A.strides[0] // T.int64(16), B.data, B.elem_offset // B.strides[0] // T.int64(16) * (B.strides[0] // T.int64(16)) + B.elem_offset % B.strides[0] // T.int64(16), C.data, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16))
                        for ax0_0, ax1_0 in T.grid(T.int64(2), T.int64(2)):
                            with T.block("NT_matmul_intermediate_reindex_pad_shared.dyn_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(T.int64(32), ax0)
                                v1_o = T.axis.spatial(T.int64(8) * ((n + T.int64(127)) // T.int64(128)), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax0_0)
                                v2_o = T.axis.spatial(T.int64(8) * ((m + T.int64(127)) // T.int64(128)), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax1_0)
                                T.reads(NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                T.writes(NT_matmul_intermediate_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                A = T.match_buffer(NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=16)
                                C = T.match_buffer(NT_matmul_intermediate_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=16)
                                T.tvm_store_matrix_sync(A.data, 16, 16, 16, A.elem_offset // A.strides[0] // T.int64(16) * (A.strides[0] // T.int64(16)) + A.elem_offset % A.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), C.data, C.elem_offset, C.strides[0] * T.int64(16), 2), C.strides[0], "row_major")
                        for ax0_ax1_fused_0 in range(T.int64(8)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("NT_matmul_intermediate_reindex_pad_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(32), ax0)
                                        v1 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), ax1_0_0_ax2_0_0_fused * T.int64(128) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) // T.int64(32))
                                        v2 = T.axis.spatial((m + T.int64(127)) // T.int64(128) * T.int64(128), ax1_0_1_ax2_0_1_fused * T.int64(128) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) % T.int64(32))
                                        T.reads(NT_matmul_intermediate_reindex_pad_shared_dyn[v0, v1, v2], lv5[T.int64(0), T.int64(0), v1, v2])
                                        T.writes(compute_intermediate[T.int64(0), v0, v1, v2])
                                        T.block_attr({"buffer_dim_align": [[0, 1, 16, 4]]})
                                        if v1 < n and v2 < m:
                                            compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float32", T.min(T.max(NT_matmul_intermediate_reindex_pad_shared_dyn[v0, v1, v2] * T.float16(0.11179039301310044), T.float16(-65504)), lv5[T.int64(0), T.int64(0), v1, v2]))

    @T.prim_func(private=True)
    def fused_NT_matmul_divide1_maximum_minimum_cast2(lv1800: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(80)), "float16"), p_lv1801: T.handle, p_lv1775: T.handle, p_output0: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        m = T.int64()
        lv1801 = T.match_buffer(p_lv1801, (T.int64(1), T.int64(32), m, T.int64(80)), "float16")
        lv1775 = T.match_buffer(p_lv1775, (T.int64(1), T.int64(1), T.int64(1), m), "float16")
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), T.int64(1), m))
        # with T.block("root"):
        NT_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), m), "float16", scope="local")
        NT_matmul_intermediate_rf_local = T.alloc_buffer((T.int64(64), T.int64(1), T.int64(32), T.int64(1), m), "float16", scope="local")
        NT_matmul_intermediate_rf_local_1 = T.alloc_buffer((T.int64(64), T.int64(1), T.int64(32), T.int64(1), m), "float16", scope="local")
        lv1801_local = T.alloc_buffer((T.int64(1), T.int64(32), m, T.int64(80)), "float16", scope="local")
        lv1800_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(80)), "float16", scope="shared")
        for ax0_fused_ax1_fused_fused_0 in T.thread_binding(m * T.int64(32), thread="blockIdx.x"):
            for ax0_fused_ax1_fused_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                        for ax3_0 in T.serial(T.int64(2), annotations={"pragma_unroll_explicit": 256, "pragma_vectorize": 1}):
                            for ax3_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                                for ax3_2 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                    for ax3_3 in T.vectorized(T.int64(1)):
                                        with T.block("lv1800_shared"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(32), ax0_fused_ax1_fused_fused_0 // m + ax1)
                                            v2 = T.axis.spatial(T.int64(1), ax2)
                                            v3 = T.axis.spatial(T.int64(80), ax3_0 * T.int64(64) + ax3_1 * T.int64(64) + ax3_2 + ax3_3)
                                            T.where((ax3_0 + ax3_1) * T.int64(64) + ax3_2 + ax3_3 < T.int64(80))
                                            T.reads(lv1800[v0, v1, v2, v3])
                                            T.writes(lv1800_shared[v0, v1, v2, v3])
                                            lv1800_shared[v0, v1, v2, v3] = lv1800[v0, v1, v2, v3]
                    for ax0_fused_ax1_fused_fused_2_init in range(T.int64(1)):
                        for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1_init in T.vectorized(T.int64(1)):
                            with T.block("NT_matmul_rf_init"):
                                vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused = T.axis.spatial(T.int64(64), ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1_init)
                                v0 = T.axis.spatial(T.int64(32), (ax0_fused_ax1_fused_fused_0 + ax0_fused_ax1_fused_fused_1 + ax0_fused_ax1_fused_fused_2_init) // m)
                                v1 = T.axis.spatial(m, (ax0_fused_ax1_fused_fused_0 + ax0_fused_ax1_fused_fused_1 + ax0_fused_ax1_fused_fused_2_init) % m)
                                T.reads()
                                T.writes(NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, T.int64(0), v0, T.int64(0), v1])
                                NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, T.int64(0), v0, T.int64(0), v1] = T.float16(0)
                    for ax2_fused_u_fused_0 in T.serial(T.int64(2), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        for ax0, ax1, ax2_0, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                            for ax2_1 in T.vectorized(T.int64(1)):
                                with T.block("lv1801_local"):
                                    v0 = T.axis.spatial(T.int64(1), ax0)
                                    v1 = T.axis.spatial(T.int64(32), ax0_fused_ax1_fused_fused_0 // m + ax1)
                                    v2 = T.axis.spatial(m, ax0_fused_ax1_fused_fused_0 % m + ax2_0 + ax2_1)
                                    v3 = T.axis.spatial(T.int64(80), ax2_fused_u_fused_0 * T.int64(64) + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 + ax3)
                                    T.where(ax2_fused_u_fused_0 * T.int64(64) + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 < T.int64(80))
                                    T.reads(lv1801[v0, v1, v2, v3])
                                    T.writes(lv1801_local[v0, v1, v2, v3])
                                    lv1801_local[v0, v1, v2, v3] = lv1801[v0, v1, v2, v3]
                        for ax0_fused_ax1_fused_fused_2, ax2_fused_u_fused_2 in T.grid(T.int64(1), T.int64(1)):
                            for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1 in T.vectorized(T.int64(1)):
                                with T.block("NT_matmul_rf_update"):
                                    vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused = T.axis.spatial(T.int64(64), ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1)
                                    v0 = T.axis.spatial(T.int64(32), (ax0_fused_ax1_fused_fused_0 + ax0_fused_ax1_fused_fused_1 + ax0_fused_ax1_fused_fused_2) // m)
                                    v1 = T.axis.spatial(m, (ax0_fused_ax1_fused_fused_0 + ax0_fused_ax1_fused_fused_1 + ax0_fused_ax1_fused_fused_2) % m)
                                    vax2_fused_u_fused_0, vax2_fused_u_fused_2 = T.axis.remap("RR", [ax2_fused_u_fused_0, ax2_fused_u_fused_2])
                                    T.where(T.Add(ax2_fused_u_fused_0 * T.int64(64) + (ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1) + ax2_fused_u_fused_2, T.int64(0)) < T.int64(80))
                                    T.reads(NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, T.int64(0), v0, T.int64(0), v1], lv1800_shared[T.int64(0), v0, T.int64(0), vax2_fused_u_fused_0 * T.int64(64) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused + vax2_fused_u_fused_2], lv1801_local[T.int64(0), v0, v1, vax2_fused_u_fused_0 * T.int64(64) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused + vax2_fused_u_fused_2])
                                    T.writes(NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, T.int64(0), v0, T.int64(0), v1])
                                    NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, T.int64(0), v0, T.int64(0), v1] = NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, T.int64(0), v0, T.int64(0), v1] + lv1800_shared[T.int64(0), v0, T.int64(0), vax2_fused_u_fused_0 * T.int64(64) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused + vax2_fused_u_fused_2] * lv1801_local[T.int64(0), v0, v1, vax2_fused_u_fused_0 * T.int64(64) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused + vax2_fused_u_fused_2]
            for ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                for ax0 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                    for ax2_ax3_fused_1_0 in T.serial(T.int64(1), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        for ax2_ax3_fused_1_1 in T.vectorized(T.int64(1)):
                            with T.block("NT_matmul_rf_init"):
                                vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 = T.axis.spatial(T.int64(64), ax0)
                                v0 = T.axis.spatial(T.int64(32), ax0_fused_ax1_fused_fused_0 // m)
                                v1 = T.axis.spatial(m, ax0_fused_ax1_fused_fused_0 % m)
                                T.reads()
                                T.writes(NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, T.int64(0), v1])
                                NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, T.int64(0), v1] = T.float16(0)
                            for ax1 in range(T.int64(1)):
                                with T.block("NT_matmul_rf_update"):
                                    vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1 = T.axis.remap("SR", [ax0, ax1])
                                    v0 = T.axis.spatial(T.int64(32), ax0_fused_ax1_fused_fused_0 // m)
                                    v1 = T.axis.spatial(m, ax0_fused_ax1_fused_fused_0 % m)
                                    T.reads(NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, T.int64(0), v1], NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1, T.int64(0), v0, T.int64(0), v1])
                                    T.writes(NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, T.int64(0), v1])
                                    NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, T.int64(0), v1] = NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, T.int64(0), v1] + NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1, T.int64(0), v0, T.int64(0), v1]
            for ax1_ax2_fused_1 in range(T.int64(1)):
                for ax1_ax2_fused_0 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                    for ax0 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                        with T.block("NT_matmul"):
                            vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 = T.axis.reduce(T.int64(64), ax0)
                            v0 = T.axis.spatial(T.int64(32), ax0_fused_ax1_fused_fused_0 // m)
                            v1 = T.axis.spatial(m, ax0_fused_ax1_fused_fused_0 % m)
                            T.reads(NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, T.int64(0), v1])
                            T.writes(NT_matmul_intermediate_local[T.int64(0), v0, T.int64(0), v1])
                            with T.init():
                                NT_matmul_intermediate_local[T.int64(0), v0, T.int64(0), v1] = T.float16(0)
                            NT_matmul_intermediate_local[T.int64(0), v0, T.int64(0), v1] = NT_matmul_intermediate_local[T.int64(0), v0, T.int64(0), v1] + NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, T.int64(0), v1]
            for ax0_ax1_fused_0 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                for ax0_ax1_fused_1 in range(T.int64(1)):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_fused_ax1_fused_fused_0 // m)
                        v1 = T.axis.spatial(m, ax0_fused_ax1_fused_fused_0 % m)
                        T.reads(NT_matmul_intermediate_local[T.int64(0), v0, T.int64(0), v1], lv1775[T.int64(0), T.int64(0), T.int64(0), v1])
                        T.writes(compute_intermediate[T.int64(0), v0, T.int64(0), v1])
                        compute_intermediate[T.int64(0), v0, T.int64(0), v1] = T.Cast("float32", T.min(T.max(NT_matmul_intermediate_local[T.int64(0), v0, T.int64(0), v1] * T.float16(0.11179039301310044), T.float16(-65504)), lv1775[T.int64(0), T.int64(0), T.int64(0), v1]))

    @T.prim_func(private=True)
    def fused_decode8(model_params_9: T.Buffer((T.int64(2560), T.int64(2560)), "int8"), model_params_10: T.Buffer((T.int64(1), T.int64(2560)), "float16"), decode_intermediate: T.Buffer((T.int64(2560), T.int64(2560)), "float16")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(6400), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("decode"):
                    v0 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(2560))
                    v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(2560))
                    T.reads(model_params_9[v0, v1], model_params_10[T.int64(0), v1])
                    T.writes(decode_intermediate[v0, v1])
                    decode_intermediate[v0, v1] = T.Cast("float16", model_params_9[v0, v1]) * model_params_10[T.int64(0), v1]

    @T.prim_func(private=True)
    def fused_fused_decode10_fused_matmul12_add5(lv796: T.Buffer((T.int64(10240), T.int64(2560)), "int8"), lv797: T.Buffer((T.int64(1), T.int64(2560)), "float16"), p_lv795: T.handle, param_1710: T.Buffer((T.int64(2560),), "float16"), p_output0: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        lv795 = T.match_buffer(p_lv795, (T.int64(1), n, T.int64(10240)), "float16")
        T_add_intermediate_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        lv795_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(10240)), "float16", scope="shared.dyn")
        decode_intermediate_intermediate_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(2560), T.int64(10240)), "float16", scope="shared.dyn")
        lv795_reindex_pad_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(10240)), "float16", scope="wmma.matrix_a")
        decode_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(1), T.int64(2560), T.int64(10240)), "float16", scope="wmma.matrix_b")
        matmul_intermediate_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(2560)), "float16", scope="shared.dyn")
        matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(2560)), "float16", scope="wmma.accumulator")
        for ax0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for ax1_0_0_ax2_0_0_fused in T.thread_binding((n + T.int64(127)) // T.int64(128), thread="blockIdx.x"):
                for ax1_0_1_ax2_0_1_fused in T.thread_binding(T.int64(20), thread="blockIdx.y"):
                    for ax2_0_2_ax1_0_2_fused in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                        for ax1_0_3_init, ax2_0_3_init in T.grid(T.int64(2), T.int64(2)):
                            with T.block("matmul_o_init"):
                                v0_o = T.axis.spatial(T.int64(1), ax0)
                                v1_o = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(8), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax1_0_3_init)
                                v2_o = T.axis.spatial(T.int64(160), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax2_0_3_init)
                                T.reads()
                                T.writes(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                with T.block("matmul_init_o"):
                                    v1_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v2_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads()
                                    T.writes(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    C = T.match_buffer(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_fill_fragment(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.float32(0))
                        for ax3_0_0 in T.serial(T.int64(160), annotations={"software_pipeline_order": [0, 3, 1, 4, 5, 2, 6], "software_pipeline_stage": [0, 0, 0, 0, 0, 1, 1]}):
                            for ax0_ax1_fused_0 in range(T.int64(4)):
                                for ax0_ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                                            with T.block("lv795_reindex_pad_shared.dyn"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), ax1_0_0_ax2_0_0_fused * T.int64(128) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(64))
                                                v2 = T.axis.spatial(T.int64(10240), ax3_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(64))
                                                T.reads(lv795[v0, v1, v2])
                                                T.writes(lv795_reindex_pad_shared_dyn[v0, v1, v2])
                                                T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "double_buffer_scope": 0, "tir.manifest_shared_memory_local_stage": 1})
                                                lv795_reindex_pad_shared_dyn[v0, v1, v2] = T.if_then_else(v1 < n, lv795[v0, v1, v2], T.float16(0))
                            for ax0_ax1_fused_0 in range(T.int64(4)):
                                for ax0_ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                                            with T.block("decode_intermediate_intermediate_reindex_shared.dyn"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(2560), ax1_0_1_ax2_0_1_fused * T.int64(128) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(64))
                                                v2 = T.axis.spatial(T.int64(10240), ax3_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(64))
                                                T.reads(lv796[v2, v1], lv797[v2 // T.int64(10240), v1])
                                                T.writes(decode_intermediate_intermediate_reindex_shared_dyn[v0, v1, v2])
                                                T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "double_buffer_scope": 0, "tir.manifest_shared_memory_local_stage": 1})
                                                decode_intermediate_intermediate_reindex_shared_dyn[v0, v1, v2] = T.Cast("float16", lv796[v2, v1]) * lv797[v2 // T.int64(10240), v1]
                            for ax3_0_1 in T.serial(T.int64(4), annotations={"software_pipeline_order": [0, 1, 2], "software_pipeline_stage": [0, 0, 1]}):
                                for ax0_0 in T.unroll(T.int64(2)):
                                    for ax1_0 in T.unroll(T.int64(1)):
                                        with T.block("lv795_reindex_pad_shared.dyn_wmma.matrix_a_o"):
                                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1_o = T.axis.spatial(T.int64(8) * ((n + T.int64(127)) // T.int64(128)), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax0_0)
                                            v2_o = T.axis.spatial(T.int64(640), ax3_0_0 * T.int64(4) + ax3_0_1 + ax1_0)
                                            T.reads(lv795_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            T.writes(lv795_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A = T.match_buffer(lv795_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                            C = T.match_buffer(lv795_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                            T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(16), 1), A.strides[0], "row_major")
                                for ax0_0 in T.unroll(T.int64(2)):
                                    for ax1_0 in T.unroll(T.int64(1)):
                                        with T.block("decode_intermediate_intermediate_reindex_shared.dyn_wmma.matrix_b_o"):
                                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1_o = T.axis.spatial(T.int64(160), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax0_0)
                                            v2_o = T.axis.spatial(T.int64(640), ax3_0_0 * T.int64(4) + ax3_0_1 + ax1_0)
                                            T.reads(decode_intermediate_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            T.writes(decode_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A = T.match_buffer(decode_intermediate_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                            C = T.match_buffer(decode_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                            T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(16), 1), A.strides[0], "col_major")
                                for ax1_0_3, ax2_0_3 in T.grid(T.int64(2), T.int64(2)):
                                    with T.block("matmul_o_update"):
                                        v0_o = T.axis.spatial(T.int64(1), ax0)
                                        v1_o = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(8), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax1_0_3)
                                        v2_o = T.axis.spatial(T.int64(160), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax2_0_3)
                                        v3_o = T.axis.reduce(T.int64(640), ax3_0_0 * T.int64(4) + ax3_0_1)
                                        T.reads(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], lv795_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], decode_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                        T.writes(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                        with T.block("matmul_o"):
                                            v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v3_i_o = T.axis.reduce(T.int64(1), T.int64(0))
                                            T.reads(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], lv795_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], decode_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                            T.writes(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A = T.match_buffer(lv795_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                            B = T.match_buffer(decode_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                            C = T.match_buffer(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                            T.tvm_mma_sync(C.data, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), A.data, A.elem_offset // A.strides[0] // T.int64(16) * (A.strides[0] // T.int64(16)) + A.elem_offset % A.strides[0] // T.int64(16), B.data, B.elem_offset // B.strides[0] // T.int64(16) * (B.strides[0] // T.int64(16)) + B.elem_offset % B.strides[0] // T.int64(16), C.data, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16))
                        for ax0_0, ax1_0 in T.grid(T.int64(2), T.int64(2)):
                            with T.block("matmul_intermediate_reindex_pad_shared.dyn_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v1_o = T.axis.spatial(T.int64(8) * ((n + T.int64(127)) // T.int64(128)), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax0_0)
                                v2_o = T.axis.spatial(T.int64(160), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax1_0)
                                T.reads(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                T.writes(matmul_intermediate_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                A = T.match_buffer(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=16)
                                C = T.match_buffer(matmul_intermediate_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=16)
                                T.tvm_store_matrix_sync(A.data, 16, 16, 16, A.elem_offset // A.strides[0] // T.int64(16) * (A.strides[0] // T.int64(16)) + A.elem_offset % A.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), C.data, C.elem_offset, C.strides[0] * T.int64(16), 2), C.strides[0], "row_major")
                        for ax0_ax1_fused_0 in range(T.int64(8)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("matmul_intermediate_reindex_pad_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), ax1_0_0_ax2_0_0_fused * T.int64(128) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) // T.int64(32))
                                        v2 = T.axis.spatial(T.int64(2560), ax1_0_1_ax2_0_1_fused * T.int64(128) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) % T.int64(32))
                                        T.reads(matmul_intermediate_reindex_pad_shared_dyn[v0, v1, v2], param_1710[v2])
                                        T.writes(T_add_intermediate_intermediate[T.int64(0), v1, v2])
                                        T.block_attr({"buffer_dim_align": [[0, 1, 16, 4]]})
                                        if v1 < n:
                                            T_add_intermediate_intermediate[T.int64(0), v1, v2] = matmul_intermediate_reindex_pad_shared_dyn[v0, v1, v2] + param_1710[v2]

    @T.prim_func(private=True)
    def fused_fused_decode10_fused_matmul4_add1(lv22: T.Buffer((T.int64(10240), T.int64(2560)), "int8"), lv23: T.Buffer((T.int64(1), T.int64(2560)), "float16"), lv21: T.Buffer((T.int64(1), T.int64(1), T.int64(10240)), "float16"), param_17: T.Buffer((T.int64(2560),), "float16"), T_add_intermediate_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16", scope="local")
        matmul_intermediate_rf_local = T.alloc_buffer((T.int64(16), T.int64(1), T.int64(1), T.int64(2560)), "float16", scope="local")
        for ax0_fused_0 in T.thread_binding(T.int64(160), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                for ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                    with T.block("matmul_rf_init"):
                        vax1_fused_1 = T.axis.spatial(T.int64(16), ax1_fused_1)
                        v0 = T.axis.spatial(T.int64(2560), ax0_fused_0 * T.int64(16) + ax0_fused_1)
                        T.reads()
                        T.writes(matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0])
                        matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0] = T.float16(0)
                    for ax1_fused_0, u in T.grid(T.int64(640), 1):
                        with T.block("matmul_rf_update"):
                            vax1_fused_1 = T.axis.spatial(T.int64(16), ax1_fused_1)
                            v0 = T.axis.spatial(T.int64(2560), ax0_fused_0 * T.int64(16) + ax0_fused_1)
                            vax1_fused_0 = T.axis.reduce(T.int64(640), ax1_fused_0)
                            T.reads(matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0], lv21[T.int64(0), T.int64(0), vax1_fused_0 * T.int64(16) + vax1_fused_1], lv22[vax1_fused_0 * T.int64(16) + vax1_fused_1, v0], lv23[T.int64(0), v0])
                            T.writes(matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0])
                            matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0] = matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0] + lv21[T.int64(0), T.int64(0), vax1_fused_0 * T.int64(16) + vax1_fused_1] * (T.Cast("float16", lv22[vax1_fused_0 * T.int64(16) + vax1_fused_1, v0]) * lv23[T.int64(0), v0])
            for ax1_fused in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                for ax0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                    with T.block("matmul"):
                        vax1_fused_1 = T.axis.reduce(T.int64(16), ax0)
                        v0 = T.axis.spatial(T.int64(2560), ax0_fused_0 * T.int64(16) + ax1_fused)
                        T.reads(matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0])
                        T.writes(matmul_intermediate_local[T.int64(0), T.int64(0), v0])
                        with T.init():
                            matmul_intermediate_local[T.int64(0), T.int64(0), v0] = T.float16(0)
                        matmul_intermediate_local[T.int64(0), T.int64(0), v0] = matmul_intermediate_local[T.int64(0), T.int64(0), v0] + matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0]
            for ax0_fused_0_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                for ax0_fused_1 in range(T.int64(1)):
                    with T.block("T_add"):
                        v0 = T.axis.spatial(T.int64(2560), ax0_fused_0 * T.int64(16) + ax0_fused_0_1 + ax0_fused_1)
                        T.reads(matmul_intermediate_local[T.int64(0), T.int64(0), v0], param_17[v0])
                        T.writes(T_add_intermediate_intermediate[T.int64(0), T.int64(0), v0])
                        T_add_intermediate_intermediate[T.int64(0), T.int64(0), v0] = matmul_intermediate_local[T.int64(0), T.int64(0), v0] + param_17[v0]

    @T.prim_func(private=True)
    def fused_fused_decode1_take(lv: T.Buffer((50280, 640), "uint32"), lv1: T.Buffer((50280, 80), "float16"), lv1772: T.Buffer((1,), "int32"), T_take_intermediate: T.Buffer((1, 2560), "float16")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0_fused_0 in T.thread_binding(3, thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(1024, thread="threadIdx.x"):
                with T.block("T_take"):
                    v0 = T.axis.spatial(2560, ax0_fused_0 * 1024 + ax0_fused_1)
                    T.where(ax0_fused_0 * 1024 + ax0_fused_1 < 2560)
                    T.reads(lv1772[0], lv[lv1772[0], v0 // 4], lv1[lv1772[0], v0 // 32])
                    T.writes(T_take_intermediate[0, v0])
                    T_take_intermediate[0, v0] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv[lv1772[0], v0 // 4], T.Cast("uint32", v0 % 4) * T.uint32(8)), T.uint32(255))) - T.float16(127)) * lv1[lv1772[0], v0 // 32]

    @T.prim_func(private=True)
    def fused_fused_decode1_take1(lv775: T.Buffer((50280, 640), "uint32"), lv776: T.Buffer((50280, 80), "float16"), p_lv: T.handle, p_output0: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int32()
        lv = T.match_buffer(p_lv, (n,), "int32")
        T_take_intermediate = T.match_buffer(p_output0, (n, 2560), "float16")
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding((n * 2560 + 1023) // 1024, thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(1024, thread="threadIdx.x"):
                with T.block("T_take"):
                    v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1) // 2560)
                    v1 = T.axis.spatial(2560, (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1) % 2560)
                    T.where(ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 < n * 2560)
                    T.reads(lv[v0], lv775[lv[v0], v1 // 4], lv776[lv[v0], v1 // 32])
                    T.writes(T_take_intermediate[v0, v1])
                    T_take_intermediate[v0, v1] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv775[lv[v0], v1 // 4], T.Cast("uint32", v1 % 4) * T.uint32(8)), T.uint32(255))) - T.float16(127)) * lv776[lv[v0], v1 // 32]

    @T.prim_func(private=True)
    def fused_fused_decode6_NT_matmul1(lv772: T.Buffer((T.int64(50280), T.int64(320)), "uint32"), lv773: T.Buffer((T.int64(50280), T.int64(80)), "float32"), lv771: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), NT_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(50280)), "float32")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        NT_matmul_intermediate_rf_local = T.alloc_buffer((T.int64(256), T.int64(1), T.int64(1), T.int64(50280)), scope="local")
        NT_matmul_intermediate_rf_local_1 = T.alloc_buffer((T.int64(64), T.int64(1), T.int64(1), T.int64(50280)), scope="local")
        lv772_local = T.alloc_buffer((T.int64(50280), T.int64(320)), "uint32", scope="local")
        lv771_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)), scope="shared")
        for u_fused_ax0_fused_fused_0 in T.thread_binding(T.int64(12570), thread="blockIdx.x"):
            for u_fused_ax0_fused_fused_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                    for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                        for ax2_0 in T.serial(T.int64(5), annotations={"pragma_unroll_explicit": 256, "pragma_vectorize": 1}):
                            for ax2_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                                for ax2_2 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                    for ax2_3 in T.vectorized(T.int64(2)):
                                        with T.block("lv771_shared"):
                                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(128) + ax2_2 * T.int64(2) + ax2_3)
                                            T.reads(lv771[v0, v1, v2])
                                            T.writes(lv771_shared[v0, v1, v2])
                                            lv771_shared[v0, v1, v2] = lv771[v0, v1, v2]
                    for u_fused_ax0_fused_fused_2_init in range(T.int64(1)):
                        for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1_init in T.vectorized(T.int64(4)):
                            with T.block("NT_matmul_rf_init"):
                                vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused = T.axis.spatial(T.int64(256), ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * T.int64(4) + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1_init)
                                v0 = T.axis.spatial(T.int64(50280), u_fused_ax0_fused_fused_0 * T.int64(4) + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2_init)
                                T.reads()
                                T.writes(NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0])
                                NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0] = T.float32(0)
                    for ax1_0_fused_ax1_1_fused_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        for ax0_0, ax1 in T.grid(T.int64(1), T.int64(1)):
                            for ax0_1 in T.vectorized(T.int64(1)):
                                with T.block("lv772_local"):
                                    v0 = T.axis.spatial(T.int64(50280), u_fused_ax0_fused_fused_0 * T.int64(4) + u_fused_ax0_fused_fused_1 + ax0_0 + ax0_1)
                                    v1 = T.axis.spatial(T.int64(320), ax1_0_fused_ax1_1_fused_0 * T.int64(64) + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 + ax1)
                                    T.reads(lv772[v0, v1])
                                    T.writes(lv772_local[v0, v1])
                                    lv772_local[v0, v1] = lv772[v0, v1]
                        for u_fused_ax0_fused_fused_2, ax1_0_fused_ax1_1_fused_2 in T.grid(T.int64(1), T.int64(2)):
                            for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1 in T.vectorized(T.int64(4)):
                                with T.block("NT_matmul_rf_update"):
                                    vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused = T.axis.spatial(T.int64(256), ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * T.int64(4) + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1)
                                    v0 = T.axis.spatial(T.int64(50280), u_fused_ax0_fused_fused_0 * T.int64(4) + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2)
                                    vax1_0_fused_ax1_1_fused_0, vax1_0_fused_ax1_1_fused_2 = T.axis.remap("RR", [ax1_0_fused_ax1_1_fused_0, ax1_0_fused_ax1_1_fused_2])
                                    T.reads(NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0], lv771_shared[T.int64(0), T.int64(0), vax1_0_fused_ax1_1_fused_0 * T.int64(512) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) * T.int64(8) + vax1_0_fused_ax1_1_fused_2 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % T.int64(4)], lv772_local[v0, vax1_0_fused_ax1_1_fused_0 * T.int64(64) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) + vax1_0_fused_ax1_1_fused_2 // T.int64(2)], lv773[v0, (vax1_0_fused_ax1_1_fused_0 * T.int64(512) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) * T.int64(8) + vax1_0_fused_ax1_1_fused_2 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % T.int64(4)) // T.int64(32)])
                                    T.writes(NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0])
                                    NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0] = NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0] + lv771_shared[T.int64(0), T.int64(0), vax1_0_fused_ax1_1_fused_0 * T.int64(512) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) * T.int64(8) + vax1_0_fused_ax1_1_fused_2 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % T.int64(4)] * ((T.Cast("float32", T.bitwise_and(T.shift_right(lv772_local[v0, vax1_0_fused_ax1_1_fused_0 * T.int64(64) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) + vax1_0_fused_ax1_1_fused_2 // T.int64(2)], T.Cast("uint32", (vax1_0_fused_ax1_1_fused_0 * T.int64(512) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) * T.int64(8) + vax1_0_fused_ax1_1_fused_2 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % T.int64(4)) % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float32(7)) * lv773[v0, (vax1_0_fused_ax1_1_fused_0 * T.int64(512) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) * T.int64(8) + vax1_0_fused_ax1_1_fused_2 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % T.int64(4)) // T.int64(32)])
            for ax2_fused_0 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                for ax0 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                    for ax2_fused_1_0 in T.serial(T.int64(1), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        for ax2_fused_1_1 in T.vectorized(T.int64(1)):
                            with T.block("NT_matmul_rf_init"):
                                vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 = T.axis.spatial(T.int64(64), ax0)
                                v0 = T.axis.spatial(T.int64(50280), u_fused_ax0_fused_fused_0 * T.int64(4) + ax2_fused_0 + ax2_fused_1_0 + ax2_fused_1_1)
                                T.reads()
                                T.writes(NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0])
                                NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0] = T.float32(0)
                            for ax1 in range(T.int64(4)):
                                with T.block("NT_matmul_rf_update"):
                                    vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1 = T.axis.remap("SR", [ax0, ax1])
                                    v0 = T.axis.spatial(T.int64(50280), u_fused_ax0_fused_fused_0 * T.int64(4) + ax2_fused_0 + ax2_fused_1_0 + ax2_fused_1_1)
                                    T.reads(NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0], NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1, T.int64(0), T.int64(0), v0])
                                    T.writes(NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0])
                                    NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0] = NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0] + NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1, T.int64(0), T.int64(0), v0]
            for ax1_fused_1 in range(T.int64(1)):
                for ax1_fused_0 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                    for ax0 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                        with T.block("NT_matmul"):
                            vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 = T.axis.reduce(T.int64(64), ax0)
                            v0 = T.axis.spatial(T.int64(50280), u_fused_ax0_fused_fused_0 * T.int64(4) + ax1_fused_0 + ax1_fused_1)
                            T.reads(NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0])
                            T.writes(NT_matmul_intermediate[T.int64(0), T.int64(0), v0])
                            with T.init():
                                NT_matmul_intermediate[T.int64(0), T.int64(0), v0] = T.float32(0)
                            NT_matmul_intermediate[T.int64(0), T.int64(0), v0] = NT_matmul_intermediate[T.int64(0), T.int64(0), v0] + NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0]

    @T.prim_func(private=True)
    def fused_fused_decode7_fused_matmul8_add4(lv780: T.Buffer((T.int64(2560), T.int64(7680)), "int8"), lv781: T.Buffer((T.int64(1), T.int64(7680)), "float16"), p_lv779: T.handle, param_810: T.Buffer((T.int64(7680),), "float16"), p_output0: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        lv779 = T.match_buffer(p_lv779, (T.int64(1), n, T.int64(2560)), "float16")
        T_add_intermediate_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(7680)), "float16")
        # with T.block("root"):
        lv779_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(2560)), "float16", scope="shared.dyn")
        decode_intermediate_intermediate_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(7680), T.int64(2560)), "float16", scope="shared.dyn")
        lv779_reindex_pad_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(2560)), "float16", scope="wmma.matrix_a")
        decode_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(1), T.int64(7680), T.int64(2560)), "float16", scope="wmma.matrix_b")
        matmul_intermediate_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(7680)), "float16", scope="shared.dyn")
        matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(7680)), "float16", scope="wmma.accumulator")
        for ax0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for ax1_0_0_ax2_0_0_fused in T.thread_binding((n + T.int64(127)) // T.int64(128), thread="blockIdx.x"):
                for ax1_0_1_ax2_0_1_fused in T.thread_binding(T.int64(60), thread="blockIdx.y"):
                    for ax2_0_2_ax1_0_2_fused in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                        for ax1_0_3_init, ax2_0_3_init in T.grid(T.int64(2), T.int64(2)):
                            with T.block("matmul_o_init"):
                                v0_o = T.axis.spatial(T.int64(1), ax0)
                                v1_o = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(8), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax1_0_3_init)
                                v2_o = T.axis.spatial(T.int64(480), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax2_0_3_init)
                                T.reads()
                                T.writes(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                with T.block("matmul_init_o"):
                                    v1_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v2_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads()
                                    T.writes(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    C = T.match_buffer(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_fill_fragment(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.float32(0))
                        for ax3_0_0 in T.serial(T.int64(40), annotations={"software_pipeline_order": [0, 3, 1, 4, 5, 2, 6], "software_pipeline_stage": [0, 0, 0, 0, 0, 1, 1]}):
                            for ax0_ax1_fused_0 in range(T.int64(4)):
                                for ax0_ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                                            with T.block("lv779_reindex_pad_shared.dyn"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), ax1_0_0_ax2_0_0_fused * T.int64(128) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(64))
                                                v2 = T.axis.spatial(T.int64(2560), ax3_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(64))
                                                T.reads(lv779[v0, v1, v2])
                                                T.writes(lv779_reindex_pad_shared_dyn[v0, v1, v2])
                                                T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "double_buffer_scope": 0, "tir.manifest_shared_memory_local_stage": 1})
                                                lv779_reindex_pad_shared_dyn[v0, v1, v2] = T.if_then_else(v1 < n, lv779[v0, v1, v2], T.float16(0))
                            for ax0_ax1_fused_0 in range(T.int64(4)):
                                for ax0_ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                                            with T.block("decode_intermediate_intermediate_reindex_shared.dyn"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(7680), ax1_0_1_ax2_0_1_fused * T.int64(128) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(64))
                                                v2 = T.axis.spatial(T.int64(2560), ax3_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(64))
                                                T.reads(lv780[v2, v1], lv781[v2 // T.int64(2560), v1])
                                                T.writes(decode_intermediate_intermediate_reindex_shared_dyn[v0, v1, v2])
                                                T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "double_buffer_scope": 0, "tir.manifest_shared_memory_local_stage": 1})
                                                decode_intermediate_intermediate_reindex_shared_dyn[v0, v1, v2] = T.Cast("float16", lv780[v2, v1]) * lv781[v2 // T.int64(2560), v1]
                            for ax3_0_1 in T.serial(T.int64(4), annotations={"software_pipeline_order": [0, 1, 2], "software_pipeline_stage": [0, 0, 1]}):
                                for ax0_0 in T.unroll(T.int64(2)):
                                    for ax1_0 in T.unroll(T.int64(1)):
                                        with T.block("lv779_reindex_pad_shared.dyn_wmma.matrix_a_o"):
                                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1_o = T.axis.spatial(T.int64(8) * ((n + T.int64(127)) // T.int64(128)), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax0_0)
                                            v2_o = T.axis.spatial(T.int64(160), ax3_0_0 * T.int64(4) + ax3_0_1 + ax1_0)
                                            T.reads(lv779_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            T.writes(lv779_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A = T.match_buffer(lv779_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                            C = T.match_buffer(lv779_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                            T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(16), 1), A.strides[0], "row_major")
                                for ax0_0 in T.unroll(T.int64(2)):
                                    for ax1_0 in T.unroll(T.int64(1)):
                                        with T.block("decode_intermediate_intermediate_reindex_shared.dyn_wmma.matrix_b_o"):
                                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1_o = T.axis.spatial(T.int64(480), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax0_0)
                                            v2_o = T.axis.spatial(T.int64(160), ax3_0_0 * T.int64(4) + ax3_0_1 + ax1_0)
                                            T.reads(decode_intermediate_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            T.writes(decode_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A = T.match_buffer(decode_intermediate_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                            C = T.match_buffer(decode_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                            T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(16), 1), A.strides[0], "col_major")
                                for ax1_0_3, ax2_0_3 in T.grid(T.int64(2), T.int64(2)):
                                    with T.block("matmul_o_update"):
                                        v0_o = T.axis.spatial(T.int64(1), ax0)
                                        v1_o = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(8), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax1_0_3)
                                        v2_o = T.axis.spatial(T.int64(480), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax2_0_3)
                                        v3_o = T.axis.reduce(T.int64(160), ax3_0_0 * T.int64(4) + ax3_0_1)
                                        T.reads(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], lv779_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], decode_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                        T.writes(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                        with T.block("matmul_o"):
                                            v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v3_i_o = T.axis.reduce(T.int64(1), T.int64(0))
                                            T.reads(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], lv779_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], decode_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                            T.writes(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A = T.match_buffer(lv779_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                            B = T.match_buffer(decode_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                            C = T.match_buffer(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                            T.tvm_mma_sync(C.data, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), A.data, A.elem_offset // A.strides[0] // T.int64(16) * (A.strides[0] // T.int64(16)) + A.elem_offset % A.strides[0] // T.int64(16), B.data, B.elem_offset // B.strides[0] // T.int64(16) * (B.strides[0] // T.int64(16)) + B.elem_offset % B.strides[0] // T.int64(16), C.data, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16))
                        for ax0_0, ax1_0 in T.grid(T.int64(2), T.int64(2)):
                            with T.block("matmul_intermediate_reindex_pad_shared.dyn_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v1_o = T.axis.spatial(T.int64(8) * ((n + T.int64(127)) // T.int64(128)), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax0_0)
                                v2_o = T.axis.spatial(T.int64(480), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax1_0)
                                T.reads(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                T.writes(matmul_intermediate_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                A = T.match_buffer(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=16)
                                C = T.match_buffer(matmul_intermediate_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=16)
                                T.tvm_store_matrix_sync(A.data, 16, 16, 16, A.elem_offset // A.strides[0] // T.int64(16) * (A.strides[0] // T.int64(16)) + A.elem_offset % A.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), C.data, C.elem_offset, C.strides[0] * T.int64(16), 2), C.strides[0], "row_major")
                        for ax0_ax1_fused_0 in range(T.int64(8)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("matmul_intermediate_reindex_pad_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), ax1_0_0_ax2_0_0_fused * T.int64(128) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) // T.int64(32))
                                        v2 = T.axis.spatial(T.int64(7680), ax1_0_1_ax2_0_1_fused * T.int64(128) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) % T.int64(32))
                                        T.reads(matmul_intermediate_reindex_pad_shared_dyn[v0, v1, v2], param_810[v2])
                                        T.writes(T_add_intermediate_intermediate[T.int64(0), v1, v2])
                                        T.block_attr({"buffer_dim_align": [[0, 1, 16, 4]]})
                                        if v1 < n:
                                            T_add_intermediate_intermediate[T.int64(0), v1, v2] = matmul_intermediate_reindex_pad_shared_dyn[v0, v1, v2] + param_810[v2]

    @T.prim_func(private=True)
    def fused_fused_decode7_fused_matmul_add(lv4: T.Buffer((T.int64(2560), T.int64(7680)), "int8"), lv5: T.Buffer((T.int64(1), T.int64(7680)), "float16"), lv3: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), param_8: T.Buffer((T.int64(7680),), "float16"), T_add_intermediate_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(7680)), "float16")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(7680)), "float16", scope="local")
        matmul_intermediate_rf_local = T.alloc_buffer((T.int64(16), T.int64(1), T.int64(1), T.int64(7680)), "float16", scope="local")
        for ax0_fused_0 in T.thread_binding(T.int64(480), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                for ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                    with T.block("matmul_rf_init"):
                        vax1_fused_1 = T.axis.spatial(T.int64(16), ax1_fused_1)
                        v0 = T.axis.spatial(T.int64(7680), ax0_fused_0 * T.int64(16) + ax0_fused_1)
                        T.reads()
                        T.writes(matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0])
                        matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0] = T.float16(0)
                    for ax1_fused_0, u in T.grid(T.int64(160), 1):
                        with T.block("matmul_rf_update"):
                            vax1_fused_1 = T.axis.spatial(T.int64(16), ax1_fused_1)
                            v0 = T.axis.spatial(T.int64(7680), ax0_fused_0 * T.int64(16) + ax0_fused_1)
                            vax1_fused_0 = T.axis.reduce(T.int64(160), ax1_fused_0)
                            T.reads(matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0], lv3[T.int64(0), T.int64(0), vax1_fused_0 * T.int64(16) + vax1_fused_1], lv4[vax1_fused_0 * T.int64(16) + vax1_fused_1, v0], lv5[T.int64(0), v0])
                            T.writes(matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0])
                            matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0] = matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0] + lv3[T.int64(0), T.int64(0), vax1_fused_0 * T.int64(16) + vax1_fused_1] * (T.Cast("float16", lv4[vax1_fused_0 * T.int64(16) + vax1_fused_1, v0]) * lv5[T.int64(0), v0])
            for ax1_fused in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                for ax0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                    with T.block("matmul"):
                        vax1_fused_1 = T.axis.reduce(T.int64(16), ax0)
                        v0 = T.axis.spatial(T.int64(7680), ax0_fused_0 * T.int64(16) + ax1_fused)
                        T.reads(matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0])
                        T.writes(matmul_intermediate_local[T.int64(0), T.int64(0), v0])
                        with T.init():
                            matmul_intermediate_local[T.int64(0), T.int64(0), v0] = T.float16(0)
                        matmul_intermediate_local[T.int64(0), T.int64(0), v0] = matmul_intermediate_local[T.int64(0), T.int64(0), v0] + matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0]
            for ax0_fused_0_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                for ax0_fused_1 in range(T.int64(1)):
                    with T.block("T_add"):
                        v0 = T.axis.spatial(T.int64(7680), ax0_fused_0 * T.int64(16) + ax0_fused_0_1 + ax0_fused_1)
                        T.reads(matmul_intermediate_local[T.int64(0), T.int64(0), v0], param_8[v0])
                        T.writes(T_add_intermediate_intermediate[T.int64(0), T.int64(0), v0])
                        T_add_intermediate_intermediate[T.int64(0), T.int64(0), v0] = matmul_intermediate_local[T.int64(0), T.int64(0), v0] + param_8[v0]

    @T.prim_func(private=True)
    def fused_fused_decode9_fused_matmul11_add6_gelu1(lv792: T.Buffer((T.int64(2560), T.int64(10240)), "int8"), lv793: T.Buffer((T.int64(1), T.int64(10240)), "float16"), p_lv791: T.handle, param_1410: T.Buffer((T.int64(10240),), "float16"), p_output0: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        lv791 = T.match_buffer(p_lv791, (T.int64(1), n, T.int64(2560)), "float16")
        T_multiply_intermediate_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(10240)), "float16")
        # with T.block("root"):
        lv791_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(2560)), "float16", scope="shared.dyn")
        decode_intermediate_intermediate_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(10240), T.int64(2560)), "float16", scope="shared.dyn")
        lv791_reindex_pad_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(2560)), "float16", scope="wmma.matrix_a")
        decode_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(1), T.int64(10240), T.int64(2560)), "float16", scope="wmma.matrix_b")
        matmul_intermediate_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(10240)), "float16", scope="shared.dyn")
        matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(10240)), "float16", scope="wmma.accumulator")
        for ax0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for ax1_0_0_ax2_0_0_fused in T.thread_binding((n + T.int64(127)) // T.int64(128), thread="blockIdx.x"):
                for ax1_0_1_ax2_0_1_fused in T.thread_binding(T.int64(80), thread="blockIdx.y"):
                    for ax2_0_2_ax1_0_2_fused in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                        for ax1_0_3_init, ax2_0_3_init in T.grid(T.int64(2), T.int64(2)):
                            with T.block("matmul_o_init"):
                                v0_o = T.axis.spatial(T.int64(1), ax0)
                                v1_o = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(8), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax1_0_3_init)
                                v2_o = T.axis.spatial(T.int64(640), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax2_0_3_init)
                                T.reads()
                                T.writes(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                with T.block("matmul_init_o"):
                                    v1_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v2_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads()
                                    T.writes(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    C = T.match_buffer(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_fill_fragment(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.float32(0))
                        for ax3_0_0 in T.serial(T.int64(40), annotations={"software_pipeline_order": [0, 3, 1, 4, 5, 2, 6], "software_pipeline_stage": [0, 0, 0, 0, 0, 1, 1]}):
                            for ax0_ax1_fused_0 in range(T.int64(4)):
                                for ax0_ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                                            with T.block("lv791_reindex_pad_shared.dyn"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), ax1_0_0_ax2_0_0_fused * T.int64(128) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(64))
                                                v2 = T.axis.spatial(T.int64(2560), ax3_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(64))
                                                T.reads(lv791[v0, v1, v2])
                                                T.writes(lv791_reindex_pad_shared_dyn[v0, v1, v2])
                                                T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "double_buffer_scope": 0, "tir.manifest_shared_memory_local_stage": 1})
                                                lv791_reindex_pad_shared_dyn[v0, v1, v2] = T.if_then_else(v1 < n, lv791[v0, v1, v2], T.float16(0))
                            for ax0_ax1_fused_0 in range(T.int64(4)):
                                for ax0_ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                                            with T.block("decode_intermediate_intermediate_reindex_shared.dyn"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(10240), ax1_0_1_ax2_0_1_fused * T.int64(128) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(64))
                                                v2 = T.axis.spatial(T.int64(2560), ax3_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(64))
                                                T.reads(lv792[v2, v1], lv793[v2 // T.int64(2560), v1])
                                                T.writes(decode_intermediate_intermediate_reindex_shared_dyn[v0, v1, v2])
                                                T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "double_buffer_scope": 0, "tir.manifest_shared_memory_local_stage": 1})
                                                decode_intermediate_intermediate_reindex_shared_dyn[v0, v1, v2] = T.Cast("float16", lv792[v2, v1]) * lv793[v2 // T.int64(2560), v1]
                            for ax3_0_1 in T.serial(T.int64(4), annotations={"software_pipeline_order": [0, 1, 2], "software_pipeline_stage": [0, 0, 1]}):
                                for ax0_0 in T.unroll(T.int64(2)):
                                    for ax1_0 in T.unroll(T.int64(1)):
                                        with T.block("lv791_reindex_pad_shared.dyn_wmma.matrix_a_o"):
                                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1_o = T.axis.spatial(T.int64(8) * ((n + T.int64(127)) // T.int64(128)), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax0_0)
                                            v2_o = T.axis.spatial(T.int64(160), ax3_0_0 * T.int64(4) + ax3_0_1 + ax1_0)
                                            T.reads(lv791_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            T.writes(lv791_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A = T.match_buffer(lv791_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                            C = T.match_buffer(lv791_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                            T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(16), 1), A.strides[0], "row_major")
                                for ax0_0 in T.unroll(T.int64(2)):
                                    for ax1_0 in T.unroll(T.int64(1)):
                                        with T.block("decode_intermediate_intermediate_reindex_shared.dyn_wmma.matrix_b_o"):
                                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1_o = T.axis.spatial(T.int64(640), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax0_0)
                                            v2_o = T.axis.spatial(T.int64(160), ax3_0_0 * T.int64(4) + ax3_0_1 + ax1_0)
                                            T.reads(decode_intermediate_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            T.writes(decode_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A = T.match_buffer(decode_intermediate_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                            C = T.match_buffer(decode_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                            T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(16), 1), A.strides[0], "col_major")
                                for ax1_0_3, ax2_0_3 in T.grid(T.int64(2), T.int64(2)):
                                    with T.block("matmul_o_update"):
                                        v0_o = T.axis.spatial(T.int64(1), ax0)
                                        v1_o = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(8), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax1_0_3)
                                        v2_o = T.axis.spatial(T.int64(640), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax2_0_3)
                                        v3_o = T.axis.reduce(T.int64(160), ax3_0_0 * T.int64(4) + ax3_0_1)
                                        T.reads(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], lv791_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], decode_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                        T.writes(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                        with T.block("matmul_o"):
                                            v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v3_i_o = T.axis.reduce(T.int64(1), T.int64(0))
                                            T.reads(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], lv791_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], decode_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                            T.writes(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A = T.match_buffer(lv791_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                            B = T.match_buffer(decode_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                            C = T.match_buffer(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                            T.tvm_mma_sync(C.data, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), A.data, A.elem_offset // A.strides[0] // T.int64(16) * (A.strides[0] // T.int64(16)) + A.elem_offset % A.strides[0] // T.int64(16), B.data, B.elem_offset // B.strides[0] // T.int64(16) * (B.strides[0] // T.int64(16)) + B.elem_offset % B.strides[0] // T.int64(16), C.data, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16))
                        for ax0_0, ax1_0 in T.grid(T.int64(2), T.int64(2)):
                            with T.block("matmul_intermediate_reindex_pad_shared.dyn_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v1_o = T.axis.spatial(T.int64(8) * ((n + T.int64(127)) // T.int64(128)), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax0_0)
                                v2_o = T.axis.spatial(T.int64(640), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax1_0)
                                T.reads(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                T.writes(matmul_intermediate_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                A = T.match_buffer(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=16)
                                C = T.match_buffer(matmul_intermediate_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=16)
                                T.tvm_store_matrix_sync(A.data, 16, 16, 16, A.elem_offset // A.strides[0] // T.int64(16) * (A.strides[0] // T.int64(16)) + A.elem_offset % A.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), C.data, C.elem_offset, C.strides[0] * T.int64(16), 2), C.strides[0], "row_major")
                        for ax0_ax1_fused_0 in range(T.int64(8)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("matmul_intermediate_reindex_pad_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), ax1_0_0_ax2_0_0_fused * T.int64(128) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) // T.int64(32))
                                        v2 = T.axis.spatial(T.int64(10240), ax1_0_1_ax2_0_1_fused * T.int64(128) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) % T.int64(32))
                                        T.reads(matmul_intermediate_reindex_pad_shared_dyn[v0, v1, v2], param_1410[v2])
                                        T.writes(T_multiply_intermediate_intermediate[T.int64(0), v1, v2])
                                        T.block_attr({"buffer_dim_align": [[0, 1, 16, 4]]})
                                        if v1 < n:
                                            T_multiply_intermediate_intermediate[T.int64(0), v1, v2] = (matmul_intermediate_reindex_pad_shared_dyn[v0, v1, v2] + param_1410[v2]) * (T.float16(0.5) + T.Cast("float16", T.erf(T.Cast("float32", (matmul_intermediate_reindex_pad_shared_dyn[v0, v1, v2] + param_1410[v2]) * T.float16(0.70710678118654757)))) * T.float16(0.5))

    @T.prim_func(private=True)
    def fused_fused_decode9_fused_matmul3_add2_gelu(lv18: T.Buffer((T.int64(2560), T.int64(10240)), "int8"), lv19: T.Buffer((T.int64(1), T.int64(10240)), "float16"), lv17: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), param_14: T.Buffer((T.int64(10240),), "float16"), T_multiply_intermediate_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(10240)), "float16")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)), "float16", scope="local")
        matmul_intermediate_rf_local = T.alloc_buffer((T.int64(16), T.int64(1), T.int64(1), T.int64(10240)), "float16", scope="local")
        for ax0_fused_0 in T.thread_binding(T.int64(640), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                for ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                    with T.block("matmul_rf_init"):
                        vax1_fused_1 = T.axis.spatial(T.int64(16), ax1_fused_1)
                        v0 = T.axis.spatial(T.int64(10240), ax0_fused_0 * T.int64(16) + ax0_fused_1)
                        T.reads()
                        T.writes(matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0])
                        matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0] = T.float16(0)
                    for ax1_fused_0, u in T.grid(T.int64(160), 1):
                        with T.block("matmul_rf_update"):
                            vax1_fused_1 = T.axis.spatial(T.int64(16), ax1_fused_1)
                            v0 = T.axis.spatial(T.int64(10240), ax0_fused_0 * T.int64(16) + ax0_fused_1)
                            vax1_fused_0 = T.axis.reduce(T.int64(160), ax1_fused_0)
                            T.reads(matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0], lv17[T.int64(0), T.int64(0), vax1_fused_0 * T.int64(16) + vax1_fused_1], lv18[vax1_fused_0 * T.int64(16) + vax1_fused_1, v0], lv19[T.int64(0), v0])
                            T.writes(matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0])
                            matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0] = matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0] + lv17[T.int64(0), T.int64(0), vax1_fused_0 * T.int64(16) + vax1_fused_1] * (T.Cast("float16", lv18[vax1_fused_0 * T.int64(16) + vax1_fused_1, v0]) * lv19[T.int64(0), v0])
            for ax1_fused in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                for ax0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                    with T.block("matmul"):
                        vax1_fused_1 = T.axis.reduce(T.int64(16), ax0)
                        v0 = T.axis.spatial(T.int64(10240), ax0_fused_0 * T.int64(16) + ax1_fused)
                        T.reads(matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0])
                        T.writes(matmul_intermediate_local[T.int64(0), T.int64(0), v0])
                        with T.init():
                            matmul_intermediate_local[T.int64(0), T.int64(0), v0] = T.float16(0)
                        matmul_intermediate_local[T.int64(0), T.int64(0), v0] = matmul_intermediate_local[T.int64(0), T.int64(0), v0] + matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0]
            for ax0_fused_0_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                for ax0_fused_1 in range(T.int64(1)):
                    with T.block("T_multiply_2"):
                        v0 = T.axis.spatial(T.int64(10240), ax0_fused_0 * T.int64(16) + ax0_fused_0_1 + ax0_fused_1)
                        T.reads(matmul_intermediate_local[T.int64(0), T.int64(0), v0], param_14[v0])
                        T.writes(T_multiply_intermediate_intermediate[T.int64(0), T.int64(0), v0])
                        T_multiply_intermediate_intermediate[T.int64(0), T.int64(0), v0] = (matmul_intermediate_local[T.int64(0), T.int64(0), v0] + param_14[v0]) * (T.float16(0.5) + T.Cast("float16", T.erf(T.Cast("float32", (matmul_intermediate_local[T.int64(0), T.int64(0), v0] + param_14[v0]) * T.float16(0.70710678118654757)))) * T.float16(0.5))

    @T.prim_func(private=True)
    def fused_layer_norm1_cast6(p_lv6: T.handle, param_2100: T.Buffer((T.int64(2560),), "float32"), param_3100: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(2560)))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        A_red_temp_v0_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        A_red_temp_v1_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        for ax0_fused in T.thread_binding(n, thread="blockIdx.x"):
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.reads(lv6[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6[T.int64(0), v0, v1] * lv6[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(256) + ax1_1)
                        T.reads(lv6[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_2100[v1], param_3100[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_2100[v1] + param_3100[v1])

    @T.prim_func(private=True)
    def fused_layer_norm_cast1(lv1776: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), param_2: T.Buffer((T.int64(2560),), "float32"), param_3: T.Buffer((T.int64(2560),), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_red_temp_v0_shared = T.alloc_buffer((T.int64(1), T.int64(1)), scope="shared")
        A_red_temp_v1_shared = T.alloc_buffer((T.int64(1), T.int64(1)), scope="shared")
        for ax0_fused in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.reads(lv1776[T.int64(0), T.int64(0), v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), T.int64(0)], A_red_temp_v1_shared[T.int64(0), T.int64(0)])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), T.int64(0)] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), T.int64(0)] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), T.int64(0)] + lv1776[T.int64(0), T.int64(0), v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), T.int64(0)] + lv1776[T.int64(0), T.int64(0), v1] * lv1776[T.int64(0), T.int64(0), v1]
                            A_red_temp_v0_shared[T.int64(0), T.int64(0)] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), T.int64(0)] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(256) + ax1_1)
                        T.reads(lv1776[T.int64(0), T.int64(0), v1], A_red_temp_v0_shared[T.int64(0), T.int64(0)], A_red_temp_v1_shared[T.int64(0), T.int64(0)], param_2[v1], param_3[v1])
                        T.writes(compute_intermediate[T.int64(0), T.int64(0), v1])
                        compute_intermediate[T.int64(0), T.int64(0), v1] = T.Cast("float16", (lv1776[T.int64(0), T.int64(0), v1] - A_red_temp_v0_shared[T.int64(0), T.int64(0)] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), T.int64(0)] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), T.int64(0)] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), T.int64(0)] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_2[v1] + param_3[v1])

    @T.prim_func(private=True)
    def fused_matmul10_add5_add7_add7(p_lv43: T.handle, lv129: T.Buffer((T.int64(2560), T.int64(2560)), "float16"), param_1110: T.Buffer((T.int64(2560),), "float16"), p_lv58: T.handle, p_lv2: T.handle, p_output0: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        lv43 = T.match_buffer(p_lv43, (T.int64(1), n, T.int64(2560)), "float16")
        lv58 = T.match_buffer(p_lv58, (T.int64(1), n, T.int64(2560)), "float16")
        lv2 = T.match_buffer(p_lv2, (T.int64(1), n, T.int64(2560)), "float16")
        T_add_intermediate_1_2 = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        lv43_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(2560)), "float16", scope="shared.dyn")
        lv129_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(2560), T.int64(2560)), "float16", scope="shared.dyn")
        lv43_reindex_pad_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(2560)), "float16", scope="wmma.matrix_a")
        lv129_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(1), T.int64(2560), T.int64(2560)), "float16", scope="wmma.matrix_b")
        matmul_intermediate_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(2560)), "float16", scope="shared.dyn")
        matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(2560)), "float16", scope="wmma.accumulator")
        for ax0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for ax1_0_0_ax2_0_0_fused in T.thread_binding((n + T.int64(127)) // T.int64(128), thread="blockIdx.x"):
                for ax1_0_1_ax2_0_1_fused in T.thread_binding(T.int64(20), thread="blockIdx.y"):
                    for ax2_0_2_ax1_0_2_fused in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                        for ax1_0_3_init, ax2_0_3_init in T.grid(T.int64(2), T.int64(2)):
                            with T.block("matmul_o_init"):
                                v0_o = T.axis.spatial(T.int64(1), ax0)
                                v1_o = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(8), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax1_0_3_init)
                                v2_o = T.axis.spatial(T.int64(160), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax2_0_3_init)
                                T.reads()
                                T.writes(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                with T.block("matmul_init_o"):
                                    v1_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v2_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads()
                                    T.writes(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    C = T.match_buffer(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_fill_fragment(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.float32(0))
                        for ax3_0_0 in T.serial(T.int64(40), annotations={"software_pipeline_order": [0, 3, 1, 4, 5, 2, 6], "software_pipeline_stage": [0, 0, 0, 0, 0, 1, 1]}):
                            for ax0_ax1_fused_0 in range(T.int64(4)):
                                for ax0_ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                                            with T.block("lv43_reindex_pad_shared.dyn"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), ax1_0_0_ax2_0_0_fused * T.int64(128) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(64))
                                                v2 = T.axis.spatial(T.int64(2560), ax3_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(64))
                                                T.reads(lv43[v0, v1, v2])
                                                T.writes(lv43_reindex_pad_shared_dyn[v0, v1, v2])
                                                T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "double_buffer_scope": 0, "tir.manifest_shared_memory_local_stage": 1})
                                                lv43_reindex_pad_shared_dyn[v0, v1, v2] = T.if_then_else(v1 < n, lv43[v0, v1, v2], T.float16(0))
                            for ax0_ax1_fused_0 in range(T.int64(4)):
                                for ax0_ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                                            with T.block("lv129_reindex_shared.dyn"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(2560), ax1_0_1_ax2_0_1_fused * T.int64(128) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(64))
                                                v2 = T.axis.spatial(T.int64(2560), ax3_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(64))
                                                T.reads(lv129[v2, v1])
                                                T.writes(lv129_reindex_shared_dyn[v0, v1, v2])
                                                T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "double_buffer_scope": 0, "tir.manifest_shared_memory_local_stage": 1})
                                                lv129_reindex_shared_dyn[v0, v1, v2] = lv129[v2, v1]
                            for ax3_0_1 in T.serial(T.int64(4), annotations={"software_pipeline_order": [0, 1, 2], "software_pipeline_stage": [0, 0, 1]}):
                                for ax0_0 in T.unroll(T.int64(2)):
                                    for ax1_0 in T.unroll(T.int64(1)):
                                        with T.block("lv43_reindex_pad_shared.dyn_wmma.matrix_a_o"):
                                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1_o = T.axis.spatial(T.int64(8) * ((n + T.int64(127)) // T.int64(128)), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax0_0)
                                            v2_o = T.axis.spatial(T.int64(160), ax3_0_0 * T.int64(4) + ax3_0_1 + ax1_0)
                                            T.reads(lv43_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            T.writes(lv43_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A = T.match_buffer(lv43_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                            C = T.match_buffer(lv43_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                            T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(16), 1), A.strides[0], "row_major")
                                for ax0_0 in T.unroll(T.int64(2)):
                                    for ax1_0 in T.unroll(T.int64(1)):
                                        with T.block("lv129_reindex_shared.dyn_wmma.matrix_b_o"):
                                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1_o = T.axis.spatial(T.int64(160), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax0_0)
                                            v2_o = T.axis.spatial(T.int64(160), ax3_0_0 * T.int64(4) + ax3_0_1 + ax1_0)
                                            T.reads(lv129_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            T.writes(lv129_reindex_shared_dyn_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A = T.match_buffer(lv129_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                            C = T.match_buffer(lv129_reindex_shared_dyn_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                            T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(16), 1), A.strides[0], "col_major")
                                for ax1_0_3, ax2_0_3 in T.grid(T.int64(2), T.int64(2)):
                                    with T.block("matmul_o_update"):
                                        v0_o = T.axis.spatial(T.int64(1), ax0)
                                        v1_o = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(8), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax1_0_3)
                                        v2_o = T.axis.spatial(T.int64(160), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax2_0_3)
                                        v3_o = T.axis.reduce(T.int64(160), ax3_0_0 * T.int64(4) + ax3_0_1)
                                        T.reads(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], lv43_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], lv129_reindex_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                        T.writes(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                        with T.block("matmul_o"):
                                            v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v3_i_o = T.axis.reduce(T.int64(1), T.int64(0))
                                            T.reads(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], lv43_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], lv129_reindex_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                            T.writes(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A = T.match_buffer(lv43_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                            B = T.match_buffer(lv129_reindex_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                            C = T.match_buffer(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                            T.tvm_mma_sync(C.data, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), A.data, A.elem_offset // A.strides[0] // T.int64(16) * (A.strides[0] // T.int64(16)) + A.elem_offset % A.strides[0] // T.int64(16), B.data, B.elem_offset // B.strides[0] // T.int64(16) * (B.strides[0] // T.int64(16)) + B.elem_offset % B.strides[0] // T.int64(16), C.data, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16))
                        for ax0_0, ax1_0 in T.grid(T.int64(2), T.int64(2)):
                            with T.block("matmul_intermediate_reindex_pad_shared.dyn_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v1_o = T.axis.spatial(T.int64(8) * ((n + T.int64(127)) // T.int64(128)), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax0_0)
                                v2_o = T.axis.spatial(T.int64(160), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax1_0)
                                T.reads(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                T.writes(matmul_intermediate_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                A = T.match_buffer(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=16)
                                C = T.match_buffer(matmul_intermediate_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=16)
                                T.tvm_store_matrix_sync(A.data, 16, 16, 16, A.elem_offset // A.strides[0] // T.int64(16) * (A.strides[0] // T.int64(16)) + A.elem_offset % A.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), C.data, C.elem_offset, C.strides[0] * T.int64(16), 2), C.strides[0], "row_major")
                        for ax0_ax1_fused_0 in range(T.int64(8)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("matmul_intermediate_reindex_pad_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), ax1_0_0_ax2_0_0_fused * T.int64(128) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) // T.int64(32))
                                        v2 = T.axis.spatial(T.int64(2560), ax1_0_1_ax2_0_1_fused * T.int64(128) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) % T.int64(32))
                                        T.reads(lv58[T.int64(0), v1, v2], matmul_intermediate_reindex_pad_shared_dyn[v0, v1, v2], param_1110[v2], lv2[T.int64(0), v1, v2])
                                        T.writes(T_add_intermediate_1_2[T.int64(0), v1, v2])
                                        T.block_attr({"buffer_dim_align": [[0, 1, 16, 4]]})
                                        if v1 < n:
                                            T_add_intermediate_1_2[T.int64(0), v1, v2] = lv58[T.int64(0), v1, v2] + (matmul_intermediate_reindex_pad_shared_dyn[v0, v1, v2] + param_1110[v2]) + lv2[T.int64(0), v1, v2]

    @T.prim_func(private=True)
    def fused_matmul10_add5_add7_add7_cast5(p_lv1748: T.handle, lv253: T.Buffer((T.int64(2560), T.int64(2560)), "float16"), param_5071: T.Buffer((T.int64(2560),), "float16"), p_lv1763: T.handle, p_lv1710: T.handle, p_output0: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        lv1748 = T.match_buffer(p_lv1748, (T.int64(1), n, T.int64(2560)), "float16")
        lv1763 = T.match_buffer(p_lv1763, (T.int64(1), n, T.int64(2560)), "float16")
        lv1710 = T.match_buffer(p_lv1710, (T.int64(1), n, T.int64(2560)), "float16")
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        lv1748_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(2560)), "float16", scope="shared.dyn")
        lv253_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(2560), T.int64(2560)), "float16", scope="shared.dyn")
        lv1748_reindex_pad_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(2560)), "float16", scope="wmma.matrix_a")
        lv253_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(1), T.int64(2560), T.int64(2560)), "float16", scope="wmma.matrix_b")
        matmul_intermediate_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(2560)), "float16", scope="shared.dyn")
        matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(2560)), "float16", scope="wmma.accumulator")
        for ax0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for ax1_0_0_ax2_0_0_fused in T.thread_binding((n + T.int64(127)) // T.int64(128), thread="blockIdx.x"):
                for ax1_0_1_ax2_0_1_fused in T.thread_binding(T.int64(20), thread="blockIdx.y"):
                    for ax2_0_2_ax1_0_2_fused in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                        for ax1_0_3_init, ax2_0_3_init in T.grid(T.int64(2), T.int64(2)):
                            with T.block("matmul_o_init"):
                                v0_o = T.axis.spatial(T.int64(1), ax0)
                                v1_o = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(8), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax1_0_3_init)
                                v2_o = T.axis.spatial(T.int64(160), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax2_0_3_init)
                                T.reads()
                                T.writes(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                with T.block("matmul_init_o"):
                                    v1_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v2_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads()
                                    T.writes(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    C = T.match_buffer(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_fill_fragment(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.float32(0))
                        for ax3_0_0 in T.serial(T.int64(40), annotations={"software_pipeline_order": [0, 3, 1, 4, 5, 2, 6], "software_pipeline_stage": [0, 0, 0, 0, 0, 1, 1]}):
                            for ax0_ax1_fused_0 in range(T.int64(4)):
                                for ax0_ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                                            with T.block("lv1748_reindex_pad_shared.dyn"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), ax1_0_0_ax2_0_0_fused * T.int64(128) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(64))
                                                v2 = T.axis.spatial(T.int64(2560), ax3_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(64))
                                                T.reads(lv1748[v0, v1, v2])
                                                T.writes(lv1748_reindex_pad_shared_dyn[v0, v1, v2])
                                                T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "double_buffer_scope": 0, "tir.manifest_shared_memory_local_stage": 1})
                                                lv1748_reindex_pad_shared_dyn[v0, v1, v2] = T.if_then_else(v1 < n, lv1748[v0, v1, v2], T.float16(0))
                            for ax0_ax1_fused_0 in range(T.int64(4)):
                                for ax0_ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                                            with T.block("lv253_reindex_shared.dyn"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(2560), ax1_0_1_ax2_0_1_fused * T.int64(128) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(64))
                                                v2 = T.axis.spatial(T.int64(2560), ax3_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(64))
                                                T.reads(lv253[v2, v1])
                                                T.writes(lv253_reindex_shared_dyn[v0, v1, v2])
                                                T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "double_buffer_scope": 0, "tir.manifest_shared_memory_local_stage": 1})
                                                lv253_reindex_shared_dyn[v0, v1, v2] = lv253[v2, v1]
                            for ax3_0_1 in T.serial(T.int64(4), annotations={"software_pipeline_order": [0, 1, 2], "software_pipeline_stage": [0, 0, 1]}):
                                for ax0_0 in T.unroll(T.int64(2)):
                                    for ax1_0 in T.unroll(T.int64(1)):
                                        with T.block("lv1748_reindex_pad_shared.dyn_wmma.matrix_a_o"):
                                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1_o = T.axis.spatial(T.int64(8) * ((n + T.int64(127)) // T.int64(128)), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax0_0)
                                            v2_o = T.axis.spatial(T.int64(160), ax3_0_0 * T.int64(4) + ax3_0_1 + ax1_0)
                                            T.reads(lv1748_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            T.writes(lv1748_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A = T.match_buffer(lv1748_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                            C = T.match_buffer(lv1748_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                            T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(16), 1), A.strides[0], "row_major")
                                for ax0_0 in T.unroll(T.int64(2)):
                                    for ax1_0 in T.unroll(T.int64(1)):
                                        with T.block("lv253_reindex_shared.dyn_wmma.matrix_b_o"):
                                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1_o = T.axis.spatial(T.int64(160), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax0_0)
                                            v2_o = T.axis.spatial(T.int64(160), ax3_0_0 * T.int64(4) + ax3_0_1 + ax1_0)
                                            T.reads(lv253_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            T.writes(lv253_reindex_shared_dyn_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A = T.match_buffer(lv253_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                            C = T.match_buffer(lv253_reindex_shared_dyn_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                            T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(16), 1), A.strides[0], "col_major")
                                for ax1_0_3, ax2_0_3 in T.grid(T.int64(2), T.int64(2)):
                                    with T.block("matmul_o_update"):
                                        v0_o = T.axis.spatial(T.int64(1), ax0)
                                        v1_o = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(8), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax1_0_3)
                                        v2_o = T.axis.spatial(T.int64(160), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax2_0_3)
                                        v3_o = T.axis.reduce(T.int64(160), ax3_0_0 * T.int64(4) + ax3_0_1)
                                        T.reads(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], lv1748_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], lv253_reindex_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                        T.writes(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                        with T.block("matmul_o"):
                                            v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v3_i_o = T.axis.reduce(T.int64(1), T.int64(0))
                                            T.reads(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], lv1748_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], lv253_reindex_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                            T.writes(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A = T.match_buffer(lv1748_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                            B = T.match_buffer(lv253_reindex_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                            C = T.match_buffer(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                            T.tvm_mma_sync(C.data, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), A.data, A.elem_offset // A.strides[0] // T.int64(16) * (A.strides[0] // T.int64(16)) + A.elem_offset % A.strides[0] // T.int64(16), B.data, B.elem_offset // B.strides[0] // T.int64(16) * (B.strides[0] // T.int64(16)) + B.elem_offset % B.strides[0] // T.int64(16), C.data, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16))
                        for ax0_0, ax1_0 in T.grid(T.int64(2), T.int64(2)):
                            with T.block("matmul_intermediate_reindex_pad_shared.dyn_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v1_o = T.axis.spatial(T.int64(8) * ((n + T.int64(127)) // T.int64(128)), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax0_0)
                                v2_o = T.axis.spatial(T.int64(160), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax1_0)
                                T.reads(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                T.writes(matmul_intermediate_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                A = T.match_buffer(matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=16)
                                C = T.match_buffer(matmul_intermediate_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=16)
                                T.tvm_store_matrix_sync(A.data, 16, 16, 16, A.elem_offset // A.strides[0] // T.int64(16) * (A.strides[0] // T.int64(16)) + A.elem_offset % A.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), C.data, C.elem_offset, C.strides[0] * T.int64(16), 2), C.strides[0], "row_major")
                        for ax0_ax1_fused_0 in range(T.int64(8)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("matmul_intermediate_reindex_pad_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), ax1_0_0_ax2_0_0_fused * T.int64(128) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) // T.int64(32))
                                        v2 = T.axis.spatial(T.int64(2560), ax1_0_1_ax2_0_1_fused * T.int64(128) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) % T.int64(32))
                                        T.reads(lv1763[T.int64(0), v1, v2], matmul_intermediate_reindex_pad_shared_dyn[v0, v1, v2], param_5071[v2], lv1710[T.int64(0), v1, v2])
                                        T.writes(compute_intermediate[T.int64(0), v1, v2])
                                        T.block_attr({"buffer_dim_align": [[0, 1, 16, 4]]})
                                        if v1 < n:
                                            compute_intermediate[T.int64(0), v1, v2] = T.Cast("float32", lv1763[T.int64(0), v1, v2] + (matmul_intermediate_reindex_pad_shared_dyn[v0, v1, v2] + param_5071[v2]) + lv1710[T.int64(0), v1, v2])

    @T.prim_func(private=True)
    def fused_matmul2_add1_add3_add3(lv1813: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), lv1: T.Buffer((T.int64(2560), T.int64(2560)), "float16"), param_11: T.Buffer((T.int64(2560),), "float16"), lv1828: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), lv1774: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), T_add_intermediate_1_2: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16", scope="local")
        matmul_intermediate_rf_local = T.alloc_buffer((T.int64(16), T.int64(1), T.int64(1), T.int64(2560)), "float16", scope="local")
        for ax0_fused_0 in T.thread_binding(T.int64(160), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                for ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                    with T.block("matmul_rf_init"):
                        vax1_fused_1 = T.axis.spatial(T.int64(16), ax1_fused_1)
                        v0 = T.axis.spatial(T.int64(2560), ax0_fused_0 * T.int64(16) + ax0_fused_1)
                        T.reads()
                        T.writes(matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0])
                        matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0] = T.float16(0)
                    for ax1_fused_0, u in T.grid(T.int64(160), 1):
                        with T.block("matmul_rf_update"):
                            vax1_fused_1 = T.axis.spatial(T.int64(16), ax1_fused_1)
                            v0 = T.axis.spatial(T.int64(2560), ax0_fused_0 * T.int64(16) + ax0_fused_1)
                            vax1_fused_0 = T.axis.reduce(T.int64(160), ax1_fused_0)
                            T.reads(matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0], lv1813[T.int64(0), T.int64(0), vax1_fused_0 * T.int64(16) + vax1_fused_1], lv1[vax1_fused_0 * T.int64(16) + vax1_fused_1, v0])
                            T.writes(matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0])
                            matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0] = matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0] + lv1813[T.int64(0), T.int64(0), vax1_fused_0 * T.int64(16) + vax1_fused_1] * lv1[vax1_fused_0 * T.int64(16) + vax1_fused_1, v0]
            for ax1_fused in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                for ax0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                    with T.block("matmul"):
                        vax1_fused_1 = T.axis.reduce(T.int64(16), ax0)
                        v0 = T.axis.spatial(T.int64(2560), ax0_fused_0 * T.int64(16) + ax1_fused)
                        T.reads(matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0])
                        T.writes(matmul_intermediate_local[T.int64(0), T.int64(0), v0])
                        with T.init():
                            matmul_intermediate_local[T.int64(0), T.int64(0), v0] = T.float16(0)
                        matmul_intermediate_local[T.int64(0), T.int64(0), v0] = matmul_intermediate_local[T.int64(0), T.int64(0), v0] + matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0]
            for ax0_fused_0_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                for ax0_fused_1 in range(T.int64(1)):
                    with T.block("T_add_2"):
                        v0 = T.axis.spatial(T.int64(2560), ax0_fused_0 * T.int64(16) + ax0_fused_0_1 + ax0_fused_1)
                        T.reads(lv1828[T.int64(0), T.int64(0), v0], matmul_intermediate_local[T.int64(0), T.int64(0), v0], param_11[v0], lv1774[T.int64(0), T.int64(0), v0])
                        T.writes(T_add_intermediate_1_2[T.int64(0), T.int64(0), v0])
                        T_add_intermediate_1_2[T.int64(0), T.int64(0), v0] = lv1828[T.int64(0), T.int64(0), v0] + (matmul_intermediate_local[T.int64(0), T.int64(0), v0] + param_11[v0]) + lv1774[T.int64(0), T.int64(0), v0]

    @T.prim_func(private=True)
    def fused_matmul2_add1_add3_add3_cast(lv3518: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), lv125: T.Buffer((T.int64(2560), T.int64(2560)), "float16"), param_507: T.Buffer((T.int64(2560),), "float16"), lv3533: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), lv3480: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), compute_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16", scope="local")
        matmul_intermediate_rf_local = T.alloc_buffer((T.int64(16), T.int64(1), T.int64(1), T.int64(2560)), "float16", scope="local")
        for ax0_fused_0 in T.thread_binding(T.int64(160), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                for ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                    with T.block("matmul_rf_init"):
                        vax1_fused_1 = T.axis.spatial(T.int64(16), ax1_fused_1)
                        v0 = T.axis.spatial(T.int64(2560), ax0_fused_0 * T.int64(16) + ax0_fused_1)
                        T.reads()
                        T.writes(matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0])
                        matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0] = T.float16(0)
                    for ax1_fused_0, u in T.grid(T.int64(160), 1):
                        with T.block("matmul_rf_update"):
                            vax1_fused_1 = T.axis.spatial(T.int64(16), ax1_fused_1)
                            v0 = T.axis.spatial(T.int64(2560), ax0_fused_0 * T.int64(16) + ax0_fused_1)
                            vax1_fused_0 = T.axis.reduce(T.int64(160), ax1_fused_0)
                            T.reads(matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0], lv3518[T.int64(0), T.int64(0), vax1_fused_0 * T.int64(16) + vax1_fused_1], lv125[vax1_fused_0 * T.int64(16) + vax1_fused_1, v0])
                            T.writes(matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0])
                            matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0] = matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0] + lv3518[T.int64(0), T.int64(0), vax1_fused_0 * T.int64(16) + vax1_fused_1] * lv125[vax1_fused_0 * T.int64(16) + vax1_fused_1, v0]
            for ax1_fused in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                for ax0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                    with T.block("matmul"):
                        vax1_fused_1 = T.axis.reduce(T.int64(16), ax0)
                        v0 = T.axis.spatial(T.int64(2560), ax0_fused_0 * T.int64(16) + ax1_fused)
                        T.reads(matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0])
                        T.writes(matmul_intermediate_local[T.int64(0), T.int64(0), v0])
                        with T.init():
                            matmul_intermediate_local[T.int64(0), T.int64(0), v0] = T.float16(0)
                        matmul_intermediate_local[T.int64(0), T.int64(0), v0] = matmul_intermediate_local[T.int64(0), T.int64(0), v0] + matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0]
            for ax0_fused_0_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                for ax0_fused_1 in range(T.int64(1)):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(2560), ax0_fused_0 * T.int64(16) + ax0_fused_0_1 + ax0_fused_1)
                        T.reads(lv3533[T.int64(0), T.int64(0), v0], matmul_intermediate_local[T.int64(0), T.int64(0), v0], param_507[v0], lv3480[T.int64(0), T.int64(0), v0])
                        T.writes(compute_intermediate[T.int64(0), T.int64(0), v0])
                        compute_intermediate[T.int64(0), T.int64(0), v0] = T.Cast("float32", lv3533[T.int64(0), T.int64(0), v0] + (matmul_intermediate_local[T.int64(0), T.int64(0), v0] + param_507[v0]) + lv3480[T.int64(0), T.int64(0), v0])

    @T.prim_func(private=True)
    def fused_min_max_triu_te_broadcast_to(p_output0: T.handle, n: T.int64):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        T_broadcast_to_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(1), n, n), "float16")
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding((n * n + T.int64(1023)) // T.int64(1024), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_broadcast_to"):
                    v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // n)
                    v1 = T.axis.spatial(n, (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % n)
                    T.where(ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 < n * n)
                    T.reads()
                    T.writes(T_broadcast_to_intermediate[T.int64(0), T.int64(0), v0, v1])
                    T_broadcast_to_intermediate[T.int64(0), T.int64(0), v0, v1] = T.Select(v0 < v1, T.float16(-65504), T.float16(65504))

    @T.prim_func(private=True)
    def fused_reshape2_split(lv1782: T.Buffer((T.int64(1), T.int64(1), T.int64(7680)), "float16"), T_split_sections_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float16"), T_split_sections_intermediate_1: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float16"), T_split_sections_intermediate_1_2: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float16")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(3), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_split_sections"):
                    v0 = T.axis.spatial(T.int64(32), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(80))
                    v1 = T.axis.spatial(T.int64(80), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(80))
                    T.where(ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 < T.int64(2560))
                    T.reads(lv1782[T.int64(0), T.int64(0), v0 * T.int64(240) + v1])
                    T.writes(T_split_sections_intermediate[T.int64(0), T.int64(0), v0, v1])
                    T_split_sections_intermediate[T.int64(0), T.int64(0), v0, v1] = lv1782[T.int64(0), T.int64(0), v0 * T.int64(240) + v1]
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(3), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_split_sections_1"):
                    v0 = T.axis.spatial(T.int64(32), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(80))
                    v1 = T.axis.spatial(T.int64(80), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(80))
                    T.where(ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 < T.int64(2560))
                    T.reads(lv1782[T.int64(0), T.int64(0), v0 * T.int64(240) + (v1 + T.int64(80))])
                    T.writes(T_split_sections_intermediate_1[T.int64(0), T.int64(0), v0, v1])
                    T_split_sections_intermediate_1[T.int64(0), T.int64(0), v0, v1] = lv1782[T.int64(0), T.int64(0), v0 * T.int64(240) + (v1 + T.int64(80))]
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(3), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_split_sections_2"):
                    v0 = T.axis.spatial(T.int64(32), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(80))
                    v1 = T.axis.spatial(T.int64(80), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(80))
                    T.where(ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 < T.int64(2560))
                    T.reads(lv1782[T.int64(0), T.int64(0), v0 * T.int64(240) + (v1 + T.int64(160))])
                    T.writes(T_split_sections_intermediate_1_2[T.int64(0), T.int64(0), v0, v1])
                    T_split_sections_intermediate_1_2[T.int64(0), T.int64(0), v0, v1] = lv1782[T.int64(0), T.int64(0), v0 * T.int64(240) + (v1 + T.int64(160))]

    @T.prim_func(private=True)
    def fused_slice1_cast4(lv3537: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0_fused_0 in T.thread_binding(T.int64(3), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(2560), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(2560))
                    T.reads(lv3537[T.int64(0), T.int64(0), v0])
                    T.writes(compute_intermediate[T.int64(0), T.int64(0), v0])
                    compute_intermediate[T.int64(0), T.int64(0), v0] = lv3537[T.int64(0), T.int64(0), v0]

    @T.prim_func(private=True)
    def fused_softmax1_cast3(p_lv1808: T.handle, p_output0: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        m = T.int64()
        lv1808 = T.match_buffer(p_lv1808, (T.int64(1), T.int64(32), T.int64(1), m))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), T.int64(1), m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)), scope="shared")
        for ax0_fused in T.thread_binding(T.int64(32), thread="blockIdx.x"):
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((m + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_fused + ax0)
                            v1 = T.axis.reduce(m, ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(256) + ax1_fused_1 < m)
                            T.reads(lv1808[T.int64(0), v0, T.int64(0), v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, T.int64(0)])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, T.int64(0)] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, T.int64(0)] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, T.int64(0)], lv1808[T.int64(0), v0, T.int64(0), v1])
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((m + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_fused + ax0)
                            v1 = T.axis.reduce(m, ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(256) + ax1_fused_1 < m)
                            T.reads(lv1808[T.int64(0), v0, T.int64(0), v1], T_softmax_maxelem_shared[T.int64(0), v0, T.int64(0)])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, T.int64(0)])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, T.int64(0)] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, T.int64(0)] = T_softmax_expsum_shared[T.int64(0), v0, T.int64(0)] + T.exp(lv1808[T.int64(0), v0, T.int64(0), v1] - T_softmax_maxelem_shared[T.int64(0), v0, T.int64(0)])
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial((m + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_fused)
                        v1 = T.axis.spatial(m, ax1_0 * T.int64(256) + ax1_1)
                        T.where(ax1_0 * T.int64(256) + ax1_1 < m)
                        T.reads(lv1808[T.int64(0), v0, T.int64(0), v1], T_softmax_maxelem_shared[T.int64(0), v0, T.int64(0)], T_softmax_expsum_shared[T.int64(0), v0, T.int64(0)])
                        T.writes(compute_intermediate[T.int64(0), v0, T.int64(0), v1])
                        compute_intermediate[T.int64(0), v0, T.int64(0), v1] = T.Cast("float16", T.exp(lv1808[T.int64(0), v0, T.int64(0), v1] - T_softmax_maxelem_shared[T.int64(0), v0, T.int64(0)]) / T_softmax_expsum_shared[T.int64(0), v0, T.int64(0)])

    @T.prim_func(private=True)
    def fused_softmax2_cast8(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(256) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(256) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(256) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(256) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(256) + ax2_1)
                        T.where(ax2_0 * T.int64(256) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1, v2])
                        compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])

    @T.prim_func(private=True)
    def fused_squeeze(lv1784_2: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float16"), T_squeeze_intermediate: T.Buffer((T.int64(1), T.int64(32), T.int64(80)), "float16")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(3), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_squeeze"):
                    v0 = T.axis.spatial(T.int64(32), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(80))
                    v1 = T.axis.spatial(T.int64(80), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(80))
                    T.where(ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 < T.int64(2560))
                    T.reads(lv1784_2[T.int64(0), T.int64(0), v0, v1])
                    T.writes(T_squeeze_intermediate[T.int64(0), v0, v1])
                    T_squeeze_intermediate[T.int64(0), v0, v1] = lv1784_2[T.int64(0), T.int64(0), v0, v1]

    @T.prim_func(private=True)
    def fused_squeeze1(p_lv14_2: T.handle, p_output0: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        lv14_2 = T.match_buffer(p_lv14_2, (T.int64(1), n, T.int64(32), T.int64(80)), "float16")
        T_squeeze_intermediate = T.match_buffer(p_output0, (n, T.int64(32), T.int64(80)), "float16")
        # with T.block("root"):
        for ax0_ax1_ax2_fused_0 in T.thread_binding((n * T.int64(2560) + T.int64(1023)) // T.int64(1024), thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_squeeze"):
                    v0 = T.axis.spatial(n, (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(2560))
                    v1 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(2560) // T.int64(80))
                    v2 = T.axis.spatial(T.int64(80), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(80))
                    T.where(ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1 < n * T.int64(2560))
                    T.reads(lv14_2[T.int64(0), v0, v1, v2])
                    T.writes(T_squeeze_intermediate[v0, v1, v2])
                    T_squeeze_intermediate[v0, v1, v2] = lv14_2[T.int64(0), v0, v1, v2]

    @T.prim_func(private=True)
    def fused_transpose2_reshape4(lv1811: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(80)), "float16"), T_reshape_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0_fused_0 in T.thread_binding(T.int64(3), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(T.int64(2560), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(2560))
                    T.reads(lv1811[T.int64(0), v0 // T.int64(80), T.int64(0), v0 % T.int64(80)])
                    T.writes(T_reshape_intermediate[T.int64(0), T.int64(0), v0])
                    T_reshape_intermediate[T.int64(0), T.int64(0), v0] = lv1811[T.int64(0), v0 // T.int64(80), T.int64(0), v0 % T.int64(80)]

    @T.prim_func(private=True)
    def layer_norm(A: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), B: T.Buffer((T.int64(2560),), "float32"), C: T.Buffer((T.int64(2560),), "float32"), T_layer_norm: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_red_temp_v0_shared = T.alloc_buffer((T.int64(1), T.int64(1)), scope="shared")
        A_red_temp_v1_shared = T.alloc_buffer((T.int64(1), T.int64(1)), scope="shared")
        for ax0_fused in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.reads(A[T.int64(0), T.int64(0), v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), T.int64(0)], A_red_temp_v1_shared[T.int64(0), T.int64(0)])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), T.int64(0)] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), T.int64(0)] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), T.int64(0)] + A[T.int64(0), T.int64(0), v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), T.int64(0)] + A[T.int64(0), T.int64(0), v1] * A[T.int64(0), T.int64(0), v1]
                            A_red_temp_v0_shared[T.int64(0), T.int64(0)] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), T.int64(0)] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_layer_norm"):
                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(256) + ax1_1)
                        T.reads(A[T.int64(0), T.int64(0), v1], A_red_temp_v0_shared[T.int64(0), T.int64(0)], A_red_temp_v1_shared[T.int64(0), T.int64(0)], B[v1], C[v1])
                        T.writes(T_layer_norm[T.int64(0), T.int64(0), v1])
                        T_layer_norm[T.int64(0), T.int64(0), v1] = (A[T.int64(0), T.int64(0), v1] - A_red_temp_v0_shared[T.int64(0), T.int64(0)] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), T.int64(0)] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), T.int64(0)] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), T.int64(0)] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * B[v1] + C[v1]

    @T.prim_func(private=True)
    def layer_norm1(var_A: T.handle, B: T.Buffer((T.int64(2560),), "float32"), C: T.Buffer((T.int64(2560),), "float32"), var_T_layer_norm: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        T_layer_norm = T.match_buffer(var_T_layer_norm, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        A_red_temp_v0_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        A_red_temp_v1_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        for ax0_fused in T.thread_binding(n, thread="blockIdx.x"):
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + A[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + A[T.int64(0), v0, v1] * A[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_layer_norm"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(256) + ax1_1)
                        T.reads(A[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], B[v1], C[v1])
                        T.writes(T_layer_norm[T.int64(0), v0, v1])
                        T_layer_norm[T.int64(0), v0, v1] = (A[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * B[v1] + C[v1]

    @T.prim_func(private=True)
    def matmul1(var_A: T.handle, var_B: T.handle, matmul: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(80)), "float16")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        m = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(32), T.int64(1), m), "float16")
        B = T.match_buffer(var_B, (T.int64(1), T.int64(32), m, T.int64(80)), "float16")
        # with T.block("root"):
        matmul_rf_local = T.alloc_buffer((T.int64(16), T.int64(1), T.int64(32), T.int64(1), T.int64(80)), "float16", scope="local")
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(160), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                for ax2_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                    with T.block("matmul_rf_init"):
                        vax2_fused_1 = T.axis.spatial(T.int64(16), ax2_fused_1)
                        v0 = T.axis.spatial(T.int64(32), (ax0_ax1_fused_0 * T.int64(16) + ax0_ax1_fused_1) // T.int64(80))
                        v1 = T.axis.spatial(T.int64(80), (ax0_ax1_fused_0 * T.int64(16) + ax0_ax1_fused_1) % T.int64(80))
                        T.reads()
                        T.writes(matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1])
                        matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1] = T.float16(0)
                    for ax2_fused_0, u in T.grid((m + T.int64(15)) // T.int64(16), 1):
                        with T.block("matmul_rf_update"):
                            vax2_fused_1 = T.axis.spatial(T.int64(16), ax2_fused_1)
                            v0 = T.axis.spatial(T.int64(32), (ax0_ax1_fused_0 * T.int64(16) + ax0_ax1_fused_1) // T.int64(80))
                            v1 = T.axis.spatial(T.int64(80), (ax0_ax1_fused_0 * T.int64(16) + ax0_ax1_fused_1) % T.int64(80))
                            vax2_fused_0 = T.axis.reduce((m + T.int64(15)) // T.int64(16), ax2_fused_0)
                            T.where(ax2_fused_0 * T.int64(16) + ax2_fused_1 < m)
                            T.reads(matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1], A[T.int64(0), v0, T.int64(0), vax2_fused_0 * T.int64(16) + vax2_fused_1], B[T.int64(0), v0, vax2_fused_0 * T.int64(16) + vax2_fused_1, v1])
                            T.writes(matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1])
                            matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1] = matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1] + A[T.int64(0), v0, T.int64(0), vax2_fused_0 * T.int64(16) + vax2_fused_1] * B[T.int64(0), v0, vax2_fused_0 * T.int64(16) + vax2_fused_1, v1]
            for ax1_ax2_fused in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                for ax0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                    with T.block("matmul"):
                        vax2_fused_1 = T.axis.reduce(T.int64(16), ax0)
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused_0 // T.int64(5))
                        v1 = T.axis.spatial(T.int64(80), ax0_ax1_fused_0 % T.int64(5) * T.int64(16) + ax1_ax2_fused)
                        T.reads(matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1])
                        T.writes(matmul[T.int64(0), v0, T.int64(0), v1])
                        with T.init():
                            matmul[T.int64(0), v0, T.int64(0), v1] = T.float16(0)
                        matmul[T.int64(0), v0, T.int64(0), v1] = matmul[T.int64(0), v0, T.int64(0), v1] + matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1]

    @T.prim_func(private=True)
    def matmul9(var_A: T.handle, var_B: T.handle, var_matmul: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(32), n, m), "float16")
        B = T.match_buffer(var_B, (T.int64(1), T.int64(32), m, T.int64(80)), "float16")
        matmul = T.match_buffer(var_matmul, (T.int64(1), T.int64(32), n, T.int64(80)), "float16")
        # with T.block("root"):
        A_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(32), (n + T.int64(127)) // T.int64(128) * T.int64(128), (m + T.int64(63)) // T.int64(64) * T.int64(64)), "float16", scope="shared.dyn")
        B_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(32), T.int64(128), (m + T.int64(63)) // T.int64(64) * T.int64(64)), "float16", scope="shared.dyn")
        A_reindex_pad_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(32), (n + T.int64(127)) // T.int64(128) * T.int64(128), (m + T.int64(63)) // T.int64(64) * T.int64(64)), "float16", scope="wmma.matrix_a")
        B_reindex_pad_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(32), T.int64(128), (m + T.int64(63)) // T.int64(64) * T.int64(64)), "float16", scope="wmma.matrix_b")
        matmul_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(32), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(128)), "float16", scope="shared.dyn")
        matmul_reindex_pad_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(32), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(128)), "float16", scope="wmma.accumulator")
        for ax0 in T.thread_binding(T.int64(32), thread="blockIdx.z"):
            for ax1_0_0_ax2_0_0_fused in T.thread_binding((n + T.int64(127)) // T.int64(128), thread="blockIdx.x"):
                for ax1_0_1_ax2_0_1_fused in T.thread_binding(T.int64(1), thread="blockIdx.y"):
                    for ax2_0_2_ax1_0_2_fused in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                        for ax1_0_3_init, ax2_0_3_init in T.grid(T.int64(2), T.int64(2)):
                            with T.block("matmul_o_init"):
                                v0_o = T.axis.spatial(T.int64(32), ax0)
                                v1_o = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(8), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax1_0_3_init)
                                v2_o = T.axis.spatial(T.int64(8), ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax2_0_3_init)
                                T.reads()
                                T.writes(matmul_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                with T.block("matmul_init_o"):
                                    v1_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v2_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads()
                                    T.writes(matmul_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    C = T.match_buffer(matmul_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_fill_fragment(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.float32(0))
                        for ax3_0_0 in T.serial((m + T.int64(63)) // T.int64(64), annotations={"software_pipeline_order": [0, 3, 1, 4, 5, 2, 6], "software_pipeline_stage": [0, 0, 0, 0, 0, 1, 1]}):
                            for ax0_ax1_fused_0 in range(T.int64(4)):
                                for ax0_ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                                            with T.block("A_reindex_pad_shared.dyn"):
                                                v0 = T.axis.spatial(T.int64(32), ax0)
                                                v1 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), ax1_0_0_ax2_0_0_fused * T.int64(128) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(64))
                                                v2 = T.axis.spatial((m + T.int64(63)) // T.int64(64) * T.int64(64), ax3_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(64))
                                                T.reads(A[T.int64(0), v0, v1, v2])
                                                T.writes(A_reindex_pad_shared_dyn[v0, v1, v2])
                                                T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "double_buffer_scope": 0, "tir.manifest_shared_memory_local_stage": 1})
                                                A_reindex_pad_shared_dyn[v0, v1, v2] = T.if_then_else(v1 < n and v2 < m, A[T.int64(0), v0, v1, v2], T.float16(0))
                            for ax0_ax1_fused_0 in range(T.int64(4)):
                                for ax0_ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                                            with T.block("B_reindex_pad_shared.dyn"):
                                                v0 = T.axis.spatial(T.int64(32), ax0)
                                                v1 = T.axis.spatial(T.int64(128), (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(64))
                                                v2 = T.axis.spatial((m + T.int64(63)) // T.int64(64) * T.int64(64), ax3_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(64))
                                                T.reads(B[T.int64(0), v0, v2, v1])
                                                T.writes(B_reindex_pad_shared_dyn[v0, v1, v2])
                                                T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "double_buffer_scope": 0, "tir.manifest_shared_memory_local_stage": 1})
                                                B_reindex_pad_shared_dyn[v0, v1, v2] = T.if_then_else(v1 < T.int64(80) and v2 < m, B[T.int64(0), v0, v2, v1], T.float16(0))
                            for ax3_0_1 in T.serial(T.int64(4), annotations={"software_pipeline_order": [0, 1, 2], "software_pipeline_stage": [0, 0, 1]}):
                                for ax0_0 in T.unroll(T.int64(2)):
                                    for ax1_0 in T.unroll(T.int64(1)):
                                        with T.block("A_reindex_pad_shared.dyn_wmma.matrix_a_o"):
                                            v0_o = T.axis.spatial(T.int64(32), ax0)
                                            v1_o = T.axis.spatial(T.int64(8) * ((n + T.int64(127)) // T.int64(128)), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax0_0)
                                            v2_o = T.axis.spatial(T.int64(4) * ((m + T.int64(63)) // T.int64(64)), ax3_0_0 * T.int64(4) + ax3_0_1 + ax1_0)
                                            T.reads(A_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            T.writes(A_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A_1 = T.match_buffer(A_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                            C = T.match_buffer(A_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                            T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "row_major")
                                for ax0_0 in T.unroll(T.int64(2)):
                                    for ax1_0 in T.unroll(T.int64(1)):
                                        with T.block("B_reindex_pad_shared.dyn_wmma.matrix_b_o"):
                                            v0_o = T.axis.spatial(T.int64(32), ax0)
                                            v1_o = T.axis.spatial(T.int64(8), ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax0_0)
                                            v2_o = T.axis.spatial(T.int64(4) * ((m + T.int64(63)) // T.int64(64)), ax3_0_0 * T.int64(4) + ax3_0_1 + ax1_0)
                                            T.reads(B_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            T.writes(B_reindex_pad_shared_dyn_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A_1 = T.match_buffer(B_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                            C = T.match_buffer(B_reindex_pad_shared_dyn_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                            T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "col_major")
                                for ax1_0_3, ax2_0_3 in T.grid(T.int64(2), T.int64(2)):
                                    with T.block("matmul_o_update"):
                                        v0_o = T.axis.spatial(T.int64(32), ax0)
                                        v1_o = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(8), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax1_0_3)
                                        v2_o = T.axis.spatial(T.int64(8), ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax2_0_3)
                                        v3_o = T.axis.reduce((m + T.int64(63)) // T.int64(64) * T.int64(4), ax3_0_0 * T.int64(4) + ax3_0_1)
                                        T.reads(matmul_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], A_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], B_reindex_pad_shared_dyn_wmma_matrix_b[v0_o, v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                        T.writes(matmul_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                        with T.block("matmul_o"):
                                            v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v3_i_o = T.axis.reduce(T.int64(1), T.int64(0))
                                            T.reads(matmul_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], A_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], B_reindex_pad_shared_dyn_wmma_matrix_b[v0_o, v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                            T.writes(matmul_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A_1 = T.match_buffer(A_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                            B_1 = T.match_buffer(B_reindex_pad_shared_dyn_wmma_matrix_b[v0_o, v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                            C = T.match_buffer(matmul_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                            T.tvm_mma_sync(C.data, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), A_1.data, A_1.elem_offset // A_1.strides[0] // T.int64(16) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), B_1.data, B_1.elem_offset // B_1.strides[0] // T.int64(16) * (B_1.strides[0] // T.int64(16)) + B_1.elem_offset % B_1.strides[0] // T.int64(16), C.data, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16))
                        for ax0_0, ax1_0 in T.grid(T.int64(2), T.int64(2)):
                            with T.block("matmul_reindex_pad_shared.dyn_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(T.int64(32), ax0)
                                v1_o = T.axis.spatial(T.int64(8) * ((n + T.int64(127)) // T.int64(128)), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax0_0)
                                v2_o = T.axis.spatial(T.int64(8), ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax1_0)
                                T.reads(matmul_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                T.writes(matmul_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                A_1 = T.match_buffer(matmul_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=16)
                                C = T.match_buffer(matmul_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=16)
                                T.tvm_store_matrix_sync(A_1.data, 16, 16, 16, A_1.elem_offset // A_1.strides[0] // T.int64(16) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), C.data, C.elem_offset, C.strides[0] * T.int64(16), 2), C.strides[0], "row_major")
                        for ax0_ax1_fused_0 in range(T.int64(8)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("matmul_reindex_pad_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(32), ax0)
                                        v1 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), ax1_0_0_ax2_0_0_fused * T.int64(128) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) // T.int64(32))
                                        v2 = T.axis.spatial(T.int64(128), ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) % T.int64(32))
                                        T.reads(matmul_reindex_pad_shared_dyn[v0, v1, v2])
                                        T.writes(matmul[T.int64(0), v0, v1, v2])
                                        T.block_attr({"buffer_dim_align": [[0, 1, 16, 4]]})
                                        if v1 < n and v2 < T.int64(80):
                                            matmul[T.int64(0), v0, v1, v2] = matmul_reindex_pad_shared_dyn[v0, v1, v2]

    @T.prim_func(private=True)
    def reshape(A: T.Buffer((T.int64(1), T.int64(1)), "int32"), T_reshape: T.Buffer((T.int64(1),), "int32")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(1))
                    T.reads(A[T.int64(0), T.int64(0)])
                    T.writes(T_reshape[T.int64(0)])
                    T_reshape[T.int64(0)] = A[T.int64(0), T.int64(0)]

    @T.prim_func(private=True)
    def reshape1(A: T.Buffer((T.int64(1), T.int64(2560)), "float16"), T_reshape: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0_fused_0 in T.thread_binding(T.int64(3), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(T.int64(2560), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(2560))
                    T.reads(A[T.int64(0), v0])
                    T.writes(T_reshape[T.int64(0), T.int64(0), v0])
                    T_reshape[T.int64(0), T.int64(0), v0] = A[T.int64(0), v0]

    @T.prim_func(private=True)
    def reshape3(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        m = T.int64()
        A = T.match_buffer(var_A, (m, T.int64(32), T.int64(80)), "float16")
        T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), m, T.int64(32), T.int64(80)), "float16")
        # with T.block("root"):
        for ax0_ax1_ax2_fused_0 in T.thread_binding((m * T.int64(2560) + T.int64(1023)) // T.int64(1024), thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(m, (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(2560))
                    v1 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(2560) // T.int64(80))
                    v2 = T.axis.spatial(T.int64(80), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(80))
                    T.where(ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1 < m * T.int64(2560))
                    T.reads(A[v0, v1, v2])
                    T.writes(T_reshape[T.int64(0), v0, v1, v2])
                    T_reshape[T.int64(0), v0, v1, v2] = A[v0, v1, v2]

    @T.prim_func(private=True)
    def reshape5(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n), "int32")
        T_reshape = T.match_buffer(var_T_reshape, (n,), "int32")
        # with T.block("root"):
        for ax0_fused_0 in T.thread_binding((n + T.int64(1023)) // T.int64(1024), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(n, ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < n)
                    T.reads(A[T.int64(0), v0])
                    T.writes(T_reshape[v0])
                    T_reshape[v0] = A[T.int64(0), v0]

    @T.prim_func(private=True)
    def reshape6(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (n, T.int64(2560)), "float16")
        T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding((n * T.int64(2560) + T.int64(1023)) // T.int64(1024), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(2560))
                    v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(2560))
                    T.where(ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 < n * T.int64(2560))
                    T.reads(A[v0, v1])
                    T.writes(T_reshape[T.int64(0), v0, v1])
                    T_reshape[T.int64(0), v0, v1] = A[v0, v1]

    @T.prim_func(private=True)
    def reshape7(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(7680)), "float16")
        T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), n, T.int64(32), T.int64(240)), "float16")
        # with T.block("root"):
        for ax0_ax1_ax2_fused_0 in T.thread_binding((n * T.int64(7680) + T.int64(1023)) // T.int64(1024), thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(n, (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(7680))
                    v1 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(7680) // T.int64(240))
                    v2 = T.axis.spatial(T.int64(240), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(240))
                    T.where(ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1 < n * T.int64(7680))
                    T.reads(A[T.int64(0), v0, v1 * T.int64(240) + v2])
                    T.writes(T_reshape[T.int64(0), v0, v1, v2])
                    T_reshape[T.int64(0), v0, v1, v2] = A[T.int64(0), v0, v1 * T.int64(240) + v2]

    @T.prim_func(private=True)
    def reshape8(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(80)), "float16")
        T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding((n * T.int64(2560) + T.int64(1023)) // T.int64(1024), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(2560))
                    v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(2560))
                    T.where(ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 < n * T.int64(2560))
                    T.reads(A[T.int64(0), v0, v1 // T.int64(80), v1 % T.int64(80)])
                    T.writes(T_reshape[T.int64(0), v0, v1])
                    T_reshape[T.int64(0), v0, v1] = A[T.int64(0), v0, v1 // T.int64(80), v1 % T.int64(80)]

    @T.prim_func(private=True)
    def rotary_embedding(var_A: T.handle, B: T.Buffer((T.int64(4096), T.int64(20)), "float16"), C: T.Buffer((T.int64(4096), T.int64(20)), "float16"), var_rotary: T.handle, m: T.int64):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(80)), "float16")
        rotary = T.match_buffer(var_rotary, (T.int64(1), n, T.int64(32), T.int64(80)), "float16")
        # with T.block("root"):
        for ax0_ax1_ax2_fused_0 in T.thread_binding((n * T.int64(2560) + T.int64(1023)) // T.int64(1024), thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("rotary"):
                    v0 = T.axis.spatial(n, (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(2560))
                    v1 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(2560) // T.int64(80))
                    v2 = T.axis.spatial(T.int64(80), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(80))
                    T.where(ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1 < n * T.int64(2560))
                    T.reads(B[v0 + (m - n), v2], A[T.int64(0), v0, v1, v2 + T.int64(-10):v2 + T.int64(-10) + T.int64(21)], C[v0 + (m - n), v2])
                    T.writes(rotary[T.int64(0), v0, v1, v2])
                    rotary[T.int64(0), v0, v1, v2] = T.Select(v2 < T.int64(20), B[v0 + (m - n), v2] * A[T.int64(0), v0, v1, v2] + C[v0 + (m - n), v2] * T.Select(v2 < T.int64(10), A[T.int64(0), v0, v1, v2 + T.int64(10)] * T.float16(-1), A[T.int64(0), v0, v1, v2 + T.int64(-10)]), A[T.int64(0), v0, v1, v2])

    @T.prim_func(private=True)
    def rotary_embedding1(A: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float16"), B: T.Buffer((T.int64(4096), T.int64(20)), "float16"), C: T.Buffer((T.int64(4096), T.int64(20)), "float16"), rotary: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float16"), m: T.int64):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(3), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("rotary"):
                    v0 = T.axis.spatial(T.int64(32), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(80))
                    v1 = T.axis.spatial(T.int64(80), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(80))
                    T.where(ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 < T.int64(2560))
                    T.reads(B[m - T.int64(1), v1], A[T.int64(0), T.int64(0), v0, v1 + T.int64(-10):v1 + T.int64(-10) + T.int64(21)], C[m - T.int64(1), v1])
                    T.writes(rotary[T.int64(0), T.int64(0), v0, v1])
                    rotary[T.int64(0), T.int64(0), v0, v1] = T.Select(v1 < T.int64(20), B[m - T.int64(1), v1] * A[T.int64(0), T.int64(0), v0, v1] + C[m - T.int64(1), v1] * T.Select(v1 < T.int64(10), A[T.int64(0), T.int64(0), v0, v1 + T.int64(10)] * T.float16(-1), A[T.int64(0), T.int64(0), v0, v1 + T.int64(-10)]), A[T.int64(0), T.int64(0), v0, v1])

    @T.prim_func(private=True)
    def slice(var_A: T.handle, slice: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_fused_0 in T.thread_binding(T.int64(3), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("slice"):
                    v0 = T.axis.spatial(T.int64(2560), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(2560))
                    T.reads(A[T.int64(0), n - T.int64(1), v0])
                    T.writes(slice[T.int64(0), T.int64(0), v0])
                    slice[T.int64(0), T.int64(0), v0] = A[T.int64(0), n - T.int64(1), v0]

    @T.prim_func(private=True)
    def softmax(A: T.Buffer((T.int64(1), T.int64(1), T.int64(50280)), "float32"), T_softmax_norm: T.Buffer((T.int64(1), T.int64(1), T.int64(50280)), "float32")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(1)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(1)), scope="shared")
        for ax0_fused in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(197), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.reduce(T.int64(50280), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(256) + ax1_fused_1 < T.int64(50280))
                            T.reads(A[T.int64(0), T.int64(0), v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), T.int64(0)])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), T.int64(0)] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), T.int64(0)] = T.max(T_softmax_maxelem_shared[T.int64(0), T.int64(0)], A[T.int64(0), T.int64(0), v1])
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(197), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.reduce(T.int64(50280), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(256) + ax1_fused_1 < T.int64(50280))
                            T.reads(A[T.int64(0), T.int64(0), v1], T_softmax_maxelem_shared[T.int64(0), T.int64(0)])
                            T.writes(T_softmax_expsum_shared[T.int64(0), T.int64(0)])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), T.int64(0)] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), T.int64(0)] = T_softmax_expsum_shared[T.int64(0), T.int64(0)] + T.exp(A[T.int64(0), T.int64(0), v1] - T_softmax_maxelem_shared[T.int64(0), T.int64(0)])
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(197), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_norm"):
                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v1 = T.axis.spatial(T.int64(50280), ax1_0 * T.int64(256) + ax1_1)
                        T.where(ax1_0 * T.int64(256) + ax1_1 < T.int64(50280))
                        T.reads(A[T.int64(0), T.int64(0), v1], T_softmax_maxelem_shared[T.int64(0), T.int64(0)], T_softmax_expsum_shared[T.int64(0), T.int64(0)])
                        T.writes(T_softmax_norm[T.int64(0), T.int64(0), v1])
                        T.block_attr({"axis": 2})
                        T_softmax_norm[T.int64(0), T.int64(0), v1] = T.exp(A[T.int64(0), T.int64(0), v1] - T_softmax_maxelem_shared[T.int64(0), T.int64(0)]) / T_softmax_expsum_shared[T.int64(0), T.int64(0)]

    @T.prim_func(private=True)
    def split1(var_A: T.handle, var_T_split_sections: T.handle, var_T_split_sections_1: T.handle, var_T_split_sections_2: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(240)), "float16")
        T_split_sections = T.match_buffer(var_T_split_sections, (T.int64(1), n, T.int64(32), T.int64(80)), "float16")
        T_split_sections_1 = T.match_buffer(var_T_split_sections_1, (T.int64(1), n, T.int64(32), T.int64(80)), "float16")
        T_split_sections_2 = T.match_buffer(var_T_split_sections_2, (T.int64(1), n, T.int64(32), T.int64(80)), "float16")
        # with T.block("root"):
        for ax0_ax1_ax2_fused_0 in T.thread_binding((n * T.int64(2560) + T.int64(1023)) // T.int64(1024), thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_split_sections"):
                    v0 = T.axis.spatial(n, (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(2560))
                    v1 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(2560) // T.int64(80))
                    v2 = T.axis.spatial(T.int64(80), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(80))
                    T.where(ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1 < n * T.int64(2560))
                    T.reads(A[T.int64(0), v0, v1, v2])
                    T.writes(T_split_sections[T.int64(0), v0, v1, v2])
                    T_split_sections[T.int64(0), v0, v1, v2] = A[T.int64(0), v0, v1, v2]
        for ax0_ax1_ax2_fused_0 in T.thread_binding((n * T.int64(2560) + T.int64(1023)) // T.int64(1024), thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_split_sections_1"):
                    v0 = T.axis.spatial(n, (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(2560))
                    v1 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(2560) // T.int64(80))
                    v2 = T.axis.spatial(T.int64(80), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(80))
                    T.where(ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1 < n * T.int64(2560))
                    T.reads(A[T.int64(0), v0, v1, v2 + T.int64(80)])
                    T.writes(T_split_sections_1[T.int64(0), v0, v1, v2])
                    T_split_sections_1[T.int64(0), v0, v1, v2] = A[T.int64(0), v0, v1, v2 + T.int64(80)]
        for ax0_ax1_ax2_fused_0 in T.thread_binding((n * T.int64(2560) + T.int64(1023)) // T.int64(1024), thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_split_sections_2"):
                    v0 = T.axis.spatial(n, (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(2560))
                    v1 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(2560) // T.int64(80))
                    v2 = T.axis.spatial(T.int64(80), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(80))
                    T.where(ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1 < n * T.int64(2560))
                    T.reads(A[T.int64(0), v0, v1, v2 + T.int64(160)])
                    T.writes(T_split_sections_2[T.int64(0), v0, v1, v2])
                    T_split_sections_2[T.int64(0), v0, v1, v2] = A[T.int64(0), v0, v1, v2 + T.int64(160)]

    @T.prim_func(private=True)
    def squeeze(A: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float16"), T_squeeze: T.Buffer((T.int64(1), T.int64(32), T.int64(80)), "float16")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(3), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_squeeze"):
                    v0 = T.axis.spatial(T.int64(32), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(80))
                    v1 = T.axis.spatial(T.int64(80), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(80))
                    T.where(ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 < T.int64(2560))
                    T.reads(A[T.int64(0), T.int64(0), v0, v1])
                    T.writes(T_squeeze[T.int64(0), v0, v1])
                    T_squeeze[T.int64(0), v0, v1] = A[T.int64(0), T.int64(0), v0, v1]

    @T.prim_func(private=True)
    def squeeze1(var_A: T.handle, var_T_squeeze: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(80)), "float16")
        T_squeeze = T.match_buffer(var_T_squeeze, (n, T.int64(32), T.int64(80)), "float16")
        # with T.block("root"):
        for ax0_ax1_ax2_fused_0 in T.thread_binding((n * T.int64(2560) + T.int64(1023)) // T.int64(1024), thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_squeeze"):
                    v0 = T.axis.spatial(n, (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(2560))
                    v1 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(2560) // T.int64(80))
                    v2 = T.axis.spatial(T.int64(80), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(80))
                    T.where(ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1 < n * T.int64(2560))
                    T.reads(A[T.int64(0), v0, v1, v2])
                    T.writes(T_squeeze[v0, v1, v2])
                    T_squeeze[v0, v1, v2] = A[T.int64(0), v0, v1, v2]

    @T.prim_func(private=True)
    def transpose(A: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float16"), T_transpose: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(80)), "float16")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(3), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_transpose"):
                    v0 = T.axis.spatial(T.int64(32), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(80))
                    v1 = T.axis.spatial(T.int64(80), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(80))
                    T.where(ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 < T.int64(2560))
                    T.reads(A[T.int64(0), T.int64(0), v0, v1])
                    T.writes(T_transpose[T.int64(0), v0, T.int64(0), v1])
                    T_transpose[T.int64(0), v0, T.int64(0), v1] = A[T.int64(0), T.int64(0), v0, v1]

    @T.prim_func(private=True)
    def transpose1(var_A: T.handle, var_T_transpose: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        m = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), m, T.int64(32), T.int64(80)), "float16")
        T_transpose = T.match_buffer(var_T_transpose, (T.int64(1), T.int64(32), m, T.int64(80)), "float16")
        # with T.block("root"):
        for ax0_ax1_ax2_fused_0 in T.thread_binding((m * T.int64(2560) + T.int64(1023)) // T.int64(1024), thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_transpose"):
                    v0 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // (T.int64(80) * m))
                    v1 = T.axis.spatial(m, (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % (T.int64(80) * m) // T.int64(80))
                    v2 = T.axis.spatial(T.int64(80), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(80))
                    T.where(ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1 < m * T.int64(2560))
                    T.reads(A[T.int64(0), v1, v0, v2])
                    T.writes(T_transpose[T.int64(0), v0, v1, v2])
                    T_transpose[T.int64(0), v0, v1, v2] = A[T.int64(0), v1, v0, v2]

    @T.prim_func(private=True)
    def transpose5(var_A: T.handle, var_T_transpose: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(32), n, T.int64(80)), "float16")
        T_transpose = T.match_buffer(var_T_transpose, (T.int64(1), n, T.int64(32), T.int64(80)), "float16")
        # with T.block("root"):
        for ax0_ax1_ax2_fused_0 in T.thread_binding((n * T.int64(2560) + T.int64(1023)) // T.int64(1024), thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_transpose"):
                    v0 = T.axis.spatial(n, (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(2560))
                    v1 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(2560) // T.int64(80))
                    v2 = T.axis.spatial(T.int64(80), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(80))
                    T.where(ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1 < n * T.int64(2560))
                    T.reads(A[T.int64(0), v1, v0, v2])
                    T.writes(T_transpose[T.int64(0), v0, v1, v2])
                    T_transpose[T.int64(0), v0, v1, v2] = A[T.int64(0), v1, v0, v2]

    @R.function
    def create_kv_cache() -> R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object):
        R.func_attr({"tir_var_upper_bound": {"m": 4096, "n": 4096}})
        with R.dataflow():
            lv3543: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3544: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3545: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3546: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3547: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3548: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3549: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3550: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3551: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3552: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3553: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3554: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3555: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3556: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3557: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3558: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3559: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3560: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3561: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3562: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3563: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3564: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3565: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3566: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3567: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3568: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3569: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3570: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3571: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3572: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3573: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3574: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3575: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3576: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3577: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3578: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3579: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3580: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3581: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3582: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3583: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3584: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3585: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3586: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3587: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3588: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3589: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3590: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3591: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3592: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3593: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3594: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3595: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3596: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3597: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3598: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3599: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3600: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3601: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3602: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3603: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3604: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3605: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3606: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            gv2: R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object) = lv3543, lv3544, lv3545, lv3546, lv3547, lv3548, lv3549, lv3550, lv3551, lv3552, lv3553, lv3554, lv3555, lv3556, lv3557, lv3558, lv3559, lv3560, lv3561, lv3562, lv3563, lv3564, lv3565, lv3566, lv3567, lv3568, lv3569, lv3570, lv3571, lv3572, lv3573, lv3574, lv3575, lv3576, lv3577, lv3578, lv3579, lv3580, lv3581, lv3582, lv3583, lv3584, lv3585, lv3586, lv3587, lv3588, lv3589, lv3590, lv3591, lv3592, lv3593, lv3594, lv3595, lv3596, lv3597, lv3598, lv3599, lv3600, lv3601, lv3602, lv3603, lv3604, lv3605, lv3606
            R.output(gv2)
        return gv2

    @R.function
    def decode(input_ids1: R.Tensor((1, 1), dtype="int32"), all_seq_len: R.Shape(["m"]), kv_cache: R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object), model_params: R.Tuple(R.Tensor((50280, 640), dtype="uint32"), R.Tensor((50280, 80), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((50280, 320), dtype="uint32"), R.Tensor((50280, 80), dtype="float32"))) -> R.Tuple(R.Tensor((1, 1, 50280), dtype="float32"), R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object)):
        m = T.int64()
        R.func_attr({"num_input": 3, "tir_var_upper_bound": {"m": 4096, "n": 4096}})
        cls = Module
        with R.dataflow():
            lv1772 = R.call_tir(cls.reshape, (input_ids1,), out_sinfo=R.Tensor((1,), dtype="int32"))
            lv: R.Tensor((50280, 640), dtype="uint32") = model_params[0]
            lv1: R.Tensor((50280, 80), dtype="float16") = model_params[1]
            lv1_1 = R.call_tir(cls.fused_fused_decode1_take, (lv, lv1, lv1772), out_sinfo=R.Tensor((1, 2560), dtype="float16"))
            lv1774 = R.call_tir(cls.reshape1, (lv1_1,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv1775 = R.call_tir(cls.full, R.tuple(), out_sinfo=R.Tensor((1, 1, 1, m), dtype="float16"))
            lv1776 = R.call_tir(cls.cast, (lv1774,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_2: R.Tensor((2560,), dtype="float32") = model_params[2]
            param_3: R.Tensor((2560,), dtype="float32") = model_params[3]
            lv3 = R.call_tir(cls.fused_layer_norm_cast1, (lv1776, param_2, param_3), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv4: R.Tensor((2560, 7680), dtype="int8") = model_params[6]
            lv5: R.Tensor((1, 7680), dtype="float16") = model_params[7]
            param_8: R.Tensor((7680,), dtype="float16") = model_params[8]
            lv_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv4, lv5, lv3, param_8), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv8 = R.call_tir(cls.fused_reshape2_split, (lv_1,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv1785: R.Tensor((1, 1, 32, 80), dtype="float16") = lv8[0]
            lv1786 = R.call_tir(cls.rotary_embedding1, (lv1785, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1787: R.Tensor((1, 1, 32, 80), dtype="float16") = lv8[1]
            lv1788 = R.call_tir(cls.rotary_embedding1, (lv1787, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1789: R.Object = kv_cache[0]
            lv1790 = R.call_tir(cls.squeeze, (lv1788,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv1791: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1789, lv1790, sinfo_args=(R.Object,))
            lv1792: R.Object = kv_cache[1]
            lv9: R.Tensor((1, 1, 32, 80), dtype="float16") = lv8[2]
            lv10 = R.call_tir(cls.fused_squeeze, (lv9,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv1795: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1792, lv10, sinfo_args=(R.Object,))
            lv1796: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1791, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1797: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1795, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1798 = R.call_tir(cls.reshape3, (lv1796,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1799 = R.call_tir(cls.reshape3, (lv1797,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1800 = R.call_tir(cls.transpose, (lv1786,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv1801 = R.call_tir(cls.transpose1, (lv1798,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1802 = R.call_tir(cls.transpose1, (lv1799,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv11 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv1800, lv1801, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv12 = R.call_tir(cls.fused_softmax1_cast3, (lv11,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv1811 = R.call_tir(cls.matmul1, (lv12, lv1802), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv13 = R.call_tir(cls.fused_transpose2_reshape4, (lv1811,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv14: R.Tensor((2560, 2560), dtype="int8") = model_params[9]
            lv15: R.Tensor((1, 2560), dtype="float16") = model_params[10]
            lv16 = R.call_tir(cls.fused_decode8, (lv14, lv15), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_11: R.Tensor((2560,), dtype="float16") = model_params[11]
            lv1817 = R.call_tir(cls.cast, (lv1774,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_4: R.Tensor((2560,), dtype="float32") = model_params[4]
            param_5: R.Tensor((2560,), dtype="float32") = model_params[5]
            lv17 = R.call_tir(cls.fused_layer_norm_cast1, (lv1817, param_4, param_5), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv18: R.Tensor((2560, 10240), dtype="int8") = model_params[12]
            lv19: R.Tensor((1, 10240), dtype="float16") = model_params[13]
            param_14: R.Tensor((10240,), dtype="float16") = model_params[14]
            lv1_2 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv18, lv19, lv17, param_14), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv22: R.Tensor((10240, 2560), dtype="int8") = model_params[15]
            lv23: R.Tensor((1, 2560), dtype="float16") = model_params[16]
            param_17: R.Tensor((2560,), dtype="float16") = model_params[17]
            lv2 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv22, lv23, lv1_2, param_17), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv26 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv13, lv16, param_11, lv2, lv1774), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv1831 = R.call_tir(cls.cast, (lv26,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_18: R.Tensor((2560,), dtype="float32") = model_params[18]
            param_19: R.Tensor((2560,), dtype="float32") = model_params[19]
            lv27 = R.call_tir(cls.fused_layer_norm_cast1, (lv1831, param_18, param_19), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv28: R.Tensor((2560, 7680), dtype="int8") = model_params[22]
            lv29: R.Tensor((1, 7680), dtype="float16") = model_params[23]
            param_24: R.Tensor((7680,), dtype="float16") = model_params[24]
            lv3_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv28, lv29, lv27, param_24), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv32 = R.call_tir(cls.fused_reshape2_split, (lv3_1,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv1840: R.Tensor((1, 1, 32, 80), dtype="float16") = lv32[0]
            lv1841 = R.call_tir(cls.rotary_embedding1, (lv1840, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1842: R.Tensor((1, 1, 32, 80), dtype="float16") = lv32[1]
            lv1843 = R.call_tir(cls.rotary_embedding1, (lv1842, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1844: R.Object = kv_cache[2]
            lv1845 = R.call_tir(cls.squeeze, (lv1843,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv1846: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1844, lv1845, sinfo_args=(R.Object,))
            lv1847: R.Object = kv_cache[3]
            lv33: R.Tensor((1, 1, 32, 80), dtype="float16") = lv32[2]
            lv34 = R.call_tir(cls.fused_squeeze, (lv33,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv1850: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1847, lv34, sinfo_args=(R.Object,))
            lv1851: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1846, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1852: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1850, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1853 = R.call_tir(cls.reshape3, (lv1851,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1854 = R.call_tir(cls.reshape3, (lv1852,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1855 = R.call_tir(cls.transpose, (lv1841,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv1856 = R.call_tir(cls.transpose1, (lv1853,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1857 = R.call_tir(cls.transpose1, (lv1854,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv35 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv1855, lv1856, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv36 = R.call_tir(cls.fused_softmax1_cast3, (lv35,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv1866 = R.call_tir(cls.matmul1, (lv36, lv1857), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv37 = R.call_tir(cls.fused_transpose2_reshape4, (lv1866,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv38: R.Tensor((2560, 2560), dtype="int8") = model_params[25]
            lv39: R.Tensor((1, 2560), dtype="float16") = model_params[26]
            lv40 = R.call_tir(cls.fused_decode8, (lv38, lv39), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_27: R.Tensor((2560,), dtype="float16") = model_params[27]
            lv1872 = R.call_tir(cls.cast, (lv26,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_20: R.Tensor((2560,), dtype="float32") = model_params[20]
            param_21: R.Tensor((2560,), dtype="float32") = model_params[21]
            lv41 = R.call_tir(cls.fused_layer_norm_cast1, (lv1872, param_20, param_21), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv42: R.Tensor((2560, 10240), dtype="int8") = model_params[28]
            lv43: R.Tensor((1, 10240), dtype="float16") = model_params[29]
            param_30: R.Tensor((10240,), dtype="float16") = model_params[30]
            lv4_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv42, lv43, lv41, param_30), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv46: R.Tensor((10240, 2560), dtype="int8") = model_params[31]
            lv47: R.Tensor((1, 2560), dtype="float16") = model_params[32]
            param_33: R.Tensor((2560,), dtype="float16") = model_params[33]
            lv5_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv46, lv47, lv4_1, param_33), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv50 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv37, lv40, param_27, lv5_1, lv26), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv1886 = R.call_tir(cls.cast, (lv50,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_34: R.Tensor((2560,), dtype="float32") = model_params[34]
            param_35: R.Tensor((2560,), dtype="float32") = model_params[35]
            lv51 = R.call_tir(cls.fused_layer_norm_cast1, (lv1886, param_34, param_35), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv52: R.Tensor((2560, 7680), dtype="int8") = model_params[38]
            lv53: R.Tensor((1, 7680), dtype="float16") = model_params[39]
            param_40: R.Tensor((7680,), dtype="float16") = model_params[40]
            lv6 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv52, lv53, lv51, param_40), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv56 = R.call_tir(cls.fused_reshape2_split, (lv6,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv1895: R.Tensor((1, 1, 32, 80), dtype="float16") = lv56[0]
            lv1896 = R.call_tir(cls.rotary_embedding1, (lv1895, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1897: R.Tensor((1, 1, 32, 80), dtype="float16") = lv56[1]
            lv1898 = R.call_tir(cls.rotary_embedding1, (lv1897, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1899: R.Object = kv_cache[4]
            lv1900 = R.call_tir(cls.squeeze, (lv1898,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv1901: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1899, lv1900, sinfo_args=(R.Object,))
            lv1902: R.Object = kv_cache[5]
            lv57: R.Tensor((1, 1, 32, 80), dtype="float16") = lv56[2]
            lv58 = R.call_tir(cls.fused_squeeze, (lv57,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv1905: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1902, lv58, sinfo_args=(R.Object,))
            lv1906: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1901, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1907: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1905, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1908 = R.call_tir(cls.reshape3, (lv1906,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1909 = R.call_tir(cls.reshape3, (lv1907,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1910 = R.call_tir(cls.transpose, (lv1896,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv1911 = R.call_tir(cls.transpose1, (lv1908,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1912 = R.call_tir(cls.transpose1, (lv1909,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv59 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv1910, lv1911, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv60 = R.call_tir(cls.fused_softmax1_cast3, (lv59,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv1921 = R.call_tir(cls.matmul1, (lv60, lv1912), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv61 = R.call_tir(cls.fused_transpose2_reshape4, (lv1921,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv62: R.Tensor((2560, 2560), dtype="int8") = model_params[41]
            lv63: R.Tensor((1, 2560), dtype="float16") = model_params[42]
            lv64 = R.call_tir(cls.fused_decode8, (lv62, lv63), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_43: R.Tensor((2560,), dtype="float16") = model_params[43]
            lv1927 = R.call_tir(cls.cast, (lv50,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_36: R.Tensor((2560,), dtype="float32") = model_params[36]
            param_37: R.Tensor((2560,), dtype="float32") = model_params[37]
            lv65 = R.call_tir(cls.fused_layer_norm_cast1, (lv1927, param_36, param_37), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv66: R.Tensor((2560, 10240), dtype="int8") = model_params[44]
            lv67: R.Tensor((1, 10240), dtype="float16") = model_params[45]
            param_46: R.Tensor((10240,), dtype="float16") = model_params[46]
            lv7 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv66, lv67, lv65, param_46), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv70: R.Tensor((10240, 2560), dtype="int8") = model_params[47]
            lv71: R.Tensor((1, 2560), dtype="float16") = model_params[48]
            param_49: R.Tensor((2560,), dtype="float16") = model_params[49]
            lv8_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv70, lv71, lv7, param_49), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv74 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv61, lv64, param_43, lv8_1, lv50), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv1941 = R.call_tir(cls.cast, (lv74,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_50: R.Tensor((2560,), dtype="float32") = model_params[50]
            param_51: R.Tensor((2560,), dtype="float32") = model_params[51]
            lv75 = R.call_tir(cls.fused_layer_norm_cast1, (lv1941, param_50, param_51), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv76: R.Tensor((2560, 7680), dtype="int8") = model_params[54]
            lv77: R.Tensor((1, 7680), dtype="float16") = model_params[55]
            param_56: R.Tensor((7680,), dtype="float16") = model_params[56]
            lv9_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv76, lv77, lv75, param_56), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv80 = R.call_tir(cls.fused_reshape2_split, (lv9_1,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv1950: R.Tensor((1, 1, 32, 80), dtype="float16") = lv80[0]
            lv1951 = R.call_tir(cls.rotary_embedding1, (lv1950, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1952: R.Tensor((1, 1, 32, 80), dtype="float16") = lv80[1]
            lv1953 = R.call_tir(cls.rotary_embedding1, (lv1952, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1954: R.Object = kv_cache[6]
            lv1955 = R.call_tir(cls.squeeze, (lv1953,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv1956: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1954, lv1955, sinfo_args=(R.Object,))
            lv1957: R.Object = kv_cache[7]
            lv81: R.Tensor((1, 1, 32, 80), dtype="float16") = lv80[2]
            lv82 = R.call_tir(cls.fused_squeeze, (lv81,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv1960: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1957, lv82, sinfo_args=(R.Object,))
            lv1961: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1956, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1962: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1960, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1963 = R.call_tir(cls.reshape3, (lv1961,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1964 = R.call_tir(cls.reshape3, (lv1962,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1965 = R.call_tir(cls.transpose, (lv1951,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv1966 = R.call_tir(cls.transpose1, (lv1963,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1967 = R.call_tir(cls.transpose1, (lv1964,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv83 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv1965, lv1966, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv84 = R.call_tir(cls.fused_softmax1_cast3, (lv83,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv1976 = R.call_tir(cls.matmul1, (lv84, lv1967), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv85 = R.call_tir(cls.fused_transpose2_reshape4, (lv1976,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv86: R.Tensor((2560, 2560), dtype="int8") = model_params[57]
            lv87: R.Tensor((1, 2560), dtype="float16") = model_params[58]
            lv88 = R.call_tir(cls.fused_decode8, (lv86, lv87), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_59: R.Tensor((2560,), dtype="float16") = model_params[59]
            lv1982 = R.call_tir(cls.cast, (lv74,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_52: R.Tensor((2560,), dtype="float32") = model_params[52]
            param_53: R.Tensor((2560,), dtype="float32") = model_params[53]
            lv89 = R.call_tir(cls.fused_layer_norm_cast1, (lv1982, param_52, param_53), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv90: R.Tensor((2560, 10240), dtype="int8") = model_params[60]
            lv91: R.Tensor((1, 10240), dtype="float16") = model_params[61]
            param_62: R.Tensor((10240,), dtype="float16") = model_params[62]
            lv10_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv90, lv91, lv89, param_62), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv94: R.Tensor((10240, 2560), dtype="int8") = model_params[63]
            lv95: R.Tensor((1, 2560), dtype="float16") = model_params[64]
            param_65: R.Tensor((2560,), dtype="float16") = model_params[65]
            lv11_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv94, lv95, lv10_1, param_65), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv98 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv85, lv88, param_59, lv11_1, lv74), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv1996 = R.call_tir(cls.cast, (lv98,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_66: R.Tensor((2560,), dtype="float32") = model_params[66]
            param_67: R.Tensor((2560,), dtype="float32") = model_params[67]
            lv99 = R.call_tir(cls.fused_layer_norm_cast1, (lv1996, param_66, param_67), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv100: R.Tensor((2560, 7680), dtype="int8") = model_params[70]
            lv101: R.Tensor((1, 7680), dtype="float16") = model_params[71]
            param_72: R.Tensor((7680,), dtype="float16") = model_params[72]
            lv12_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv100, lv101, lv99, param_72), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv104 = R.call_tir(cls.fused_reshape2_split, (lv12_1,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2005: R.Tensor((1, 1, 32, 80), dtype="float16") = lv104[0]
            lv2006 = R.call_tir(cls.rotary_embedding1, (lv2005, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2007: R.Tensor((1, 1, 32, 80), dtype="float16") = lv104[1]
            lv2008 = R.call_tir(cls.rotary_embedding1, (lv2007, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2009: R.Object = kv_cache[8]
            lv2010 = R.call_tir(cls.squeeze, (lv2008,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2011: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2009, lv2010, sinfo_args=(R.Object,))
            lv2012: R.Object = kv_cache[9]
            lv105: R.Tensor((1, 1, 32, 80), dtype="float16") = lv104[2]
            lv106 = R.call_tir(cls.fused_squeeze, (lv105,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2015: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2012, lv106, sinfo_args=(R.Object,))
            lv2016: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2011, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2017: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2015, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2018 = R.call_tir(cls.reshape3, (lv2016,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2019 = R.call_tir(cls.reshape3, (lv2017,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2020 = R.call_tir(cls.transpose, (lv2006,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2021 = R.call_tir(cls.transpose1, (lv2018,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2022 = R.call_tir(cls.transpose1, (lv2019,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv107 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv2020, lv2021, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv108 = R.call_tir(cls.fused_softmax1_cast3, (lv107,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2031 = R.call_tir(cls.matmul1, (lv108, lv2022), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv109 = R.call_tir(cls.fused_transpose2_reshape4, (lv2031,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv110: R.Tensor((2560, 2560), dtype="int8") = model_params[73]
            lv111: R.Tensor((1, 2560), dtype="float16") = model_params[74]
            lv112 = R.call_tir(cls.fused_decode8, (lv110, lv111), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_75: R.Tensor((2560,), dtype="float16") = model_params[75]
            lv2037 = R.call_tir(cls.cast, (lv98,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_68: R.Tensor((2560,), dtype="float32") = model_params[68]
            param_69: R.Tensor((2560,), dtype="float32") = model_params[69]
            lv113 = R.call_tir(cls.fused_layer_norm_cast1, (lv2037, param_68, param_69), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv114: R.Tensor((2560, 10240), dtype="int8") = model_params[76]
            lv115: R.Tensor((1, 10240), dtype="float16") = model_params[77]
            param_78: R.Tensor((10240,), dtype="float16") = model_params[78]
            lv13_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv114, lv115, lv113, param_78), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv118: R.Tensor((10240, 2560), dtype="int8") = model_params[79]
            lv119: R.Tensor((1, 2560), dtype="float16") = model_params[80]
            param_81: R.Tensor((2560,), dtype="float16") = model_params[81]
            lv14_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv118, lv119, lv13_1, param_81), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv122 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv109, lv112, param_75, lv14_1, lv98), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2051 = R.call_tir(cls.cast, (lv122,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_82: R.Tensor((2560,), dtype="float32") = model_params[82]
            param_83: R.Tensor((2560,), dtype="float32") = model_params[83]
            lv123 = R.call_tir(cls.fused_layer_norm_cast1, (lv2051, param_82, param_83), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv124: R.Tensor((2560, 7680), dtype="int8") = model_params[86]
            lv125: R.Tensor((1, 7680), dtype="float16") = model_params[87]
            param_88: R.Tensor((7680,), dtype="float16") = model_params[88]
            lv15_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv124, lv125, lv123, param_88), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv128 = R.call_tir(cls.fused_reshape2_split, (lv15_1,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2060: R.Tensor((1, 1, 32, 80), dtype="float16") = lv128[0]
            lv2061 = R.call_tir(cls.rotary_embedding1, (lv2060, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2062: R.Tensor((1, 1, 32, 80), dtype="float16") = lv128[1]
            lv2063 = R.call_tir(cls.rotary_embedding1, (lv2062, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2064: R.Object = kv_cache[10]
            lv2065 = R.call_tir(cls.squeeze, (lv2063,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2066: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2064, lv2065, sinfo_args=(R.Object,))
            lv2067: R.Object = kv_cache[11]
            lv129: R.Tensor((1, 1, 32, 80), dtype="float16") = lv128[2]
            lv130 = R.call_tir(cls.fused_squeeze, (lv129,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2070: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2067, lv130, sinfo_args=(R.Object,))
            lv2071: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2066, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2072: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2070, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2073 = R.call_tir(cls.reshape3, (lv2071,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2074 = R.call_tir(cls.reshape3, (lv2072,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2075 = R.call_tir(cls.transpose, (lv2061,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2076 = R.call_tir(cls.transpose1, (lv2073,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2077 = R.call_tir(cls.transpose1, (lv2074,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv131 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv2075, lv2076, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv132 = R.call_tir(cls.fused_softmax1_cast3, (lv131,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2086 = R.call_tir(cls.matmul1, (lv132, lv2077), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv133 = R.call_tir(cls.fused_transpose2_reshape4, (lv2086,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv134: R.Tensor((2560, 2560), dtype="int8") = model_params[89]
            lv135: R.Tensor((1, 2560), dtype="float16") = model_params[90]
            lv136 = R.call_tir(cls.fused_decode8, (lv134, lv135), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_91: R.Tensor((2560,), dtype="float16") = model_params[91]
            lv2092 = R.call_tir(cls.cast, (lv122,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_84: R.Tensor((2560,), dtype="float32") = model_params[84]
            param_85: R.Tensor((2560,), dtype="float32") = model_params[85]
            lv137 = R.call_tir(cls.fused_layer_norm_cast1, (lv2092, param_84, param_85), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv138: R.Tensor((2560, 10240), dtype="int8") = model_params[92]
            lv139: R.Tensor((1, 10240), dtype="float16") = model_params[93]
            param_94: R.Tensor((10240,), dtype="float16") = model_params[94]
            lv16_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv138, lv139, lv137, param_94), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv142: R.Tensor((10240, 2560), dtype="int8") = model_params[95]
            lv143: R.Tensor((1, 2560), dtype="float16") = model_params[96]
            param_97: R.Tensor((2560,), dtype="float16") = model_params[97]
            lv17_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv142, lv143, lv16_1, param_97), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv146 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv133, lv136, param_91, lv17_1, lv122), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2106 = R.call_tir(cls.cast, (lv146,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_98: R.Tensor((2560,), dtype="float32") = model_params[98]
            param_99: R.Tensor((2560,), dtype="float32") = model_params[99]
            lv147 = R.call_tir(cls.fused_layer_norm_cast1, (lv2106, param_98, param_99), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv148: R.Tensor((2560, 7680), dtype="int8") = model_params[102]
            lv149: R.Tensor((1, 7680), dtype="float16") = model_params[103]
            param_104: R.Tensor((7680,), dtype="float16") = model_params[104]
            lv18_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv148, lv149, lv147, param_104), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv152 = R.call_tir(cls.fused_reshape2_split, (lv18_1,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2115: R.Tensor((1, 1, 32, 80), dtype="float16") = lv152[0]
            lv2116 = R.call_tir(cls.rotary_embedding1, (lv2115, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2117: R.Tensor((1, 1, 32, 80), dtype="float16") = lv152[1]
            lv2118 = R.call_tir(cls.rotary_embedding1, (lv2117, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2119: R.Object = kv_cache[12]
            lv2120 = R.call_tir(cls.squeeze, (lv2118,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2121: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2119, lv2120, sinfo_args=(R.Object,))
            lv2122: R.Object = kv_cache[13]
            lv153: R.Tensor((1, 1, 32, 80), dtype="float16") = lv152[2]
            lv154 = R.call_tir(cls.fused_squeeze, (lv153,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2125: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2122, lv154, sinfo_args=(R.Object,))
            lv2126: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2121, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2127: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2125, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2128 = R.call_tir(cls.reshape3, (lv2126,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2129 = R.call_tir(cls.reshape3, (lv2127,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2130 = R.call_tir(cls.transpose, (lv2116,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2131 = R.call_tir(cls.transpose1, (lv2128,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2132 = R.call_tir(cls.transpose1, (lv2129,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv155 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv2130, lv2131, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv156 = R.call_tir(cls.fused_softmax1_cast3, (lv155,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2141 = R.call_tir(cls.matmul1, (lv156, lv2132), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv157 = R.call_tir(cls.fused_transpose2_reshape4, (lv2141,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv158: R.Tensor((2560, 2560), dtype="int8") = model_params[105]
            lv159: R.Tensor((1, 2560), dtype="float16") = model_params[106]
            lv160 = R.call_tir(cls.fused_decode8, (lv158, lv159), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_107: R.Tensor((2560,), dtype="float16") = model_params[107]
            lv2147 = R.call_tir(cls.cast, (lv146,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_100: R.Tensor((2560,), dtype="float32") = model_params[100]
            param_101: R.Tensor((2560,), dtype="float32") = model_params[101]
            lv161 = R.call_tir(cls.fused_layer_norm_cast1, (lv2147, param_100, param_101), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv162: R.Tensor((2560, 10240), dtype="int8") = model_params[108]
            lv163: R.Tensor((1, 10240), dtype="float16") = model_params[109]
            param_110: R.Tensor((10240,), dtype="float16") = model_params[110]
            lv19_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv162, lv163, lv161, param_110), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv166: R.Tensor((10240, 2560), dtype="int8") = model_params[111]
            lv167: R.Tensor((1, 2560), dtype="float16") = model_params[112]
            param_113: R.Tensor((2560,), dtype="float16") = model_params[113]
            lv20 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv166, lv167, lv19_1, param_113), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv170 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv157, lv160, param_107, lv20, lv146), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2161 = R.call_tir(cls.cast, (lv170,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_114: R.Tensor((2560,), dtype="float32") = model_params[114]
            param_115: R.Tensor((2560,), dtype="float32") = model_params[115]
            lv171 = R.call_tir(cls.fused_layer_norm_cast1, (lv2161, param_114, param_115), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv172: R.Tensor((2560, 7680), dtype="int8") = model_params[118]
            lv173: R.Tensor((1, 7680), dtype="float16") = model_params[119]
            param_120: R.Tensor((7680,), dtype="float16") = model_params[120]
            lv21 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv172, lv173, lv171, param_120), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv176 = R.call_tir(cls.fused_reshape2_split, (lv21,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2170: R.Tensor((1, 1, 32, 80), dtype="float16") = lv176[0]
            lv2171 = R.call_tir(cls.rotary_embedding1, (lv2170, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2172: R.Tensor((1, 1, 32, 80), dtype="float16") = lv176[1]
            lv2173 = R.call_tir(cls.rotary_embedding1, (lv2172, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2174: R.Object = kv_cache[14]
            lv2175 = R.call_tir(cls.squeeze, (lv2173,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2176: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2174, lv2175, sinfo_args=(R.Object,))
            lv2177: R.Object = kv_cache[15]
            lv177: R.Tensor((1, 1, 32, 80), dtype="float16") = lv176[2]
            lv178 = R.call_tir(cls.fused_squeeze, (lv177,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2180: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2177, lv178, sinfo_args=(R.Object,))
            lv2181: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2176, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2182: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2180, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2183 = R.call_tir(cls.reshape3, (lv2181,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2184 = R.call_tir(cls.reshape3, (lv2182,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2185 = R.call_tir(cls.transpose, (lv2171,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2186 = R.call_tir(cls.transpose1, (lv2183,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2187 = R.call_tir(cls.transpose1, (lv2184,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv179 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv2185, lv2186, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv180 = R.call_tir(cls.fused_softmax1_cast3, (lv179,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2196 = R.call_tir(cls.matmul1, (lv180, lv2187), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv181 = R.call_tir(cls.fused_transpose2_reshape4, (lv2196,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv182: R.Tensor((2560, 2560), dtype="int8") = model_params[121]
            lv183: R.Tensor((1, 2560), dtype="float16") = model_params[122]
            lv184 = R.call_tir(cls.fused_decode8, (lv182, lv183), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_123: R.Tensor((2560,), dtype="float16") = model_params[123]
            lv2202 = R.call_tir(cls.cast, (lv170,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_116: R.Tensor((2560,), dtype="float32") = model_params[116]
            param_117: R.Tensor((2560,), dtype="float32") = model_params[117]
            lv185 = R.call_tir(cls.fused_layer_norm_cast1, (lv2202, param_116, param_117), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv186: R.Tensor((2560, 10240), dtype="int8") = model_params[124]
            lv187: R.Tensor((1, 10240), dtype="float16") = model_params[125]
            param_126: R.Tensor((10240,), dtype="float16") = model_params[126]
            lv22_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv186, lv187, lv185, param_126), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv190: R.Tensor((10240, 2560), dtype="int8") = model_params[127]
            lv191: R.Tensor((1, 2560), dtype="float16") = model_params[128]
            param_129: R.Tensor((2560,), dtype="float16") = model_params[129]
            lv23_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv190, lv191, lv22_1, param_129), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv194 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv181, lv184, param_123, lv23_1, lv170), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2216 = R.call_tir(cls.cast, (lv194,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_130: R.Tensor((2560,), dtype="float32") = model_params[130]
            param_131: R.Tensor((2560,), dtype="float32") = model_params[131]
            lv195 = R.call_tir(cls.fused_layer_norm_cast1, (lv2216, param_130, param_131), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv196: R.Tensor((2560, 7680), dtype="int8") = model_params[134]
            lv197: R.Tensor((1, 7680), dtype="float16") = model_params[135]
            param_136: R.Tensor((7680,), dtype="float16") = model_params[136]
            lv24 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv196, lv197, lv195, param_136), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv200 = R.call_tir(cls.fused_reshape2_split, (lv24,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2225: R.Tensor((1, 1, 32, 80), dtype="float16") = lv200[0]
            lv2226 = R.call_tir(cls.rotary_embedding1, (lv2225, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2227: R.Tensor((1, 1, 32, 80), dtype="float16") = lv200[1]
            lv2228 = R.call_tir(cls.rotary_embedding1, (lv2227, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2229: R.Object = kv_cache[16]
            lv2230 = R.call_tir(cls.squeeze, (lv2228,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2231: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2229, lv2230, sinfo_args=(R.Object,))
            lv2232: R.Object = kv_cache[17]
            lv201: R.Tensor((1, 1, 32, 80), dtype="float16") = lv200[2]
            lv202 = R.call_tir(cls.fused_squeeze, (lv201,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2235: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2232, lv202, sinfo_args=(R.Object,))
            lv2236: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2231, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2237: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2235, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2238 = R.call_tir(cls.reshape3, (lv2236,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2239 = R.call_tir(cls.reshape3, (lv2237,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2240 = R.call_tir(cls.transpose, (lv2226,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2241 = R.call_tir(cls.transpose1, (lv2238,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2242 = R.call_tir(cls.transpose1, (lv2239,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv203 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv2240, lv2241, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv204 = R.call_tir(cls.fused_softmax1_cast3, (lv203,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2251 = R.call_tir(cls.matmul1, (lv204, lv2242), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv205 = R.call_tir(cls.fused_transpose2_reshape4, (lv2251,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv206: R.Tensor((2560, 2560), dtype="int8") = model_params[137]
            lv207: R.Tensor((1, 2560), dtype="float16") = model_params[138]
            lv208 = R.call_tir(cls.fused_decode8, (lv206, lv207), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_139: R.Tensor((2560,), dtype="float16") = model_params[139]
            lv2257 = R.call_tir(cls.cast, (lv194,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_132: R.Tensor((2560,), dtype="float32") = model_params[132]
            param_133: R.Tensor((2560,), dtype="float32") = model_params[133]
            lv209 = R.call_tir(cls.fused_layer_norm_cast1, (lv2257, param_132, param_133), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv210: R.Tensor((2560, 10240), dtype="int8") = model_params[140]
            lv211: R.Tensor((1, 10240), dtype="float16") = model_params[141]
            param_142: R.Tensor((10240,), dtype="float16") = model_params[142]
            lv25 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv210, lv211, lv209, param_142), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv214: R.Tensor((10240, 2560), dtype="int8") = model_params[143]
            lv215: R.Tensor((1, 2560), dtype="float16") = model_params[144]
            param_145: R.Tensor((2560,), dtype="float16") = model_params[145]
            lv26_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv214, lv215, lv25, param_145), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv218 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv205, lv208, param_139, lv26_1, lv194), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2271 = R.call_tir(cls.cast, (lv218,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_146: R.Tensor((2560,), dtype="float32") = model_params[146]
            param_147: R.Tensor((2560,), dtype="float32") = model_params[147]
            lv219 = R.call_tir(cls.fused_layer_norm_cast1, (lv2271, param_146, param_147), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv220: R.Tensor((2560, 7680), dtype="int8") = model_params[150]
            lv221: R.Tensor((1, 7680), dtype="float16") = model_params[151]
            param_152: R.Tensor((7680,), dtype="float16") = model_params[152]
            lv27_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv220, lv221, lv219, param_152), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv224 = R.call_tir(cls.fused_reshape2_split, (lv27_1,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2280: R.Tensor((1, 1, 32, 80), dtype="float16") = lv224[0]
            lv2281 = R.call_tir(cls.rotary_embedding1, (lv2280, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2282: R.Tensor((1, 1, 32, 80), dtype="float16") = lv224[1]
            lv2283 = R.call_tir(cls.rotary_embedding1, (lv2282, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2284: R.Object = kv_cache[18]
            lv2285 = R.call_tir(cls.squeeze, (lv2283,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2286: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2284, lv2285, sinfo_args=(R.Object,))
            lv2287: R.Object = kv_cache[19]
            lv225: R.Tensor((1, 1, 32, 80), dtype="float16") = lv224[2]
            lv226 = R.call_tir(cls.fused_squeeze, (lv225,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2290: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2287, lv226, sinfo_args=(R.Object,))
            lv2291: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2286, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2292: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2290, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2293 = R.call_tir(cls.reshape3, (lv2291,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2294 = R.call_tir(cls.reshape3, (lv2292,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2295 = R.call_tir(cls.transpose, (lv2281,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2296 = R.call_tir(cls.transpose1, (lv2293,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2297 = R.call_tir(cls.transpose1, (lv2294,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv227 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv2295, lv2296, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv228 = R.call_tir(cls.fused_softmax1_cast3, (lv227,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2306 = R.call_tir(cls.matmul1, (lv228, lv2297), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv229 = R.call_tir(cls.fused_transpose2_reshape4, (lv2306,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv230: R.Tensor((2560, 2560), dtype="int8") = model_params[153]
            lv231: R.Tensor((1, 2560), dtype="float16") = model_params[154]
            lv232 = R.call_tir(cls.fused_decode8, (lv230, lv231), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_155: R.Tensor((2560,), dtype="float16") = model_params[155]
            lv2312 = R.call_tir(cls.cast, (lv218,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_148: R.Tensor((2560,), dtype="float32") = model_params[148]
            param_149: R.Tensor((2560,), dtype="float32") = model_params[149]
            lv233 = R.call_tir(cls.fused_layer_norm_cast1, (lv2312, param_148, param_149), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv234: R.Tensor((2560, 10240), dtype="int8") = model_params[156]
            lv235: R.Tensor((1, 10240), dtype="float16") = model_params[157]
            param_158: R.Tensor((10240,), dtype="float16") = model_params[158]
            lv28_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv234, lv235, lv233, param_158), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv238: R.Tensor((10240, 2560), dtype="int8") = model_params[159]
            lv239: R.Tensor((1, 2560), dtype="float16") = model_params[160]
            param_161: R.Tensor((2560,), dtype="float16") = model_params[161]
            lv29_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv238, lv239, lv28_1, param_161), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv242 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv229, lv232, param_155, lv29_1, lv218), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2326 = R.call_tir(cls.cast, (lv242,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_162: R.Tensor((2560,), dtype="float32") = model_params[162]
            param_163: R.Tensor((2560,), dtype="float32") = model_params[163]
            lv243 = R.call_tir(cls.fused_layer_norm_cast1, (lv2326, param_162, param_163), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv244: R.Tensor((2560, 7680), dtype="int8") = model_params[166]
            lv245: R.Tensor((1, 7680), dtype="float16") = model_params[167]
            param_168: R.Tensor((7680,), dtype="float16") = model_params[168]
            lv30 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv244, lv245, lv243, param_168), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv248 = R.call_tir(cls.fused_reshape2_split, (lv30,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2335: R.Tensor((1, 1, 32, 80), dtype="float16") = lv248[0]
            lv2336 = R.call_tir(cls.rotary_embedding1, (lv2335, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2337: R.Tensor((1, 1, 32, 80), dtype="float16") = lv248[1]
            lv2338 = R.call_tir(cls.rotary_embedding1, (lv2337, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2339: R.Object = kv_cache[20]
            lv2340 = R.call_tir(cls.squeeze, (lv2338,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2341: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2339, lv2340, sinfo_args=(R.Object,))
            lv2342: R.Object = kv_cache[21]
            lv249: R.Tensor((1, 1, 32, 80), dtype="float16") = lv248[2]
            lv250 = R.call_tir(cls.fused_squeeze, (lv249,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2345: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2342, lv250, sinfo_args=(R.Object,))
            lv2346: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2341, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2347: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2345, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2348 = R.call_tir(cls.reshape3, (lv2346,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2349 = R.call_tir(cls.reshape3, (lv2347,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2350 = R.call_tir(cls.transpose, (lv2336,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2351 = R.call_tir(cls.transpose1, (lv2348,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2352 = R.call_tir(cls.transpose1, (lv2349,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv251 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv2350, lv2351, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv252 = R.call_tir(cls.fused_softmax1_cast3, (lv251,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2361 = R.call_tir(cls.matmul1, (lv252, lv2352), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv253 = R.call_tir(cls.fused_transpose2_reshape4, (lv2361,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv254: R.Tensor((2560, 2560), dtype="int8") = model_params[169]
            lv255: R.Tensor((1, 2560), dtype="float16") = model_params[170]
            lv256 = R.call_tir(cls.fused_decode8, (lv254, lv255), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_171: R.Tensor((2560,), dtype="float16") = model_params[171]
            lv2367 = R.call_tir(cls.cast, (lv242,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_164: R.Tensor((2560,), dtype="float32") = model_params[164]
            param_165: R.Tensor((2560,), dtype="float32") = model_params[165]
            lv257 = R.call_tir(cls.fused_layer_norm_cast1, (lv2367, param_164, param_165), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv258: R.Tensor((2560, 10240), dtype="int8") = model_params[172]
            lv259: R.Tensor((1, 10240), dtype="float16") = model_params[173]
            param_174: R.Tensor((10240,), dtype="float16") = model_params[174]
            lv31 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv258, lv259, lv257, param_174), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv262: R.Tensor((10240, 2560), dtype="int8") = model_params[175]
            lv263: R.Tensor((1, 2560), dtype="float16") = model_params[176]
            param_177: R.Tensor((2560,), dtype="float16") = model_params[177]
            lv32_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv262, lv263, lv31, param_177), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv266 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv253, lv256, param_171, lv32_1, lv242), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2381 = R.call_tir(cls.cast, (lv266,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_178: R.Tensor((2560,), dtype="float32") = model_params[178]
            param_179: R.Tensor((2560,), dtype="float32") = model_params[179]
            lv267 = R.call_tir(cls.fused_layer_norm_cast1, (lv2381, param_178, param_179), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv268: R.Tensor((2560, 7680), dtype="int8") = model_params[182]
            lv269: R.Tensor((1, 7680), dtype="float16") = model_params[183]
            param_184: R.Tensor((7680,), dtype="float16") = model_params[184]
            lv33_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv268, lv269, lv267, param_184), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv272 = R.call_tir(cls.fused_reshape2_split, (lv33_1,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2390: R.Tensor((1, 1, 32, 80), dtype="float16") = lv272[0]
            lv2391 = R.call_tir(cls.rotary_embedding1, (lv2390, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2392: R.Tensor((1, 1, 32, 80), dtype="float16") = lv272[1]
            lv2393 = R.call_tir(cls.rotary_embedding1, (lv2392, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2394: R.Object = kv_cache[22]
            lv2395 = R.call_tir(cls.squeeze, (lv2393,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2396: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2394, lv2395, sinfo_args=(R.Object,))
            lv2397: R.Object = kv_cache[23]
            lv273: R.Tensor((1, 1, 32, 80), dtype="float16") = lv272[2]
            lv274 = R.call_tir(cls.fused_squeeze, (lv273,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2400: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2397, lv274, sinfo_args=(R.Object,))
            lv2401: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2396, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2402: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2400, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2403 = R.call_tir(cls.reshape3, (lv2401,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2404 = R.call_tir(cls.reshape3, (lv2402,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2405 = R.call_tir(cls.transpose, (lv2391,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2406 = R.call_tir(cls.transpose1, (lv2403,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2407 = R.call_tir(cls.transpose1, (lv2404,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv275 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv2405, lv2406, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv276 = R.call_tir(cls.fused_softmax1_cast3, (lv275,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2416 = R.call_tir(cls.matmul1, (lv276, lv2407), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv277 = R.call_tir(cls.fused_transpose2_reshape4, (lv2416,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv278: R.Tensor((2560, 2560), dtype="int8") = model_params[185]
            lv279: R.Tensor((1, 2560), dtype="float16") = model_params[186]
            lv280 = R.call_tir(cls.fused_decode8, (lv278, lv279), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_187: R.Tensor((2560,), dtype="float16") = model_params[187]
            lv2422 = R.call_tir(cls.cast, (lv266,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_180: R.Tensor((2560,), dtype="float32") = model_params[180]
            param_181: R.Tensor((2560,), dtype="float32") = model_params[181]
            lv281 = R.call_tir(cls.fused_layer_norm_cast1, (lv2422, param_180, param_181), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv282: R.Tensor((2560, 10240), dtype="int8") = model_params[188]
            lv283: R.Tensor((1, 10240), dtype="float16") = model_params[189]
            param_190: R.Tensor((10240,), dtype="float16") = model_params[190]
            lv34_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv282, lv283, lv281, param_190), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv286: R.Tensor((10240, 2560), dtype="int8") = model_params[191]
            lv287: R.Tensor((1, 2560), dtype="float16") = model_params[192]
            param_193: R.Tensor((2560,), dtype="float16") = model_params[193]
            lv35_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv286, lv287, lv34_1, param_193), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv290 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv277, lv280, param_187, lv35_1, lv266), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2436 = R.call_tir(cls.cast, (lv290,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_194: R.Tensor((2560,), dtype="float32") = model_params[194]
            param_195: R.Tensor((2560,), dtype="float32") = model_params[195]
            lv291 = R.call_tir(cls.fused_layer_norm_cast1, (lv2436, param_194, param_195), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv292: R.Tensor((2560, 7680), dtype="int8") = model_params[198]
            lv293: R.Tensor((1, 7680), dtype="float16") = model_params[199]
            param_200: R.Tensor((7680,), dtype="float16") = model_params[200]
            lv36_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv292, lv293, lv291, param_200), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv296 = R.call_tir(cls.fused_reshape2_split, (lv36_1,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2445: R.Tensor((1, 1, 32, 80), dtype="float16") = lv296[0]
            lv2446 = R.call_tir(cls.rotary_embedding1, (lv2445, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2447: R.Tensor((1, 1, 32, 80), dtype="float16") = lv296[1]
            lv2448 = R.call_tir(cls.rotary_embedding1, (lv2447, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2449: R.Object = kv_cache[24]
            lv2450 = R.call_tir(cls.squeeze, (lv2448,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2451: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2449, lv2450, sinfo_args=(R.Object,))
            lv2452: R.Object = kv_cache[25]
            lv297: R.Tensor((1, 1, 32, 80), dtype="float16") = lv296[2]
            lv298 = R.call_tir(cls.fused_squeeze, (lv297,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2455: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2452, lv298, sinfo_args=(R.Object,))
            lv2456: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2451, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2457: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2455, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2458 = R.call_tir(cls.reshape3, (lv2456,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2459 = R.call_tir(cls.reshape3, (lv2457,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2460 = R.call_tir(cls.transpose, (lv2446,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2461 = R.call_tir(cls.transpose1, (lv2458,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2462 = R.call_tir(cls.transpose1, (lv2459,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv299 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv2460, lv2461, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv300 = R.call_tir(cls.fused_softmax1_cast3, (lv299,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2471 = R.call_tir(cls.matmul1, (lv300, lv2462), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv301 = R.call_tir(cls.fused_transpose2_reshape4, (lv2471,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv302: R.Tensor((2560, 2560), dtype="int8") = model_params[201]
            lv303: R.Tensor((1, 2560), dtype="float16") = model_params[202]
            lv304 = R.call_tir(cls.fused_decode8, (lv302, lv303), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_203: R.Tensor((2560,), dtype="float16") = model_params[203]
            lv2477 = R.call_tir(cls.cast, (lv290,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_196: R.Tensor((2560,), dtype="float32") = model_params[196]
            param_197: R.Tensor((2560,), dtype="float32") = model_params[197]
            lv305 = R.call_tir(cls.fused_layer_norm_cast1, (lv2477, param_196, param_197), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv306: R.Tensor((2560, 10240), dtype="int8") = model_params[204]
            lv307: R.Tensor((1, 10240), dtype="float16") = model_params[205]
            param_206: R.Tensor((10240,), dtype="float16") = model_params[206]
            lv37_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv306, lv307, lv305, param_206), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv310: R.Tensor((10240, 2560), dtype="int8") = model_params[207]
            lv311: R.Tensor((1, 2560), dtype="float16") = model_params[208]
            param_209: R.Tensor((2560,), dtype="float16") = model_params[209]
            lv38_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv310, lv311, lv37_1, param_209), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv314 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv301, lv304, param_203, lv38_1, lv290), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2491 = R.call_tir(cls.cast, (lv314,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_210: R.Tensor((2560,), dtype="float32") = model_params[210]
            param_211: R.Tensor((2560,), dtype="float32") = model_params[211]
            lv315 = R.call_tir(cls.fused_layer_norm_cast1, (lv2491, param_210, param_211), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv316: R.Tensor((2560, 7680), dtype="int8") = model_params[214]
            lv317: R.Tensor((1, 7680), dtype="float16") = model_params[215]
            param_216: R.Tensor((7680,), dtype="float16") = model_params[216]
            lv39_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv316, lv317, lv315, param_216), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv320 = R.call_tir(cls.fused_reshape2_split, (lv39_1,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2500: R.Tensor((1, 1, 32, 80), dtype="float16") = lv320[0]
            lv2501 = R.call_tir(cls.rotary_embedding1, (lv2500, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2502: R.Tensor((1, 1, 32, 80), dtype="float16") = lv320[1]
            lv2503 = R.call_tir(cls.rotary_embedding1, (lv2502, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2504: R.Object = kv_cache[26]
            lv2505 = R.call_tir(cls.squeeze, (lv2503,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2506: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2504, lv2505, sinfo_args=(R.Object,))
            lv2507: R.Object = kv_cache[27]
            lv321: R.Tensor((1, 1, 32, 80), dtype="float16") = lv320[2]
            lv322 = R.call_tir(cls.fused_squeeze, (lv321,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2510: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2507, lv322, sinfo_args=(R.Object,))
            lv2511: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2506, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2512: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2510, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2513 = R.call_tir(cls.reshape3, (lv2511,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2514 = R.call_tir(cls.reshape3, (lv2512,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2515 = R.call_tir(cls.transpose, (lv2501,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2516 = R.call_tir(cls.transpose1, (lv2513,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2517 = R.call_tir(cls.transpose1, (lv2514,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv323 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv2515, lv2516, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv324 = R.call_tir(cls.fused_softmax1_cast3, (lv323,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2526 = R.call_tir(cls.matmul1, (lv324, lv2517), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv325 = R.call_tir(cls.fused_transpose2_reshape4, (lv2526,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv326: R.Tensor((2560, 2560), dtype="int8") = model_params[217]
            lv327: R.Tensor((1, 2560), dtype="float16") = model_params[218]
            lv328 = R.call_tir(cls.fused_decode8, (lv326, lv327), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_219: R.Tensor((2560,), dtype="float16") = model_params[219]
            lv2532 = R.call_tir(cls.cast, (lv314,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_212: R.Tensor((2560,), dtype="float32") = model_params[212]
            param_213: R.Tensor((2560,), dtype="float32") = model_params[213]
            lv329 = R.call_tir(cls.fused_layer_norm_cast1, (lv2532, param_212, param_213), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv330: R.Tensor((2560, 10240), dtype="int8") = model_params[220]
            lv331: R.Tensor((1, 10240), dtype="float16") = model_params[221]
            param_222: R.Tensor((10240,), dtype="float16") = model_params[222]
            lv40_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv330, lv331, lv329, param_222), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv334: R.Tensor((10240, 2560), dtype="int8") = model_params[223]
            lv335: R.Tensor((1, 2560), dtype="float16") = model_params[224]
            param_225: R.Tensor((2560,), dtype="float16") = model_params[225]
            lv41_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv334, lv335, lv40_1, param_225), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv338 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv325, lv328, param_219, lv41_1, lv314), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2546 = R.call_tir(cls.cast, (lv338,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_226: R.Tensor((2560,), dtype="float32") = model_params[226]
            param_227: R.Tensor((2560,), dtype="float32") = model_params[227]
            lv339 = R.call_tir(cls.fused_layer_norm_cast1, (lv2546, param_226, param_227), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv340: R.Tensor((2560, 7680), dtype="int8") = model_params[230]
            lv341: R.Tensor((1, 7680), dtype="float16") = model_params[231]
            param_232: R.Tensor((7680,), dtype="float16") = model_params[232]
            lv42_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv340, lv341, lv339, param_232), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv344 = R.call_tir(cls.fused_reshape2_split, (lv42_1,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2555: R.Tensor((1, 1, 32, 80), dtype="float16") = lv344[0]
            lv2556 = R.call_tir(cls.rotary_embedding1, (lv2555, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2557: R.Tensor((1, 1, 32, 80), dtype="float16") = lv344[1]
            lv2558 = R.call_tir(cls.rotary_embedding1, (lv2557, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2559: R.Object = kv_cache[28]
            lv2560 = R.call_tir(cls.squeeze, (lv2558,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2561: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2559, lv2560, sinfo_args=(R.Object,))
            lv2562: R.Object = kv_cache[29]
            lv345: R.Tensor((1, 1, 32, 80), dtype="float16") = lv344[2]
            lv346 = R.call_tir(cls.fused_squeeze, (lv345,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2565: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2562, lv346, sinfo_args=(R.Object,))
            lv2566: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2561, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2567: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2565, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2568 = R.call_tir(cls.reshape3, (lv2566,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2569 = R.call_tir(cls.reshape3, (lv2567,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2570 = R.call_tir(cls.transpose, (lv2556,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2571 = R.call_tir(cls.transpose1, (lv2568,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2572 = R.call_tir(cls.transpose1, (lv2569,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv347 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv2570, lv2571, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv348 = R.call_tir(cls.fused_softmax1_cast3, (lv347,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2581 = R.call_tir(cls.matmul1, (lv348, lv2572), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv349 = R.call_tir(cls.fused_transpose2_reshape4, (lv2581,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv350: R.Tensor((2560, 2560), dtype="int8") = model_params[233]
            lv351: R.Tensor((1, 2560), dtype="float16") = model_params[234]
            lv352 = R.call_tir(cls.fused_decode8, (lv350, lv351), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_235: R.Tensor((2560,), dtype="float16") = model_params[235]
            lv2587 = R.call_tir(cls.cast, (lv338,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_228: R.Tensor((2560,), dtype="float32") = model_params[228]
            param_229: R.Tensor((2560,), dtype="float32") = model_params[229]
            lv353 = R.call_tir(cls.fused_layer_norm_cast1, (lv2587, param_228, param_229), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv354: R.Tensor((2560, 10240), dtype="int8") = model_params[236]
            lv355: R.Tensor((1, 10240), dtype="float16") = model_params[237]
            param_238: R.Tensor((10240,), dtype="float16") = model_params[238]
            lv43_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv354, lv355, lv353, param_238), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv358: R.Tensor((10240, 2560), dtype="int8") = model_params[239]
            lv359: R.Tensor((1, 2560), dtype="float16") = model_params[240]
            param_241: R.Tensor((2560,), dtype="float16") = model_params[241]
            lv44 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv358, lv359, lv43_1, param_241), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv362 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv349, lv352, param_235, lv44, lv338), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2601 = R.call_tir(cls.cast, (lv362,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_242: R.Tensor((2560,), dtype="float32") = model_params[242]
            param_243: R.Tensor((2560,), dtype="float32") = model_params[243]
            lv363 = R.call_tir(cls.fused_layer_norm_cast1, (lv2601, param_242, param_243), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv364: R.Tensor((2560, 7680), dtype="int8") = model_params[246]
            lv365: R.Tensor((1, 7680), dtype="float16") = model_params[247]
            param_248: R.Tensor((7680,), dtype="float16") = model_params[248]
            lv45 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv364, lv365, lv363, param_248), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv368 = R.call_tir(cls.fused_reshape2_split, (lv45,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2610: R.Tensor((1, 1, 32, 80), dtype="float16") = lv368[0]
            lv2611 = R.call_tir(cls.rotary_embedding1, (lv2610, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2612: R.Tensor((1, 1, 32, 80), dtype="float16") = lv368[1]
            lv2613 = R.call_tir(cls.rotary_embedding1, (lv2612, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2614: R.Object = kv_cache[30]
            lv2615 = R.call_tir(cls.squeeze, (lv2613,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2616: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2614, lv2615, sinfo_args=(R.Object,))
            lv2617: R.Object = kv_cache[31]
            lv369: R.Tensor((1, 1, 32, 80), dtype="float16") = lv368[2]
            lv370 = R.call_tir(cls.fused_squeeze, (lv369,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2620: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2617, lv370, sinfo_args=(R.Object,))
            lv2621: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2616, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2622: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2620, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2623 = R.call_tir(cls.reshape3, (lv2621,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2624 = R.call_tir(cls.reshape3, (lv2622,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2625 = R.call_tir(cls.transpose, (lv2611,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2626 = R.call_tir(cls.transpose1, (lv2623,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2627 = R.call_tir(cls.transpose1, (lv2624,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv371 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv2625, lv2626, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv372 = R.call_tir(cls.fused_softmax1_cast3, (lv371,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2636 = R.call_tir(cls.matmul1, (lv372, lv2627), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv373 = R.call_tir(cls.fused_transpose2_reshape4, (lv2636,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv374: R.Tensor((2560, 2560), dtype="int8") = model_params[249]
            lv375: R.Tensor((1, 2560), dtype="float16") = model_params[250]
            lv376 = R.call_tir(cls.fused_decode8, (lv374, lv375), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_251: R.Tensor((2560,), dtype="float16") = model_params[251]
            lv2642 = R.call_tir(cls.cast, (lv362,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_244: R.Tensor((2560,), dtype="float32") = model_params[244]
            param_245: R.Tensor((2560,), dtype="float32") = model_params[245]
            lv377 = R.call_tir(cls.fused_layer_norm_cast1, (lv2642, param_244, param_245), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv378: R.Tensor((2560, 10240), dtype="int8") = model_params[252]
            lv379: R.Tensor((1, 10240), dtype="float16") = model_params[253]
            param_254: R.Tensor((10240,), dtype="float16") = model_params[254]
            lv46_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv378, lv379, lv377, param_254), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv382: R.Tensor((10240, 2560), dtype="int8") = model_params[255]
            lv383: R.Tensor((1, 2560), dtype="float16") = model_params[256]
            param_257: R.Tensor((2560,), dtype="float16") = model_params[257]
            lv47_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv382, lv383, lv46_1, param_257), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv386 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv373, lv376, param_251, lv47_1, lv362), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2656 = R.call_tir(cls.cast, (lv386,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_258: R.Tensor((2560,), dtype="float32") = model_params[258]
            param_259: R.Tensor((2560,), dtype="float32") = model_params[259]
            lv387 = R.call_tir(cls.fused_layer_norm_cast1, (lv2656, param_258, param_259), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv388: R.Tensor((2560, 7680), dtype="int8") = model_params[262]
            lv389: R.Tensor((1, 7680), dtype="float16") = model_params[263]
            param_264: R.Tensor((7680,), dtype="float16") = model_params[264]
            lv48 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv388, lv389, lv387, param_264), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv392 = R.call_tir(cls.fused_reshape2_split, (lv48,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2665: R.Tensor((1, 1, 32, 80), dtype="float16") = lv392[0]
            lv2666 = R.call_tir(cls.rotary_embedding1, (lv2665, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2667: R.Tensor((1, 1, 32, 80), dtype="float16") = lv392[1]
            lv2668 = R.call_tir(cls.rotary_embedding1, (lv2667, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2669: R.Object = kv_cache[32]
            lv2670 = R.call_tir(cls.squeeze, (lv2668,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2671: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2669, lv2670, sinfo_args=(R.Object,))
            lv2672: R.Object = kv_cache[33]
            lv393: R.Tensor((1, 1, 32, 80), dtype="float16") = lv392[2]
            lv394 = R.call_tir(cls.fused_squeeze, (lv393,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2675: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2672, lv394, sinfo_args=(R.Object,))
            lv2676: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2671, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2677: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2675, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2678 = R.call_tir(cls.reshape3, (lv2676,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2679 = R.call_tir(cls.reshape3, (lv2677,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2680 = R.call_tir(cls.transpose, (lv2666,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2681 = R.call_tir(cls.transpose1, (lv2678,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2682 = R.call_tir(cls.transpose1, (lv2679,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv395 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv2680, lv2681, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv396 = R.call_tir(cls.fused_softmax1_cast3, (lv395,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2691 = R.call_tir(cls.matmul1, (lv396, lv2682), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv397 = R.call_tir(cls.fused_transpose2_reshape4, (lv2691,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv398: R.Tensor((2560, 2560), dtype="int8") = model_params[265]
            lv399: R.Tensor((1, 2560), dtype="float16") = model_params[266]
            lv400 = R.call_tir(cls.fused_decode8, (lv398, lv399), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_267: R.Tensor((2560,), dtype="float16") = model_params[267]
            lv2697 = R.call_tir(cls.cast, (lv386,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_260: R.Tensor((2560,), dtype="float32") = model_params[260]
            param_261: R.Tensor((2560,), dtype="float32") = model_params[261]
            lv401 = R.call_tir(cls.fused_layer_norm_cast1, (lv2697, param_260, param_261), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv402: R.Tensor((2560, 10240), dtype="int8") = model_params[268]
            lv403: R.Tensor((1, 10240), dtype="float16") = model_params[269]
            param_270: R.Tensor((10240,), dtype="float16") = model_params[270]
            lv49 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv402, lv403, lv401, param_270), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv406: R.Tensor((10240, 2560), dtype="int8") = model_params[271]
            lv407: R.Tensor((1, 2560), dtype="float16") = model_params[272]
            param_273: R.Tensor((2560,), dtype="float16") = model_params[273]
            lv50_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv406, lv407, lv49, param_273), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv410 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv397, lv400, param_267, lv50_1, lv386), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2711 = R.call_tir(cls.cast, (lv410,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_274: R.Tensor((2560,), dtype="float32") = model_params[274]
            param_275: R.Tensor((2560,), dtype="float32") = model_params[275]
            lv411 = R.call_tir(cls.fused_layer_norm_cast1, (lv2711, param_274, param_275), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv412: R.Tensor((2560, 7680), dtype="int8") = model_params[278]
            lv413: R.Tensor((1, 7680), dtype="float16") = model_params[279]
            param_280: R.Tensor((7680,), dtype="float16") = model_params[280]
            lv51_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv412, lv413, lv411, param_280), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv416 = R.call_tir(cls.fused_reshape2_split, (lv51_1,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2720: R.Tensor((1, 1, 32, 80), dtype="float16") = lv416[0]
            lv2721 = R.call_tir(cls.rotary_embedding1, (lv2720, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2722: R.Tensor((1, 1, 32, 80), dtype="float16") = lv416[1]
            lv2723 = R.call_tir(cls.rotary_embedding1, (lv2722, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2724: R.Object = kv_cache[34]
            lv2725 = R.call_tir(cls.squeeze, (lv2723,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2726: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2724, lv2725, sinfo_args=(R.Object,))
            lv2727: R.Object = kv_cache[35]
            lv417: R.Tensor((1, 1, 32, 80), dtype="float16") = lv416[2]
            lv418 = R.call_tir(cls.fused_squeeze, (lv417,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2730: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2727, lv418, sinfo_args=(R.Object,))
            lv2731: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2726, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2732: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2730, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2733 = R.call_tir(cls.reshape3, (lv2731,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2734 = R.call_tir(cls.reshape3, (lv2732,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2735 = R.call_tir(cls.transpose, (lv2721,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2736 = R.call_tir(cls.transpose1, (lv2733,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2737 = R.call_tir(cls.transpose1, (lv2734,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv419 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv2735, lv2736, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv420 = R.call_tir(cls.fused_softmax1_cast3, (lv419,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2746 = R.call_tir(cls.matmul1, (lv420, lv2737), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv421 = R.call_tir(cls.fused_transpose2_reshape4, (lv2746,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv422: R.Tensor((2560, 2560), dtype="int8") = model_params[281]
            lv423: R.Tensor((1, 2560), dtype="float16") = model_params[282]
            lv424 = R.call_tir(cls.fused_decode8, (lv422, lv423), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_283: R.Tensor((2560,), dtype="float16") = model_params[283]
            lv2752 = R.call_tir(cls.cast, (lv410,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_276: R.Tensor((2560,), dtype="float32") = model_params[276]
            param_277: R.Tensor((2560,), dtype="float32") = model_params[277]
            lv425 = R.call_tir(cls.fused_layer_norm_cast1, (lv2752, param_276, param_277), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv426: R.Tensor((2560, 10240), dtype="int8") = model_params[284]
            lv427: R.Tensor((1, 10240), dtype="float16") = model_params[285]
            param_286: R.Tensor((10240,), dtype="float16") = model_params[286]
            lv52_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv426, lv427, lv425, param_286), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv430: R.Tensor((10240, 2560), dtype="int8") = model_params[287]
            lv431: R.Tensor((1, 2560), dtype="float16") = model_params[288]
            param_289: R.Tensor((2560,), dtype="float16") = model_params[289]
            lv53_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv430, lv431, lv52_1, param_289), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv434 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv421, lv424, param_283, lv53_1, lv410), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2766 = R.call_tir(cls.cast, (lv434,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_290: R.Tensor((2560,), dtype="float32") = model_params[290]
            param_291: R.Tensor((2560,), dtype="float32") = model_params[291]
            lv435 = R.call_tir(cls.fused_layer_norm_cast1, (lv2766, param_290, param_291), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv436: R.Tensor((2560, 7680), dtype="int8") = model_params[294]
            lv437: R.Tensor((1, 7680), dtype="float16") = model_params[295]
            param_296: R.Tensor((7680,), dtype="float16") = model_params[296]
            lv54 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv436, lv437, lv435, param_296), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv440 = R.call_tir(cls.fused_reshape2_split, (lv54,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2775: R.Tensor((1, 1, 32, 80), dtype="float16") = lv440[0]
            lv2776 = R.call_tir(cls.rotary_embedding1, (lv2775, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2777: R.Tensor((1, 1, 32, 80), dtype="float16") = lv440[1]
            lv2778 = R.call_tir(cls.rotary_embedding1, (lv2777, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2779: R.Object = kv_cache[36]
            lv2780 = R.call_tir(cls.squeeze, (lv2778,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2781: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2779, lv2780, sinfo_args=(R.Object,))
            lv2782: R.Object = kv_cache[37]
            lv441: R.Tensor((1, 1, 32, 80), dtype="float16") = lv440[2]
            lv442 = R.call_tir(cls.fused_squeeze, (lv441,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2785: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2782, lv442, sinfo_args=(R.Object,))
            lv2786: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2781, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2787: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2785, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2788 = R.call_tir(cls.reshape3, (lv2786,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2789 = R.call_tir(cls.reshape3, (lv2787,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2790 = R.call_tir(cls.transpose, (lv2776,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2791 = R.call_tir(cls.transpose1, (lv2788,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2792 = R.call_tir(cls.transpose1, (lv2789,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv443 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv2790, lv2791, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv444 = R.call_tir(cls.fused_softmax1_cast3, (lv443,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2801 = R.call_tir(cls.matmul1, (lv444, lv2792), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv445 = R.call_tir(cls.fused_transpose2_reshape4, (lv2801,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv446: R.Tensor((2560, 2560), dtype="int8") = model_params[297]
            lv447: R.Tensor((1, 2560), dtype="float16") = model_params[298]
            lv448 = R.call_tir(cls.fused_decode8, (lv446, lv447), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_299: R.Tensor((2560,), dtype="float16") = model_params[299]
            lv2807 = R.call_tir(cls.cast, (lv434,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_292: R.Tensor((2560,), dtype="float32") = model_params[292]
            param_293: R.Tensor((2560,), dtype="float32") = model_params[293]
            lv449 = R.call_tir(cls.fused_layer_norm_cast1, (lv2807, param_292, param_293), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv450: R.Tensor((2560, 10240), dtype="int8") = model_params[300]
            lv451: R.Tensor((1, 10240), dtype="float16") = model_params[301]
            param_302: R.Tensor((10240,), dtype="float16") = model_params[302]
            lv55 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv450, lv451, lv449, param_302), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv454: R.Tensor((10240, 2560), dtype="int8") = model_params[303]
            lv455: R.Tensor((1, 2560), dtype="float16") = model_params[304]
            param_305: R.Tensor((2560,), dtype="float16") = model_params[305]
            lv56_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv454, lv455, lv55, param_305), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv458 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv445, lv448, param_299, lv56_1, lv434), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2821 = R.call_tir(cls.cast, (lv458,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_306: R.Tensor((2560,), dtype="float32") = model_params[306]
            param_307: R.Tensor((2560,), dtype="float32") = model_params[307]
            lv459 = R.call_tir(cls.fused_layer_norm_cast1, (lv2821, param_306, param_307), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv460: R.Tensor((2560, 7680), dtype="int8") = model_params[310]
            lv461: R.Tensor((1, 7680), dtype="float16") = model_params[311]
            param_312: R.Tensor((7680,), dtype="float16") = model_params[312]
            lv57_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv460, lv461, lv459, param_312), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv464 = R.call_tir(cls.fused_reshape2_split, (lv57_1,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2830: R.Tensor((1, 1, 32, 80), dtype="float16") = lv464[0]
            lv2831 = R.call_tir(cls.rotary_embedding1, (lv2830, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2832: R.Tensor((1, 1, 32, 80), dtype="float16") = lv464[1]
            lv2833 = R.call_tir(cls.rotary_embedding1, (lv2832, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2834: R.Object = kv_cache[38]
            lv2835 = R.call_tir(cls.squeeze, (lv2833,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2836: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2834, lv2835, sinfo_args=(R.Object,))
            lv2837: R.Object = kv_cache[39]
            lv465: R.Tensor((1, 1, 32, 80), dtype="float16") = lv464[2]
            lv466 = R.call_tir(cls.fused_squeeze, (lv465,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2840: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2837, lv466, sinfo_args=(R.Object,))
            lv2841: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2836, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2842: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2840, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2843 = R.call_tir(cls.reshape3, (lv2841,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2844 = R.call_tir(cls.reshape3, (lv2842,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2845 = R.call_tir(cls.transpose, (lv2831,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2846 = R.call_tir(cls.transpose1, (lv2843,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2847 = R.call_tir(cls.transpose1, (lv2844,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv467 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv2845, lv2846, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv468 = R.call_tir(cls.fused_softmax1_cast3, (lv467,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2856 = R.call_tir(cls.matmul1, (lv468, lv2847), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv469 = R.call_tir(cls.fused_transpose2_reshape4, (lv2856,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv470: R.Tensor((2560, 2560), dtype="int8") = model_params[313]
            lv471: R.Tensor((1, 2560), dtype="float16") = model_params[314]
            lv472 = R.call_tir(cls.fused_decode8, (lv470, lv471), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_315: R.Tensor((2560,), dtype="float16") = model_params[315]
            lv2862 = R.call_tir(cls.cast, (lv458,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_308: R.Tensor((2560,), dtype="float32") = model_params[308]
            param_309: R.Tensor((2560,), dtype="float32") = model_params[309]
            lv473 = R.call_tir(cls.fused_layer_norm_cast1, (lv2862, param_308, param_309), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv474: R.Tensor((2560, 10240), dtype="int8") = model_params[316]
            lv475: R.Tensor((1, 10240), dtype="float16") = model_params[317]
            param_318: R.Tensor((10240,), dtype="float16") = model_params[318]
            lv58_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv474, lv475, lv473, param_318), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv478: R.Tensor((10240, 2560), dtype="int8") = model_params[319]
            lv479: R.Tensor((1, 2560), dtype="float16") = model_params[320]
            param_321: R.Tensor((2560,), dtype="float16") = model_params[321]
            lv59_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv478, lv479, lv58_1, param_321), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv482 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv469, lv472, param_315, lv59_1, lv458), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2876 = R.call_tir(cls.cast, (lv482,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_322: R.Tensor((2560,), dtype="float32") = model_params[322]
            param_323: R.Tensor((2560,), dtype="float32") = model_params[323]
            lv483 = R.call_tir(cls.fused_layer_norm_cast1, (lv2876, param_322, param_323), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv484: R.Tensor((2560, 7680), dtype="int8") = model_params[326]
            lv485: R.Tensor((1, 7680), dtype="float16") = model_params[327]
            param_328: R.Tensor((7680,), dtype="float16") = model_params[328]
            lv60_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv484, lv485, lv483, param_328), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv488 = R.call_tir(cls.fused_reshape2_split, (lv60_1,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2885: R.Tensor((1, 1, 32, 80), dtype="float16") = lv488[0]
            lv2886 = R.call_tir(cls.rotary_embedding1, (lv2885, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2887: R.Tensor((1, 1, 32, 80), dtype="float16") = lv488[1]
            lv2888 = R.call_tir(cls.rotary_embedding1, (lv2887, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2889: R.Object = kv_cache[40]
            lv2890 = R.call_tir(cls.squeeze, (lv2888,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2891: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2889, lv2890, sinfo_args=(R.Object,))
            lv2892: R.Object = kv_cache[41]
            lv489: R.Tensor((1, 1, 32, 80), dtype="float16") = lv488[2]
            lv490 = R.call_tir(cls.fused_squeeze, (lv489,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2895: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2892, lv490, sinfo_args=(R.Object,))
            lv2896: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2891, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2897: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2895, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2898 = R.call_tir(cls.reshape3, (lv2896,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2899 = R.call_tir(cls.reshape3, (lv2897,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2900 = R.call_tir(cls.transpose, (lv2886,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2901 = R.call_tir(cls.transpose1, (lv2898,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2902 = R.call_tir(cls.transpose1, (lv2899,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv491 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv2900, lv2901, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv492 = R.call_tir(cls.fused_softmax1_cast3, (lv491,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2911 = R.call_tir(cls.matmul1, (lv492, lv2902), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv493 = R.call_tir(cls.fused_transpose2_reshape4, (lv2911,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv494: R.Tensor((2560, 2560), dtype="int8") = model_params[329]
            lv495: R.Tensor((1, 2560), dtype="float16") = model_params[330]
            lv496 = R.call_tir(cls.fused_decode8, (lv494, lv495), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_331: R.Tensor((2560,), dtype="float16") = model_params[331]
            lv2917 = R.call_tir(cls.cast, (lv482,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_324: R.Tensor((2560,), dtype="float32") = model_params[324]
            param_325: R.Tensor((2560,), dtype="float32") = model_params[325]
            lv497 = R.call_tir(cls.fused_layer_norm_cast1, (lv2917, param_324, param_325), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv498: R.Tensor((2560, 10240), dtype="int8") = model_params[332]
            lv499: R.Tensor((1, 10240), dtype="float16") = model_params[333]
            param_334: R.Tensor((10240,), dtype="float16") = model_params[334]
            lv61_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv498, lv499, lv497, param_334), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv502: R.Tensor((10240, 2560), dtype="int8") = model_params[335]
            lv503: R.Tensor((1, 2560), dtype="float16") = model_params[336]
            param_337: R.Tensor((2560,), dtype="float16") = model_params[337]
            lv62_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv502, lv503, lv61_1, param_337), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv506 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv493, lv496, param_331, lv62_1, lv482), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2931 = R.call_tir(cls.cast, (lv506,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_338: R.Tensor((2560,), dtype="float32") = model_params[338]
            param_339: R.Tensor((2560,), dtype="float32") = model_params[339]
            lv507 = R.call_tir(cls.fused_layer_norm_cast1, (lv2931, param_338, param_339), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv508: R.Tensor((2560, 7680), dtype="int8") = model_params[342]
            lv509: R.Tensor((1, 7680), dtype="float16") = model_params[343]
            param_344: R.Tensor((7680,), dtype="float16") = model_params[344]
            lv63_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv508, lv509, lv507, param_344), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv512 = R.call_tir(cls.fused_reshape2_split, (lv63_1,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2940: R.Tensor((1, 1, 32, 80), dtype="float16") = lv512[0]
            lv2941 = R.call_tir(cls.rotary_embedding1, (lv2940, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2942: R.Tensor((1, 1, 32, 80), dtype="float16") = lv512[1]
            lv2943 = R.call_tir(cls.rotary_embedding1, (lv2942, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2944: R.Object = kv_cache[42]
            lv2945 = R.call_tir(cls.squeeze, (lv2943,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2946: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2944, lv2945, sinfo_args=(R.Object,))
            lv2947: R.Object = kv_cache[43]
            lv513: R.Tensor((1, 1, 32, 80), dtype="float16") = lv512[2]
            lv514 = R.call_tir(cls.fused_squeeze, (lv513,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2950: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2947, lv514, sinfo_args=(R.Object,))
            lv2951: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2946, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2952: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2950, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2953 = R.call_tir(cls.reshape3, (lv2951,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2954 = R.call_tir(cls.reshape3, (lv2952,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2955 = R.call_tir(cls.transpose, (lv2941,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2956 = R.call_tir(cls.transpose1, (lv2953,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2957 = R.call_tir(cls.transpose1, (lv2954,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv515 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv2955, lv2956, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv516 = R.call_tir(cls.fused_softmax1_cast3, (lv515,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2966 = R.call_tir(cls.matmul1, (lv516, lv2957), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv517 = R.call_tir(cls.fused_transpose2_reshape4, (lv2966,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv518: R.Tensor((2560, 2560), dtype="int8") = model_params[345]
            lv519: R.Tensor((1, 2560), dtype="float16") = model_params[346]
            lv520 = R.call_tir(cls.fused_decode8, (lv518, lv519), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_347: R.Tensor((2560,), dtype="float16") = model_params[347]
            lv2972 = R.call_tir(cls.cast, (lv506,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_340: R.Tensor((2560,), dtype="float32") = model_params[340]
            param_341: R.Tensor((2560,), dtype="float32") = model_params[341]
            lv521 = R.call_tir(cls.fused_layer_norm_cast1, (lv2972, param_340, param_341), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv522: R.Tensor((2560, 10240), dtype="int8") = model_params[348]
            lv523: R.Tensor((1, 10240), dtype="float16") = model_params[349]
            param_350: R.Tensor((10240,), dtype="float16") = model_params[350]
            lv64_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv522, lv523, lv521, param_350), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv526: R.Tensor((10240, 2560), dtype="int8") = model_params[351]
            lv527: R.Tensor((1, 2560), dtype="float16") = model_params[352]
            param_353: R.Tensor((2560,), dtype="float16") = model_params[353]
            lv65_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv526, lv527, lv64_1, param_353), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv530 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv517, lv520, param_347, lv65_1, lv506), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2986 = R.call_tir(cls.cast, (lv530,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_354: R.Tensor((2560,), dtype="float32") = model_params[354]
            param_355: R.Tensor((2560,), dtype="float32") = model_params[355]
            lv531 = R.call_tir(cls.fused_layer_norm_cast1, (lv2986, param_354, param_355), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv532: R.Tensor((2560, 7680), dtype="int8") = model_params[358]
            lv533: R.Tensor((1, 7680), dtype="float16") = model_params[359]
            param_360: R.Tensor((7680,), dtype="float16") = model_params[360]
            lv66_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv532, lv533, lv531, param_360), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv536 = R.call_tir(cls.fused_reshape2_split, (lv66_1,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2995: R.Tensor((1, 1, 32, 80), dtype="float16") = lv536[0]
            lv2996 = R.call_tir(cls.rotary_embedding1, (lv2995, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2997: R.Tensor((1, 1, 32, 80), dtype="float16") = lv536[1]
            lv2998 = R.call_tir(cls.rotary_embedding1, (lv2997, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2999: R.Object = kv_cache[44]
            lv3000 = R.call_tir(cls.squeeze, (lv2998,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3001: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2999, lv3000, sinfo_args=(R.Object,))
            lv3002: R.Object = kv_cache[45]
            lv537: R.Tensor((1, 1, 32, 80), dtype="float16") = lv536[2]
            lv538 = R.call_tir(cls.fused_squeeze, (lv537,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3005: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3002, lv538, sinfo_args=(R.Object,))
            lv3006: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3001, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3007: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3005, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3008 = R.call_tir(cls.reshape3, (lv3006,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3009 = R.call_tir(cls.reshape3, (lv3007,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3010 = R.call_tir(cls.transpose, (lv2996,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv3011 = R.call_tir(cls.transpose1, (lv3008,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv3012 = R.call_tir(cls.transpose1, (lv3009,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv539 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv3010, lv3011, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv540 = R.call_tir(cls.fused_softmax1_cast3, (lv539,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv3021 = R.call_tir(cls.matmul1, (lv540, lv3012), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv541 = R.call_tir(cls.fused_transpose2_reshape4, (lv3021,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv542: R.Tensor((2560, 2560), dtype="int8") = model_params[361]
            lv543: R.Tensor((1, 2560), dtype="float16") = model_params[362]
            lv544 = R.call_tir(cls.fused_decode8, (lv542, lv543), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_363: R.Tensor((2560,), dtype="float16") = model_params[363]
            lv3027 = R.call_tir(cls.cast, (lv530,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_356: R.Tensor((2560,), dtype="float32") = model_params[356]
            param_357: R.Tensor((2560,), dtype="float32") = model_params[357]
            lv545 = R.call_tir(cls.fused_layer_norm_cast1, (lv3027, param_356, param_357), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv546: R.Tensor((2560, 10240), dtype="int8") = model_params[364]
            lv547: R.Tensor((1, 10240), dtype="float16") = model_params[365]
            param_366: R.Tensor((10240,), dtype="float16") = model_params[366]
            lv67_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv546, lv547, lv545, param_366), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv550: R.Tensor((10240, 2560), dtype="int8") = model_params[367]
            lv551: R.Tensor((1, 2560), dtype="float16") = model_params[368]
            param_369: R.Tensor((2560,), dtype="float16") = model_params[369]
            lv68 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv550, lv551, lv67_1, param_369), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv554 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv541, lv544, param_363, lv68, lv530), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv3041 = R.call_tir(cls.cast, (lv554,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_370: R.Tensor((2560,), dtype="float32") = model_params[370]
            param_371: R.Tensor((2560,), dtype="float32") = model_params[371]
            lv555 = R.call_tir(cls.fused_layer_norm_cast1, (lv3041, param_370, param_371), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv556: R.Tensor((2560, 7680), dtype="int8") = model_params[374]
            lv557: R.Tensor((1, 7680), dtype="float16") = model_params[375]
            param_376: R.Tensor((7680,), dtype="float16") = model_params[376]
            lv69 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv556, lv557, lv555, param_376), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv560 = R.call_tir(cls.fused_reshape2_split, (lv69,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv3050: R.Tensor((1, 1, 32, 80), dtype="float16") = lv560[0]
            lv3051 = R.call_tir(cls.rotary_embedding1, (lv3050, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3052: R.Tensor((1, 1, 32, 80), dtype="float16") = lv560[1]
            lv3053 = R.call_tir(cls.rotary_embedding1, (lv3052, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3054: R.Object = kv_cache[46]
            lv3055 = R.call_tir(cls.squeeze, (lv3053,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3056: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3054, lv3055, sinfo_args=(R.Object,))
            lv3057: R.Object = kv_cache[47]
            lv561: R.Tensor((1, 1, 32, 80), dtype="float16") = lv560[2]
            lv562 = R.call_tir(cls.fused_squeeze, (lv561,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3060: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3057, lv562, sinfo_args=(R.Object,))
            lv3061: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3056, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3062: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3060, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3063 = R.call_tir(cls.reshape3, (lv3061,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3064 = R.call_tir(cls.reshape3, (lv3062,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3065 = R.call_tir(cls.transpose, (lv3051,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv3066 = R.call_tir(cls.transpose1, (lv3063,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv3067 = R.call_tir(cls.transpose1, (lv3064,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv563 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv3065, lv3066, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv564 = R.call_tir(cls.fused_softmax1_cast3, (lv563,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv3076 = R.call_tir(cls.matmul1, (lv564, lv3067), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv565 = R.call_tir(cls.fused_transpose2_reshape4, (lv3076,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv566: R.Tensor((2560, 2560), dtype="int8") = model_params[377]
            lv567: R.Tensor((1, 2560), dtype="float16") = model_params[378]
            lv568 = R.call_tir(cls.fused_decode8, (lv566, lv567), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_379: R.Tensor((2560,), dtype="float16") = model_params[379]
            lv3082 = R.call_tir(cls.cast, (lv554,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_372: R.Tensor((2560,), dtype="float32") = model_params[372]
            param_373: R.Tensor((2560,), dtype="float32") = model_params[373]
            lv569 = R.call_tir(cls.fused_layer_norm_cast1, (lv3082, param_372, param_373), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv570: R.Tensor((2560, 10240), dtype="int8") = model_params[380]
            lv571: R.Tensor((1, 10240), dtype="float16") = model_params[381]
            param_382: R.Tensor((10240,), dtype="float16") = model_params[382]
            lv70_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv570, lv571, lv569, param_382), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv574: R.Tensor((10240, 2560), dtype="int8") = model_params[383]
            lv575: R.Tensor((1, 2560), dtype="float16") = model_params[384]
            param_385: R.Tensor((2560,), dtype="float16") = model_params[385]
            lv71_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv574, lv575, lv70_1, param_385), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv578 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv565, lv568, param_379, lv71_1, lv554), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv3096 = R.call_tir(cls.cast, (lv578,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_386: R.Tensor((2560,), dtype="float32") = model_params[386]
            param_387: R.Tensor((2560,), dtype="float32") = model_params[387]
            lv579 = R.call_tir(cls.fused_layer_norm_cast1, (lv3096, param_386, param_387), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv580: R.Tensor((2560, 7680), dtype="int8") = model_params[390]
            lv581: R.Tensor((1, 7680), dtype="float16") = model_params[391]
            param_392: R.Tensor((7680,), dtype="float16") = model_params[392]
            lv72 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv580, lv581, lv579, param_392), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv584 = R.call_tir(cls.fused_reshape2_split, (lv72,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv3105: R.Tensor((1, 1, 32, 80), dtype="float16") = lv584[0]
            lv3106 = R.call_tir(cls.rotary_embedding1, (lv3105, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3107: R.Tensor((1, 1, 32, 80), dtype="float16") = lv584[1]
            lv3108 = R.call_tir(cls.rotary_embedding1, (lv3107, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3109: R.Object = kv_cache[48]
            lv3110 = R.call_tir(cls.squeeze, (lv3108,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3111: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3109, lv3110, sinfo_args=(R.Object,))
            lv3112: R.Object = kv_cache[49]
            lv585: R.Tensor((1, 1, 32, 80), dtype="float16") = lv584[2]
            lv586 = R.call_tir(cls.fused_squeeze, (lv585,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3115: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3112, lv586, sinfo_args=(R.Object,))
            lv3116: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3111, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3117: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3115, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3118 = R.call_tir(cls.reshape3, (lv3116,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3119 = R.call_tir(cls.reshape3, (lv3117,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3120 = R.call_tir(cls.transpose, (lv3106,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv3121 = R.call_tir(cls.transpose1, (lv3118,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv3122 = R.call_tir(cls.transpose1, (lv3119,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv587 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv3120, lv3121, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv588 = R.call_tir(cls.fused_softmax1_cast3, (lv587,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv3131 = R.call_tir(cls.matmul1, (lv588, lv3122), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv589 = R.call_tir(cls.fused_transpose2_reshape4, (lv3131,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv590: R.Tensor((2560, 2560), dtype="int8") = model_params[393]
            lv591: R.Tensor((1, 2560), dtype="float16") = model_params[394]
            lv592 = R.call_tir(cls.fused_decode8, (lv590, lv591), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_395: R.Tensor((2560,), dtype="float16") = model_params[395]
            lv3137 = R.call_tir(cls.cast, (lv578,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_388: R.Tensor((2560,), dtype="float32") = model_params[388]
            param_389: R.Tensor((2560,), dtype="float32") = model_params[389]
            lv593 = R.call_tir(cls.fused_layer_norm_cast1, (lv3137, param_388, param_389), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv594: R.Tensor((2560, 10240), dtype="int8") = model_params[396]
            lv595: R.Tensor((1, 10240), dtype="float16") = model_params[397]
            param_398: R.Tensor((10240,), dtype="float16") = model_params[398]
            lv73 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv594, lv595, lv593, param_398), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv598: R.Tensor((10240, 2560), dtype="int8") = model_params[399]
            lv599: R.Tensor((1, 2560), dtype="float16") = model_params[400]
            param_401: R.Tensor((2560,), dtype="float16") = model_params[401]
            lv74_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv598, lv599, lv73, param_401), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv602 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv589, lv592, param_395, lv74_1, lv578), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv3151 = R.call_tir(cls.cast, (lv602,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_402: R.Tensor((2560,), dtype="float32") = model_params[402]
            param_403: R.Tensor((2560,), dtype="float32") = model_params[403]
            lv603 = R.call_tir(cls.fused_layer_norm_cast1, (lv3151, param_402, param_403), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv604: R.Tensor((2560, 7680), dtype="int8") = model_params[406]
            lv605: R.Tensor((1, 7680), dtype="float16") = model_params[407]
            param_408: R.Tensor((7680,), dtype="float16") = model_params[408]
            lv75_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv604, lv605, lv603, param_408), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv608 = R.call_tir(cls.fused_reshape2_split, (lv75_1,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv3160: R.Tensor((1, 1, 32, 80), dtype="float16") = lv608[0]
            lv3161 = R.call_tir(cls.rotary_embedding1, (lv3160, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3162: R.Tensor((1, 1, 32, 80), dtype="float16") = lv608[1]
            lv3163 = R.call_tir(cls.rotary_embedding1, (lv3162, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3164: R.Object = kv_cache[50]
            lv3165 = R.call_tir(cls.squeeze, (lv3163,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3166: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3164, lv3165, sinfo_args=(R.Object,))
            lv3167: R.Object = kv_cache[51]
            lv609: R.Tensor((1, 1, 32, 80), dtype="float16") = lv608[2]
            lv610 = R.call_tir(cls.fused_squeeze, (lv609,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3170: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3167, lv610, sinfo_args=(R.Object,))
            lv3171: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3166, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3172: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3170, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3173 = R.call_tir(cls.reshape3, (lv3171,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3174 = R.call_tir(cls.reshape3, (lv3172,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3175 = R.call_tir(cls.transpose, (lv3161,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv3176 = R.call_tir(cls.transpose1, (lv3173,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv3177 = R.call_tir(cls.transpose1, (lv3174,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv611 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv3175, lv3176, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv612 = R.call_tir(cls.fused_softmax1_cast3, (lv611,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv3186 = R.call_tir(cls.matmul1, (lv612, lv3177), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv613 = R.call_tir(cls.fused_transpose2_reshape4, (lv3186,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv614: R.Tensor((2560, 2560), dtype="int8") = model_params[409]
            lv615: R.Tensor((1, 2560), dtype="float16") = model_params[410]
            lv616 = R.call_tir(cls.fused_decode8, (lv614, lv615), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_411: R.Tensor((2560,), dtype="float16") = model_params[411]
            lv3192 = R.call_tir(cls.cast, (lv602,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_404: R.Tensor((2560,), dtype="float32") = model_params[404]
            param_405: R.Tensor((2560,), dtype="float32") = model_params[405]
            lv617 = R.call_tir(cls.fused_layer_norm_cast1, (lv3192, param_404, param_405), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv618: R.Tensor((2560, 10240), dtype="int8") = model_params[412]
            lv619: R.Tensor((1, 10240), dtype="float16") = model_params[413]
            param_414: R.Tensor((10240,), dtype="float16") = model_params[414]
            lv76_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv618, lv619, lv617, param_414), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv622: R.Tensor((10240, 2560), dtype="int8") = model_params[415]
            lv623: R.Tensor((1, 2560), dtype="float16") = model_params[416]
            param_417: R.Tensor((2560,), dtype="float16") = model_params[417]
            lv77_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv622, lv623, lv76_1, param_417), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv626 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv613, lv616, param_411, lv77_1, lv602), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv3206 = R.call_tir(cls.cast, (lv626,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_418: R.Tensor((2560,), dtype="float32") = model_params[418]
            param_419: R.Tensor((2560,), dtype="float32") = model_params[419]
            lv627 = R.call_tir(cls.fused_layer_norm_cast1, (lv3206, param_418, param_419), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv628: R.Tensor((2560, 7680), dtype="int8") = model_params[422]
            lv629: R.Tensor((1, 7680), dtype="float16") = model_params[423]
            param_424: R.Tensor((7680,), dtype="float16") = model_params[424]
            lv78 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv628, lv629, lv627, param_424), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv632 = R.call_tir(cls.fused_reshape2_split, (lv78,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv3215: R.Tensor((1, 1, 32, 80), dtype="float16") = lv632[0]
            lv3216 = R.call_tir(cls.rotary_embedding1, (lv3215, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3217: R.Tensor((1, 1, 32, 80), dtype="float16") = lv632[1]
            lv3218 = R.call_tir(cls.rotary_embedding1, (lv3217, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3219: R.Object = kv_cache[52]
            lv3220 = R.call_tir(cls.squeeze, (lv3218,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3221: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3219, lv3220, sinfo_args=(R.Object,))
            lv3222: R.Object = kv_cache[53]
            lv633: R.Tensor((1, 1, 32, 80), dtype="float16") = lv632[2]
            lv634 = R.call_tir(cls.fused_squeeze, (lv633,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3225: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3222, lv634, sinfo_args=(R.Object,))
            lv3226: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3221, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3227: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3225, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3228 = R.call_tir(cls.reshape3, (lv3226,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3229 = R.call_tir(cls.reshape3, (lv3227,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3230 = R.call_tir(cls.transpose, (lv3216,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv3231 = R.call_tir(cls.transpose1, (lv3228,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv3232 = R.call_tir(cls.transpose1, (lv3229,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv635 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv3230, lv3231, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv636 = R.call_tir(cls.fused_softmax1_cast3, (lv635,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv3241 = R.call_tir(cls.matmul1, (lv636, lv3232), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv637 = R.call_tir(cls.fused_transpose2_reshape4, (lv3241,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv638: R.Tensor((2560, 2560), dtype="int8") = model_params[425]
            lv639: R.Tensor((1, 2560), dtype="float16") = model_params[426]
            lv640 = R.call_tir(cls.fused_decode8, (lv638, lv639), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_427: R.Tensor((2560,), dtype="float16") = model_params[427]
            lv3247 = R.call_tir(cls.cast, (lv626,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_420: R.Tensor((2560,), dtype="float32") = model_params[420]
            param_421: R.Tensor((2560,), dtype="float32") = model_params[421]
            lv641 = R.call_tir(cls.fused_layer_norm_cast1, (lv3247, param_420, param_421), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv642: R.Tensor((2560, 10240), dtype="int8") = model_params[428]
            lv643: R.Tensor((1, 10240), dtype="float16") = model_params[429]
            param_430: R.Tensor((10240,), dtype="float16") = model_params[430]
            lv79 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv642, lv643, lv641, param_430), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv646: R.Tensor((10240, 2560), dtype="int8") = model_params[431]
            lv647: R.Tensor((1, 2560), dtype="float16") = model_params[432]
            param_433: R.Tensor((2560,), dtype="float16") = model_params[433]
            lv80_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv646, lv647, lv79, param_433), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv650 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv637, lv640, param_427, lv80_1, lv626), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv3261 = R.call_tir(cls.cast, (lv650,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_434: R.Tensor((2560,), dtype="float32") = model_params[434]
            param_435: R.Tensor((2560,), dtype="float32") = model_params[435]
            lv651 = R.call_tir(cls.fused_layer_norm_cast1, (lv3261, param_434, param_435), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv652: R.Tensor((2560, 7680), dtype="int8") = model_params[438]
            lv653: R.Tensor((1, 7680), dtype="float16") = model_params[439]
            param_440: R.Tensor((7680,), dtype="float16") = model_params[440]
            lv81_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv652, lv653, lv651, param_440), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv656 = R.call_tir(cls.fused_reshape2_split, (lv81_1,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv3270: R.Tensor((1, 1, 32, 80), dtype="float16") = lv656[0]
            lv3271 = R.call_tir(cls.rotary_embedding1, (lv3270, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3272: R.Tensor((1, 1, 32, 80), dtype="float16") = lv656[1]
            lv3273 = R.call_tir(cls.rotary_embedding1, (lv3272, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3274: R.Object = kv_cache[54]
            lv3275 = R.call_tir(cls.squeeze, (lv3273,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3276: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3274, lv3275, sinfo_args=(R.Object,))
            lv3277: R.Object = kv_cache[55]
            lv657: R.Tensor((1, 1, 32, 80), dtype="float16") = lv656[2]
            lv658 = R.call_tir(cls.fused_squeeze, (lv657,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3280: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3277, lv658, sinfo_args=(R.Object,))
            lv3281: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3276, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3282: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3280, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3283 = R.call_tir(cls.reshape3, (lv3281,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3284 = R.call_tir(cls.reshape3, (lv3282,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3285 = R.call_tir(cls.transpose, (lv3271,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv3286 = R.call_tir(cls.transpose1, (lv3283,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv3287 = R.call_tir(cls.transpose1, (lv3284,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv659 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv3285, lv3286, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv660 = R.call_tir(cls.fused_softmax1_cast3, (lv659,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv3296 = R.call_tir(cls.matmul1, (lv660, lv3287), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv661 = R.call_tir(cls.fused_transpose2_reshape4, (lv3296,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv662: R.Tensor((2560, 2560), dtype="int8") = model_params[441]
            lv663: R.Tensor((1, 2560), dtype="float16") = model_params[442]
            lv664 = R.call_tir(cls.fused_decode8, (lv662, lv663), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_443: R.Tensor((2560,), dtype="float16") = model_params[443]
            lv3302 = R.call_tir(cls.cast, (lv650,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_436: R.Tensor((2560,), dtype="float32") = model_params[436]
            param_437: R.Tensor((2560,), dtype="float32") = model_params[437]
            lv665 = R.call_tir(cls.fused_layer_norm_cast1, (lv3302, param_436, param_437), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv666: R.Tensor((2560, 10240), dtype="int8") = model_params[444]
            lv667: R.Tensor((1, 10240), dtype="float16") = model_params[445]
            param_446: R.Tensor((10240,), dtype="float16") = model_params[446]
            lv82_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv666, lv667, lv665, param_446), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv670: R.Tensor((10240, 2560), dtype="int8") = model_params[447]
            lv671: R.Tensor((1, 2560), dtype="float16") = model_params[448]
            param_449: R.Tensor((2560,), dtype="float16") = model_params[449]
            lv83_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv670, lv671, lv82_1, param_449), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv674 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv661, lv664, param_443, lv83_1, lv650), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv3316 = R.call_tir(cls.cast, (lv674,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_450: R.Tensor((2560,), dtype="float32") = model_params[450]
            param_451: R.Tensor((2560,), dtype="float32") = model_params[451]
            lv675 = R.call_tir(cls.fused_layer_norm_cast1, (lv3316, param_450, param_451), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv676: R.Tensor((2560, 7680), dtype="int8") = model_params[454]
            lv677: R.Tensor((1, 7680), dtype="float16") = model_params[455]
            param_456: R.Tensor((7680,), dtype="float16") = model_params[456]
            lv84_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv676, lv677, lv675, param_456), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv680 = R.call_tir(cls.fused_reshape2_split, (lv84_1,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv3325: R.Tensor((1, 1, 32, 80), dtype="float16") = lv680[0]
            lv3326 = R.call_tir(cls.rotary_embedding1, (lv3325, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3327: R.Tensor((1, 1, 32, 80), dtype="float16") = lv680[1]
            lv3328 = R.call_tir(cls.rotary_embedding1, (lv3327, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3329: R.Object = kv_cache[56]
            lv3330 = R.call_tir(cls.squeeze, (lv3328,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3331: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3329, lv3330, sinfo_args=(R.Object,))
            lv3332: R.Object = kv_cache[57]
            lv681: R.Tensor((1, 1, 32, 80), dtype="float16") = lv680[2]
            lv682 = R.call_tir(cls.fused_squeeze, (lv681,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3335: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3332, lv682, sinfo_args=(R.Object,))
            lv3336: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3331, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3337: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3335, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3338 = R.call_tir(cls.reshape3, (lv3336,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3339 = R.call_tir(cls.reshape3, (lv3337,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3340 = R.call_tir(cls.transpose, (lv3326,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv3341 = R.call_tir(cls.transpose1, (lv3338,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv3342 = R.call_tir(cls.transpose1, (lv3339,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv683 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv3340, lv3341, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv684 = R.call_tir(cls.fused_softmax1_cast3, (lv683,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv3351 = R.call_tir(cls.matmul1, (lv684, lv3342), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv685 = R.call_tir(cls.fused_transpose2_reshape4, (lv3351,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv686: R.Tensor((2560, 2560), dtype="int8") = model_params[457]
            lv687: R.Tensor((1, 2560), dtype="float16") = model_params[458]
            lv688 = R.call_tir(cls.fused_decode8, (lv686, lv687), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_459: R.Tensor((2560,), dtype="float16") = model_params[459]
            lv3357 = R.call_tir(cls.cast, (lv674,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_452: R.Tensor((2560,), dtype="float32") = model_params[452]
            param_453: R.Tensor((2560,), dtype="float32") = model_params[453]
            lv689 = R.call_tir(cls.fused_layer_norm_cast1, (lv3357, param_452, param_453), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv690: R.Tensor((2560, 10240), dtype="int8") = model_params[460]
            lv691: R.Tensor((1, 10240), dtype="float16") = model_params[461]
            param_462: R.Tensor((10240,), dtype="float16") = model_params[462]
            lv85_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv690, lv691, lv689, param_462), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv694: R.Tensor((10240, 2560), dtype="int8") = model_params[463]
            lv695: R.Tensor((1, 2560), dtype="float16") = model_params[464]
            param_465: R.Tensor((2560,), dtype="float16") = model_params[465]
            lv86_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv694, lv695, lv85_1, param_465), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv698 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv685, lv688, param_459, lv86_1, lv674), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv3371 = R.call_tir(cls.cast, (lv698,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_466: R.Tensor((2560,), dtype="float32") = model_params[466]
            param_467: R.Tensor((2560,), dtype="float32") = model_params[467]
            lv699 = R.call_tir(cls.fused_layer_norm_cast1, (lv3371, param_466, param_467), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv700: R.Tensor((2560, 7680), dtype="int8") = model_params[470]
            lv701: R.Tensor((1, 7680), dtype="float16") = model_params[471]
            param_472: R.Tensor((7680,), dtype="float16") = model_params[472]
            lv87_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv700, lv701, lv699, param_472), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv704 = R.call_tir(cls.fused_reshape2_split, (lv87_1,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv3380: R.Tensor((1, 1, 32, 80), dtype="float16") = lv704[0]
            lv3381 = R.call_tir(cls.rotary_embedding1, (lv3380, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3382: R.Tensor((1, 1, 32, 80), dtype="float16") = lv704[1]
            lv3383 = R.call_tir(cls.rotary_embedding1, (lv3382, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3384: R.Object = kv_cache[58]
            lv3385 = R.call_tir(cls.squeeze, (lv3383,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3386: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3384, lv3385, sinfo_args=(R.Object,))
            lv3387: R.Object = kv_cache[59]
            lv705: R.Tensor((1, 1, 32, 80), dtype="float16") = lv704[2]
            lv706 = R.call_tir(cls.fused_squeeze, (lv705,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3390: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3387, lv706, sinfo_args=(R.Object,))
            lv3391: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3386, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3392: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3390, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3393 = R.call_tir(cls.reshape3, (lv3391,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3394 = R.call_tir(cls.reshape3, (lv3392,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3395 = R.call_tir(cls.transpose, (lv3381,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv3396 = R.call_tir(cls.transpose1, (lv3393,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv3397 = R.call_tir(cls.transpose1, (lv3394,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv707 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv3395, lv3396, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv708 = R.call_tir(cls.fused_softmax1_cast3, (lv707,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv3406 = R.call_tir(cls.matmul1, (lv708, lv3397), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv709 = R.call_tir(cls.fused_transpose2_reshape4, (lv3406,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv710: R.Tensor((2560, 2560), dtype="int8") = model_params[473]
            lv711: R.Tensor((1, 2560), dtype="float16") = model_params[474]
            lv712 = R.call_tir(cls.fused_decode8, (lv710, lv711), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_475: R.Tensor((2560,), dtype="float16") = model_params[475]
            lv3412 = R.call_tir(cls.cast, (lv698,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_468: R.Tensor((2560,), dtype="float32") = model_params[468]
            param_469: R.Tensor((2560,), dtype="float32") = model_params[469]
            lv713 = R.call_tir(cls.fused_layer_norm_cast1, (lv3412, param_468, param_469), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv714: R.Tensor((2560, 10240), dtype="int8") = model_params[476]
            lv715: R.Tensor((1, 10240), dtype="float16") = model_params[477]
            param_478: R.Tensor((10240,), dtype="float16") = model_params[478]
            lv88_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv714, lv715, lv713, param_478), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv718: R.Tensor((10240, 2560), dtype="int8") = model_params[479]
            lv719: R.Tensor((1, 2560), dtype="float16") = model_params[480]
            param_481: R.Tensor((2560,), dtype="float16") = model_params[481]
            lv89_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv718, lv719, lv88_1, param_481), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv722 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv709, lv712, param_475, lv89_1, lv698), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv3426 = R.call_tir(cls.cast, (lv722,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_482: R.Tensor((2560,), dtype="float32") = model_params[482]
            param_483: R.Tensor((2560,), dtype="float32") = model_params[483]
            lv723 = R.call_tir(cls.fused_layer_norm_cast1, (lv3426, param_482, param_483), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv724: R.Tensor((2560, 7680), dtype="int8") = model_params[486]
            lv725: R.Tensor((1, 7680), dtype="float16") = model_params[487]
            param_488: R.Tensor((7680,), dtype="float16") = model_params[488]
            lv90_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv724, lv725, lv723, param_488), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv728 = R.call_tir(cls.fused_reshape2_split, (lv90_1,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv3435: R.Tensor((1, 1, 32, 80), dtype="float16") = lv728[0]
            lv3436 = R.call_tir(cls.rotary_embedding1, (lv3435, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3437: R.Tensor((1, 1, 32, 80), dtype="float16") = lv728[1]
            lv3438 = R.call_tir(cls.rotary_embedding1, (lv3437, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3439: R.Object = kv_cache[60]
            lv3440 = R.call_tir(cls.squeeze, (lv3438,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3441: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3439, lv3440, sinfo_args=(R.Object,))
            lv3442: R.Object = kv_cache[61]
            lv729: R.Tensor((1, 1, 32, 80), dtype="float16") = lv728[2]
            lv730 = R.call_tir(cls.fused_squeeze, (lv729,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3445: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3442, lv730, sinfo_args=(R.Object,))
            lv3446: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3441, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3447: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3445, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3448 = R.call_tir(cls.reshape3, (lv3446,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3449 = R.call_tir(cls.reshape3, (lv3447,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3450 = R.call_tir(cls.transpose, (lv3436,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv3451 = R.call_tir(cls.transpose1, (lv3448,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv3452 = R.call_tir(cls.transpose1, (lv3449,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv731 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv3450, lv3451, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv732 = R.call_tir(cls.fused_softmax1_cast3, (lv731,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv3461 = R.call_tir(cls.matmul1, (lv732, lv3452), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv733 = R.call_tir(cls.fused_transpose2_reshape4, (lv3461,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv734: R.Tensor((2560, 2560), dtype="int8") = model_params[489]
            lv735: R.Tensor((1, 2560), dtype="float16") = model_params[490]
            lv736 = R.call_tir(cls.fused_decode8, (lv734, lv735), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_491: R.Tensor((2560,), dtype="float16") = model_params[491]
            lv3467 = R.call_tir(cls.cast, (lv722,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_484: R.Tensor((2560,), dtype="float32") = model_params[484]
            param_485: R.Tensor((2560,), dtype="float32") = model_params[485]
            lv737 = R.call_tir(cls.fused_layer_norm_cast1, (lv3467, param_484, param_485), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv738: R.Tensor((2560, 10240), dtype="int8") = model_params[492]
            lv739: R.Tensor((1, 10240), dtype="float16") = model_params[493]
            param_494: R.Tensor((10240,), dtype="float16") = model_params[494]
            lv91_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv738, lv739, lv737, param_494), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv742: R.Tensor((10240, 2560), dtype="int8") = model_params[495]
            lv743: R.Tensor((1, 2560), dtype="float16") = model_params[496]
            param_497: R.Tensor((2560,), dtype="float16") = model_params[497]
            lv92 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv742, lv743, lv91_1, param_497), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv746 = R.call_tir(cls.fused_matmul2_add1_add3_add3, (lv733, lv736, param_491, lv92, lv722), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv3481 = R.call_tir(cls.cast, (lv746,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_498: R.Tensor((2560,), dtype="float32") = model_params[498]
            param_499: R.Tensor((2560,), dtype="float32") = model_params[499]
            lv747 = R.call_tir(cls.fused_layer_norm_cast1, (lv3481, param_498, param_499), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv748: R.Tensor((2560, 7680), dtype="int8") = model_params[502]
            lv749: R.Tensor((1, 7680), dtype="float16") = model_params[503]
            param_504: R.Tensor((7680,), dtype="float16") = model_params[504]
            lv93 = R.call_tir(cls.fused_fused_decode7_fused_matmul_add, (lv748, lv749, lv747, param_504), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv752 = R.call_tir(cls.fused_reshape2_split, (lv93,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv3490: R.Tensor((1, 1, 32, 80), dtype="float16") = lv752[0]
            lv3491 = R.call_tir(cls.rotary_embedding1, (lv3490, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3492: R.Tensor((1, 1, 32, 80), dtype="float16") = lv752[1]
            lv3493 = R.call_tir(cls.rotary_embedding1, (lv3492, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3494: R.Object = kv_cache[62]
            lv3495 = R.call_tir(cls.squeeze, (lv3493,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3496: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3494, lv3495, sinfo_args=(R.Object,))
            lv3497: R.Object = kv_cache[63]
            lv753: R.Tensor((1, 1, 32, 80), dtype="float16") = lv752[2]
            lv754 = R.call_tir(cls.fused_squeeze, (lv753,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3500: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3497, lv754, sinfo_args=(R.Object,))
            lv3501: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3496, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3502: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3500, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3503 = R.call_tir(cls.reshape3, (lv3501,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3504 = R.call_tir(cls.reshape3, (lv3502,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3505 = R.call_tir(cls.transpose, (lv3491,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv3506 = R.call_tir(cls.transpose1, (lv3503,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv3507 = R.call_tir(cls.transpose1, (lv3504,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv755 = R.call_tir(cls.fused_NT_matmul_divide1_maximum_minimum_cast2, (lv3505, lv3506, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv756 = R.call_tir(cls.fused_softmax1_cast3, (lv755,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv3516 = R.call_tir(cls.matmul1, (lv756, lv3507), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv757 = R.call_tir(cls.fused_transpose2_reshape4, (lv3516,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv758: R.Tensor((2560, 2560), dtype="int8") = model_params[505]
            lv759: R.Tensor((1, 2560), dtype="float16") = model_params[506]
            lv760 = R.call_tir(cls.fused_decode8, (lv758, lv759), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_507: R.Tensor((2560,), dtype="float16") = model_params[507]
            lv3522 = R.call_tir(cls.cast, (lv746,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_500: R.Tensor((2560,), dtype="float32") = model_params[500]
            param_501: R.Tensor((2560,), dtype="float32") = model_params[501]
            lv761 = R.call_tir(cls.fused_layer_norm_cast1, (lv3522, param_500, param_501), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv762: R.Tensor((2560, 10240), dtype="int8") = model_params[508]
            lv763: R.Tensor((1, 10240), dtype="float16") = model_params[509]
            param_510: R.Tensor((10240,), dtype="float16") = model_params[510]
            lv94_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul3_add2_gelu, (lv762, lv763, lv761, param_510), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            lv766: R.Tensor((10240, 2560), dtype="int8") = model_params[511]
            lv767: R.Tensor((1, 2560), dtype="float16") = model_params[512]
            param_513: R.Tensor((2560,), dtype="float16") = model_params[513]
            lv95_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul4_add1, (lv766, lv767, lv94_1, param_513), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv770 = R.call_tir(cls.fused_matmul2_add1_add3_add3_cast, (lv757, lv760, param_507, lv95_1, lv746), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_514: R.Tensor((2560,), dtype="float32") = model_params[514]
            param_515: R.Tensor((2560,), dtype="float32") = model_params[515]
            lv3537 = R.call_tir(cls.layer_norm, (lv770, param_514, param_515), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            lv771 = R.call_tir(cls.fused_slice1_cast4, (lv3537,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            lv772: R.Tensor((50280, 320), dtype="uint32") = model_params[516]
            lv773: R.Tensor((50280, 80), dtype="float32") = model_params[517]
            lv_2 = R.call_tir(cls.fused_fused_decode6_NT_matmul1, (lv772, lv773, lv771), out_sinfo=R.Tensor((1, 1, 50280), dtype="float32"))
            gv1: R.Tuple(R.Tensor((1, 1, 50280), dtype="float32"), R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object)) = lv_2, (lv1791, lv1795, lv1846, lv1850, lv1901, lv1905, lv1956, lv1960, lv2011, lv2015, lv2066, lv2070, lv2121, lv2125, lv2176, lv2180, lv2231, lv2235, lv2286, lv2290, lv2341, lv2345, lv2396, lv2400, lv2451, lv2455, lv2506, lv2510, lv2561, lv2565, lv2616, lv2620, lv2671, lv2675, lv2726, lv2730, lv2781, lv2785, lv2836, lv2840, lv2891, lv2895, lv2946, lv2950, lv3001, lv3005, lv3056, lv3060, lv3111, lv3115, lv3166, lv3170, lv3221, lv3225, lv3276, lv3280, lv3331, lv3335, lv3386, lv3390, lv3441, lv3445, lv3496, lv3500)
            R.output(gv1)
        return gv1

    @R.function
    def get_metadata() -> R.Object:
        R.func_attr({"tir_var_upper_bound": {"m": 4096, "n": 4096}})
        return R.str("{\"model_name\": \"dolly-v2-3b\", \"max_window_size\": 4096, \"stop_tokens\": [2], \"add_prefix_space\": false, \"prefill_chunk_size\": -1, \"sliding_window\": -1}")

    @R.function
    def prefill(input_ids: R.Tensor((1, "n"), dtype="int32"), all_seq_len: R.Shape(["m"]), kv_cache: R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object), model_params: R.Tuple(R.Tensor((50280, 640), dtype="uint32"), R.Tensor((50280, 80), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560, 7680), dtype="int8"), R.Tensor((1, 7680), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560, 10240), dtype="int8"), R.Tensor((1, 10240), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((10240, 2560), dtype="int8"), R.Tensor((1, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((50280, 320), dtype="uint32"), R.Tensor((50280, 80), dtype="float32"))) -> R.Tuple(R.Tensor((1, 1, 50280), dtype="float32"), R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object)):
        n = T.int64()
        m = T.int64()
        R.func_attr({"num_input": 3, "tir_var_upper_bound": {"m": 4096, "n": 4096}})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.reshape5, (input_ids,), out_sinfo=R.Tensor((n,), dtype="int32"))
            lv775: R.Tensor((50280, 640), dtype="uint32") = model_params[0]
            lv776: R.Tensor((50280, 80), dtype="float16") = model_params[1]
            lv_1 = R.call_tir(cls.fused_fused_decode1_take1, (lv775, lv776, lv), out_sinfo=R.Tensor((n, 2560), dtype="float16"))
            lv2 = R.call_tir(cls.reshape6, (lv_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv778 = R.call_tir(cls.fused_min_max_triu_te_broadcast_to, R.tuple(), out_sinfo=R.Tensor((1, 1, n, n), dtype="float16"), tir_vars=R.shape([n]))
            lv5 = R.call_tir(cls.extend_te, (lv778,), out_sinfo=R.Tensor((1, 1, n, m), dtype="float16"))
            lv6 = R.call_tir(cls.cast5, (lv2,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_2100: R.Tensor((2560,), dtype="float32") = model_params[2]
            param_3100: R.Tensor((2560,), dtype="float32") = model_params[3]
            lv779 = R.call_tir(cls.fused_layer_norm1_cast6, (lv6, param_2100, param_3100), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv780: R.Tensor((2560, 7680), dtype="int8") = model_params[6]
            lv781: R.Tensor((1, 7680), dtype="float16") = model_params[7]
            param_810: R.Tensor((7680,), dtype="float16") = model_params[8]
            lv96 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv780, lv781, lv779, param_810), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv13 = R.call_tir(cls.reshape7, (lv96,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv14 = R.call_tir(cls.split1, (lv13,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv15: R.Tensor((1, n, 32, 80), dtype="float16") = lv14[0]
            lv16 = R.call_tir(cls.rotary_embedding, (lv15, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv17: R.Tensor((1, n, 32, 80), dtype="float16") = lv14[1]
            lv18 = R.call_tir(cls.rotary_embedding, (lv17, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv19: R.Object = kv_cache[0]
            lv20 = R.call_tir(cls.squeeze1, (lv18,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv21: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv19, lv20, sinfo_args=(R.Object,))
            lv22: R.Object = kv_cache[1]
            lv784: R.Tensor((1, n, 32, 80), dtype="float16") = lv14[2]
            lv785 = R.call_tir(cls.fused_squeeze1, (lv784,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv25: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv22, lv785, sinfo_args=(R.Object,))
            lv26: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv21, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv27: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv25, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv28 = R.call_tir(cls.reshape3, (lv26,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv29 = R.call_tir(cls.reshape3, (lv27,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv30 = R.call_tir(cls.transpose1, (lv16,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv31 = R.call_tir(cls.transpose1, (lv28,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv32 = R.call_tir(cls.transpose1, (lv29,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv786 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv30, lv31, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv787 = R.call_tir(cls.fused_softmax2_cast8, (lv786,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv41 = R.call_tir(cls.matmul9, (lv787, lv32), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv42 = R.call_tir(cls.transpose5, (lv41,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv43 = R.call_tir(cls.reshape8, (lv42,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv788: R.Tensor((2560, 2560), dtype="int8") = model_params[9]
            lv789: R.Tensor((1, 2560), dtype="float16") = model_params[10]
            lv790 = R.call_tir(cls.fused_decode8, (lv788, lv789), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_1110: R.Tensor((2560,), dtype="float16") = model_params[11]
            lv47 = R.call_tir(cls.cast5, (lv2,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_4100: R.Tensor((2560,), dtype="float32") = model_params[4]
            param_518: R.Tensor((2560,), dtype="float32") = model_params[5]
            lv791 = R.call_tir(cls.fused_layer_norm1_cast6, (lv47, param_4100, param_518), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv792: R.Tensor((2560, 10240), dtype="int8") = model_params[12]
            lv793: R.Tensor((1, 10240), dtype="float16") = model_params[13]
            param_1410: R.Tensor((10240,), dtype="float16") = model_params[14]
            lv97 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv792, lv793, lv791, param_1410), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv796: R.Tensor((10240, 2560), dtype="int8") = model_params[15]
            lv797: R.Tensor((1, 2560), dtype="float16") = model_params[16]
            param_1710: R.Tensor((2560,), dtype="float16") = model_params[17]
            lv98 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv796, lv797, lv97, param_1710), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv800 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv43, lv790, param_1110, lv98, lv2), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv61 = R.call_tir(cls.cast5, (lv800,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_1810: R.Tensor((2560,), dtype="float32") = model_params[18]
            param_1910: R.Tensor((2560,), dtype="float32") = model_params[19]
            lv801 = R.call_tir(cls.fused_layer_norm1_cast6, (lv61, param_1810, param_1910), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv802: R.Tensor((2560, 7680), dtype="int8") = model_params[22]
            lv803: R.Tensor((1, 7680), dtype="float16") = model_params[23]
            param_2410: R.Tensor((7680,), dtype="float16") = model_params[24]
            lv99 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv802, lv803, lv801, param_2410), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv68 = R.call_tir(cls.reshape7, (lv99,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv69 = R.call_tir(cls.split1, (lv68,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv70: R.Tensor((1, n, 32, 80), dtype="float16") = lv69[0]
            lv71 = R.call_tir(cls.rotary_embedding, (lv70, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv72: R.Tensor((1, n, 32, 80), dtype="float16") = lv69[1]
            lv73 = R.call_tir(cls.rotary_embedding, (lv72, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv74: R.Object = kv_cache[2]
            lv75 = R.call_tir(cls.squeeze1, (lv73,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv76: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv74, lv75, sinfo_args=(R.Object,))
            lv77: R.Object = kv_cache[3]
            lv806: R.Tensor((1, n, 32, 80), dtype="float16") = lv69[2]
            lv807 = R.call_tir(cls.fused_squeeze1, (lv806,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv80: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv77, lv807, sinfo_args=(R.Object,))
            lv81: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv76, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv82: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv80, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv83 = R.call_tir(cls.reshape3, (lv81,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv84 = R.call_tir(cls.reshape3, (lv82,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv85 = R.call_tir(cls.transpose1, (lv71,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv86 = R.call_tir(cls.transpose1, (lv83,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv87 = R.call_tir(cls.transpose1, (lv84,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv808 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv85, lv86, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv809 = R.call_tir(cls.fused_softmax2_cast8, (lv808,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv96_1 = R.call_tir(cls.matmul9, (lv809, lv87), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv97_1 = R.call_tir(cls.transpose5, (lv96_1,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv98_1 = R.call_tir(cls.reshape8, (lv97_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv810: R.Tensor((2560, 2560), dtype="int8") = model_params[25]
            lv811: R.Tensor((1, 2560), dtype="float16") = model_params[26]
            lv812 = R.call_tir(cls.fused_decode8, (lv810, lv811), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_2710: R.Tensor((2560,), dtype="float16") = model_params[27]
            lv102 = R.call_tir(cls.cast5, (lv800,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_2010: R.Tensor((2560,), dtype="float32") = model_params[20]
            param_2110: R.Tensor((2560,), dtype="float32") = model_params[21]
            lv813 = R.call_tir(cls.fused_layer_norm1_cast6, (lv102, param_2010, param_2110), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv814: R.Tensor((2560, 10240), dtype="int8") = model_params[28]
            lv815: R.Tensor((1, 10240), dtype="float16") = model_params[29]
            param_3010: R.Tensor((10240,), dtype="float16") = model_params[30]
            lv100 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv814, lv815, lv813, param_3010), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv818: R.Tensor((10240, 2560), dtype="int8") = model_params[31]
            lv819: R.Tensor((1, 2560), dtype="float16") = model_params[32]
            param_3310: R.Tensor((2560,), dtype="float16") = model_params[33]
            lv101 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv818, lv819, lv100, param_3310), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv822 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv98_1, lv812, param_2710, lv101, lv800), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv116 = R.call_tir(cls.cast5, (lv822,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_3410: R.Tensor((2560,), dtype="float32") = model_params[34]
            param_3510: R.Tensor((2560,), dtype="float32") = model_params[35]
            lv823 = R.call_tir(cls.fused_layer_norm1_cast6, (lv116, param_3410, param_3510), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv824: R.Tensor((2560, 7680), dtype="int8") = model_params[38]
            lv825: R.Tensor((1, 7680), dtype="float16") = model_params[39]
            param_4010: R.Tensor((7680,), dtype="float16") = model_params[40]
            lv102_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv824, lv825, lv823, param_4010), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv123 = R.call_tir(cls.reshape7, (lv102_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv124 = R.call_tir(cls.split1, (lv123,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv125: R.Tensor((1, n, 32, 80), dtype="float16") = lv124[0]
            lv126 = R.call_tir(cls.rotary_embedding, (lv125, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv127: R.Tensor((1, n, 32, 80), dtype="float16") = lv124[1]
            lv128 = R.call_tir(cls.rotary_embedding, (lv127, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv129: R.Object = kv_cache[4]
            lv130 = R.call_tir(cls.squeeze1, (lv128,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv131: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv129, lv130, sinfo_args=(R.Object,))
            lv132: R.Object = kv_cache[5]
            lv828: R.Tensor((1, n, 32, 80), dtype="float16") = lv124[2]
            lv829 = R.call_tir(cls.fused_squeeze1, (lv828,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv135: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv132, lv829, sinfo_args=(R.Object,))
            lv136: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv131, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv137: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv135, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv138 = R.call_tir(cls.reshape3, (lv136,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv139 = R.call_tir(cls.reshape3, (lv137,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv140 = R.call_tir(cls.transpose1, (lv126,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv141 = R.call_tir(cls.transpose1, (lv138,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv142 = R.call_tir(cls.transpose1, (lv139,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv830 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv140, lv141, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv831 = R.call_tir(cls.fused_softmax2_cast8, (lv830,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv151 = R.call_tir(cls.matmul9, (lv831, lv142), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv152 = R.call_tir(cls.transpose5, (lv151,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv153 = R.call_tir(cls.reshape8, (lv152,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv832: R.Tensor((2560, 2560), dtype="int8") = model_params[41]
            lv833: R.Tensor((1, 2560), dtype="float16") = model_params[42]
            lv834 = R.call_tir(cls.fused_decode8, (lv832, lv833), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_4310: R.Tensor((2560,), dtype="float16") = model_params[43]
            lv157 = R.call_tir(cls.cast5, (lv822,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_3610: R.Tensor((2560,), dtype="float32") = model_params[36]
            param_3710: R.Tensor((2560,), dtype="float32") = model_params[37]
            lv835 = R.call_tir(cls.fused_layer_norm1_cast6, (lv157, param_3610, param_3710), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv836: R.Tensor((2560, 10240), dtype="int8") = model_params[44]
            lv837: R.Tensor((1, 10240), dtype="float16") = model_params[45]
            param_4610: R.Tensor((10240,), dtype="float16") = model_params[46]
            lv103 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv836, lv837, lv835, param_4610), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv840: R.Tensor((10240, 2560), dtype="int8") = model_params[47]
            lv841: R.Tensor((1, 2560), dtype="float16") = model_params[48]
            param_4910: R.Tensor((2560,), dtype="float16") = model_params[49]
            lv104 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv840, lv841, lv103, param_4910), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv844 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv153, lv834, param_4310, lv104, lv822), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv171 = R.call_tir(cls.cast5, (lv844,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_5010: R.Tensor((2560,), dtype="float32") = model_params[50]
            param_519: R.Tensor((2560,), dtype="float32") = model_params[51]
            lv845 = R.call_tir(cls.fused_layer_norm1_cast6, (lv171, param_5010, param_519), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv846: R.Tensor((2560, 7680), dtype="int8") = model_params[54]
            lv847: R.Tensor((1, 7680), dtype="float16") = model_params[55]
            param_561: R.Tensor((7680,), dtype="float16") = model_params[56]
            lv105 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv846, lv847, lv845, param_561), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv178 = R.call_tir(cls.reshape7, (lv105,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv179 = R.call_tir(cls.split1, (lv178,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv180: R.Tensor((1, n, 32, 80), dtype="float16") = lv179[0]
            lv181 = R.call_tir(cls.rotary_embedding, (lv180, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv182: R.Tensor((1, n, 32, 80), dtype="float16") = lv179[1]
            lv183 = R.call_tir(cls.rotary_embedding, (lv182, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv184: R.Object = kv_cache[6]
            lv185 = R.call_tir(cls.squeeze1, (lv183,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv186: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv184, lv185, sinfo_args=(R.Object,))
            lv187: R.Object = kv_cache[7]
            lv850: R.Tensor((1, n, 32, 80), dtype="float16") = lv179[2]
            lv851 = R.call_tir(cls.fused_squeeze1, (lv850,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv190: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv187, lv851, sinfo_args=(R.Object,))
            lv191: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv186, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv192: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv190, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv193 = R.call_tir(cls.reshape3, (lv191,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv194 = R.call_tir(cls.reshape3, (lv192,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv195 = R.call_tir(cls.transpose1, (lv181,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv196 = R.call_tir(cls.transpose1, (lv193,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv197 = R.call_tir(cls.transpose1, (lv194,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv852 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv195, lv196, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv853 = R.call_tir(cls.fused_softmax2_cast8, (lv852,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv206 = R.call_tir(cls.matmul9, (lv853, lv197), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv207 = R.call_tir(cls.transpose5, (lv206,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv208 = R.call_tir(cls.reshape8, (lv207,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv854: R.Tensor((2560, 2560), dtype="int8") = model_params[57]
            lv855: R.Tensor((1, 2560), dtype="float16") = model_params[58]
            lv856 = R.call_tir(cls.fused_decode8, (lv854, lv855), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_591: R.Tensor((2560,), dtype="float16") = model_params[59]
            lv212 = R.call_tir(cls.cast5, (lv844,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_521: R.Tensor((2560,), dtype="float32") = model_params[52]
            param_531: R.Tensor((2560,), dtype="float32") = model_params[53]
            lv857 = R.call_tir(cls.fused_layer_norm1_cast6, (lv212, param_521, param_531), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv858: R.Tensor((2560, 10240), dtype="int8") = model_params[60]
            lv859: R.Tensor((1, 10240), dtype="float16") = model_params[61]
            param_621: R.Tensor((10240,), dtype="float16") = model_params[62]
            lv106 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv858, lv859, lv857, param_621), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv862: R.Tensor((10240, 2560), dtype="int8") = model_params[63]
            lv863: R.Tensor((1, 2560), dtype="float16") = model_params[64]
            param_651: R.Tensor((2560,), dtype="float16") = model_params[65]
            lv107 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv862, lv863, lv106, param_651), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv866 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv208, lv856, param_591, lv107, lv844), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv226 = R.call_tir(cls.cast5, (lv866,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_661: R.Tensor((2560,), dtype="float32") = model_params[66]
            param_671: R.Tensor((2560,), dtype="float32") = model_params[67]
            lv867 = R.call_tir(cls.fused_layer_norm1_cast6, (lv226, param_661, param_671), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv868: R.Tensor((2560, 7680), dtype="int8") = model_params[70]
            lv869: R.Tensor((1, 7680), dtype="float16") = model_params[71]
            param_721: R.Tensor((7680,), dtype="float16") = model_params[72]
            lv108 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv868, lv869, lv867, param_721), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv233 = R.call_tir(cls.reshape7, (lv108,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv234 = R.call_tir(cls.split1, (lv233,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv235: R.Tensor((1, n, 32, 80), dtype="float16") = lv234[0]
            lv236 = R.call_tir(cls.rotary_embedding, (lv235, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv237: R.Tensor((1, n, 32, 80), dtype="float16") = lv234[1]
            lv238 = R.call_tir(cls.rotary_embedding, (lv237, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv239: R.Object = kv_cache[8]
            lv240 = R.call_tir(cls.squeeze1, (lv238,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv241: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv239, lv240, sinfo_args=(R.Object,))
            lv242: R.Object = kv_cache[9]
            lv872: R.Tensor((1, n, 32, 80), dtype="float16") = lv234[2]
            lv873 = R.call_tir(cls.fused_squeeze1, (lv872,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv245: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv242, lv873, sinfo_args=(R.Object,))
            lv246: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv241, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv247: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv245, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv248 = R.call_tir(cls.reshape3, (lv246,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv249 = R.call_tir(cls.reshape3, (lv247,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv250 = R.call_tir(cls.transpose1, (lv236,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv251 = R.call_tir(cls.transpose1, (lv248,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv252 = R.call_tir(cls.transpose1, (lv249,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv874 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv250, lv251, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv875 = R.call_tir(cls.fused_softmax2_cast8, (lv874,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv261 = R.call_tir(cls.matmul9, (lv875, lv252), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv262 = R.call_tir(cls.transpose5, (lv261,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv263 = R.call_tir(cls.reshape8, (lv262,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv876: R.Tensor((2560, 2560), dtype="int8") = model_params[73]
            lv877: R.Tensor((1, 2560), dtype="float16") = model_params[74]
            lv878 = R.call_tir(cls.fused_decode8, (lv876, lv877), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_751: R.Tensor((2560,), dtype="float16") = model_params[75]
            lv267 = R.call_tir(cls.cast5, (lv866,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_681: R.Tensor((2560,), dtype="float32") = model_params[68]
            param_691: R.Tensor((2560,), dtype="float32") = model_params[69]
            lv879 = R.call_tir(cls.fused_layer_norm1_cast6, (lv267, param_681, param_691), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv880: R.Tensor((2560, 10240), dtype="int8") = model_params[76]
            lv881: R.Tensor((1, 10240), dtype="float16") = model_params[77]
            param_781: R.Tensor((10240,), dtype="float16") = model_params[78]
            lv109 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv880, lv881, lv879, param_781), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv884: R.Tensor((10240, 2560), dtype="int8") = model_params[79]
            lv885: R.Tensor((1, 2560), dtype="float16") = model_params[80]
            param_811: R.Tensor((2560,), dtype="float16") = model_params[81]
            lv110 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv884, lv885, lv109, param_811), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv888 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv263, lv878, param_751, lv110, lv866), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv281 = R.call_tir(cls.cast5, (lv888,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_821: R.Tensor((2560,), dtype="float32") = model_params[82]
            param_831: R.Tensor((2560,), dtype="float32") = model_params[83]
            lv889 = R.call_tir(cls.fused_layer_norm1_cast6, (lv281, param_821, param_831), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv890: R.Tensor((2560, 7680), dtype="int8") = model_params[86]
            lv891: R.Tensor((1, 7680), dtype="float16") = model_params[87]
            param_881: R.Tensor((7680,), dtype="float16") = model_params[88]
            lv111 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv890, lv891, lv889, param_881), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv288 = R.call_tir(cls.reshape7, (lv111,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv289 = R.call_tir(cls.split1, (lv288,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv290: R.Tensor((1, n, 32, 80), dtype="float16") = lv289[0]
            lv291 = R.call_tir(cls.rotary_embedding, (lv290, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv292: R.Tensor((1, n, 32, 80), dtype="float16") = lv289[1]
            lv293 = R.call_tir(cls.rotary_embedding, (lv292, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv294: R.Object = kv_cache[10]
            lv295 = R.call_tir(cls.squeeze1, (lv293,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv296: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv294, lv295, sinfo_args=(R.Object,))
            lv297: R.Object = kv_cache[11]
            lv894: R.Tensor((1, n, 32, 80), dtype="float16") = lv289[2]
            lv895 = R.call_tir(cls.fused_squeeze1, (lv894,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv300: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv297, lv895, sinfo_args=(R.Object,))
            lv301: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv296, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv302: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv300, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv303 = R.call_tir(cls.reshape3, (lv301,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv304 = R.call_tir(cls.reshape3, (lv302,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv305 = R.call_tir(cls.transpose1, (lv291,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv306 = R.call_tir(cls.transpose1, (lv303,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv307 = R.call_tir(cls.transpose1, (lv304,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv896 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv305, lv306, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv897 = R.call_tir(cls.fused_softmax2_cast8, (lv896,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv316 = R.call_tir(cls.matmul9, (lv897, lv307), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv317 = R.call_tir(cls.transpose5, (lv316,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv318 = R.call_tir(cls.reshape8, (lv317,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv898: R.Tensor((2560, 2560), dtype="int8") = model_params[89]
            lv899: R.Tensor((1, 2560), dtype="float16") = model_params[90]
            lv900 = R.call_tir(cls.fused_decode8, (lv898, lv899), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_911: R.Tensor((2560,), dtype="float16") = model_params[91]
            lv322 = R.call_tir(cls.cast5, (lv888,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_841: R.Tensor((2560,), dtype="float32") = model_params[84]
            param_851: R.Tensor((2560,), dtype="float32") = model_params[85]
            lv901 = R.call_tir(cls.fused_layer_norm1_cast6, (lv322, param_841, param_851), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv902: R.Tensor((2560, 10240), dtype="int8") = model_params[92]
            lv903: R.Tensor((1, 10240), dtype="float16") = model_params[93]
            param_941: R.Tensor((10240,), dtype="float16") = model_params[94]
            lv112 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv902, lv903, lv901, param_941), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv906: R.Tensor((10240, 2560), dtype="int8") = model_params[95]
            lv907: R.Tensor((1, 2560), dtype="float16") = model_params[96]
            param_971: R.Tensor((2560,), dtype="float16") = model_params[97]
            lv113 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv906, lv907, lv112, param_971), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv910 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv318, lv900, param_911, lv113, lv888), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv336 = R.call_tir(cls.cast5, (lv910,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_981: R.Tensor((2560,), dtype="float32") = model_params[98]
            param_991: R.Tensor((2560,), dtype="float32") = model_params[99]
            lv911 = R.call_tir(cls.fused_layer_norm1_cast6, (lv336, param_981, param_991), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv912: R.Tensor((2560, 7680), dtype="int8") = model_params[102]
            lv913: R.Tensor((1, 7680), dtype="float16") = model_params[103]
            param_1041: R.Tensor((7680,), dtype="float16") = model_params[104]
            lv114 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv912, lv913, lv911, param_1041), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv343 = R.call_tir(cls.reshape7, (lv114,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv344 = R.call_tir(cls.split1, (lv343,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv345: R.Tensor((1, n, 32, 80), dtype="float16") = lv344[0]
            lv346 = R.call_tir(cls.rotary_embedding, (lv345, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv347: R.Tensor((1, n, 32, 80), dtype="float16") = lv344[1]
            lv348 = R.call_tir(cls.rotary_embedding, (lv347, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv349: R.Object = kv_cache[12]
            lv350 = R.call_tir(cls.squeeze1, (lv348,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv351: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv349, lv350, sinfo_args=(R.Object,))
            lv352: R.Object = kv_cache[13]
            lv916: R.Tensor((1, n, 32, 80), dtype="float16") = lv344[2]
            lv917 = R.call_tir(cls.fused_squeeze1, (lv916,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv355: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv352, lv917, sinfo_args=(R.Object,))
            lv356: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv351, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv357: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv355, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv358 = R.call_tir(cls.reshape3, (lv356,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv359 = R.call_tir(cls.reshape3, (lv357,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv360 = R.call_tir(cls.transpose1, (lv346,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv361 = R.call_tir(cls.transpose1, (lv358,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv362 = R.call_tir(cls.transpose1, (lv359,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv918 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv360, lv361, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv919 = R.call_tir(cls.fused_softmax2_cast8, (lv918,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv371 = R.call_tir(cls.matmul9, (lv919, lv362), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv372 = R.call_tir(cls.transpose5, (lv371,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv373 = R.call_tir(cls.reshape8, (lv372,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv920: R.Tensor((2560, 2560), dtype="int8") = model_params[105]
            lv921: R.Tensor((1, 2560), dtype="float16") = model_params[106]
            lv922 = R.call_tir(cls.fused_decode8, (lv920, lv921), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_1071: R.Tensor((2560,), dtype="float16") = model_params[107]
            lv377 = R.call_tir(cls.cast5, (lv910,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_1001: R.Tensor((2560,), dtype="float32") = model_params[100]
            param_1011: R.Tensor((2560,), dtype="float32") = model_params[101]
            lv923 = R.call_tir(cls.fused_layer_norm1_cast6, (lv377, param_1001, param_1011), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv924: R.Tensor((2560, 10240), dtype="int8") = model_params[108]
            lv925: R.Tensor((1, 10240), dtype="float16") = model_params[109]
            param_1101: R.Tensor((10240,), dtype="float16") = model_params[110]
            lv115 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv924, lv925, lv923, param_1101), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv928: R.Tensor((10240, 2560), dtype="int8") = model_params[111]
            lv929: R.Tensor((1, 2560), dtype="float16") = model_params[112]
            param_1131: R.Tensor((2560,), dtype="float16") = model_params[113]
            lv116_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv928, lv929, lv115, param_1131), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv932 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv373, lv922, param_1071, lv116_1, lv910), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv391 = R.call_tir(cls.cast5, (lv932,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_1141: R.Tensor((2560,), dtype="float32") = model_params[114]
            param_1151: R.Tensor((2560,), dtype="float32") = model_params[115]
            lv933 = R.call_tir(cls.fused_layer_norm1_cast6, (lv391, param_1141, param_1151), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv934: R.Tensor((2560, 7680), dtype="int8") = model_params[118]
            lv935: R.Tensor((1, 7680), dtype="float16") = model_params[119]
            param_1201: R.Tensor((7680,), dtype="float16") = model_params[120]
            lv117 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv934, lv935, lv933, param_1201), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv398 = R.call_tir(cls.reshape7, (lv117,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv399 = R.call_tir(cls.split1, (lv398,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv400: R.Tensor((1, n, 32, 80), dtype="float16") = lv399[0]
            lv401 = R.call_tir(cls.rotary_embedding, (lv400, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv402: R.Tensor((1, n, 32, 80), dtype="float16") = lv399[1]
            lv403 = R.call_tir(cls.rotary_embedding, (lv402, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv404: R.Object = kv_cache[14]
            lv405 = R.call_tir(cls.squeeze1, (lv403,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv406: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv404, lv405, sinfo_args=(R.Object,))
            lv407: R.Object = kv_cache[15]
            lv938: R.Tensor((1, n, 32, 80), dtype="float16") = lv399[2]
            lv939 = R.call_tir(cls.fused_squeeze1, (lv938,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv410: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv407, lv939, sinfo_args=(R.Object,))
            lv411: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv406, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv412: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv410, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv413 = R.call_tir(cls.reshape3, (lv411,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv414 = R.call_tir(cls.reshape3, (lv412,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv415 = R.call_tir(cls.transpose1, (lv401,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv416 = R.call_tir(cls.transpose1, (lv413,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv417 = R.call_tir(cls.transpose1, (lv414,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv940 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv415, lv416, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv941 = R.call_tir(cls.fused_softmax2_cast8, (lv940,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv426 = R.call_tir(cls.matmul9, (lv941, lv417), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv427 = R.call_tir(cls.transpose5, (lv426,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv428 = R.call_tir(cls.reshape8, (lv427,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv942: R.Tensor((2560, 2560), dtype="int8") = model_params[121]
            lv943: R.Tensor((1, 2560), dtype="float16") = model_params[122]
            lv944 = R.call_tir(cls.fused_decode8, (lv942, lv943), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_1231: R.Tensor((2560,), dtype="float16") = model_params[123]
            lv432 = R.call_tir(cls.cast5, (lv932,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_1161: R.Tensor((2560,), dtype="float32") = model_params[116]
            param_1171: R.Tensor((2560,), dtype="float32") = model_params[117]
            lv945 = R.call_tir(cls.fused_layer_norm1_cast6, (lv432, param_1161, param_1171), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv946: R.Tensor((2560, 10240), dtype="int8") = model_params[124]
            lv947: R.Tensor((1, 10240), dtype="float16") = model_params[125]
            param_1261: R.Tensor((10240,), dtype="float16") = model_params[126]
            lv118 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv946, lv947, lv945, param_1261), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv950: R.Tensor((10240, 2560), dtype="int8") = model_params[127]
            lv951: R.Tensor((1, 2560), dtype="float16") = model_params[128]
            param_1291: R.Tensor((2560,), dtype="float16") = model_params[129]
            lv119 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv950, lv951, lv118, param_1291), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv954 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv428, lv944, param_1231, lv119, lv932), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv446 = R.call_tir(cls.cast5, (lv954,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_1301: R.Tensor((2560,), dtype="float32") = model_params[130]
            param_1311: R.Tensor((2560,), dtype="float32") = model_params[131]
            lv955 = R.call_tir(cls.fused_layer_norm1_cast6, (lv446, param_1301, param_1311), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv956: R.Tensor((2560, 7680), dtype="int8") = model_params[134]
            lv957: R.Tensor((1, 7680), dtype="float16") = model_params[135]
            param_1361: R.Tensor((7680,), dtype="float16") = model_params[136]
            lv120 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv956, lv957, lv955, param_1361), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv453 = R.call_tir(cls.reshape7, (lv120,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv454 = R.call_tir(cls.split1, (lv453,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv455: R.Tensor((1, n, 32, 80), dtype="float16") = lv454[0]
            lv456 = R.call_tir(cls.rotary_embedding, (lv455, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv457: R.Tensor((1, n, 32, 80), dtype="float16") = lv454[1]
            lv458 = R.call_tir(cls.rotary_embedding, (lv457, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv459: R.Object = kv_cache[16]
            lv460 = R.call_tir(cls.squeeze1, (lv458,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv461: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv459, lv460, sinfo_args=(R.Object,))
            lv462: R.Object = kv_cache[17]
            lv960: R.Tensor((1, n, 32, 80), dtype="float16") = lv454[2]
            lv961 = R.call_tir(cls.fused_squeeze1, (lv960,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv465: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv462, lv961, sinfo_args=(R.Object,))
            lv466: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv461, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv467: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv465, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv468 = R.call_tir(cls.reshape3, (lv466,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv469 = R.call_tir(cls.reshape3, (lv467,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv470 = R.call_tir(cls.transpose1, (lv456,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv471 = R.call_tir(cls.transpose1, (lv468,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv472 = R.call_tir(cls.transpose1, (lv469,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv962 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv470, lv471, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv963 = R.call_tir(cls.fused_softmax2_cast8, (lv962,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv481 = R.call_tir(cls.matmul9, (lv963, lv472), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv482 = R.call_tir(cls.transpose5, (lv481,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv483 = R.call_tir(cls.reshape8, (lv482,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv964: R.Tensor((2560, 2560), dtype="int8") = model_params[137]
            lv965: R.Tensor((1, 2560), dtype="float16") = model_params[138]
            lv966 = R.call_tir(cls.fused_decode8, (lv964, lv965), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_1391: R.Tensor((2560,), dtype="float16") = model_params[139]
            lv487 = R.call_tir(cls.cast5, (lv954,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_1321: R.Tensor((2560,), dtype="float32") = model_params[132]
            param_1331: R.Tensor((2560,), dtype="float32") = model_params[133]
            lv967 = R.call_tir(cls.fused_layer_norm1_cast6, (lv487, param_1321, param_1331), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv968: R.Tensor((2560, 10240), dtype="int8") = model_params[140]
            lv969: R.Tensor((1, 10240), dtype="float16") = model_params[141]
            param_1421: R.Tensor((10240,), dtype="float16") = model_params[142]
            lv121 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv968, lv969, lv967, param_1421), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv972: R.Tensor((10240, 2560), dtype="int8") = model_params[143]
            lv973: R.Tensor((1, 2560), dtype="float16") = model_params[144]
            param_1451: R.Tensor((2560,), dtype="float16") = model_params[145]
            lv122 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv972, lv973, lv121, param_1451), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv976 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv483, lv966, param_1391, lv122, lv954), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv501 = R.call_tir(cls.cast5, (lv976,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_1461: R.Tensor((2560,), dtype="float32") = model_params[146]
            param_1471: R.Tensor((2560,), dtype="float32") = model_params[147]
            lv977 = R.call_tir(cls.fused_layer_norm1_cast6, (lv501, param_1461, param_1471), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv978: R.Tensor((2560, 7680), dtype="int8") = model_params[150]
            lv979: R.Tensor((1, 7680), dtype="float16") = model_params[151]
            param_1521: R.Tensor((7680,), dtype="float16") = model_params[152]
            lv123_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv978, lv979, lv977, param_1521), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv508 = R.call_tir(cls.reshape7, (lv123_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv509 = R.call_tir(cls.split1, (lv508,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv510: R.Tensor((1, n, 32, 80), dtype="float16") = lv509[0]
            lv511 = R.call_tir(cls.rotary_embedding, (lv510, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv512: R.Tensor((1, n, 32, 80), dtype="float16") = lv509[1]
            lv513 = R.call_tir(cls.rotary_embedding, (lv512, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv514: R.Object = kv_cache[18]
            lv515 = R.call_tir(cls.squeeze1, (lv513,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv516: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv514, lv515, sinfo_args=(R.Object,))
            lv517: R.Object = kv_cache[19]
            lv982: R.Tensor((1, n, 32, 80), dtype="float16") = lv509[2]
            lv983 = R.call_tir(cls.fused_squeeze1, (lv982,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv520: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv517, lv983, sinfo_args=(R.Object,))
            lv521: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv516, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv522: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv520, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv523 = R.call_tir(cls.reshape3, (lv521,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv524 = R.call_tir(cls.reshape3, (lv522,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv525 = R.call_tir(cls.transpose1, (lv511,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv526 = R.call_tir(cls.transpose1, (lv523,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv527 = R.call_tir(cls.transpose1, (lv524,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv984 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv525, lv526, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv985 = R.call_tir(cls.fused_softmax2_cast8, (lv984,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv536 = R.call_tir(cls.matmul9, (lv985, lv527), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv537 = R.call_tir(cls.transpose5, (lv536,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv538 = R.call_tir(cls.reshape8, (lv537,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv986: R.Tensor((2560, 2560), dtype="int8") = model_params[153]
            lv987: R.Tensor((1, 2560), dtype="float16") = model_params[154]
            lv988 = R.call_tir(cls.fused_decode8, (lv986, lv987), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_1551: R.Tensor((2560,), dtype="float16") = model_params[155]
            lv542 = R.call_tir(cls.cast5, (lv976,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_1481: R.Tensor((2560,), dtype="float32") = model_params[148]
            param_1491: R.Tensor((2560,), dtype="float32") = model_params[149]
            lv989 = R.call_tir(cls.fused_layer_norm1_cast6, (lv542, param_1481, param_1491), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv990: R.Tensor((2560, 10240), dtype="int8") = model_params[156]
            lv991: R.Tensor((1, 10240), dtype="float16") = model_params[157]
            param_1581: R.Tensor((10240,), dtype="float16") = model_params[158]
            lv124_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv990, lv991, lv989, param_1581), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv994: R.Tensor((10240, 2560), dtype="int8") = model_params[159]
            lv995: R.Tensor((1, 2560), dtype="float16") = model_params[160]
            param_1611: R.Tensor((2560,), dtype="float16") = model_params[161]
            lv125_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv994, lv995, lv124_1, param_1611), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv998 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv538, lv988, param_1551, lv125_1, lv976), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv556 = R.call_tir(cls.cast5, (lv998,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_1621: R.Tensor((2560,), dtype="float32") = model_params[162]
            param_1631: R.Tensor((2560,), dtype="float32") = model_params[163]
            lv999 = R.call_tir(cls.fused_layer_norm1_cast6, (lv556, param_1621, param_1631), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1000: R.Tensor((2560, 7680), dtype="int8") = model_params[166]
            lv1001: R.Tensor((1, 7680), dtype="float16") = model_params[167]
            param_1681: R.Tensor((7680,), dtype="float16") = model_params[168]
            lv126_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv1000, lv1001, lv999, param_1681), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv563 = R.call_tir(cls.reshape7, (lv126_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv564 = R.call_tir(cls.split1, (lv563,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv565: R.Tensor((1, n, 32, 80), dtype="float16") = lv564[0]
            lv566 = R.call_tir(cls.rotary_embedding, (lv565, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv567: R.Tensor((1, n, 32, 80), dtype="float16") = lv564[1]
            lv568 = R.call_tir(cls.rotary_embedding, (lv567, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv569: R.Object = kv_cache[20]
            lv570 = R.call_tir(cls.squeeze1, (lv568,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv571: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv569, lv570, sinfo_args=(R.Object,))
            lv572: R.Object = kv_cache[21]
            lv1004: R.Tensor((1, n, 32, 80), dtype="float16") = lv564[2]
            lv1005 = R.call_tir(cls.fused_squeeze1, (lv1004,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv575: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv572, lv1005, sinfo_args=(R.Object,))
            lv576: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv571, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv577: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv575, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv578 = R.call_tir(cls.reshape3, (lv576,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv579 = R.call_tir(cls.reshape3, (lv577,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv580 = R.call_tir(cls.transpose1, (lv566,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv581 = R.call_tir(cls.transpose1, (lv578,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv582 = R.call_tir(cls.transpose1, (lv579,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1006 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv580, lv581, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv1007 = R.call_tir(cls.fused_softmax2_cast8, (lv1006,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv591 = R.call_tir(cls.matmul9, (lv1007, lv582), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv592 = R.call_tir(cls.transpose5, (lv591,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv593 = R.call_tir(cls.reshape8, (lv592,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1008: R.Tensor((2560, 2560), dtype="int8") = model_params[169]
            lv1009: R.Tensor((1, 2560), dtype="float16") = model_params[170]
            lv1010 = R.call_tir(cls.fused_decode8, (lv1008, lv1009), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_1711: R.Tensor((2560,), dtype="float16") = model_params[171]
            lv597 = R.call_tir(cls.cast5, (lv998,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_1641: R.Tensor((2560,), dtype="float32") = model_params[164]
            param_1651: R.Tensor((2560,), dtype="float32") = model_params[165]
            lv1011 = R.call_tir(cls.fused_layer_norm1_cast6, (lv597, param_1641, param_1651), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1012: R.Tensor((2560, 10240), dtype="int8") = model_params[172]
            lv1013: R.Tensor((1, 10240), dtype="float16") = model_params[173]
            param_1741: R.Tensor((10240,), dtype="float16") = model_params[174]
            lv127_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv1012, lv1013, lv1011, param_1741), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv1016: R.Tensor((10240, 2560), dtype="int8") = model_params[175]
            lv1017: R.Tensor((1, 2560), dtype="float16") = model_params[176]
            param_1771: R.Tensor((2560,), dtype="float16") = model_params[177]
            lv128_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv1016, lv1017, lv127_1, param_1771), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1020 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv593, lv1010, param_1711, lv128_1, lv998), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv611 = R.call_tir(cls.cast5, (lv1020,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_1781: R.Tensor((2560,), dtype="float32") = model_params[178]
            param_1791: R.Tensor((2560,), dtype="float32") = model_params[179]
            lv1021 = R.call_tir(cls.fused_layer_norm1_cast6, (lv611, param_1781, param_1791), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1022: R.Tensor((2560, 7680), dtype="int8") = model_params[182]
            lv1023: R.Tensor((1, 7680), dtype="float16") = model_params[183]
            param_1841: R.Tensor((7680,), dtype="float16") = model_params[184]
            lv129_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv1022, lv1023, lv1021, param_1841), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv618 = R.call_tir(cls.reshape7, (lv129_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv619 = R.call_tir(cls.split1, (lv618,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv620: R.Tensor((1, n, 32, 80), dtype="float16") = lv619[0]
            lv621 = R.call_tir(cls.rotary_embedding, (lv620, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv622: R.Tensor((1, n, 32, 80), dtype="float16") = lv619[1]
            lv623 = R.call_tir(cls.rotary_embedding, (lv622, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv624: R.Object = kv_cache[22]
            lv625 = R.call_tir(cls.squeeze1, (lv623,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv626: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv624, lv625, sinfo_args=(R.Object,))
            lv627: R.Object = kv_cache[23]
            lv1026: R.Tensor((1, n, 32, 80), dtype="float16") = lv619[2]
            lv1027 = R.call_tir(cls.fused_squeeze1, (lv1026,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv630: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv627, lv1027, sinfo_args=(R.Object,))
            lv631: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv626, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv632: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv630, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv633 = R.call_tir(cls.reshape3, (lv631,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv634 = R.call_tir(cls.reshape3, (lv632,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv635 = R.call_tir(cls.transpose1, (lv621,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv636 = R.call_tir(cls.transpose1, (lv633,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv637 = R.call_tir(cls.transpose1, (lv634,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1028 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv635, lv636, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv1029 = R.call_tir(cls.fused_softmax2_cast8, (lv1028,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv646 = R.call_tir(cls.matmul9, (lv1029, lv637), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv647 = R.call_tir(cls.transpose5, (lv646,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv648 = R.call_tir(cls.reshape8, (lv647,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1030: R.Tensor((2560, 2560), dtype="int8") = model_params[185]
            lv1031: R.Tensor((1, 2560), dtype="float16") = model_params[186]
            lv1032 = R.call_tir(cls.fused_decode8, (lv1030, lv1031), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_1871: R.Tensor((2560,), dtype="float16") = model_params[187]
            lv652 = R.call_tir(cls.cast5, (lv1020,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_1801: R.Tensor((2560,), dtype="float32") = model_params[180]
            param_1811: R.Tensor((2560,), dtype="float32") = model_params[181]
            lv1033 = R.call_tir(cls.fused_layer_norm1_cast6, (lv652, param_1801, param_1811), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1034: R.Tensor((2560, 10240), dtype="int8") = model_params[188]
            lv1035: R.Tensor((1, 10240), dtype="float16") = model_params[189]
            param_1901: R.Tensor((10240,), dtype="float16") = model_params[190]
            lv130_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv1034, lv1035, lv1033, param_1901), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv1038: R.Tensor((10240, 2560), dtype="int8") = model_params[191]
            lv1039: R.Tensor((1, 2560), dtype="float16") = model_params[192]
            param_1931: R.Tensor((2560,), dtype="float16") = model_params[193]
            lv131_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv1038, lv1039, lv130_1, param_1931), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1042 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv648, lv1032, param_1871, lv131_1, lv1020), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv666 = R.call_tir(cls.cast5, (lv1042,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_1941: R.Tensor((2560,), dtype="float32") = model_params[194]
            param_1951: R.Tensor((2560,), dtype="float32") = model_params[195]
            lv1043 = R.call_tir(cls.fused_layer_norm1_cast6, (lv666, param_1941, param_1951), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1044: R.Tensor((2560, 7680), dtype="int8") = model_params[198]
            lv1045: R.Tensor((1, 7680), dtype="float16") = model_params[199]
            param_2001: R.Tensor((7680,), dtype="float16") = model_params[200]
            lv132_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv1044, lv1045, lv1043, param_2001), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv673 = R.call_tir(cls.reshape7, (lv132_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv674 = R.call_tir(cls.split1, (lv673,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv675: R.Tensor((1, n, 32, 80), dtype="float16") = lv674[0]
            lv676 = R.call_tir(cls.rotary_embedding, (lv675, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv677: R.Tensor((1, n, 32, 80), dtype="float16") = lv674[1]
            lv678 = R.call_tir(cls.rotary_embedding, (lv677, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv679: R.Object = kv_cache[24]
            lv680 = R.call_tir(cls.squeeze1, (lv678,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv681: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv679, lv680, sinfo_args=(R.Object,))
            lv682: R.Object = kv_cache[25]
            lv1048: R.Tensor((1, n, 32, 80), dtype="float16") = lv674[2]
            lv1049 = R.call_tir(cls.fused_squeeze1, (lv1048,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv685: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv682, lv1049, sinfo_args=(R.Object,))
            lv686: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv681, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv687: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv685, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv688 = R.call_tir(cls.reshape3, (lv686,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv689 = R.call_tir(cls.reshape3, (lv687,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv690 = R.call_tir(cls.transpose1, (lv676,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv691 = R.call_tir(cls.transpose1, (lv688,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv692 = R.call_tir(cls.transpose1, (lv689,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1050 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv690, lv691, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv1051 = R.call_tir(cls.fused_softmax2_cast8, (lv1050,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv701 = R.call_tir(cls.matmul9, (lv1051, lv692), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv702 = R.call_tir(cls.transpose5, (lv701,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv703 = R.call_tir(cls.reshape8, (lv702,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1052: R.Tensor((2560, 2560), dtype="int8") = model_params[201]
            lv1053: R.Tensor((1, 2560), dtype="float16") = model_params[202]
            lv1054 = R.call_tir(cls.fused_decode8, (lv1052, lv1053), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_2031: R.Tensor((2560,), dtype="float16") = model_params[203]
            lv707 = R.call_tir(cls.cast5, (lv1042,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_1961: R.Tensor((2560,), dtype="float32") = model_params[196]
            param_1971: R.Tensor((2560,), dtype="float32") = model_params[197]
            lv1055 = R.call_tir(cls.fused_layer_norm1_cast6, (lv707, param_1961, param_1971), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1056: R.Tensor((2560, 10240), dtype="int8") = model_params[204]
            lv1057: R.Tensor((1, 10240), dtype="float16") = model_params[205]
            param_2061: R.Tensor((10240,), dtype="float16") = model_params[206]
            lv133 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv1056, lv1057, lv1055, param_2061), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv1060: R.Tensor((10240, 2560), dtype="int8") = model_params[207]
            lv1061: R.Tensor((1, 2560), dtype="float16") = model_params[208]
            param_2091: R.Tensor((2560,), dtype="float16") = model_params[209]
            lv134 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv1060, lv1061, lv133, param_2091), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1064 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv703, lv1054, param_2031, lv134, lv1042), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv721 = R.call_tir(cls.cast5, (lv1064,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_2101: R.Tensor((2560,), dtype="float32") = model_params[210]
            param_2111: R.Tensor((2560,), dtype="float32") = model_params[211]
            lv1065 = R.call_tir(cls.fused_layer_norm1_cast6, (lv721, param_2101, param_2111), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1066: R.Tensor((2560, 7680), dtype="int8") = model_params[214]
            lv1067: R.Tensor((1, 7680), dtype="float16") = model_params[215]
            param_2161: R.Tensor((7680,), dtype="float16") = model_params[216]
            lv135_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv1066, lv1067, lv1065, param_2161), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv728 = R.call_tir(cls.reshape7, (lv135_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv729 = R.call_tir(cls.split1, (lv728,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv730: R.Tensor((1, n, 32, 80), dtype="float16") = lv729[0]
            lv731 = R.call_tir(cls.rotary_embedding, (lv730, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv732: R.Tensor((1, n, 32, 80), dtype="float16") = lv729[1]
            lv733 = R.call_tir(cls.rotary_embedding, (lv732, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv734: R.Object = kv_cache[26]
            lv735 = R.call_tir(cls.squeeze1, (lv733,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv736: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv734, lv735, sinfo_args=(R.Object,))
            lv737: R.Object = kv_cache[27]
            lv1070: R.Tensor((1, n, 32, 80), dtype="float16") = lv729[2]
            lv1071 = R.call_tir(cls.fused_squeeze1, (lv1070,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv740: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv737, lv1071, sinfo_args=(R.Object,))
            lv741: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv736, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv742: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv740, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv743 = R.call_tir(cls.reshape3, (lv741,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv744 = R.call_tir(cls.reshape3, (lv742,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv745 = R.call_tir(cls.transpose1, (lv731,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv746 = R.call_tir(cls.transpose1, (lv743,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv747 = R.call_tir(cls.transpose1, (lv744,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1072 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv745, lv746, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv1073 = R.call_tir(cls.fused_softmax2_cast8, (lv1072,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv756 = R.call_tir(cls.matmul9, (lv1073, lv747), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv757 = R.call_tir(cls.transpose5, (lv756,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv758 = R.call_tir(cls.reshape8, (lv757,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1074: R.Tensor((2560, 2560), dtype="int8") = model_params[217]
            lv1075: R.Tensor((1, 2560), dtype="float16") = model_params[218]
            lv1076 = R.call_tir(cls.fused_decode8, (lv1074, lv1075), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_2191: R.Tensor((2560,), dtype="float16") = model_params[219]
            lv762 = R.call_tir(cls.cast5, (lv1064,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_2121: R.Tensor((2560,), dtype="float32") = model_params[212]
            param_2131: R.Tensor((2560,), dtype="float32") = model_params[213]
            lv1077 = R.call_tir(cls.fused_layer_norm1_cast6, (lv762, param_2121, param_2131), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1078: R.Tensor((2560, 10240), dtype="int8") = model_params[220]
            lv1079: R.Tensor((1, 10240), dtype="float16") = model_params[221]
            param_2221: R.Tensor((10240,), dtype="float16") = model_params[222]
            lv136_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv1078, lv1079, lv1077, param_2221), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv1082: R.Tensor((10240, 2560), dtype="int8") = model_params[223]
            lv1083: R.Tensor((1, 2560), dtype="float16") = model_params[224]
            param_2251: R.Tensor((2560,), dtype="float16") = model_params[225]
            lv137_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv1082, lv1083, lv136_1, param_2251), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1086 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv758, lv1076, param_2191, lv137_1, lv1064), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv776_1 = R.call_tir(cls.cast5, (lv1086,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_2261: R.Tensor((2560,), dtype="float32") = model_params[226]
            param_2271: R.Tensor((2560,), dtype="float32") = model_params[227]
            lv1087 = R.call_tir(cls.fused_layer_norm1_cast6, (lv776_1, param_2261, param_2271), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1088: R.Tensor((2560, 7680), dtype="int8") = model_params[230]
            lv1089: R.Tensor((1, 7680), dtype="float16") = model_params[231]
            param_2321: R.Tensor((7680,), dtype="float16") = model_params[232]
            lv138_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv1088, lv1089, lv1087, param_2321), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv783 = R.call_tir(cls.reshape7, (lv138_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv784_1 = R.call_tir(cls.split1, (lv783,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv785_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv784_1[0]
            lv786_1 = R.call_tir(cls.rotary_embedding, (lv785_1, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv787_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv784_1[1]
            lv788_1 = R.call_tir(cls.rotary_embedding, (lv787_1, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv789_1: R.Object = kv_cache[28]
            lv790_1 = R.call_tir(cls.squeeze1, (lv788_1,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv791_1: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv789_1, lv790_1, sinfo_args=(R.Object,))
            lv792_1: R.Object = kv_cache[29]
            lv1092: R.Tensor((1, n, 32, 80), dtype="float16") = lv784_1[2]
            lv1093 = R.call_tir(cls.fused_squeeze1, (lv1092,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv795: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv792_1, lv1093, sinfo_args=(R.Object,))
            lv796_1: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv791_1, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv797_1: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv795, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv798 = R.call_tir(cls.reshape3, (lv796_1,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv799 = R.call_tir(cls.reshape3, (lv797_1,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv800_1 = R.call_tir(cls.transpose1, (lv786_1,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv801_1 = R.call_tir(cls.transpose1, (lv798,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv802_1 = R.call_tir(cls.transpose1, (lv799,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1094 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv800_1, lv801_1, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv1095 = R.call_tir(cls.fused_softmax2_cast8, (lv1094,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv811_1 = R.call_tir(cls.matmul9, (lv1095, lv802_1), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv812_1 = R.call_tir(cls.transpose5, (lv811_1,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv813_1 = R.call_tir(cls.reshape8, (lv812_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1096: R.Tensor((2560, 2560), dtype="int8") = model_params[233]
            lv1097: R.Tensor((1, 2560), dtype="float16") = model_params[234]
            lv1098 = R.call_tir(cls.fused_decode8, (lv1096, lv1097), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_2351: R.Tensor((2560,), dtype="float16") = model_params[235]
            lv817 = R.call_tir(cls.cast5, (lv1086,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_2281: R.Tensor((2560,), dtype="float32") = model_params[228]
            param_2291: R.Tensor((2560,), dtype="float32") = model_params[229]
            lv1099 = R.call_tir(cls.fused_layer_norm1_cast6, (lv817, param_2281, param_2291), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1100: R.Tensor((2560, 10240), dtype="int8") = model_params[236]
            lv1101: R.Tensor((1, 10240), dtype="float16") = model_params[237]
            param_2381: R.Tensor((10240,), dtype="float16") = model_params[238]
            lv139_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv1100, lv1101, lv1099, param_2381), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv1104: R.Tensor((10240, 2560), dtype="int8") = model_params[239]
            lv1105: R.Tensor((1, 2560), dtype="float16") = model_params[240]
            param_2411: R.Tensor((2560,), dtype="float16") = model_params[241]
            lv140_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv1104, lv1105, lv139_1, param_2411), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1108 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv813_1, lv1098, param_2351, lv140_1, lv1086), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv831_1 = R.call_tir(cls.cast5, (lv1108,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_2421: R.Tensor((2560,), dtype="float32") = model_params[242]
            param_2431: R.Tensor((2560,), dtype="float32") = model_params[243]
            lv1109 = R.call_tir(cls.fused_layer_norm1_cast6, (lv831_1, param_2421, param_2431), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1110: R.Tensor((2560, 7680), dtype="int8") = model_params[246]
            lv1111: R.Tensor((1, 7680), dtype="float16") = model_params[247]
            param_2481: R.Tensor((7680,), dtype="float16") = model_params[248]
            lv141_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv1110, lv1111, lv1109, param_2481), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv838 = R.call_tir(cls.reshape7, (lv141_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv839 = R.call_tir(cls.split1, (lv838,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv840_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv839[0]
            lv841_1 = R.call_tir(cls.rotary_embedding, (lv840_1, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv842: R.Tensor((1, n, 32, 80), dtype="float16") = lv839[1]
            lv843 = R.call_tir(cls.rotary_embedding, (lv842, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv844_1: R.Object = kv_cache[30]
            lv845_1 = R.call_tir(cls.squeeze1, (lv843,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv846_1: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv844_1, lv845_1, sinfo_args=(R.Object,))
            lv847_1: R.Object = kv_cache[31]
            lv1114: R.Tensor((1, n, 32, 80), dtype="float16") = lv839[2]
            lv1115 = R.call_tir(cls.fused_squeeze1, (lv1114,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv850_1: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv847_1, lv1115, sinfo_args=(R.Object,))
            lv851_1: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv846_1, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv852_1: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv850_1, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv853_1 = R.call_tir(cls.reshape3, (lv851_1,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv854_1 = R.call_tir(cls.reshape3, (lv852_1,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv855_1 = R.call_tir(cls.transpose1, (lv841_1,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv856_1 = R.call_tir(cls.transpose1, (lv853_1,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv857_1 = R.call_tir(cls.transpose1, (lv854_1,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1116 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv855_1, lv856_1, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv1117 = R.call_tir(cls.fused_softmax2_cast8, (lv1116,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv866_1 = R.call_tir(cls.matmul9, (lv1117, lv857_1), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv867_1 = R.call_tir(cls.transpose5, (lv866_1,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv868_1 = R.call_tir(cls.reshape8, (lv867_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1118: R.Tensor((2560, 2560), dtype="int8") = model_params[249]
            lv1119: R.Tensor((1, 2560), dtype="float16") = model_params[250]
            lv1120 = R.call_tir(cls.fused_decode8, (lv1118, lv1119), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_2511: R.Tensor((2560,), dtype="float16") = model_params[251]
            lv872_1 = R.call_tir(cls.cast5, (lv1108,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_2441: R.Tensor((2560,), dtype="float32") = model_params[244]
            param_2451: R.Tensor((2560,), dtype="float32") = model_params[245]
            lv1121 = R.call_tir(cls.fused_layer_norm1_cast6, (lv872_1, param_2441, param_2451), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1122: R.Tensor((2560, 10240), dtype="int8") = model_params[252]
            lv1123: R.Tensor((1, 10240), dtype="float16") = model_params[253]
            param_2541: R.Tensor((10240,), dtype="float16") = model_params[254]
            lv142_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv1122, lv1123, lv1121, param_2541), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv1126: R.Tensor((10240, 2560), dtype="int8") = model_params[255]
            lv1127: R.Tensor((1, 2560), dtype="float16") = model_params[256]
            param_2571: R.Tensor((2560,), dtype="float16") = model_params[257]
            lv143 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv1126, lv1127, lv142_1, param_2571), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1130 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv868_1, lv1120, param_2511, lv143, lv1108), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv886 = R.call_tir(cls.cast5, (lv1130,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_2581: R.Tensor((2560,), dtype="float32") = model_params[258]
            param_2591: R.Tensor((2560,), dtype="float32") = model_params[259]
            lv1131 = R.call_tir(cls.fused_layer_norm1_cast6, (lv886, param_2581, param_2591), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1132: R.Tensor((2560, 7680), dtype="int8") = model_params[262]
            lv1133: R.Tensor((1, 7680), dtype="float16") = model_params[263]
            param_2641: R.Tensor((7680,), dtype="float16") = model_params[264]
            lv144 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv1132, lv1133, lv1131, param_2641), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv893 = R.call_tir(cls.reshape7, (lv144,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv894_1 = R.call_tir(cls.split1, (lv893,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv895_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv894_1[0]
            lv896_1 = R.call_tir(cls.rotary_embedding, (lv895_1, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv897_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv894_1[1]
            lv898_1 = R.call_tir(cls.rotary_embedding, (lv897_1, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv899_1: R.Object = kv_cache[32]
            lv900_1 = R.call_tir(cls.squeeze1, (lv898_1,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv901_1: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv899_1, lv900_1, sinfo_args=(R.Object,))
            lv902_1: R.Object = kv_cache[33]
            lv1136: R.Tensor((1, n, 32, 80), dtype="float16") = lv894_1[2]
            lv1137 = R.call_tir(cls.fused_squeeze1, (lv1136,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv905: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv902_1, lv1137, sinfo_args=(R.Object,))
            lv906_1: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv901_1, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv907_1: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv905, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv908 = R.call_tir(cls.reshape3, (lv906_1,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv909 = R.call_tir(cls.reshape3, (lv907_1,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv910_1 = R.call_tir(cls.transpose1, (lv896_1,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv911_1 = R.call_tir(cls.transpose1, (lv908,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv912_1 = R.call_tir(cls.transpose1, (lv909,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1138 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv910_1, lv911_1, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv1139 = R.call_tir(cls.fused_softmax2_cast8, (lv1138,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv921_1 = R.call_tir(cls.matmul9, (lv1139, lv912_1), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv922_1 = R.call_tir(cls.transpose5, (lv921_1,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv923_1 = R.call_tir(cls.reshape8, (lv922_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1140: R.Tensor((2560, 2560), dtype="int8") = model_params[265]
            lv1141: R.Tensor((1, 2560), dtype="float16") = model_params[266]
            lv1142 = R.call_tir(cls.fused_decode8, (lv1140, lv1141), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_2671: R.Tensor((2560,), dtype="float16") = model_params[267]
            lv927 = R.call_tir(cls.cast5, (lv1130,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_2601: R.Tensor((2560,), dtype="float32") = model_params[260]
            param_2611: R.Tensor((2560,), dtype="float32") = model_params[261]
            lv1143 = R.call_tir(cls.fused_layer_norm1_cast6, (lv927, param_2601, param_2611), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1144: R.Tensor((2560, 10240), dtype="int8") = model_params[268]
            lv1145: R.Tensor((1, 10240), dtype="float16") = model_params[269]
            param_2701: R.Tensor((10240,), dtype="float16") = model_params[270]
            lv145 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv1144, lv1145, lv1143, param_2701), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv1148: R.Tensor((10240, 2560), dtype="int8") = model_params[271]
            lv1149: R.Tensor((1, 2560), dtype="float16") = model_params[272]
            param_2731: R.Tensor((2560,), dtype="float16") = model_params[273]
            lv146 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv1148, lv1149, lv145, param_2731), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1152 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv923_1, lv1142, param_2671, lv146, lv1130), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv941_1 = R.call_tir(cls.cast5, (lv1152,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_2741: R.Tensor((2560,), dtype="float32") = model_params[274]
            param_2751: R.Tensor((2560,), dtype="float32") = model_params[275]
            lv1153 = R.call_tir(cls.fused_layer_norm1_cast6, (lv941_1, param_2741, param_2751), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1154: R.Tensor((2560, 7680), dtype="int8") = model_params[278]
            lv1155: R.Tensor((1, 7680), dtype="float16") = model_params[279]
            param_2801: R.Tensor((7680,), dtype="float16") = model_params[280]
            lv147 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv1154, lv1155, lv1153, param_2801), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv948 = R.call_tir(cls.reshape7, (lv147,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv949 = R.call_tir(cls.split1, (lv948,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv950_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv949[0]
            lv951_1 = R.call_tir(cls.rotary_embedding, (lv950_1, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv952: R.Tensor((1, n, 32, 80), dtype="float16") = lv949[1]
            lv953 = R.call_tir(cls.rotary_embedding, (lv952, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv954_1: R.Object = kv_cache[34]
            lv955_1 = R.call_tir(cls.squeeze1, (lv953,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv956_1: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv954_1, lv955_1, sinfo_args=(R.Object,))
            lv957_1: R.Object = kv_cache[35]
            lv1158: R.Tensor((1, n, 32, 80), dtype="float16") = lv949[2]
            lv1159 = R.call_tir(cls.fused_squeeze1, (lv1158,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv960_1: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv957_1, lv1159, sinfo_args=(R.Object,))
            lv961_1: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv956_1, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv962_1: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv960_1, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv963_1 = R.call_tir(cls.reshape3, (lv961_1,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv964_1 = R.call_tir(cls.reshape3, (lv962_1,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv965_1 = R.call_tir(cls.transpose1, (lv951_1,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv966_1 = R.call_tir(cls.transpose1, (lv963_1,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv967_1 = R.call_tir(cls.transpose1, (lv964_1,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1160 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv965_1, lv966_1, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv1161 = R.call_tir(cls.fused_softmax2_cast8, (lv1160,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv976_1 = R.call_tir(cls.matmul9, (lv1161, lv967_1), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv977_1 = R.call_tir(cls.transpose5, (lv976_1,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv978_1 = R.call_tir(cls.reshape8, (lv977_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1162: R.Tensor((2560, 2560), dtype="int8") = model_params[281]
            lv1163: R.Tensor((1, 2560), dtype="float16") = model_params[282]
            lv1164 = R.call_tir(cls.fused_decode8, (lv1162, lv1163), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_2831: R.Tensor((2560,), dtype="float16") = model_params[283]
            lv982_1 = R.call_tir(cls.cast5, (lv1152,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_2761: R.Tensor((2560,), dtype="float32") = model_params[276]
            param_2771: R.Tensor((2560,), dtype="float32") = model_params[277]
            lv1165 = R.call_tir(cls.fused_layer_norm1_cast6, (lv982_1, param_2761, param_2771), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1166: R.Tensor((2560, 10240), dtype="int8") = model_params[284]
            lv1167: R.Tensor((1, 10240), dtype="float16") = model_params[285]
            param_2861: R.Tensor((10240,), dtype="float16") = model_params[286]
            lv148 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv1166, lv1167, lv1165, param_2861), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv1170: R.Tensor((10240, 2560), dtype="int8") = model_params[287]
            lv1171: R.Tensor((1, 2560), dtype="float16") = model_params[288]
            param_2891: R.Tensor((2560,), dtype="float16") = model_params[289]
            lv149 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv1170, lv1171, lv148, param_2891), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1174 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv978_1, lv1164, param_2831, lv149, lv1152), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv996 = R.call_tir(cls.cast5, (lv1174,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_2901: R.Tensor((2560,), dtype="float32") = model_params[290]
            param_2911: R.Tensor((2560,), dtype="float32") = model_params[291]
            lv1175 = R.call_tir(cls.fused_layer_norm1_cast6, (lv996, param_2901, param_2911), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1176: R.Tensor((2560, 7680), dtype="int8") = model_params[294]
            lv1177: R.Tensor((1, 7680), dtype="float16") = model_params[295]
            param_2961: R.Tensor((7680,), dtype="float16") = model_params[296]
            lv150 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv1176, lv1177, lv1175, param_2961), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1003 = R.call_tir(cls.reshape7, (lv150,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1004_1 = R.call_tir(cls.split1, (lv1003,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1005_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv1004_1[0]
            lv1006_1 = R.call_tir(cls.rotary_embedding, (lv1005_1, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1007_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv1004_1[1]
            lv1008_1 = R.call_tir(cls.rotary_embedding, (lv1007_1, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1009_1: R.Object = kv_cache[36]
            lv1010_1 = R.call_tir(cls.squeeze1, (lv1008_1,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1011_1: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1009_1, lv1010_1, sinfo_args=(R.Object,))
            lv1012_1: R.Object = kv_cache[37]
            lv1180: R.Tensor((1, n, 32, 80), dtype="float16") = lv1004_1[2]
            lv1181 = R.call_tir(cls.fused_squeeze1, (lv1180,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1015: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1012_1, lv1181, sinfo_args=(R.Object,))
            lv1016_1: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1011_1, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1017_1: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1015, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1018 = R.call_tir(cls.reshape3, (lv1016_1,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1019 = R.call_tir(cls.reshape3, (lv1017_1,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1020_1 = R.call_tir(cls.transpose1, (lv1006_1,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1021_1 = R.call_tir(cls.transpose1, (lv1018,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1022_1 = R.call_tir(cls.transpose1, (lv1019,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1182 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv1020_1, lv1021_1, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv1183 = R.call_tir(cls.fused_softmax2_cast8, (lv1182,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1031_1 = R.call_tir(cls.matmul9, (lv1183, lv1022_1), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1032_1 = R.call_tir(cls.transpose5, (lv1031_1,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1033_1 = R.call_tir(cls.reshape8, (lv1032_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1184: R.Tensor((2560, 2560), dtype="int8") = model_params[297]
            lv1185: R.Tensor((1, 2560), dtype="float16") = model_params[298]
            lv1186 = R.call_tir(cls.fused_decode8, (lv1184, lv1185), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_2991: R.Tensor((2560,), dtype="float16") = model_params[299]
            lv1037 = R.call_tir(cls.cast5, (lv1174,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_2921: R.Tensor((2560,), dtype="float32") = model_params[292]
            param_2931: R.Tensor((2560,), dtype="float32") = model_params[293]
            lv1187 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1037, param_2921, param_2931), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1188: R.Tensor((2560, 10240), dtype="int8") = model_params[300]
            lv1189: R.Tensor((1, 10240), dtype="float16") = model_params[301]
            param_3021: R.Tensor((10240,), dtype="float16") = model_params[302]
            lv151_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv1188, lv1189, lv1187, param_3021), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv1192: R.Tensor((10240, 2560), dtype="int8") = model_params[303]
            lv1193: R.Tensor((1, 2560), dtype="float16") = model_params[304]
            param_3051: R.Tensor((2560,), dtype="float16") = model_params[305]
            lv152_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv1192, lv1193, lv151_1, param_3051), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1196 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv1033_1, lv1186, param_2991, lv152_1, lv1174), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1051_1 = R.call_tir(cls.cast5, (lv1196,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_3061: R.Tensor((2560,), dtype="float32") = model_params[306]
            param_3071: R.Tensor((2560,), dtype="float32") = model_params[307]
            lv1197 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1051_1, param_3061, param_3071), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1198: R.Tensor((2560, 7680), dtype="int8") = model_params[310]
            lv1199: R.Tensor((1, 7680), dtype="float16") = model_params[311]
            param_3121: R.Tensor((7680,), dtype="float16") = model_params[312]
            lv153_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv1198, lv1199, lv1197, param_3121), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1058 = R.call_tir(cls.reshape7, (lv153_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1059 = R.call_tir(cls.split1, (lv1058,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1060_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv1059[0]
            lv1061_1 = R.call_tir(cls.rotary_embedding, (lv1060_1, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1062: R.Tensor((1, n, 32, 80), dtype="float16") = lv1059[1]
            lv1063 = R.call_tir(cls.rotary_embedding, (lv1062, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1064_1: R.Object = kv_cache[38]
            lv1065_1 = R.call_tir(cls.squeeze1, (lv1063,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1066_1: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1064_1, lv1065_1, sinfo_args=(R.Object,))
            lv1067_1: R.Object = kv_cache[39]
            lv1202: R.Tensor((1, n, 32, 80), dtype="float16") = lv1059[2]
            lv1203 = R.call_tir(cls.fused_squeeze1, (lv1202,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1070_1: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1067_1, lv1203, sinfo_args=(R.Object,))
            lv1071_1: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1066_1, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1072_1: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1070_1, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1073_1 = R.call_tir(cls.reshape3, (lv1071_1,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1074_1 = R.call_tir(cls.reshape3, (lv1072_1,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1075_1 = R.call_tir(cls.transpose1, (lv1061_1,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1076_1 = R.call_tir(cls.transpose1, (lv1073_1,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1077_1 = R.call_tir(cls.transpose1, (lv1074_1,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1204 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv1075_1, lv1076_1, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv1205 = R.call_tir(cls.fused_softmax2_cast8, (lv1204,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1086_1 = R.call_tir(cls.matmul9, (lv1205, lv1077_1), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1087_1 = R.call_tir(cls.transpose5, (lv1086_1,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1088_1 = R.call_tir(cls.reshape8, (lv1087_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1206: R.Tensor((2560, 2560), dtype="int8") = model_params[313]
            lv1207: R.Tensor((1, 2560), dtype="float16") = model_params[314]
            lv1208 = R.call_tir(cls.fused_decode8, (lv1206, lv1207), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_3151: R.Tensor((2560,), dtype="float16") = model_params[315]
            lv1092_1 = R.call_tir(cls.cast5, (lv1196,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_3081: R.Tensor((2560,), dtype="float32") = model_params[308]
            param_3091: R.Tensor((2560,), dtype="float32") = model_params[309]
            lv1209 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1092_1, param_3081, param_3091), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1210: R.Tensor((2560, 10240), dtype="int8") = model_params[316]
            lv1211: R.Tensor((1, 10240), dtype="float16") = model_params[317]
            param_3181: R.Tensor((10240,), dtype="float16") = model_params[318]
            lv154 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv1210, lv1211, lv1209, param_3181), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv1214: R.Tensor((10240, 2560), dtype="int8") = model_params[319]
            lv1215: R.Tensor((1, 2560), dtype="float16") = model_params[320]
            param_3211: R.Tensor((2560,), dtype="float16") = model_params[321]
            lv155 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv1214, lv1215, lv154, param_3211), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1218 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv1088_1, lv1208, param_3151, lv155, lv1196), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1106 = R.call_tir(cls.cast5, (lv1218,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_3221: R.Tensor((2560,), dtype="float32") = model_params[322]
            param_3231: R.Tensor((2560,), dtype="float32") = model_params[323]
            lv1219 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1106, param_3221, param_3231), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1220: R.Tensor((2560, 7680), dtype="int8") = model_params[326]
            lv1221: R.Tensor((1, 7680), dtype="float16") = model_params[327]
            param_3281: R.Tensor((7680,), dtype="float16") = model_params[328]
            lv156 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv1220, lv1221, lv1219, param_3281), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1113 = R.call_tir(cls.reshape7, (lv156,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1114_1 = R.call_tir(cls.split1, (lv1113,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1115_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv1114_1[0]
            lv1116_1 = R.call_tir(cls.rotary_embedding, (lv1115_1, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1117_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv1114_1[1]
            lv1118_1 = R.call_tir(cls.rotary_embedding, (lv1117_1, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1119_1: R.Object = kv_cache[40]
            lv1120_1 = R.call_tir(cls.squeeze1, (lv1118_1,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1121_1: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1119_1, lv1120_1, sinfo_args=(R.Object,))
            lv1122_1: R.Object = kv_cache[41]
            lv1224: R.Tensor((1, n, 32, 80), dtype="float16") = lv1114_1[2]
            lv1225 = R.call_tir(cls.fused_squeeze1, (lv1224,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1125: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1122_1, lv1225, sinfo_args=(R.Object,))
            lv1126_1: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1121_1, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1127_1: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1125, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1128 = R.call_tir(cls.reshape3, (lv1126_1,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1129 = R.call_tir(cls.reshape3, (lv1127_1,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1130_1 = R.call_tir(cls.transpose1, (lv1116_1,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1131_1 = R.call_tir(cls.transpose1, (lv1128,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1132_1 = R.call_tir(cls.transpose1, (lv1129,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1226 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv1130_1, lv1131_1, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv1227 = R.call_tir(cls.fused_softmax2_cast8, (lv1226,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1141_1 = R.call_tir(cls.matmul9, (lv1227, lv1132_1), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1142_1 = R.call_tir(cls.transpose5, (lv1141_1,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1143_1 = R.call_tir(cls.reshape8, (lv1142_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1228: R.Tensor((2560, 2560), dtype="int8") = model_params[329]
            lv1229: R.Tensor((1, 2560), dtype="float16") = model_params[330]
            lv1230 = R.call_tir(cls.fused_decode8, (lv1228, lv1229), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_3311: R.Tensor((2560,), dtype="float16") = model_params[331]
            lv1147 = R.call_tir(cls.cast5, (lv1218,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_3241: R.Tensor((2560,), dtype="float32") = model_params[324]
            param_3251: R.Tensor((2560,), dtype="float32") = model_params[325]
            lv1231 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1147, param_3241, param_3251), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1232: R.Tensor((2560, 10240), dtype="int8") = model_params[332]
            lv1233: R.Tensor((1, 10240), dtype="float16") = model_params[333]
            param_3341: R.Tensor((10240,), dtype="float16") = model_params[334]
            lv157_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv1232, lv1233, lv1231, param_3341), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv1236: R.Tensor((10240, 2560), dtype="int8") = model_params[335]
            lv1237: R.Tensor((1, 2560), dtype="float16") = model_params[336]
            param_3371: R.Tensor((2560,), dtype="float16") = model_params[337]
            lv158 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv1236, lv1237, lv157_1, param_3371), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1240 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv1143_1, lv1230, param_3311, lv158, lv1218), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1161_1 = R.call_tir(cls.cast5, (lv1240,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_3381: R.Tensor((2560,), dtype="float32") = model_params[338]
            param_3391: R.Tensor((2560,), dtype="float32") = model_params[339]
            lv1241 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1161_1, param_3381, param_3391), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1242: R.Tensor((2560, 7680), dtype="int8") = model_params[342]
            lv1243: R.Tensor((1, 7680), dtype="float16") = model_params[343]
            param_3441: R.Tensor((7680,), dtype="float16") = model_params[344]
            lv159 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv1242, lv1243, lv1241, param_3441), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1168 = R.call_tir(cls.reshape7, (lv159,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1169 = R.call_tir(cls.split1, (lv1168,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1170_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv1169[0]
            lv1171_1 = R.call_tir(cls.rotary_embedding, (lv1170_1, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1172: R.Tensor((1, n, 32, 80), dtype="float16") = lv1169[1]
            lv1173 = R.call_tir(cls.rotary_embedding, (lv1172, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1174_1: R.Object = kv_cache[42]
            lv1175_1 = R.call_tir(cls.squeeze1, (lv1173,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1176_1: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1174_1, lv1175_1, sinfo_args=(R.Object,))
            lv1177_1: R.Object = kv_cache[43]
            lv1246: R.Tensor((1, n, 32, 80), dtype="float16") = lv1169[2]
            lv1247 = R.call_tir(cls.fused_squeeze1, (lv1246,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1180_1: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1177_1, lv1247, sinfo_args=(R.Object,))
            lv1181_1: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1176_1, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1182_1: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1180_1, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1183_1 = R.call_tir(cls.reshape3, (lv1181_1,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1184_1 = R.call_tir(cls.reshape3, (lv1182_1,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1185_1 = R.call_tir(cls.transpose1, (lv1171_1,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1186_1 = R.call_tir(cls.transpose1, (lv1183_1,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1187_1 = R.call_tir(cls.transpose1, (lv1184_1,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1248 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv1185_1, lv1186_1, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv1249 = R.call_tir(cls.fused_softmax2_cast8, (lv1248,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1196_1 = R.call_tir(cls.matmul9, (lv1249, lv1187_1), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1197_1 = R.call_tir(cls.transpose5, (lv1196_1,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1198_1 = R.call_tir(cls.reshape8, (lv1197_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1250: R.Tensor((2560, 2560), dtype="int8") = model_params[345]
            lv1251: R.Tensor((1, 2560), dtype="float16") = model_params[346]
            lv1252 = R.call_tir(cls.fused_decode8, (lv1250, lv1251), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_3471: R.Tensor((2560,), dtype="float16") = model_params[347]
            lv1202_1 = R.call_tir(cls.cast5, (lv1240,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_3401: R.Tensor((2560,), dtype="float32") = model_params[340]
            param_3411: R.Tensor((2560,), dtype="float32") = model_params[341]
            lv1253 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1202_1, param_3401, param_3411), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1254: R.Tensor((2560, 10240), dtype="int8") = model_params[348]
            lv1255: R.Tensor((1, 10240), dtype="float16") = model_params[349]
            param_3501: R.Tensor((10240,), dtype="float16") = model_params[350]
            lv160 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv1254, lv1255, lv1253, param_3501), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv1258: R.Tensor((10240, 2560), dtype="int8") = model_params[351]
            lv1259: R.Tensor((1, 2560), dtype="float16") = model_params[352]
            param_3531: R.Tensor((2560,), dtype="float16") = model_params[353]
            lv161 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv1258, lv1259, lv160, param_3531), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1262 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv1198_1, lv1252, param_3471, lv161, lv1240), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1216 = R.call_tir(cls.cast5, (lv1262,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_3541: R.Tensor((2560,), dtype="float32") = model_params[354]
            param_3551: R.Tensor((2560,), dtype="float32") = model_params[355]
            lv1263 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1216, param_3541, param_3551), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1264: R.Tensor((2560, 7680), dtype="int8") = model_params[358]
            lv1265: R.Tensor((1, 7680), dtype="float16") = model_params[359]
            param_3601: R.Tensor((7680,), dtype="float16") = model_params[360]
            lv162 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv1264, lv1265, lv1263, param_3601), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1223 = R.call_tir(cls.reshape7, (lv162,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1224_1 = R.call_tir(cls.split1, (lv1223,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1225_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv1224_1[0]
            lv1226_1 = R.call_tir(cls.rotary_embedding, (lv1225_1, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1227_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv1224_1[1]
            lv1228_1 = R.call_tir(cls.rotary_embedding, (lv1227_1, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1229_1: R.Object = kv_cache[44]
            lv1230_1 = R.call_tir(cls.squeeze1, (lv1228_1,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1231_1: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1229_1, lv1230_1, sinfo_args=(R.Object,))
            lv1232_1: R.Object = kv_cache[45]
            lv1268: R.Tensor((1, n, 32, 80), dtype="float16") = lv1224_1[2]
            lv1269 = R.call_tir(cls.fused_squeeze1, (lv1268,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1235: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1232_1, lv1269, sinfo_args=(R.Object,))
            lv1236_1: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1231_1, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1237_1: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1235, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1238 = R.call_tir(cls.reshape3, (lv1236_1,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1239 = R.call_tir(cls.reshape3, (lv1237_1,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1240_1 = R.call_tir(cls.transpose1, (lv1226_1,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1241_1 = R.call_tir(cls.transpose1, (lv1238,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1242_1 = R.call_tir(cls.transpose1, (lv1239,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1270 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv1240_1, lv1241_1, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv1271 = R.call_tir(cls.fused_softmax2_cast8, (lv1270,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1251_1 = R.call_tir(cls.matmul9, (lv1271, lv1242_1), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1252_1 = R.call_tir(cls.transpose5, (lv1251_1,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1253_1 = R.call_tir(cls.reshape8, (lv1252_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1272: R.Tensor((2560, 2560), dtype="int8") = model_params[361]
            lv1273: R.Tensor((1, 2560), dtype="float16") = model_params[362]
            lv1274 = R.call_tir(cls.fused_decode8, (lv1272, lv1273), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_3631: R.Tensor((2560,), dtype="float16") = model_params[363]
            lv1257 = R.call_tir(cls.cast5, (lv1262,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_3561: R.Tensor((2560,), dtype="float32") = model_params[356]
            param_3571: R.Tensor((2560,), dtype="float32") = model_params[357]
            lv1275 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1257, param_3561, param_3571), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1276: R.Tensor((2560, 10240), dtype="int8") = model_params[364]
            lv1277: R.Tensor((1, 10240), dtype="float16") = model_params[365]
            param_3661: R.Tensor((10240,), dtype="float16") = model_params[366]
            lv163 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv1276, lv1277, lv1275, param_3661), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv1280: R.Tensor((10240, 2560), dtype="int8") = model_params[367]
            lv1281: R.Tensor((1, 2560), dtype="float16") = model_params[368]
            param_3691: R.Tensor((2560,), dtype="float16") = model_params[369]
            lv164 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv1280, lv1281, lv163, param_3691), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1284 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv1253_1, lv1274, param_3631, lv164, lv1262), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1271_1 = R.call_tir(cls.cast5, (lv1284,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_3701: R.Tensor((2560,), dtype="float32") = model_params[370]
            param_3711: R.Tensor((2560,), dtype="float32") = model_params[371]
            lv1285 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1271_1, param_3701, param_3711), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1286: R.Tensor((2560, 7680), dtype="int8") = model_params[374]
            lv1287: R.Tensor((1, 7680), dtype="float16") = model_params[375]
            param_3761: R.Tensor((7680,), dtype="float16") = model_params[376]
            lv165 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv1286, lv1287, lv1285, param_3761), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1278 = R.call_tir(cls.reshape7, (lv165,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1279 = R.call_tir(cls.split1, (lv1278,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1280_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv1279[0]
            lv1281_1 = R.call_tir(cls.rotary_embedding, (lv1280_1, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1282: R.Tensor((1, n, 32, 80), dtype="float16") = lv1279[1]
            lv1283 = R.call_tir(cls.rotary_embedding, (lv1282, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1284_1: R.Object = kv_cache[46]
            lv1285_1 = R.call_tir(cls.squeeze1, (lv1283,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1286_1: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1284_1, lv1285_1, sinfo_args=(R.Object,))
            lv1287_1: R.Object = kv_cache[47]
            lv1290: R.Tensor((1, n, 32, 80), dtype="float16") = lv1279[2]
            lv1291 = R.call_tir(cls.fused_squeeze1, (lv1290,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1290_1: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1287_1, lv1291, sinfo_args=(R.Object,))
            lv1291_1: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1286_1, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1292: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1290_1, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1293 = R.call_tir(cls.reshape3, (lv1291_1,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1294 = R.call_tir(cls.reshape3, (lv1292,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1295 = R.call_tir(cls.transpose1, (lv1281_1,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1296 = R.call_tir(cls.transpose1, (lv1293,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1297 = R.call_tir(cls.transpose1, (lv1294,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1292_1 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv1295, lv1296, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv1293_1 = R.call_tir(cls.fused_softmax2_cast8, (lv1292_1,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1306 = R.call_tir(cls.matmul9, (lv1293_1, lv1297), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1307 = R.call_tir(cls.transpose5, (lv1306,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1308 = R.call_tir(cls.reshape8, (lv1307,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1294_1: R.Tensor((2560, 2560), dtype="int8") = model_params[377]
            lv1295_1: R.Tensor((1, 2560), dtype="float16") = model_params[378]
            lv1296_1 = R.call_tir(cls.fused_decode8, (lv1294_1, lv1295_1), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_3791: R.Tensor((2560,), dtype="float16") = model_params[379]
            lv1312 = R.call_tir(cls.cast5, (lv1284,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_3721: R.Tensor((2560,), dtype="float32") = model_params[372]
            param_3731: R.Tensor((2560,), dtype="float32") = model_params[373]
            lv1297_1 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1312, param_3721, param_3731), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1298: R.Tensor((2560, 10240), dtype="int8") = model_params[380]
            lv1299: R.Tensor((1, 10240), dtype="float16") = model_params[381]
            param_3821: R.Tensor((10240,), dtype="float16") = model_params[382]
            lv166 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv1298, lv1299, lv1297_1, param_3821), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv1302: R.Tensor((10240, 2560), dtype="int8") = model_params[383]
            lv1303: R.Tensor((1, 2560), dtype="float16") = model_params[384]
            param_3851: R.Tensor((2560,), dtype="float16") = model_params[385]
            lv167 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv1302, lv1303, lv166, param_3851), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1306_1 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv1308, lv1296_1, param_3791, lv167, lv1284), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1326 = R.call_tir(cls.cast5, (lv1306_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_3861: R.Tensor((2560,), dtype="float32") = model_params[386]
            param_3871: R.Tensor((2560,), dtype="float32") = model_params[387]
            lv1307_1 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1326, param_3861, param_3871), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1308_1: R.Tensor((2560, 7680), dtype="int8") = model_params[390]
            lv1309: R.Tensor((1, 7680), dtype="float16") = model_params[391]
            param_3921: R.Tensor((7680,), dtype="float16") = model_params[392]
            lv168 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv1308_1, lv1309, lv1307_1, param_3921), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1333 = R.call_tir(cls.reshape7, (lv168,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1334 = R.call_tir(cls.split1, (lv1333,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1335: R.Tensor((1, n, 32, 80), dtype="float16") = lv1334[0]
            lv1336 = R.call_tir(cls.rotary_embedding, (lv1335, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1337: R.Tensor((1, n, 32, 80), dtype="float16") = lv1334[1]
            lv1338 = R.call_tir(cls.rotary_embedding, (lv1337, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1339: R.Object = kv_cache[48]
            lv1340 = R.call_tir(cls.squeeze1, (lv1338,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1341: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1339, lv1340, sinfo_args=(R.Object,))
            lv1342: R.Object = kv_cache[49]
            lv1312_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv1334[2]
            lv1313 = R.call_tir(cls.fused_squeeze1, (lv1312_1,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1345: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1342, lv1313, sinfo_args=(R.Object,))
            lv1346: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1341, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1347: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1345, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1348 = R.call_tir(cls.reshape3, (lv1346,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1349 = R.call_tir(cls.reshape3, (lv1347,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1350 = R.call_tir(cls.transpose1, (lv1336,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1351 = R.call_tir(cls.transpose1, (lv1348,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1352 = R.call_tir(cls.transpose1, (lv1349,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1314 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv1350, lv1351, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv1315 = R.call_tir(cls.fused_softmax2_cast8, (lv1314,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1361 = R.call_tir(cls.matmul9, (lv1315, lv1352), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1362 = R.call_tir(cls.transpose5, (lv1361,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1363 = R.call_tir(cls.reshape8, (lv1362,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1316: R.Tensor((2560, 2560), dtype="int8") = model_params[393]
            lv1317: R.Tensor((1, 2560), dtype="float16") = model_params[394]
            lv1318 = R.call_tir(cls.fused_decode8, (lv1316, lv1317), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_3951: R.Tensor((2560,), dtype="float16") = model_params[395]
            lv1367 = R.call_tir(cls.cast5, (lv1306_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_3881: R.Tensor((2560,), dtype="float32") = model_params[388]
            param_3891: R.Tensor((2560,), dtype="float32") = model_params[389]
            lv1319 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1367, param_3881, param_3891), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1320: R.Tensor((2560, 10240), dtype="int8") = model_params[396]
            lv1321: R.Tensor((1, 10240), dtype="float16") = model_params[397]
            param_3981: R.Tensor((10240,), dtype="float16") = model_params[398]
            lv169 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv1320, lv1321, lv1319, param_3981), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv1324: R.Tensor((10240, 2560), dtype="int8") = model_params[399]
            lv1325: R.Tensor((1, 2560), dtype="float16") = model_params[400]
            param_4011: R.Tensor((2560,), dtype="float16") = model_params[401]
            lv170 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv1324, lv1325, lv169, param_4011), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1328 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv1363, lv1318, param_3951, lv170, lv1306_1), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1381 = R.call_tir(cls.cast5, (lv1328,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_4021: R.Tensor((2560,), dtype="float32") = model_params[402]
            param_4031: R.Tensor((2560,), dtype="float32") = model_params[403]
            lv1329 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1381, param_4021, param_4031), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1330: R.Tensor((2560, 7680), dtype="int8") = model_params[406]
            lv1331: R.Tensor((1, 7680), dtype="float16") = model_params[407]
            param_4081: R.Tensor((7680,), dtype="float16") = model_params[408]
            lv171_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv1330, lv1331, lv1329, param_4081), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1388 = R.call_tir(cls.reshape7, (lv171_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1389 = R.call_tir(cls.split1, (lv1388,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1390: R.Tensor((1, n, 32, 80), dtype="float16") = lv1389[0]
            lv1391 = R.call_tir(cls.rotary_embedding, (lv1390, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1392: R.Tensor((1, n, 32, 80), dtype="float16") = lv1389[1]
            lv1393 = R.call_tir(cls.rotary_embedding, (lv1392, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1394: R.Object = kv_cache[50]
            lv1395 = R.call_tir(cls.squeeze1, (lv1393,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1396: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1394, lv1395, sinfo_args=(R.Object,))
            lv1397: R.Object = kv_cache[51]
            lv1334_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv1389[2]
            lv1335_1 = R.call_tir(cls.fused_squeeze1, (lv1334_1,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1400: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1397, lv1335_1, sinfo_args=(R.Object,))
            lv1401: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1396, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1402: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1400, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1403 = R.call_tir(cls.reshape3, (lv1401,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1404 = R.call_tir(cls.reshape3, (lv1402,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1405 = R.call_tir(cls.transpose1, (lv1391,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1406 = R.call_tir(cls.transpose1, (lv1403,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1407 = R.call_tir(cls.transpose1, (lv1404,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1336_1 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv1405, lv1406, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv1337_1 = R.call_tir(cls.fused_softmax2_cast8, (lv1336_1,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1416 = R.call_tir(cls.matmul9, (lv1337_1, lv1407), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1417 = R.call_tir(cls.transpose5, (lv1416,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1418 = R.call_tir(cls.reshape8, (lv1417,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1338_1: R.Tensor((2560, 2560), dtype="int8") = model_params[409]
            lv1339_1: R.Tensor((1, 2560), dtype="float16") = model_params[410]
            lv1340_1 = R.call_tir(cls.fused_decode8, (lv1338_1, lv1339_1), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_4111: R.Tensor((2560,), dtype="float16") = model_params[411]
            lv1422 = R.call_tir(cls.cast5, (lv1328,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_4041: R.Tensor((2560,), dtype="float32") = model_params[404]
            param_4051: R.Tensor((2560,), dtype="float32") = model_params[405]
            lv1341_1 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1422, param_4041, param_4051), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1342_1: R.Tensor((2560, 10240), dtype="int8") = model_params[412]
            lv1343: R.Tensor((1, 10240), dtype="float16") = model_params[413]
            param_4141: R.Tensor((10240,), dtype="float16") = model_params[414]
            lv172 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv1342_1, lv1343, lv1341_1, param_4141), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv1346_1: R.Tensor((10240, 2560), dtype="int8") = model_params[415]
            lv1347_1: R.Tensor((1, 2560), dtype="float16") = model_params[416]
            param_4171: R.Tensor((2560,), dtype="float16") = model_params[417]
            lv173 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv1346_1, lv1347_1, lv172, param_4171), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1350_1 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv1418, lv1340_1, param_4111, lv173, lv1328), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1436 = R.call_tir(cls.cast5, (lv1350_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_4181: R.Tensor((2560,), dtype="float32") = model_params[418]
            param_4191: R.Tensor((2560,), dtype="float32") = model_params[419]
            lv1351_1 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1436, param_4181, param_4191), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1352_1: R.Tensor((2560, 7680), dtype="int8") = model_params[422]
            lv1353: R.Tensor((1, 7680), dtype="float16") = model_params[423]
            param_4241: R.Tensor((7680,), dtype="float16") = model_params[424]
            lv174 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv1352_1, lv1353, lv1351_1, param_4241), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1443 = R.call_tir(cls.reshape7, (lv174,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1444 = R.call_tir(cls.split1, (lv1443,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1445: R.Tensor((1, n, 32, 80), dtype="float16") = lv1444[0]
            lv1446 = R.call_tir(cls.rotary_embedding, (lv1445, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1447: R.Tensor((1, n, 32, 80), dtype="float16") = lv1444[1]
            lv1448 = R.call_tir(cls.rotary_embedding, (lv1447, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1449: R.Object = kv_cache[52]
            lv1450 = R.call_tir(cls.squeeze1, (lv1448,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1451: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1449, lv1450, sinfo_args=(R.Object,))
            lv1452: R.Object = kv_cache[53]
            lv1356: R.Tensor((1, n, 32, 80), dtype="float16") = lv1444[2]
            lv1357 = R.call_tir(cls.fused_squeeze1, (lv1356,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1455: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1452, lv1357, sinfo_args=(R.Object,))
            lv1456: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1451, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1457: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1455, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1458 = R.call_tir(cls.reshape3, (lv1456,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1459 = R.call_tir(cls.reshape3, (lv1457,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1460 = R.call_tir(cls.transpose1, (lv1446,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1461 = R.call_tir(cls.transpose1, (lv1458,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1462 = R.call_tir(cls.transpose1, (lv1459,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1358 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv1460, lv1461, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv1359 = R.call_tir(cls.fused_softmax2_cast8, (lv1358,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1471 = R.call_tir(cls.matmul9, (lv1359, lv1462), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1472 = R.call_tir(cls.transpose5, (lv1471,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1473 = R.call_tir(cls.reshape8, (lv1472,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1360: R.Tensor((2560, 2560), dtype="int8") = model_params[425]
            lv1361_1: R.Tensor((1, 2560), dtype="float16") = model_params[426]
            lv1362_1 = R.call_tir(cls.fused_decode8, (lv1360, lv1361_1), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_4271: R.Tensor((2560,), dtype="float16") = model_params[427]
            lv1477 = R.call_tir(cls.cast5, (lv1350_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_4201: R.Tensor((2560,), dtype="float32") = model_params[420]
            param_4211: R.Tensor((2560,), dtype="float32") = model_params[421]
            lv1363_1 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1477, param_4201, param_4211), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1364: R.Tensor((2560, 10240), dtype="int8") = model_params[428]
            lv1365: R.Tensor((1, 10240), dtype="float16") = model_params[429]
            param_4301: R.Tensor((10240,), dtype="float16") = model_params[430]
            lv175 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv1364, lv1365, lv1363_1, param_4301), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv1368: R.Tensor((10240, 2560), dtype="int8") = model_params[431]
            lv1369: R.Tensor((1, 2560), dtype="float16") = model_params[432]
            param_4331: R.Tensor((2560,), dtype="float16") = model_params[433]
            lv176 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv1368, lv1369, lv175, param_4331), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1372 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv1473, lv1362_1, param_4271, lv176, lv1350_1), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1491 = R.call_tir(cls.cast5, (lv1372,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_4341: R.Tensor((2560,), dtype="float32") = model_params[434]
            param_4351: R.Tensor((2560,), dtype="float32") = model_params[435]
            lv1373 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1491, param_4341, param_4351), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1374: R.Tensor((2560, 7680), dtype="int8") = model_params[438]
            lv1375: R.Tensor((1, 7680), dtype="float16") = model_params[439]
            param_4401: R.Tensor((7680,), dtype="float16") = model_params[440]
            lv177 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv1374, lv1375, lv1373, param_4401), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1498 = R.call_tir(cls.reshape7, (lv177,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1499 = R.call_tir(cls.split1, (lv1498,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1500: R.Tensor((1, n, 32, 80), dtype="float16") = lv1499[0]
            lv1501 = R.call_tir(cls.rotary_embedding, (lv1500, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1502: R.Tensor((1, n, 32, 80), dtype="float16") = lv1499[1]
            lv1503 = R.call_tir(cls.rotary_embedding, (lv1502, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1504: R.Object = kv_cache[54]
            lv1505 = R.call_tir(cls.squeeze1, (lv1503,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1506: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1504, lv1505, sinfo_args=(R.Object,))
            lv1507: R.Object = kv_cache[55]
            lv1378: R.Tensor((1, n, 32, 80), dtype="float16") = lv1499[2]
            lv1379 = R.call_tir(cls.fused_squeeze1, (lv1378,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1510: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1507, lv1379, sinfo_args=(R.Object,))
            lv1511: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1506, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1512: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1510, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1513 = R.call_tir(cls.reshape3, (lv1511,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1514 = R.call_tir(cls.reshape3, (lv1512,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1515 = R.call_tir(cls.transpose1, (lv1501,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1516 = R.call_tir(cls.transpose1, (lv1513,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1517 = R.call_tir(cls.transpose1, (lv1514,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1380 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv1515, lv1516, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv1381_1 = R.call_tir(cls.fused_softmax2_cast8, (lv1380,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1526 = R.call_tir(cls.matmul9, (lv1381_1, lv1517), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1527 = R.call_tir(cls.transpose5, (lv1526,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1528 = R.call_tir(cls.reshape8, (lv1527,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1382: R.Tensor((2560, 2560), dtype="int8") = model_params[441]
            lv1383: R.Tensor((1, 2560), dtype="float16") = model_params[442]
            lv1384 = R.call_tir(cls.fused_decode8, (lv1382, lv1383), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_4431: R.Tensor((2560,), dtype="float16") = model_params[443]
            lv1532 = R.call_tir(cls.cast5, (lv1372,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_4361: R.Tensor((2560,), dtype="float32") = model_params[436]
            param_4371: R.Tensor((2560,), dtype="float32") = model_params[437]
            lv1385 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1532, param_4361, param_4371), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1386: R.Tensor((2560, 10240), dtype="int8") = model_params[444]
            lv1387: R.Tensor((1, 10240), dtype="float16") = model_params[445]
            param_4461: R.Tensor((10240,), dtype="float16") = model_params[446]
            lv178_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv1386, lv1387, lv1385, param_4461), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv1390_1: R.Tensor((10240, 2560), dtype="int8") = model_params[447]
            lv1391_1: R.Tensor((1, 2560), dtype="float16") = model_params[448]
            param_4491: R.Tensor((2560,), dtype="float16") = model_params[449]
            lv179_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv1390_1, lv1391_1, lv178_1, param_4491), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1394_1 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv1528, lv1384, param_4431, lv179_1, lv1372), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1546 = R.call_tir(cls.cast5, (lv1394_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_4501: R.Tensor((2560,), dtype="float32") = model_params[450]
            param_4511: R.Tensor((2560,), dtype="float32") = model_params[451]
            lv1395_1 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1546, param_4501, param_4511), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1396_1: R.Tensor((2560, 7680), dtype="int8") = model_params[454]
            lv1397_1: R.Tensor((1, 7680), dtype="float16") = model_params[455]
            param_4561: R.Tensor((7680,), dtype="float16") = model_params[456]
            lv180_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv1396_1, lv1397_1, lv1395_1, param_4561), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1553 = R.call_tir(cls.reshape7, (lv180_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1554 = R.call_tir(cls.split1, (lv1553,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1555: R.Tensor((1, n, 32, 80), dtype="float16") = lv1554[0]
            lv1556 = R.call_tir(cls.rotary_embedding, (lv1555, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1557: R.Tensor((1, n, 32, 80), dtype="float16") = lv1554[1]
            lv1558 = R.call_tir(cls.rotary_embedding, (lv1557, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1559: R.Object = kv_cache[56]
            lv1560 = R.call_tir(cls.squeeze1, (lv1558,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1561: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1559, lv1560, sinfo_args=(R.Object,))
            lv1562: R.Object = kv_cache[57]
            lv1400_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv1554[2]
            lv1401_1 = R.call_tir(cls.fused_squeeze1, (lv1400_1,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1565: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1562, lv1401_1, sinfo_args=(R.Object,))
            lv1566: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1561, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1567: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1565, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1568 = R.call_tir(cls.reshape3, (lv1566,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1569 = R.call_tir(cls.reshape3, (lv1567,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1570 = R.call_tir(cls.transpose1, (lv1556,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1571 = R.call_tir(cls.transpose1, (lv1568,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1572 = R.call_tir(cls.transpose1, (lv1569,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1402_1 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv1570, lv1571, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv1403_1 = R.call_tir(cls.fused_softmax2_cast8, (lv1402_1,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1581 = R.call_tir(cls.matmul9, (lv1403_1, lv1572), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1582 = R.call_tir(cls.transpose5, (lv1581,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1583 = R.call_tir(cls.reshape8, (lv1582,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1404_1: R.Tensor((2560, 2560), dtype="int8") = model_params[457]
            lv1405_1: R.Tensor((1, 2560), dtype="float16") = model_params[458]
            lv1406_1 = R.call_tir(cls.fused_decode8, (lv1404_1, lv1405_1), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_4591: R.Tensor((2560,), dtype="float16") = model_params[459]
            lv1587 = R.call_tir(cls.cast5, (lv1394_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_4521: R.Tensor((2560,), dtype="float32") = model_params[452]
            param_4531: R.Tensor((2560,), dtype="float32") = model_params[453]
            lv1407_1 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1587, param_4521, param_4531), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1408: R.Tensor((2560, 10240), dtype="int8") = model_params[460]
            lv1409: R.Tensor((1, 10240), dtype="float16") = model_params[461]
            param_4621: R.Tensor((10240,), dtype="float16") = model_params[462]
            lv181_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv1408, lv1409, lv1407_1, param_4621), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv1412: R.Tensor((10240, 2560), dtype="int8") = model_params[463]
            lv1413: R.Tensor((1, 2560), dtype="float16") = model_params[464]
            param_4651: R.Tensor((2560,), dtype="float16") = model_params[465]
            lv182_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv1412, lv1413, lv181_1, param_4651), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1416_1 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv1583, lv1406_1, param_4591, lv182_1, lv1394_1), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1601 = R.call_tir(cls.cast5, (lv1416_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_4661: R.Tensor((2560,), dtype="float32") = model_params[466]
            param_4671: R.Tensor((2560,), dtype="float32") = model_params[467]
            lv1417_1 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1601, param_4661, param_4671), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1418_1: R.Tensor((2560, 7680), dtype="int8") = model_params[470]
            lv1419: R.Tensor((1, 7680), dtype="float16") = model_params[471]
            param_4721: R.Tensor((7680,), dtype="float16") = model_params[472]
            lv183_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv1418_1, lv1419, lv1417_1, param_4721), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1608 = R.call_tir(cls.reshape7, (lv183_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1609 = R.call_tir(cls.split1, (lv1608,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1610: R.Tensor((1, n, 32, 80), dtype="float16") = lv1609[0]
            lv1611 = R.call_tir(cls.rotary_embedding, (lv1610, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1612: R.Tensor((1, n, 32, 80), dtype="float16") = lv1609[1]
            lv1613 = R.call_tir(cls.rotary_embedding, (lv1612, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1614: R.Object = kv_cache[58]
            lv1615 = R.call_tir(cls.squeeze1, (lv1613,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1616: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1614, lv1615, sinfo_args=(R.Object,))
            lv1617: R.Object = kv_cache[59]
            lv1422_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv1609[2]
            lv1423 = R.call_tir(cls.fused_squeeze1, (lv1422_1,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1620: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1617, lv1423, sinfo_args=(R.Object,))
            lv1621: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1616, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1622: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1620, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1623 = R.call_tir(cls.reshape3, (lv1621,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1624 = R.call_tir(cls.reshape3, (lv1622,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1625 = R.call_tir(cls.transpose1, (lv1611,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1626 = R.call_tir(cls.transpose1, (lv1623,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1627 = R.call_tir(cls.transpose1, (lv1624,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1424 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv1625, lv1626, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv1425 = R.call_tir(cls.fused_softmax2_cast8, (lv1424,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1636 = R.call_tir(cls.matmul9, (lv1425, lv1627), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1637 = R.call_tir(cls.transpose5, (lv1636,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1638 = R.call_tir(cls.reshape8, (lv1637,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1426: R.Tensor((2560, 2560), dtype="int8") = model_params[473]
            lv1427: R.Tensor((1, 2560), dtype="float16") = model_params[474]
            lv1428 = R.call_tir(cls.fused_decode8, (lv1426, lv1427), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_4751: R.Tensor((2560,), dtype="float16") = model_params[475]
            lv1642 = R.call_tir(cls.cast5, (lv1416_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_4681: R.Tensor((2560,), dtype="float32") = model_params[468]
            param_4691: R.Tensor((2560,), dtype="float32") = model_params[469]
            lv1429 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1642, param_4681, param_4691), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1430: R.Tensor((2560, 10240), dtype="int8") = model_params[476]
            lv1431: R.Tensor((1, 10240), dtype="float16") = model_params[477]
            param_4781: R.Tensor((10240,), dtype="float16") = model_params[478]
            lv184_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv1430, lv1431, lv1429, param_4781), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv1434: R.Tensor((10240, 2560), dtype="int8") = model_params[479]
            lv1435: R.Tensor((1, 2560), dtype="float16") = model_params[480]
            param_4811: R.Tensor((2560,), dtype="float16") = model_params[481]
            lv185_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv1434, lv1435, lv184_1, param_4811), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1438 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv1638, lv1428, param_4751, lv185_1, lv1416_1), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1656 = R.call_tir(cls.cast5, (lv1438,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_4821: R.Tensor((2560,), dtype="float32") = model_params[482]
            param_4831: R.Tensor((2560,), dtype="float32") = model_params[483]
            lv1439 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1656, param_4821, param_4831), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1440: R.Tensor((2560, 7680), dtype="int8") = model_params[486]
            lv1441: R.Tensor((1, 7680), dtype="float16") = model_params[487]
            param_4881: R.Tensor((7680,), dtype="float16") = model_params[488]
            lv186_1 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv1440, lv1441, lv1439, param_4881), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1663 = R.call_tir(cls.reshape7, (lv186_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1664 = R.call_tir(cls.split1, (lv1663,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1665: R.Tensor((1, n, 32, 80), dtype="float16") = lv1664[0]
            lv1666 = R.call_tir(cls.rotary_embedding, (lv1665, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1667: R.Tensor((1, n, 32, 80), dtype="float16") = lv1664[1]
            lv1668 = R.call_tir(cls.rotary_embedding, (lv1667, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1669: R.Object = kv_cache[60]
            lv1670 = R.call_tir(cls.squeeze1, (lv1668,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1671: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1669, lv1670, sinfo_args=(R.Object,))
            lv1672: R.Object = kv_cache[61]
            lv1444_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv1664[2]
            lv1445_1 = R.call_tir(cls.fused_squeeze1, (lv1444_1,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1675: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1672, lv1445_1, sinfo_args=(R.Object,))
            lv1676: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1671, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1677: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1675, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1678 = R.call_tir(cls.reshape3, (lv1676,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1679 = R.call_tir(cls.reshape3, (lv1677,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1680 = R.call_tir(cls.transpose1, (lv1666,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1681 = R.call_tir(cls.transpose1, (lv1678,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1682 = R.call_tir(cls.transpose1, (lv1679,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1446_1 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv1680, lv1681, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv1447_1 = R.call_tir(cls.fused_softmax2_cast8, (lv1446_1,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1691 = R.call_tir(cls.matmul9, (lv1447_1, lv1682), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1692 = R.call_tir(cls.transpose5, (lv1691,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1693 = R.call_tir(cls.reshape8, (lv1692,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1448_1: R.Tensor((2560, 2560), dtype="int8") = model_params[489]
            lv1449_1: R.Tensor((1, 2560), dtype="float16") = model_params[490]
            lv1450_1 = R.call_tir(cls.fused_decode8, (lv1448_1, lv1449_1), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_4911: R.Tensor((2560,), dtype="float16") = model_params[491]
            lv1697 = R.call_tir(cls.cast5, (lv1438,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_4841: R.Tensor((2560,), dtype="float32") = model_params[484]
            param_4851: R.Tensor((2560,), dtype="float32") = model_params[485]
            lv1451_1 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1697, param_4841, param_4851), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1452_1: R.Tensor((2560, 10240), dtype="int8") = model_params[492]
            lv1453: R.Tensor((1, 10240), dtype="float16") = model_params[493]
            param_4941: R.Tensor((10240,), dtype="float16") = model_params[494]
            lv187_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv1452_1, lv1453, lv1451_1, param_4941), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv1456_1: R.Tensor((10240, 2560), dtype="int8") = model_params[495]
            lv1457_1: R.Tensor((1, 2560), dtype="float16") = model_params[496]
            param_4971: R.Tensor((2560,), dtype="float16") = model_params[497]
            lv188 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv1456_1, lv1457_1, lv187_1, param_4971), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1460_1 = R.call_tir(cls.fused_matmul10_add5_add7_add7, (lv1693, lv1450_1, param_4911, lv188, lv1438), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1711 = R.call_tir(cls.cast5, (lv1460_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_4981: R.Tensor((2560,), dtype="float32") = model_params[498]
            param_4991: R.Tensor((2560,), dtype="float32") = model_params[499]
            lv1461_1 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1711, param_4981, param_4991), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1462_1: R.Tensor((2560, 7680), dtype="int8") = model_params[502]
            lv1463: R.Tensor((1, 7680), dtype="float16") = model_params[503]
            param_5041: R.Tensor((7680,), dtype="float16") = model_params[504]
            lv189 = R.call_tir(cls.fused_fused_decode7_fused_matmul8_add4, (lv1462_1, lv1463, lv1461_1, param_5041), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1718 = R.call_tir(cls.reshape7, (lv189,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1719 = R.call_tir(cls.split1, (lv1718,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1720: R.Tensor((1, n, 32, 80), dtype="float16") = lv1719[0]
            lv1721 = R.call_tir(cls.rotary_embedding, (lv1720, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1722: R.Tensor((1, n, 32, 80), dtype="float16") = lv1719[1]
            lv1723 = R.call_tir(cls.rotary_embedding, (lv1722, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1724: R.Object = kv_cache[62]
            lv1725 = R.call_tir(cls.squeeze1, (lv1723,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1726: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1724, lv1725, sinfo_args=(R.Object,))
            lv1727: R.Object = kv_cache[63]
            lv1466: R.Tensor((1, n, 32, 80), dtype="float16") = lv1719[2]
            lv1467 = R.call_tir(cls.fused_squeeze1, (lv1466,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1730: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1727, lv1467, sinfo_args=(R.Object,))
            lv1731: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1726, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1732: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1730, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1733 = R.call_tir(cls.reshape3, (lv1731,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1734 = R.call_tir(cls.reshape3, (lv1732,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1735 = R.call_tir(cls.transpose1, (lv1721,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1736 = R.call_tir(cls.transpose1, (lv1733,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1737 = R.call_tir(cls.transpose1, (lv1734,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1468 = R.call_tir(cls.fused_NT_matmul2_divide2_maximum1_minimum1_cast7, (lv1735, lv1736, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv1469 = R.call_tir(cls.fused_softmax2_cast8, (lv1468,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1746 = R.call_tir(cls.matmul9, (lv1469, lv1737), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1747 = R.call_tir(cls.transpose5, (lv1746,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1748 = R.call_tir(cls.reshape8, (lv1747,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1470: R.Tensor((2560, 2560), dtype="int8") = model_params[505]
            lv1471_1: R.Tensor((1, 2560), dtype="float16") = model_params[506]
            lv1472_1 = R.call_tir(cls.fused_decode8, (lv1470, lv1471_1), out_sinfo=R.Tensor((2560, 2560), dtype="float16"))
            param_5071: R.Tensor((2560,), dtype="float16") = model_params[507]
            lv1752 = R.call_tir(cls.cast5, (lv1460_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_5001: R.Tensor((2560,), dtype="float32") = model_params[500]
            param_5011: R.Tensor((2560,), dtype="float32") = model_params[501]
            lv1473_1 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1752, param_5001, param_5011), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1474: R.Tensor((2560, 10240), dtype="int8") = model_params[508]
            lv1475: R.Tensor((1, 10240), dtype="float16") = model_params[509]
            param_5101: R.Tensor((10240,), dtype="float16") = model_params[510]
            lv190_1 = R.call_tir(cls.fused_fused_decode9_fused_matmul11_add6_gelu1, (lv1474, lv1475, lv1473_1, param_5101), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            lv1478: R.Tensor((10240, 2560), dtype="int8") = model_params[511]
            lv1479: R.Tensor((1, 2560), dtype="float16") = model_params[512]
            param_5131: R.Tensor((2560,), dtype="float16") = model_params[513]
            lv191_1 = R.call_tir(cls.fused_fused_decode10_fused_matmul12_add5, (lv1478, lv1479, lv190_1, param_5131), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1482 = R.call_tir(cls.fused_matmul10_add5_add7_add7_cast5, (lv1748, lv1472_1, param_5071, lv191_1, lv1460_1), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_5141: R.Tensor((2560,), dtype="float32") = model_params[514]
            param_5151: R.Tensor((2560,), dtype="float32") = model_params[515]
            lv1767 = R.call_tir(cls.layer_norm1, (lv1482, param_5141, param_5151), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            lv1768 = R.call_tir(cls.slice, (lv1767,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            lv1769 = R.call_tir(cls.cast4, (lv1768,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            lv1483: R.Tensor((50280, 320), dtype="uint32") = model_params[516]
            lv1484: R.Tensor((50280, 80), dtype="float32") = model_params[517]
            lv1 = R.call_tir(cls.fused_fused_decode6_NT_matmul1, (lv1483, lv1484, lv1769), out_sinfo=R.Tensor((1, 1, 50280), dtype="float32"))
            gv: R.Tuple(R.Tensor((1, 1, 50280), dtype="float32"), R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object)) = lv1, (lv21, lv25, lv76, lv80, lv131, lv135, lv186, lv190, lv241, lv245, lv296, lv300, lv351, lv355, lv406, lv410, lv461, lv465, lv516, lv520, lv571, lv575, lv626, lv630, lv681, lv685, lv736, lv740, lv791_1, lv795, lv846_1, lv850_1, lv901_1, lv905, lv956_1, lv960_1, lv1011_1, lv1015, lv1066_1, lv1070_1, lv1121_1, lv1125, lv1176_1, lv1180_1, lv1231_1, lv1235, lv1286_1, lv1290_1, lv1341, lv1345, lv1396, lv1400, lv1451, lv1455, lv1506, lv1510, lv1561, lv1565, lv1616, lv1620, lv1671, lv1675, lv1726, lv1730)
            R.output(gv)
        return gv

    @R.function
    def softmax_with_temperature(logits: R.Tensor((1, 1, 50280), dtype="float32"), temperature: R.Tensor((), dtype="float32")) -> R.Tensor((1, 1, 50280), dtype="float32"):
        R.func_attr({"tir_var_upper_bound": {"m": 4096, "n": 4096}})
        cls = Module
        with R.dataflow():
            lv3607 = R.call_tir(cls.divide, (logits, temperature), out_sinfo=R.Tensor((1, 1, 50280), dtype="float32"))
            gv3 = R.call_tir(cls.softmax, (lv3607,), out_sinfo=R.Tensor((1, 1, 50280), dtype="float32"))
            R.output(gv3)
        return gv3

# Metadata omitted. Use show_meta=True in script() method to show it.