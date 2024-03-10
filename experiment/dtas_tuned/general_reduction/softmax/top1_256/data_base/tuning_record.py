#name: fused_softmax_cast
#range: (<n: Range(1, 512)>,)  latency(ms):    0.0310    
#config: ReductionConfig: {'len_tx': 1024, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 0}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(1000), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(1000), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(1000)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(1000)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused in T.thread_binding(T.int64(1000), thread="blockIdx.x"):
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(1023)) // T.int64(1024), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(1000), ax0_fused + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(1024) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(1024) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(1023)) // T.int64(1024), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(1000), ax0_fused + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(1024) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(1024) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax1_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax1_0 in T.serial((n + T.int64(1023)) // T.int64(1024), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(1000), ax0_fused)
                        v1 = T.axis.spatial(n, ax1_0 * T.int64(1024) + ax1_1)
                        T.where(ax1_0 * T.int64(1024) + ax1_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(513, 1024)>,)  latency(ms):    0.0314    
#config: ReductionConfig: {'len_tx': 1024, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 0}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(1000), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(1000), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(1000)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(1000)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused in T.thread_binding(T.int64(1000), thread="blockIdx.x"):
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(1023)) // T.int64(1024), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(1000), ax0_fused + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(1024) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(1024) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(1023)) // T.int64(1024), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(1000), ax0_fused + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(1024) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(1024) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax1_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax1_0 in T.serial((n + T.int64(1023)) // T.int64(1024), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(1000), ax0_fused)
                        v1 = T.axis.spatial(n, ax1_0 * T.int64(1024) + ax1_1)
                        T.where(ax1_0 * T.int64(1024) + ax1_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1025, 1536)>,)  latency(ms):    0.0155    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn', 'max_active_blocks_per_sm': 12}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(1000), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(1000), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(1000)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(1000)), scope="shared")
        A_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(1000), n), scope="shared.dyn")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused in T.thread_binding(T.int64(1000), thread="blockIdx.x"):
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), (T.max(T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128)) + T.int64(511) - T.min(T.int64(0), (n + T.int64(127)) // T.int64(128) * T.int64(128) - T.int64(1))) // T.int64(512)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("A_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(1000), ax0_fused + ax1)
                            v2 = T.axis.spatial(n, ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2 + (T.int64(0) - (T.int64(0) - T.min(T.int64(0), (n + T.int64(127)) // T.int64(128) * T.int64(128) - T.int64(1)))) + (T.min(T.int64(0), (n + T.int64(127)) // T.int64(128) * T.int64(128) - T.int64(1)) + T.max(T.int64(0) - T.min(T.int64(0), (n + T.int64(127)) // T.int64(128) * T.int64(128) - T.int64(1)), T.int64(0))))
                            T.where(T.int64(0) <= T.min(T.int64(0), (n + T.int64(127)) // T.int64(128) * T.int64(128) - T.int64(1)) + ((ax2_0 * T.int64(128) + ax2_1) * T.int64(4) + ax2_2) and T.min(T.int64(0), (n + T.int64(127)) // T.int64(128) * T.int64(128) - T.int64(1)) + ((ax2_0 * T.int64(128) + ax2_1) * T.int64(4) + ax2_2) < n and (ax2_0 * T.int64(128) + ax2_1) * T.int64(4) + ax2_2 < T.max(T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128)) - T.min(T.int64(0), (n + T.int64(127)) // T.int64(128) * T.int64(128) - T.int64(1)))
                            T.reads(A[v0, v1, v2])
                            T.writes(A_shared_dyn[v0, v1, v2])
                            A_shared_dyn[v0, v1, v2] = A[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(1000), ax0_fused + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(128) + ax1_fused_1 < n)
                            T.reads(A_shared_dyn[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A_shared_dyn[T.int64(0), v0, v1])
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(1000), ax0_fused + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(128) + ax1_fused_1 < n)
                            T.reads(A_shared_dyn[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A_shared_dyn[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax1_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax1_0 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(1000), ax0_fused)
                        v1 = T.axis.spatial(n, ax1_0 * T.int64(128) + ax1_1)
                        T.where(ax1_0 * T.int64(128) + ax1_1 < n)
                        T.reads(A_shared_dyn[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A_shared_dyn[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1537, 2048)>,)  latency(ms):    0.0233    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn', 'max_active_blocks_per_sm': 12}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(1000), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(1000), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(1000)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(1000)), scope="shared")
        A_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(1000), n), scope="shared.dyn")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused in T.thread_binding(T.int64(1000), thread="blockIdx.x"):
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), (T.max(T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128)) + T.int64(511) - T.min(T.int64(0), (n + T.int64(127)) // T.int64(128) * T.int64(128) - T.int64(1))) // T.int64(512)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("A_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(1000), ax0_fused + ax1)
                            v2 = T.axis.spatial(n, ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2 + (T.int64(0) - (T.int64(0) - T.min(T.int64(0), (n + T.int64(127)) // T.int64(128) * T.int64(128) - T.int64(1)))) + (T.min(T.int64(0), (n + T.int64(127)) // T.int64(128) * T.int64(128) - T.int64(1)) + T.max(T.int64(0) - T.min(T.int64(0), (n + T.int64(127)) // T.int64(128) * T.int64(128) - T.int64(1)), T.int64(0))))
                            T.where(T.int64(0) <= T.min(T.int64(0), (n + T.int64(127)) // T.int64(128) * T.int64(128) - T.int64(1)) + ((ax2_0 * T.int64(128) + ax2_1) * T.int64(4) + ax2_2) and T.min(T.int64(0), (n + T.int64(127)) // T.int64(128) * T.int64(128) - T.int64(1)) + ((ax2_0 * T.int64(128) + ax2_1) * T.int64(4) + ax2_2) < n and (ax2_0 * T.int64(128) + ax2_1) * T.int64(4) + ax2_2 < T.max(T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128)) - T.min(T.int64(0), (n + T.int64(127)) // T.int64(128) * T.int64(128) - T.int64(1)))
                            T.reads(A[v0, v1, v2])
                            T.writes(A_shared_dyn[v0, v1, v2])
                            A_shared_dyn[v0, v1, v2] = A[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(1000), ax0_fused + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(128) + ax1_fused_1 < n)
                            T.reads(A_shared_dyn[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A_shared_dyn[T.int64(0), v0, v1])
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(1000), ax0_fused + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(128) + ax1_fused_1 < n)
                            T.reads(A_shared_dyn[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A_shared_dyn[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax1_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax1_0 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(1000), ax0_fused)
                        v1 = T.axis.spatial(n, ax1_0 * T.int64(128) + ax1_1)
                        T.where(ax1_0 * T.int64(128) + ax1_1 < n)
                        T.reads(A_shared_dyn[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A_shared_dyn[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2049, 2560)>,)  latency(ms):    0.0263    
#config: ReductionConfig: {'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn', 'max_active_blocks_per_sm': 9}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(1000), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(1000), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(1000)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(1000)), scope="shared")
        A_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(1000), n), scope="shared.dyn")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused in T.thread_binding(T.int64(1000), thread="blockIdx.x"):
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), (T.max(T.int64(1), (n + T.int64(159)) // T.int64(160) * T.int64(160)) + T.int64(639) - T.min(T.int64(0), (n + T.int64(159)) // T.int64(160) * T.int64(160) - T.int64(1))) // T.int64(640)):
                for ax2_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("A_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(1000), ax0_fused + ax1)
                            v2 = T.axis.spatial(n, ax2_0 * T.int64(640) + ax2_1 * T.int64(4) + ax2_2 + (T.int64(0) - (T.int64(0) - T.min(T.int64(0), (n + T.int64(159)) // T.int64(160) * T.int64(160) - T.int64(1)))) + (T.min(T.int64(0), (n + T.int64(159)) // T.int64(160) * T.int64(160) - T.int64(1)) + T.max(T.int64(0) - T.min(T.int64(0), (n + T.int64(159)) // T.int64(160) * T.int64(160) - T.int64(1)), T.int64(0))))
                            T.where(T.int64(0) <= T.min(T.int64(0), (n + T.int64(159)) // T.int64(160) * T.int64(160) - T.int64(1)) + ((ax2_0 * T.int64(160) + ax2_1) * T.int64(4) + ax2_2) and T.min(T.int64(0), (n + T.int64(159)) // T.int64(160) * T.int64(160) - T.int64(1)) + ((ax2_0 * T.int64(160) + ax2_1) * T.int64(4) + ax2_2) < n and (ax2_0 * T.int64(160) + ax2_1) * T.int64(4) + ax2_2 < T.max(T.int64(1), (n + T.int64(159)) // T.int64(160) * T.int64(160)) - T.min(T.int64(0), (n + T.int64(159)) // T.int64(160) * T.int64(160) - T.int64(1)))
                            T.reads(A[v0, v1, v2])
                            T.writes(A_shared_dyn[v0, v1, v2])
                            A_shared_dyn[v0, v1, v2] = A[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(1000), ax0_fused + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(160) + ax1_fused_1 < n)
                            T.reads(A_shared_dyn[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A_shared_dyn[T.int64(0), v0, v1])
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(1000), ax0_fused + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(160) + ax1_fused_1 < n)
                            T.reads(A_shared_dyn[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A_shared_dyn[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax1_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax1_0 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(1000), ax0_fused)
                        v1 = T.axis.spatial(n, ax1_0 * T.int64(160) + ax1_1)
                        T.where(ax1_0 * T.int64(160) + ax1_1 < n)
                        T.reads(A_shared_dyn[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A_shared_dyn[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2561, 3072)>,)  latency(ms):    0.0306    
#config: ReductionConfig: {'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn', 'max_active_blocks_per_sm': 8}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(1000), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(1000), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(1000)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(1000)), scope="shared")
        A_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(1000), n), scope="shared.dyn")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused in T.thread_binding(T.int64(1000), thread="blockIdx.x"):
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), (T.max(T.int64(1), (n + T.int64(191)) // T.int64(192) * T.int64(192)) + T.int64(767) - T.min(T.int64(0), (n + T.int64(191)) // T.int64(192) * T.int64(192) - T.int64(1))) // T.int64(768)):
                for ax2_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("A_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(1000), ax0_fused + ax1)
                            v2 = T.axis.spatial(n, ax2_0 * T.int64(768) + ax2_1 * T.int64(4) + ax2_2 + (T.int64(0) - (T.int64(0) - T.min(T.int64(0), (n + T.int64(191)) // T.int64(192) * T.int64(192) - T.int64(1)))) + (T.min(T.int64(0), (n + T.int64(191)) // T.int64(192) * T.int64(192) - T.int64(1)) + T.max(T.int64(0) - T.min(T.int64(0), (n + T.int64(191)) // T.int64(192) * T.int64(192) - T.int64(1)), T.int64(0))))
                            T.where(T.int64(0) <= T.min(T.int64(0), (n + T.int64(191)) // T.int64(192) * T.int64(192) - T.int64(1)) + ((ax2_0 * T.int64(192) + ax2_1) * T.int64(4) + ax2_2) and T.min(T.int64(0), (n + T.int64(191)) // T.int64(192) * T.int64(192) - T.int64(1)) + ((ax2_0 * T.int64(192) + ax2_1) * T.int64(4) + ax2_2) < n and (ax2_0 * T.int64(192) + ax2_1) * T.int64(4) + ax2_2 < T.max(T.int64(1), (n + T.int64(191)) // T.int64(192) * T.int64(192)) - T.min(T.int64(0), (n + T.int64(191)) // T.int64(192) * T.int64(192) - T.int64(1)))
                            T.reads(A[v0, v1, v2])
                            T.writes(A_shared_dyn[v0, v1, v2])
                            A_shared_dyn[v0, v1, v2] = A[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(1000), ax0_fused + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < n)
                            T.reads(A_shared_dyn[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A_shared_dyn[T.int64(0), v0, v1])
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(1000), ax0_fused + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < n)
                            T.reads(A_shared_dyn[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A_shared_dyn[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax1_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax1_0 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(1000), ax0_fused)
                        v1 = T.axis.spatial(n, ax1_0 * T.int64(192) + ax1_1)
                        T.where(ax1_0 * T.int64(192) + ax1_1 < n)
                        T.reads(A_shared_dyn[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A_shared_dyn[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3073, 3584)>,)  latency(ms):    0.0356    
#config: ReductionConfig: {'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn', 'max_active_blocks_per_sm': 7}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(1000), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(1000), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(1000)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(1000)), scope="shared")
        A_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(1000), n), scope="shared.dyn")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused in T.thread_binding(T.int64(1000), thread="blockIdx.x"):
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), (T.max(T.int64(1), (n + T.int64(191)) // T.int64(192) * T.int64(192)) + T.int64(767) - T.min(T.int64(0), (n + T.int64(191)) // T.int64(192) * T.int64(192) - T.int64(1))) // T.int64(768)):
                for ax2_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("A_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(1000), ax0_fused + ax1)
                            v2 = T.axis.spatial(n, ax2_0 * T.int64(768) + ax2_1 * T.int64(4) + ax2_2 + (T.int64(0) - (T.int64(0) - T.min(T.int64(0), (n + T.int64(191)) // T.int64(192) * T.int64(192) - T.int64(1)))) + (T.min(T.int64(0), (n + T.int64(191)) // T.int64(192) * T.int64(192) - T.int64(1)) + T.max(T.int64(0) - T.min(T.int64(0), (n + T.int64(191)) // T.int64(192) * T.int64(192) - T.int64(1)), T.int64(0))))
                            T.where(T.int64(0) <= T.min(T.int64(0), (n + T.int64(191)) // T.int64(192) * T.int64(192) - T.int64(1)) + ((ax2_0 * T.int64(192) + ax2_1) * T.int64(4) + ax2_2) and T.min(T.int64(0), (n + T.int64(191)) // T.int64(192) * T.int64(192) - T.int64(1)) + ((ax2_0 * T.int64(192) + ax2_1) * T.int64(4) + ax2_2) < n and (ax2_0 * T.int64(192) + ax2_1) * T.int64(4) + ax2_2 < T.max(T.int64(1), (n + T.int64(191)) // T.int64(192) * T.int64(192)) - T.min(T.int64(0), (n + T.int64(191)) // T.int64(192) * T.int64(192) - T.int64(1)))
                            T.reads(A[v0, v1, v2])
                            T.writes(A_shared_dyn[v0, v1, v2])
                            A_shared_dyn[v0, v1, v2] = A[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(1000), ax0_fused + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < n)
                            T.reads(A_shared_dyn[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A_shared_dyn[T.int64(0), v0, v1])
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(1000), ax0_fused + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < n)
                            T.reads(A_shared_dyn[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A_shared_dyn[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax1_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax1_0 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(1000), ax0_fused)
                        v1 = T.axis.spatial(n, ax1_0 * T.int64(192) + ax1_1)
                        T.where(ax1_0 * T.int64(192) + ax1_1 < n)
                        T.reads(A_shared_dyn[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A_shared_dyn[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3585, 4096)>,)  latency(ms):    0.0424    
#config: ReductionConfig: {'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(1000), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(1000), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(1000)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(1000)), scope="shared")
        A_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(1000), n), scope="shared.dyn")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused in T.thread_binding(T.int64(1000), thread="blockIdx.x"):
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), (T.max(T.int64(1), (n + T.int64(255)) // T.int64(256) * T.int64(256)) + T.int64(1023) - T.min(T.int64(0), (n + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1))) // T.int64(1024)):
                for ax2_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("A_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(1000), ax0_fused + ax1)
                            v2 = T.axis.spatial(n, ax2_0 * T.int64(1024) + ax2_1 * T.int64(4) + ax2_2 + (T.int64(0) - (T.int64(0) - T.min(T.int64(0), (n + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1)))) + (T.min(T.int64(0), (n + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1)) + T.max(T.int64(0) - T.min(T.int64(0), (n + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1)), T.int64(0))))
                            T.where(T.int64(0) <= T.min(T.int64(0), (n + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1)) + ((ax2_0 * T.int64(256) + ax2_1) * T.int64(4) + ax2_2) and T.min(T.int64(0), (n + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1)) + ((ax2_0 * T.int64(256) + ax2_1) * T.int64(4) + ax2_2) < n and (ax2_0 * T.int64(256) + ax2_1) * T.int64(4) + ax2_2 < T.max(T.int64(1), (n + T.int64(255)) // T.int64(256) * T.int64(256)) - T.min(T.int64(0), (n + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1)))
                            T.reads(A[v0, v1, v2])
                            T.writes(A_shared_dyn[v0, v1, v2])
                            A_shared_dyn[v0, v1, v2] = A[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(1000), ax0_fused + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(256) + ax1_fused_1 < n)
                            T.reads(A_shared_dyn[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A_shared_dyn[T.int64(0), v0, v1])
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(1000), ax0_fused + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(256) + ax1_fused_1 < n)
                            T.reads(A_shared_dyn[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A_shared_dyn[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(1000), ax0_fused)
                        v1 = T.axis.spatial(n, ax1_0 * T.int64(256) + ax1_1)
                        T.where(ax1_0 * T.int64(256) + ax1_1 < n)
                        T.reads(A_shared_dyn[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A_shared_dyn[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
