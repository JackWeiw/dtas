#name: fused_softmax_cast
#range: (<n: Range(1, 256)>,)  latency(ms):    0.0487    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 8}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1, 256)>,)  latency(ms):    0.0331    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 12}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(128) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1, 256)>,)  latency(ms):    0.0425    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 9}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(160) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1, 256)>,)  latency(ms):    0.0567    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1, 256)>,)  latency(ms):    0.0557    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(256) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(257, 512)>,)  latency(ms):    0.0921    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 5}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(257, 512)>,)  latency(ms):    0.0643    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 9}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(160) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(257, 512)>,)  latency(ms):    0.1331    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(257, 512)>,)  latency(ms):    0.0811    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(257, 512)>,)  latency(ms):    0.1499    
#config: ReductionConfig: {'bx': 7872, 'bx_factor': None, 'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 3}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(7872), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(257, 512)>,)  latency(ms):    0.0716    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 8}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(257, 512)>,)  latency(ms):    0.0783    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(256) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(257, 512)>,)  latency(ms):    0.1533    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(257, 512)>,)  latency(ms):    0.0543    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 12}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(128) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(257, 512)>,)  latency(ms):    0.1591    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(513, 768)>,)  latency(ms):    0.1526    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(513, 768)>,)  latency(ms):    0.1183    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 5}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(513, 768)>,)  latency(ms):    0.1869    
#config: ReductionConfig: {'bx': 7872, 'bx_factor': None, 'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 3}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(7872), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(513, 768)>,)  latency(ms):    0.0843    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 9}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(160) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(513, 768)>,)  latency(ms):    0.0848    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 8}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(513, 768)>,)  latency(ms):    0.0999    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(256) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(513, 768)>,)  latency(ms):    0.0797    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 12}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(128) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(513, 768)>,)  latency(ms):    0.0981    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(513, 768)>,)  latency(ms):    0.1766    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(513, 768)>,)  latency(ms):    0.1832    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.1228    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 5}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.2023    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.1127    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(256) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.1130    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 8}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.1696    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.1181    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.1126    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 9}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(160) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.1023    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 12}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(128) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.2019    
#config: ReductionConfig: {'bx': 7872, 'bx_factor': None, 'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 3}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(7872), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.1968    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.2140    
#config: ReductionConfig: {'bx': 7872, 'bx_factor': None, 'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 3}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(7872), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.1445    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(256) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.2129    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.1525    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 5}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.1432    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 8}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.1440    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.2174    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.1974    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.1399    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 12}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(128) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.1335    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 9}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(160) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.2409    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.1782    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.1610    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 9}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(160) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.2319    
#config: ReductionConfig: {'bx': 7872, 'bx_factor': None, 'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 3}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(7872), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.2489    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.1805    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 5}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.1731    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 12}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(128) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.1626    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 8}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.2296    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.1747    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(256) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.2849    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.2227    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 12}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(128) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.2049    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 5}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.2560    
#config: ReductionConfig: {'bx': 7872, 'bx_factor': None, 'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 3}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(7872), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.2866    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.1971    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.1935    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(256) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.2519    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.1948    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 8}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.1979    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 9}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(160) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.2123    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 8}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.2798    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.3155    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.2743    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 12}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(128) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.2261    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 9}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(160) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.2296    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 5}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.2232    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.2705    
#config: ReductionConfig: {'bx': 7872, 'bx_factor': None, 'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 3}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(7872), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.3138    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.2125    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(256) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.2777    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 9}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(160) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.2427    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 5}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.3011    
#config: ReductionConfig: {'bx': 7872, 'bx_factor': None, 'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 3}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(7872), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.3324    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.2443    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(256) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.2479    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.2468    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 8}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.3079    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.3309    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.3300    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 12}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(128) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.3409    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.3552    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.2685    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 5}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.2596    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.3020    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 8}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.3185    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 9}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(160) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.2639    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(256) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.3203    
#config: ReductionConfig: {'bx': 7872, 'bx_factor': None, 'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 3}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(7872), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.3479    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.3808    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 12}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(128) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.2821    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(256) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.2974    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 5}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.3096    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.3599    
#config: ReductionConfig: {'bx': 7872, 'bx_factor': None, 'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 3}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(7872), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.4434    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 12}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(128) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.3431    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 8}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.3701    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.3998    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.3749    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.3705    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 9}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(160) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.4031    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.4346    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 9}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(160) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.3754    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 8}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.4202    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.3941    
#config: ReductionConfig: {'bx': 7872, 'bx_factor': None, 'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 3}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(7872), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.4509    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.3220    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(256) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.3234    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 5}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.5052    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 12}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(128) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.3404    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.5725    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 12}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(128) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.4350    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 8}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.3332    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 5}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.3864    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.4603    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.4712    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.5113    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.4330    
#config: ReductionConfig: {'bx': 7872, 'bx_factor': None, 'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 3}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(7872), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.4726    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 9}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(160) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.3700    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(256) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.3514    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 5}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.5653    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 9}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(160) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.4952    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 8}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.5773    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.5105    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.4179    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(256) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.5197    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.6508    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 12}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(128) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.4107    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.4753    
#config: ReductionConfig: {'bx': 7872, 'bx_factor': None, 'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 3}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(7872), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.6295    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.4762    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.5061    
#config: ReductionConfig: {'bx': 7872, 'bx_factor': None, 'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 3}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(7872), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.4152    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 5}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.5624    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.4854    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(256) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.7177    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 12}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(128) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.5368    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 8}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.5684    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.5987    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 9}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(160) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.6059    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(351)) // T.int64(352), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(352) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.5078    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(256) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(256) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(256) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.4534    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 5}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(288) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(288) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(287)) // T.int64(288), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.5553    
#config: ReductionConfig: {'bx': 7872, 'bx_factor': None, 'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 3}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(7872), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(511)) // T.int64(512), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(512) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.7950    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 12}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(128) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(128) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(128) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.5179    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 6}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(224) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(224) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(223)) // T.int64(224), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.6138    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(383)) // T.int64(384), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(384) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.6061    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 8}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(192) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(192) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(191)) // T.int64(192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.6631    
#config: ReductionConfig: {'bx': None, 'bx_factor': 1, 'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 9}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_maxelem"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1])
                        T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax0_fused_0_1 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(T.int64(12800), ax0_fused_0)
                        v1 = T.axis.reduce(n, ax0_fused_0_1 * T.int64(160) + ax0_fused_1)
                        T.where(ax0_fused_0_1 * T.int64(160) + ax0_fused_1 < n)
                        T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                        T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                        with T.init():
                            T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                        T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(159)) // T.int64(160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(160) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
#name: fused_softmax_cast
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.7237    
#config: ReductionConfig: {'bx': 10496, 'bx_factor': None, 'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache', 'max_active_blocks_per_sm': 4}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(12800), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(12800), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(12800)), scope="shared")
        assert n > T.int64(0), "[n] should be greater than 0"
        for ax0_fused_0 in T.thread_binding(T.int64(10496), thread="blockIdx.x"):
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0] = T.max(T_softmax_maxelem_shared[T.int64(0), v0], A[T.int64(0), v0, v1])
            for ax0 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0)
                            v1 = T.axis.reduce(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 < T.int64(6400) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0] = T_softmax_expsum_shared[T.int64(0), v0] + T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0])
            for ax0_fused_1 in range(T.int64(2)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial((n + T.int64(319)) // T.int64(320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(12800), ax0_fused_0 * T.int64(2) + ax0_fused_1)
                            v1 = T.axis.spatial(n, ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.where(ax0_fused_0 * T.int64(2) + ax0_fused_1 < T.int64(12800) and ax1_fused_0 * T.int64(320) + ax1_fused_1 < n)
                            T.reads(A[T.int64(0), v0, v1], T_softmax_maxelem_shared[T.int64(0), v0], T_softmax_expsum_shared[T.int64(0), v0])
                            T.writes(compute_intermediate[T.int64(0), v0, v1])
                            compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", T.exp(A[T.int64(0), v0, v1] - T_softmax_maxelem_shared[T.int64(0), v0]) / T_softmax_expsum_shared[T.int64(0), v0])
 
