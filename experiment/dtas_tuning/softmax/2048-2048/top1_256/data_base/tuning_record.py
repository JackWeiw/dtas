#name: softmax
#range: (<n: Range(1, 256)>, <m: Range(1, 256)>)  latency(ms):    0.0128    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1, 256)>, <m: Range(257, 512)>)  latency(ms):    0.0174    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1, 256)>, <m: Range(513, 768)>)  latency(ms):    0.0270    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1, 256)>, <m: Range(769, 1024)>)  latency(ms):    0.0364    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1, 256)>, <m: Range(1025, 1280)>)  latency(ms):    0.0436    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1, 256)>, <m: Range(1281, 1536)>)  latency(ms):    0.0503    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1, 256)>, <m: Range(1537, 1792)>)  latency(ms):    0.0658    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1, 256)>, <m: Range(1793, 2048)>)  latency(ms):    0.0714    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(257, 512)>, <m: Range(1, 256)>)  latency(ms):    0.0333    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(257, 512)>, <m: Range(257, 512)>)  latency(ms):    0.0489    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(257, 512)>, <m: Range(513, 768)>)  latency(ms):    0.0708    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(257, 512)>, <m: Range(769, 1024)>)  latency(ms):    0.0982    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(257, 512)>, <m: Range(1025, 1280)>)  latency(ms):    0.1165    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(257, 512)>, <m: Range(1281, 1536)>)  latency(ms):    0.1329    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(257, 512)>, <m: Range(1537, 1792)>)  latency(ms):    0.1755    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(257, 512)>, <m: Range(1793, 2048)>)  latency(ms):    0.1915    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(513, 768)>, <m: Range(1, 256)>)  latency(ms):    0.0530    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(513, 768)>, <m: Range(257, 512)>)  latency(ms):    0.0790    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(513, 768)>, <m: Range(513, 768)>)  latency(ms):    0.1113    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(513, 768)>, <m: Range(769, 1024)>)  latency(ms):    0.1588    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(513, 768)>, <m: Range(1025, 1280)>)  latency(ms):    0.1856    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(513, 768)>, <m: Range(1281, 1536)>)  latency(ms):    0.2146    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(513, 768)>, <m: Range(1537, 1792)>)  latency(ms):    0.2854    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(513, 768)>, <m: Range(1793, 2048)>)  latency(ms):    0.3054    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(769, 1024)>, <m: Range(1, 256)>)  latency(ms):    0.0715    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(769, 1024)>, <m: Range(257, 512)>)  latency(ms):    0.1060    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(769, 1024)>, <m: Range(513, 768)>)  latency(ms):    0.1529    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(769, 1024)>, <m: Range(769, 1024)>)  latency(ms):    0.2192    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(769, 1024)>, <m: Range(1025, 1280)>)  latency(ms):    0.2556    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(769, 1024)>, <m: Range(1281, 1536)>)  latency(ms):    0.2967    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(769, 1024)>, <m: Range(1537, 1792)>)  latency(ms):    0.3939    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(769, 1024)>, <m: Range(1793, 2048)>)  latency(ms):    0.4256    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1025, 1280)>, <m: Range(1, 256)>)  latency(ms):    0.0906    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1025, 1280)>, <m: Range(257, 512)>)  latency(ms):    0.1348    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1025, 1280)>, <m: Range(513, 768)>)  latency(ms):    0.1951    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1025, 1280)>, <m: Range(769, 1024)>)  latency(ms):    0.2799    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1025, 1280)>, <m: Range(1025, 1280)>)  latency(ms):    0.3279    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1025, 1280)>, <m: Range(1281, 1536)>)  latency(ms):    0.3786    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1025, 1280)>, <m: Range(1537, 1792)>)  latency(ms):    0.5044    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1025, 1280)>, <m: Range(1793, 2048)>)  latency(ms):    0.5345    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1281, 1536)>, <m: Range(1, 256)>)  latency(ms):    0.1106    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1281, 1536)>, <m: Range(257, 512)>)  latency(ms):    0.1643    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1281, 1536)>, <m: Range(513, 768)>)  latency(ms):    0.2362    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1281, 1536)>, <m: Range(769, 1024)>)  latency(ms):    0.3408    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1281, 1536)>, <m: Range(1025, 1280)>)  latency(ms):    0.3978    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1281, 1536)>, <m: Range(1281, 1536)>)  latency(ms):    0.4605    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1281, 1536)>, <m: Range(1537, 1792)>)  latency(ms):    0.6195    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1281, 1536)>, <m: Range(1793, 2048)>)  latency(ms):    0.6480    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1537, 1792)>, <m: Range(1, 256)>)  latency(ms):    0.1298    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1537, 1792)>, <m: Range(257, 512)>)  latency(ms):    0.1932    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1537, 1792)>, <m: Range(513, 768)>)  latency(ms):    0.2786    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1537, 1792)>, <m: Range(769, 1024)>)  latency(ms):    0.4010    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1537, 1792)>, <m: Range(1025, 1280)>)  latency(ms):    0.4679    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1537, 1792)>, <m: Range(1281, 1536)>)  latency(ms):    0.5420    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1537, 1792)>, <m: Range(1537, 1792)>)  latency(ms):    0.7311    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1537, 1792)>, <m: Range(1793, 2048)>)  latency(ms):    0.7623    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1793, 2048)>, <m: Range(1, 256)>)  latency(ms):    0.1523    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1793, 2048)>, <m: Range(257, 512)>)  latency(ms):    0.2225    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1793, 2048)>, <m: Range(513, 768)>)  latency(ms):    0.3204    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1793, 2048)>, <m: Range(769, 1024)>)  latency(ms):    0.4610    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'cache'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1793, 2048)>, <m: Range(1025, 1280)>)  latency(ms):    0.5396    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1793, 2048)>, <m: Range(1281, 1536)>)  latency(ms):    0.6245    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1793, 2048)>, <m: Range(1537, 1792)>)  latency(ms):    0.8269    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
#name: softmax
#range: (<n: Range(1793, 2048)>, <m: Range(1793, 2048)>)  latency(ms):    0.8781    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
        lv38_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(32), n, m), scope="shared.dyn")
        for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), ((m + T.int64(127)) // T.int64(128) * T.int64(128) + T.int64(511)) // T.int64(512)):
                for ax3_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(512) + ax3_1 * T.int64(4) + ax3_2)
                            T.where((ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < m and (ax3_0 * T.int64(128) + ax3_1) * T.int64(4) + ax3_2 < (m + T.int64(127)) // T.int64(128) * T.int64(128))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(128) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(128) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(127)) // T.int64(128), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(128) + ax2_1)
                        T.where(ax2_0 * T.int64(128) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
 
