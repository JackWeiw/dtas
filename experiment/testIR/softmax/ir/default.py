from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
dtype="float16"
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
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), (T.max(T.int64(1), (m + T.int64(255)) // T.int64(256) * T.int64(256)) + T.int64(1023) - T.min(T.int64(0), (m + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1))) // T.int64(1024)):
                for ax3_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(1024) + ax3_1 * T.int64(4) + ax3_2 + (T.int64(0) - (T.int64(0) - T.min(T.int64(0), (m + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1)))) + (T.min(T.int64(0), (m + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1)) + T.max(T.int64(0) - T.min(T.int64(0), (m + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1)), T.int64(0))))
                            T.where(T.int64(0) <= T.min(T.int64(0), (m + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1)) + ((ax3_0 * T.int64(256) + ax3_1) * T.int64(4) + ax3_2) and T.min(T.int64(0), (m + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1)) + ((ax3_0 * T.int64(256) + ax3_1) * T.int64(4) + ax3_2) < m and (ax3_0 * T.int64(256) + ax3_1) * T.int64(4) + ax3_2 < T.max(T.int64(1), (m + T.int64(255)) // T.int64(256) * T.int64(256)) - T.min(T.int64(0), (m + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1)))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(256) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(256) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(256) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(256) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(256) + ax2_1)
                        T.where(ax2_0 * T.int64(256) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
                        
import tvm
mod = Module
print(tvm.lower(mod))

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        blockIdx_x = T.launch_thread("blockIdx.x", n * T.int64(32))
        lv38_shared_dyn = T.allocate([T.min(m, T.max(T.min(T.int64(0), (m + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1)) + T.max(T.int64(0), T.int64(1) - (m + T.int64(255)) // T.int64(256) * T.int64(256)) + m, m) - T.min(T.min(T.int64(0), (m + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1)) + T.max(T.int64(0), T.int64(1) - (m + T.int64(255)) // T.int64(256) * T.int64(256)), T.int64(0)) * T.int64(2))], "float32", "shared.dyn")
        in_thread_T_softmax_maxelem_shared = T.allocate([1], "float32", "local")
        cross_thread_T_softmax_maxelem_shared = T.allocate([1], "float32", "local")
        T_softmax_maxelem_shared = T.allocate([1], "float32", "shared")
        in_thread_T_softmax_expsum_shared = T.allocate([1], "float32", "local")
        cross_thread_T_softmax_expsum_shared = T.allocate([1], "float32", "local")
        T_softmax_expsum_shared = T.allocate([1], "float32", "shared")
        threadIdx_x = T.launch_thread("threadIdx.x", T.int64(256))
        lv38_shared_dyn_1 = T.Buffer((T.min(m, T.max(T.max(T.int64(0), T.int64(1) - (m + T.int64(255)) // T.int64(256) * T.int64(256)) + T.min(T.int64(0), (m + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1)) + m, m) - T.min(T.max(T.int64(0), T.int64(1) - (m + T.int64(255)) // T.int64(256) * T.int64(256)) + T.min(T.int64(0), (m + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1)), T.int64(0)) * T.int64(2)),), data=lv38_shared_dyn, scope="shared.dyn")
        for ax3_0, ax3_2_s in T.grid((T.max(T.int64(1), (m + T.int64(255)) // T.int64(256) * T.int64(256)) + T.int64(1023) - T.min(T.int64(0), (m + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1))) // T.int64(1024), T.int64(4)):
            if T.int64(0) <= ax3_0 * T.int64(1024) + threadIdx_x * T.int64(4) + T.min(T.int64(0), (m + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1)) + ax3_2_s and ax3_0 * T.int64(1024) + threadIdx_x * T.int64(4) + T.min(T.int64(0), (m + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1)) + ax3_2_s < m and ax3_0 * T.int64(1024) + threadIdx_x * T.int64(4) + ax3_2_s < T.max(T.int64(1), (m + T.int64(255)) // T.int64(256) * T.int64(256)) - T.min(T.int64(0), (m + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1)):
                lv38_1 = T.Buffer((n * m * T.int64(32),), data=lv38.data)
                lv38_shared_dyn_1[ax3_0 * T.int64(1024) + threadIdx_x * T.int64(4) + T.min(T.int64(0), (m + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1)) * T.int64(3) + T.max(T.int64(0), T.int64(1) - (m + T.int64(255)) // T.int64(256) * T.int64(256)) * T.int64(2) + ax3_2_s] = lv38_1[ax3_0 * T.int64(1024) + threadIdx_x * T.int64(4) + T.min(T.int64(0), (m + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1)) * T.int64(3) + T.max(T.int64(0), T.int64(1) - (m + T.int64(255)) // T.int64(256) * T.int64(256)) * T.int64(2) + blockIdx_x * m + ax3_2_s]
        in_thread_T_softmax_maxelem_shared_1 = T.Buffer((1,), data=in_thread_T_softmax_maxelem_shared, scope="local")
        in_thread_T_softmax_maxelem_shared_1[0] = T.float32(-3.4028234663852886e+38)
        for ax2_fused_0 in range((m + T.int64(255)) // T.int64(256)):
            if ax2_fused_0 * T.int64(256) + threadIdx_x < m:
                in_thread_T_softmax_maxelem_shared_1[0] = T.max(in_thread_T_softmax_maxelem_shared_1[0], lv38_shared_dyn_1[ax2_fused_0 * T.int64(256) + threadIdx_x])
        cross_thread_T_softmax_maxelem_shared_1 = T.Buffer((1,), data=cross_thread_T_softmax_maxelem_shared, scope="local")
        with T.attr(T.comm_reducer(lambda x0, y0: T.max(x0, y0), [T.float32(-3.4028234663852886e+38)]), "reduce_scope", T.reinterpret("handle", T.uint64(0))):
            T.tvm_thread_allreduce(T.uint32(1), in_thread_T_softmax_maxelem_shared_1[0], T.bool(True), cross_thread_T_softmax_maxelem_shared_1[0], threadIdx_x)
        T_softmax_maxelem_shared_1 = T.Buffer((T.int64(1),), data=T_softmax_maxelem_shared, scope="shared")
        if threadIdx_x == T.int64(0):
            T_softmax_maxelem_shared_1[0] = cross_thread_T_softmax_maxelem_shared_1[0]
        in_thread_T_softmax_expsum_shared_1 = T.Buffer((1,), data=in_thread_T_softmax_expsum_shared, scope="local")
        in_thread_T_softmax_expsum_shared_1[0] = T.float32(0)
        for ax2_fused_0 in range((m + T.int64(255)) // T.int64(256)):
            if ax2_fused_0 * T.int64(256) + threadIdx_x < m:
                in_thread_T_softmax_expsum_shared_1[0] = in_thread_T_softmax_expsum_shared_1[0] + T.exp(lv38_shared_dyn_1[ax2_fused_0 * T.int64(256) + threadIdx_x] - T_softmax_maxelem_shared_1[0])
        cross_thread_T_softmax_expsum_shared_1 = T.Buffer((1,), data=cross_thread_T_softmax_expsum_shared, scope="local")
        with T.attr(T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]), "reduce_scope", T.reinterpret("handle", T.uint64(0))):
            T.tvm_thread_allreduce(T.uint32(1), in_thread_T_softmax_expsum_shared_1[0], T.bool(True), cross_thread_T_softmax_expsum_shared_1[0], threadIdx_x)
        T_softmax_expsum_shared_1 = T.Buffer((T.int64(1),), data=T_softmax_expsum_shared, scope="shared")
        if threadIdx_x == T.int64(0):
            T_softmax_expsum_shared_1[0] = cross_thread_T_softmax_expsum_shared_1[0]
        for ax2_0 in range((m + T.int64(255)) // T.int64(256)):
            if ax2_0 * T.int64(256) + threadIdx_x < m:
                var_compute_intermediate_1 = T.Buffer((n * m * T.int64(32),), "float16", data=var_compute_intermediate.data)
                cse_var_1: T.int64 = ax2_0 * T.int64(256)
                var_compute_intermediate_1[cse_var_1 + blockIdx_x * m + threadIdx_x] = T.Cast("float16", T.exp(lv38_shared_dyn_1[cse_var_1 + threadIdx_x] - T_softmax_maxelem_shared_1[0]) / T_softmax_expsum_shared_1[0])