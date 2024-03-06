from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
dtype="float16"  
@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv6: T.handle, lv1: T.Buffer((T.int64(2560),), "float32"), lv2: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(2560)))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
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
                        T.reads(lv6[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], lv1[v1], lv2[v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1])
                        var_compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * lv1[v1] + lv2[v1])
                        
                        
mod=Module
import tvm
from tvm import relax as rx
target = tvm.target.Target("nvidia/geforce-rtx-3090")
dev=tvm.cuda(0)
mod=rx.transform.AttachGlobalSymbol()(mod) 
lib=tvm.build(mod,target=target)                       