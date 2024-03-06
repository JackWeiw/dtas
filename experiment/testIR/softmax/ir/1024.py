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
            for ax0, ax1, ax2, ax3_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), (T.max(T.int64(1), (m + T.int64(1023)) // T.int64(1024) * T.int64(1024)) + T.int64(4095) - T.min(T.int64(0), (m + T.int64(1023)) // T.int64(1024) * T.int64(1024) - T.int64(1))) // T.int64(4096)):
                for ax3_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                    for ax3_2 in T.vectorized(T.int64(4)):
                        with T.block("lv38_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax1)
                            v2 = T.axis.spatial(n, ax0_ax1_fused % n + ax2)
                            v3 = T.axis.spatial(m, ax3_0 * T.int64(4096) + ax3_1 * T.int64(4) + ax3_2 + (T.int64(0) - (T.int64(0) - T.min(T.int64(0), (m + T.int64(1023)) // T.int64(1024) * T.int64(1024) - T.int64(1)))) + (T.min(T.int64(0), (m + T.int64(1023)) // T.int64(1024) * T.int64(1024) - T.int64(1)) + T.max(T.int64(0) - T.min(T.int64(0), (m + T.int64(1023)) // T.int64(1024) * T.int64(1024) - T.int64(1)), T.int64(0))))
                            T.where(T.int64(0) <= T.min(T.int64(0), (m + T.int64(1023)) // T.int64(1024) * T.int64(1024) - T.int64(1)) + ((ax3_0 * T.int64(1024) + ax3_1) * T.int64(4) + ax3_2) and T.min(T.int64(0), (m + T.int64(1023)) // T.int64(1024) * T.int64(1024) - T.int64(1)) + ((ax3_0 * T.int64(1024) + ax3_1) * T.int64(4) + ax3_2) < m and (ax3_0 * T.int64(1024) + ax3_1) * T.int64(4) + ax3_2 < T.max(T.int64(1), (m + T.int64(1023)) // T.int64(1024) * T.int64(1024)) - T.min(T.int64(0), (m + T.int64(1023)) // T.int64(1024) * T.int64(1024) - T.int64(1)))
                            T.reads(lv38[v0, v1, v2, v3])
                            T.writes(lv38_shared_dyn[v0, v1, v2, v3])
                            lv38_shared_dyn[v0, v1, v2, v3] = lv38[v0, v1, v2, v3]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(1023)) // T.int64(1024), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_maxelem"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(1024) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(1024) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2])
                            T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv38_shared_dyn[T.int64(0), v0, v1, v2])
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                    for ax2_fused_0 in T.serial((m + T.int64(1023)) // T.int64(1024), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_expsum"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                            v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(1024) + ax2_fused_1)
                            T.where(ax2_fused_0 * T.int64(1024) + ax2_fused_1 < m)
                            T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                            T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                            with T.init():
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                            T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
            for ax2_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax2_0 in T.serial((m + T.int64(1023)) // T.int64(1024), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                        v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                        v2 = T.axis.spatial(m, ax2_0 * T.int64(1024) + ax2_1)
                        T.where(ax2_0 * T.int64(1024) + ax2_1 < m)
                        T.reads(lv38_shared_dyn[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                        var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv38_shared_dyn[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
                        
                        
                        
    @R.function
    def WT_test(A: R.Tensor((1, 32, "n", "m"), dtype="float32")) -> R.Tensor((1, 32, "n", "m"), dtype="float16"):
                n, m = T.int64(), T.int64()
                cls = Module
                with R.dataflow():
                    lv = R.call_tir(cls.main, (A), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
                    gv: R.Tensor((1, 32, n, m), dtype="float16") = lv
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
dev=tvm.cuda(1)
# mod = tvm.lower(mod)
# mod = tir.transform.Simplify()(mod)
# print(mod)
# with target:
#     mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
#                     dl.gpu.Matmul(),
#                     dl.gpu.GEMV(),
#                     dl.gpu.Reduction(),
#                     dl.gpu.GeneralReduction(),
#                     dl.gpu.Fallback(),
#                 )(mod)
# mod=rx.transform.AttachGlobalSymbol()(mod)    
# lib=tvm.build(mod,target=target)
exe=rx.build(mod, target=target)
vm = rx.VirtualMachine(exe, dev,profile=True)

hidden_size = 400
result={}


 
# x = tvm.nd.array(np.random.uniform(0, 1, (1, 32, hidden_size, hidden_size)).astype("float32"), dev)
# vm["WT_test"](x)
# print(vm.profile("WT_test",x))
   
def save_to_json(profile, filename):
    import json

    with open(filename, "w") as f:
        f.write(json.dumps(profile, indent=4))

import json
for m in range(1, 4097):
    x = tvm.nd.array(np.random.uniform(0, 1, (1, 32, hidden_size, m)).astype("float32"), dev)
    # vm["WT_test"](x)
    s = vm.module["profile"]("WT_test", x)
    dic = json.loads(s)
    for j in range(len(dic["calls"])):
        if dic["calls"][j]["Name"]["string"] == "main":
            result[m] = dic["calls"][j]["Duration (us)"]["microseconds"]
    # print(vm.profile("MyT",a))
    del x
    
save_to_json(
    result,
    "/home/weitao/XIAG8XX/profile/testIR/softmax/data/32*400/1024.json",
)                       