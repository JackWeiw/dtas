from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
dtype="float16" 
@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_compute: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)), "float16")
        compute = T.match_buffer(var_compute, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(2560), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in range((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440)):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
                            
                            
    @R.function
    def WT_test(A: R.Tensor((1, "n", 2560), dtype="float16")) -> R.Tensor((1, "n", 2560), dtype="float32"):
                n = T.int64()
                cls = Module
                with R.dataflow():
                    lv = R.call_tir(cls.main, (A), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
                    gv: R.Tensor((1, n, 2560), dtype="float32") = lv
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
dev=tvm.cuda(7)
exe=rx.build(mod, target=target)
vm = rx.VirtualMachine(exe, dev,profile=True)

hidden_size=2560
result={}

# n = 1000
import json
for n in range(256, 257):
    x = tvm.nd.array(np.random.uniform(0, 1, (1,  n, hidden_size)).astype("float16"), dev)
    vm["WT_test"](x)
    s = vm.module["profile"]("WT_test", x)
    dic = json.loads(s)
    for j in range(len(dic["calls"])):
        if dic["calls"][j]["Name"]["string"] == "main":
            result[n] = dic["calls"][j]["Duration (us)"]["microseconds"]
    print(vm.profile("WT_test",x))
    del x
# vm["WT_test"](x, a, b)
# print(vm.profile("WT_test",x, a, b))
# def save_to_json(profile, filename):
#     import json

#     with open(filename, "w") as f:
#         f.write(json.dumps(profile, indent=4))


# save_to_json(
#     result,
#     "/home/weitao/XIAG8XX/profile/testIR/Elementwise/data/2560.json",
# )               