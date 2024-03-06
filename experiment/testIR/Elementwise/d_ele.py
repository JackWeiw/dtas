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
        for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(A[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.Cast("float32", A[v_i0, v_i1, v_i2])  
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
# mod = tvm.lower(mod)
# mod = tir.transform.Simplify()(mod)
# print(mod)
with target:
    mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
                    dl.gpu.Matmul(),
                    dl.gpu.GEMV(),
                    dl.gpu.Reduction(),
                    dl.gpu.GeneralReduction(),
                    dl.gpu.Fallback(),
                )(mod)
# mod=rx.transform.AttachGlobalSymbol()(mod)    
# lib=tvm.build(mod,target=target)
exe=rx.build(mod, target=target)
vm = rx.VirtualMachine(exe, dev,profile=True)

hidden_size=2560
result={}

# n = 1000
import json
for n in range(256, 257):
    x = tvm.nd.array(np.random.uniform(0, 1, (1,  n, hidden_size)).astype("float16"), dev)
    vm["WT_test"](x)
    # s = vm.module["profile"]("WT_test", x, a, b)
    # dic = json.loads(s)
    # for j in range(len(dic["calls"])):
    #     if dic["calls"][j]["Name"]["string"] == "main":
    #         result[n] = dic["calls"][j]["Duration (us)"]["microseconds"]
    # print(vm.profile("MyT",a))
    del x