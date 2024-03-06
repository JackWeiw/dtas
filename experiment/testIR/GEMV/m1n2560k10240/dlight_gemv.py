from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
dtype="float16"
@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv9: T.handle, lv3: T.Buffer((2560, 10240), dtype), lv4: T.Buffer((2560,), dtype), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        lv9 = T.match_buffer(p_lv9, (1, 1, 10240), dtype)
        var_T_add_intermediate = T.match_buffer(p_output0, (1, 1, 2560), dtype)
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((1, 1, 2560), dtype)
        for i0, i1, i2, k in T.grid(1, 1, 2560, 10240):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv9[v_i0, v_i1, v_k], lv3[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv9[v_i0, v_i1, v_k] * lv3[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(1, 1, 2560):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], lv4[v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + lv4[v_ax2]  
                  
    @R.function
    def WT_test(A: R.Tensor((1, 1, 10240), dtype=dtype), w_q: R.Tensor((2560, 10240), dtype=dtype), bias_q: R.Tensor((2560,), dtype=dtype)) -> R.Tensor((1, 1, 2560), dtype=dtype):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.main, (A, w_q, bias_q), out_sinfo=R.Tensor((1, 1, 2560), dtype=dtype))
            gv: R.Tensor((1, 1, 2560), dtype=dtype) = lv
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

with target:
    mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
                    dl.gpu.Matmul(),
                    dl.gpu.GEMV(),
                    dl.gpu.Reduction(),
                    dl.gpu.GeneralReduction(),
                    dl.gpu.Fallback(),
                )(mod)
    
print(mod)
mod=rx.transform.AttachGlobalSymbol()(mod)    
# lib=tvm.build(mod,target=target)
exe=rx.build(mod, target=target)
vm = rx.VirtualMachine(exe, dev,profile=True)

hidden_size=2560
result={}
import json
# m=[128,300,512,768,1024,2048,3584,4096]
   
x = tvm.nd.array(np.random.uniform(0, 1, (1, 1, 4*hidden_size)).astype(dtype), dev)
weight = tvm.nd.array(np.random.uniform(0, 1,(hidden_size, 4*hidden_size)).astype(dtype), dev)
bias = tvm.nd.array(np.random.uniform(0, 1,(hidden_size,)).astype(dtype), dev)
# z=tvm.nd.array(np.random.uniform(1, 100, (1, i, hidden_size)).astype(dtype), dev)
# lib["fused_NT_matmul4_add1"](x,weight,bias,z)
# print("here")
# vm["WT_test"](x, weight, bias)
# print("zzz")
# print(print(vm.profile("WT_test",x,weight,bias)))
# s=vm.module["profile"]("WT_test",x, weight, bias)
# vm["WT_test"](x, weight, bias)
print(vm.profile("WT_test",x, weight, bias))
# dic=json.loads(s)

# for j in range(len(dic["calls"])):
#     if dic["calls"][j]["Name"]["string"]=="main":
#         result[i]=(dic["calls"][j]["Duration (us)"]["microseconds"])
# print(vm.profile("MyT",a))
del x,weight,bias

def save_to_json(profile, filename):
    import json
    with open(filename, "w") as f:
        f.write(json.dumps(profile,indent=4))
        
# save_to_json(result,"./data/16x16x16/dlight.json")