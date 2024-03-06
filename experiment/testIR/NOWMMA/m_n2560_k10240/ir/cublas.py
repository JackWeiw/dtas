import numpy as np
import tvm
from tvm import tir
from tvm import relax
from tvm.relax.backend import get_patterns_with_prefix
from tvm.relax.backend.contrib.cutlass import annotate_workspace
import tvm.relax.backend.contrib.cublas as _
import tvm.relax as rx

dtype="int8"
bb=rx.BlockBuilder()

from tvm.relax.op import matmul,add
from tvm.relax.op.nn import gelu
from tvm.relax.testing import nn
hidden_size = 2560   
bb = rx.BlockBuilder()

n = tir.Var("n","int64")
A = rx.Var("A",rx.TensorStructInfo([1, n, 4*hidden_size], dtype)) 
B = rx.Var("B",rx.TensorStructInfo([hidden_size, 4*hidden_size], dtype))
# V = rx.Var("V",rx.TensorStructInfo([hidden_size], dtype))

with bb.function("WT_test",[A, B]):
        with bb.dataflow():
            lv0 = rx.op.permute_dims(B,[1,0])
            lv1 = nn.emit(matmul(A,lv0))
            gv0 = bb.emit_output(lv1)
        bb.emit_func_output(gv0)
mod=bb.get()

# mod=rx.get_pipeline()(mod)
# print(mod)

sm = 86
patterns = []
has_cublas = tvm.get_global_func("relax.ext.cublas", True)

if has_cublas :

    # patterns += get_patterns_with_prefix("cutlass.attention")
    # # print(patterns)
    # patterns += get_patterns_with_prefix("cutlass.layer_norm")
    # # print(patterns)
    # patterns += get_patterns_with_prefix("cutlass.rms_norm")
    # print(patterns)
    patterns += get_patterns_with_prefix("cublas")
    
model_names = ["WT_test"]

mod = tvm.transform.Sequential(
                [
                    relax.transform.FuseOpsByPattern(
                        patterns, bind_constants=False, annotate_codegen=True
                    ),
                    annotate_workspace,
                    relax.transform.AllocateWorkspace(),
                    relax.transform.RunCodegen(
                        {"cutlass": {"sm": sm, "find_first_valid": False}},
                        entry_functions=model_names,
                    ),
                ]
            )(mod)
print(mod)

target = tvm.target.Target("nvidia/geforce-rtx-3090")
dev = tvm.cuda(7)

exe = rx.build(mod, target)
vm = rx.VirtualMachine(exe, dev , profile=True)

result={}
# m=[1024,1536,2048,2560,3072,3584,4096]
import json
for i in range(4096,4097):    
    x = tvm.nd.array(np.random.uniform(0, 1, (1, i, 4*hidden_size)).astype(dtype), dev)
    weight = tvm.nd.array(np.random.uniform(0, 1,(hidden_size, 4*hidden_size)).astype(dtype), dev)
    # bias = tvm.nd.array(np.random.uniform(0, 1,(hidden_size,)).astype(dtype), dev)
    # vm["WT_test"](x, weight, bias)
    # print(print(vm.profile("WT_test",x,weight,bias)))
    s=vm.module["profile"]("WT_test",x, weight)
    dic=json.loads(s)
    print(s)
    for j in range(len(dic["calls"])):
        if dic["calls"][j]["Name"]["string"]=="fused_relax_permute_dims_relax_matmul_cublas":
           result[i]=(dic["calls"][j]["Duration (us)"]["microseconds"])
    # print(vm.profile("MyT",a))
    del x,weight

def save_to_json(profile, filename):
    import json
    with open(filename, "w") as f:
        f.write(json.dumps(profile,indent=4))
        
# save_to_json(result,"/home/weitao/XIAG8XX/profile/testIR/NOWMMA/m_n2560_k10240/data/cublas.json")