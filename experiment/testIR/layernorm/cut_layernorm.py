import numpy as np
import tvm
from tvm import tir
from tvm import relax
from tvm.relax.backend import get_patterns_with_prefix
from tvm.relax.backend.contrib.cutlass import annotate_workspace
import tvm.relax.backend.contrib.cublas as _
import tvm.relax.backend.contrib.cutlass as _
import tvm.relax as rx

dtype="float32"
bb=rx.BlockBuilder()

from tvm.relax.op import matmul,add
from tvm.relax.op.nn import layer_norm
from tvm.relax.testing import nn
hidden_size = 2560   
bb = rx.BlockBuilder()

n = tir.Var("n", "int64")
m = tir.Var("m", "int64")
A = rx.Var("A", rx.TensorStructInfo([1, n, hidden_size], dtype)) 
gama = rx.Var("gama", rx.TensorStructInfo([hidden_size], dtype))
beta = rx.Var("beta", rx.TensorStructInfo([hidden_size], dtype))

with bb.function("WT_test",[A, gama, beta]):
        with bb.dataflow():
            lv0 = nn.emit(layer_norm(A))
            gv0 = bb.emit_output(lv0)
        bb.emit_func_output(gv0)
mod=bb.finalize()
model_names = ["WT_test"]
sm = 86
patterns = []
has_cutlass = tvm.get_global_func("relax.ext.cutlass", True)

if has_cutlass :
    print("here")
    patterns += get_patterns_with_prefix("cutlass.layer_norm")

mod = tvm.transform.Sequential(
        [
            relax.transform.FuseOpsByPattern(
                patterns, bind_constants=False, annotate_codegen=True
            ),
            annotate_workspace,
            relax.transform.AllocateWorkspace(),
            relax.transform.RunCodegen(
                {"cutlass": {"sm": sm, "find_first_valid": False}},
                entry_functions= model_names,
            ),
        ]
    )(mod)

target = tvm.target.Target("nvidia/geforce-rtx-3090")
dev = tvm.cuda(6)

exe = rx.build(mod, target)
vm = rx.VirtualMachine(exe, dev , profile=True)

result={}
# m=[1024,1536,2048,2560,3072,3584,4096]
import json
for i in range(1):    
    x = tvm.nd.array(np.random.uniform(0, 1, (1, 1, 4*hidden_size)).astype(dtype), dev)
    gama = tvm.nd.array(np.random.uniform(0, 1,(hidden_size,)).astype(dtype), dev)
    beta = tvm.nd.array(np.random.uniform(0, 1,(hidden_size,)).astype(dtype), dev)
    # vm["WT_test"](x, weight, bias)
    print(print(vm.profile("WT_test",x, gama, beta)))
    # s=vm.module["profile"]("WT_test",x, gama, beta)
    # dic=json.loads(s)

    # for j in range(len(dic["calls"])):
    #     if dic["calls"][j]["Name"]["string"]=="fused_relax_permute_dims_relax_matmul_relax_add_cublas":
    #        result[i]=(dic["calls"][j]["Duration (us)"]["microseconds"])
    # # print(vm.profile("MyT",a))
    # del x,weight,bias

# def save_to_json(profile, filename):
#     import json
#     with open(filename, "w") as f:
#         f.write(json.dumps(profile,indent=4))
        
# save_to_json(result,"/home/weitao/XIAG8XX/profile/data/m_n7680_k2560/cublas.json")