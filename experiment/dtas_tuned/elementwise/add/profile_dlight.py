import tvm
from tvm import dlight as dl, relax as rx
from tvm import tir
import numpy as np
dtype="int8"
if dtype =="int8":
    byte = 1
elif dtype == "float16":
    byte = 2
elif dtype == "float32":
    byte = 4
    
# bb=rx.BlockBuilder()
# from tvm.relax.op import matmul,add
# from tvm.relax.op.nn import gelu
# from tvm.relax.testing import nn
# hidden_size = 2560   
# bb = rx.BlockBuilder()
# n = tir.Var("n","int64")
# A = rx.Var("A",rx.TensorStructInfo([1, n, hidden_size], dtype)) 
# B = rx.Var("B",rx.TensorStructInfo([1, n, hidden_size], dtype))
# with bb.function("WT_test",[A,B]):
#         with bb.dataflow():
#             lv0 = nn.emit(add(A, B))
#             gv0 = bb.emit_output(lv0)
#         bb.emit_func_output(gv0)
# mod=bb.finalize()
# mod = rx.get_pipeline()(mod)
# print(mod)
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)), dtype)
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)), dtype)
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)), dtype)
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax1, v_ax2], B[v_ax0, v_ax1, v_ax2])
                T.writes(T_add[v_ax0, v_ax1, v_ax2])
                T_add[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2] + B[v_ax0, v_ax1, v_ax2]
mod = Module                
target = tvm.target.Target("nvidia/geforce-rtx-3090")
dev = tvm.cuda(0)
with target:
    mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
                    dl.gpu.Matmul(),
                    dl.gpu.GEMV(),
                    dl.gpu.Reduction(),
                    dl.gpu.GeneralReduction(),
                    dl.gpu.Fallback(),
                )(mod)
# # print(mod)
mod = rx.transform.AttachGlobalSymbol()(mod)
rt_mod = tvm.build(mod, target=target)
timer = rt_mod.time_evaluator(func_name="add", dev=dev, number=10,  min_repeat_ms=50 )

hidden_size=2560
result={}
bandwidth={}
import json
for i in range(1, 4097):
    x = tvm.nd.array(np.random.uniform(0, 1, (1, i, hidden_size)).astype(dtype), dev)
    z = tvm.nd.array(np.random.uniform(0, 1, (1, i, hidden_size)).astype(dtype), dev)
    m = tvm.nd.array(np.random.uniform(0, 1, (1, i, hidden_size)).astype(dtype), dev)
    # rt_mod["gemm"](x, weight, bias, z)
    time = timer(x, z, m).mean
    # print(f"latency: {time} S, {byte}" )
    result[i] = time * 1e6
    bandwidth[i] = i * hidden_size * 3 * byte / (1024*1024*1024)/ time 
    # print(f"bandwidth: {bandwidth} GB/S" )
    del x, z, m
    
def save_to_json(profile, filename):
    import json
    with open(filename, "w") as f:
        f.write(json.dumps(profile, indent=4))


save_to_json(
    result,
    f"/home/weitao/XIAG8XX/profile/dtas_tuned/elementwise/add/dlight/{dtype}/latency.json",
)   
save_to_json(bandwidth, f"/home/weitao/XIAG8XX/profile/dtas_tuned/elementwise/add/dlight/{dtype}/bandwidth.json")