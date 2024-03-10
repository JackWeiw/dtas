import tvm
from tvm import dlight as dl, relax as rx, tir
import numpy as np
    
hidden_size = 2560   
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
dtype = "float32"

if dtype =="int8":
    byte = 1
elif dtype == "float16":
    byte = 2
elif dtype == "float32":
    byte = 4
    
@I.ir_module
class Module:
    @T.prim_func(private=True)
    def copy(var_A: T.handle, var_B: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)), "float32")
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)), "float32")
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax1, v_ax2], )
                T.writes(B[v_ax0, v_ax1, v_ax2])
                B[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2] 

mod = Module    


target = tvm.target.Target("nvidia/geforce-rtx-3090")
dev = tvm.cuda(5)
with target:
    mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
                    dl.gpu.Matmul(),
                    dl.gpu.GEMV(),
                    dl.gpu.Reduction(),
                    dl.gpu.GeneralReduction(),
                    dl.gpu.Fallback(),
                )(mod)
mod = rx.transform.AttachGlobalSymbol()(mod)
# print(mod)
rt_mod = tvm.build(mod, target=target)
timer = rt_mod.time_evaluator(func_name="copy", dev=dev, number=10,  min_repeat_ms=50 )

hidden_size=2560
result={}
bandwidth={}
import json
for i in range(1, 4097):    
    x = tvm.nd.array(np.random.uniform(0, 1, (1, i, hidden_size)).astype("float32"), dev)
    z = tvm.nd.array(np.random.uniform(0, 1, (1, i, hidden_size)).astype("float32"), dev)
    time = timer(x, z).mean
    # print(f"latency: {time} S, {byte}" )
    result[i] = time * 1e6
    bandwidth[i] = i * hidden_size * 2 * byte / (1024*1024*1024)/ time 
    # print(f"bandwidth: {bandwidth} GB/S" )
    del x, z
    
def save_to_json(profile, filename):
    import json
    with open(filename, "w") as f:
        f.write(json.dumps(profile, indent=4))


save_to_json(
    result,
    "/home/weitao/XIAG8XX/profile/dtas_tuned/elementwise/copy/dlight/float32/latency.json",
)   
save_to_json(bandwidth, "/home/weitao/XIAG8XX/profile/dtas_tuned/elementwise/copy/dlight/float32/bandwidth.json")