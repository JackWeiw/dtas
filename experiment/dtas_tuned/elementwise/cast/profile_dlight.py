import tvm
from tvm import dlight as dl, relax as rx
from tvm import tir
import numpy as np
    
# bb=rx.BlockBuilder()
# from tvm.relax.op import astype
# from tvm.relax.op.nn import gelu
# from tvm.relax.testing import nn
# hidden_size = 2560   
# bb = rx.BlockBuilder()

# n = tir.Var("n","int64")
# A = rx.Var("A",rx.TensorStructInfo([1, n, hidden_size], "float16")) 
# # B = rx.Var("B",rx.TensorStructInfo([1, n, hidden_size], "float32"))
# with bb.function("WT_test",[A]):
#         with bb.dataflow():
#             lv0 = nn.emit(astype(A, "float32"))
#             gv0 = bb.emit_output(lv0)
#         bb.emit_func_output(gv0)
# mod=bb.finalize()
# mod = rx.get_pipeline()(mod)
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def cast(var_A: T.handle, var_compute: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
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

mod = Module
target = tvm.target.Target("nvidia/geforce-rtx-3090")
dev = tvm.cuda(4)
with target:
    mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
                    dl.gpu.Matmul(),
                    dl.gpu.GEMV(),
                    dl.gpu.Reduction(),
                    dl.gpu.GeneralReduction(),
                    dl.gpu.Fallback(),
                )(mod)
mod = rx.transform.AttachGlobalSymbol()(mod)

rt_mod = tvm.build(mod, target=target)
timer = rt_mod.time_evaluator(func_name="cast", dev=dev, number=10,  min_repeat_ms=50 )

hidden_size=2560
result={}
bandwidth={}
import json
for i in range(1, 4097):    
    x = tvm.nd.array(np.random.uniform(0, 1, (1, i, hidden_size)).astype("float16"), dev)
    z = tvm.nd.array(np.random.uniform(0, 1, (1, i, hidden_size)).astype("float32"), dev)
    time = timer(x, z).mean
    print(f"latency: {time} S" )
    result[i] = time * 1e6
    bandwidth[i] = i * hidden_size * 6  / (1024*1024*1024)/ time 
    # print(f"bandwidth: {bandwidth} GB/S" )
    del x
    
def save_to_json(profile, filename):
    import json
    with open(filename, "w") as f:
        f.write(json.dumps(profile, indent=4))


save_to_json(
    result,
    "/home/weitao/XIAG8XX/profile/dtas_tuned/elementwise/cast/dlight/latency.json",
)   
save_to_json(bandwidth, "/home/weitao/XIAG8XX/profile/dtas_tuned/elementwise/cast/dlight/bandwidth.json")