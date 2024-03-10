from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def fused_softmax_cast(p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), T.int64(1000), n))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(1000), n), "float16")
        # with T.block("root"):
        T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(1000)))
        T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(1000), n))
        T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(1000)))
        T_softmax_norm_intermediate = T.alloc_buffer((T.int64(1), T.int64(1000), n))
        for i0, i1, k in T.grid(T.int64(1), T.int64(1000), n):
            with T.block("T_softmax_maxelem"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(A[v_i0, v_i1, v_k])
                T.writes(T_softmax_maxelem[v_i0, v_i1])
                with T.init():
                    T_softmax_maxelem[v_i0, v_i1] = T.float32(-3.4028234663852886e+38)
                T_softmax_maxelem[v_i0, v_i1] = T.max(T_softmax_maxelem[v_i0, v_i1], A[v_i0, v_i1, v_k])
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1000), n):
            with T.block("T_softmax_exp"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(A[v_i0, v_i1, v_i2], T_softmax_maxelem[v_i0, v_i1])
                T.writes(T_softmax_exp[v_i0, v_i1, v_i2])
                T_softmax_exp[v_i0, v_i1, v_i2] = T.exp(A[v_i0, v_i1, v_i2] - T_softmax_maxelem[v_i0, v_i1])
        for i0, i1, k in T.grid(T.int64(1), T.int64(1000), n):
            with T.block("T_softmax_expsum"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(T_softmax_exp[v_i0, v_i1, v_k])
                T.writes(T_softmax_expsum[v_i0, v_i1])
                with T.init():
                    T_softmax_expsum[v_i0, v_i1] = T.float32(0)
                T_softmax_expsum[v_i0, v_i1] = T_softmax_expsum[v_i0, v_i1] + T_softmax_exp[v_i0, v_i1, v_k]
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1000), n):
            with T.block("T_softmax_norm"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2], T_softmax_expsum[v_i0, v_i1])
                T.writes(T_softmax_norm_intermediate[v_i0, v_i1, v_i2])
                T.block_attr({"axis": 2})
                T_softmax_norm_intermediate[v_i0, v_i1, v_i2] = T_softmax_exp[v_i0, v_i1, v_i2] / T_softmax_expsum[v_i0, v_i1]
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1000), n):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_softmax_norm_intermediate[v_i0, v_i1, v_i2])
                T.writes(compute_intermediate[v_i0, v_i1, v_i2])
                compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", T_softmax_norm_intermediate[v_i0, v_i1, v_i2])

import tvm
from tvm import dlight as dl, relax as rx
from tvm import tir
import numpy as np  
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
# # print(mod)
mod = rx.transform.AttachGlobalSymbol()(mod)
rt_mod = tvm.build(mod, target=target)
timer = rt_mod.time_evaluator(func_name="fused_softmax_cast", dev=dev, number=10,  min_repeat_ms=50 )
result={}

import json
for n in range(1, 4097):
    x = tvm.nd.array(np.random.uniform(0, 1, (1, 1000, n)).astype("float32"), dev)
    z = tvm.nd.array(np.random.uniform(0, 1, (1, 1000, n)).astype("float16"), dev)
    time = timer(x, z).mean
    result[n] = time * 1e6
    # print(f"bandwidth: {bandwidth} GB/S" )
    del x, z
    
def save_to_json(profile, filename):
    import json
    with open(filename, "w") as f:
        f.write(json.dumps(profile, indent=4))


save_to_json(
    result,
    f"/home/weitao/XIAG8XX/profile/dtas_tuned/general_reduction/softmax/dlight/latency.json",
)   