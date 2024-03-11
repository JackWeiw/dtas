from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def fused_layer_norm_cast(p_lv6: T.handle, param_1: T.Buffer((T.int64(2560),), "float32"), param_2: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(2560)))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        A_red_temp_v0 = T.alloc_buffer((T.int64(1), n))
        A_red_temp_v1 = T.alloc_buffer((T.int64(1), n))
        T_layer_norm_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)))
        for ax0, ax1, k2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("A_red_temp"):
                v_ax0, v_ax1, v_k2 = T.axis.remap("SSR", [ax0, ax1, k2])
                T.reads(lv6[v_ax0, v_ax1, v_k2])
                T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
                with T.init():
                    A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                    A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
                v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + lv6[v_ax0, v_ax1, v_k2]
                v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + lv6[v_ax0, v_ax1, v_k2] * lv6[v_ax0, v_ax1, v_k2]
                A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
                A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_layer_norm"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv6[v_ax0, v_ax1, v_ax2], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], param_1[v_ax2], param_2[v_ax2])
                T.writes(T_layer_norm_intermediate[v_ax0, v_ax1, v_ax2])
                T_layer_norm_intermediate[v_ax0, v_ax1, v_ax2] = (lv6[v_ax0, v_ax1, v_ax2] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(0.00039062500000000002) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v_ax2] + param_2[v_ax2]
        for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_layer_norm_intermediate[v_i0, v_i1, v_i2])
                T.writes(compute_intermediate[v_i0, v_i1, v_i2])
                compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", T_layer_norm_intermediate[v_i0, v_i1, v_i2])
                
import tvm
from tvm import dlight as dl, relax as rx
from tvm import tir
import numpy as np  
mod = Module                
target = tvm.target.Target("nvidia/geforce-rtx-3090")
dev = tvm.cuda(7)
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
timer = rt_mod.time_evaluator(func_name="fused_layer_norm_cast", dev=dev, number=10,  min_repeat_ms=50 )
result={}

import json
for n in range(1, 4097):
    x = tvm.nd.array(np.random.uniform(0, 1, (1, n, 2560)).astype("float32"), dev)
    beta = tvm.nd.array(np.random.uniform(0, 1, (2560,)).astype("float32"), dev)
    gamma = tvm.nd.array(np.random.uniform(0, 1, (2560,)).astype("float32"), dev)
    z = tvm.nd.array(np.random.uniform(0, 1, (1, n, 2560)).astype("float16"), dev)
    time = timer(x,beta, gamma, z).mean
    result[n] = time * 1e6
    # print(f"bandwidth: {bandwidth} GB/S" )
    del x, z
    
def save_to_json(profile, filename):``
    import json
    with open(filename, "w") as f:
        f.write(json.dumps(profile, indent=4))


save_to_json(
    result,
    f"/home/weitao/XIAG8XX/profile/dtas_tuned/general_reduction/layernorm/dlight/latency.json",
)   