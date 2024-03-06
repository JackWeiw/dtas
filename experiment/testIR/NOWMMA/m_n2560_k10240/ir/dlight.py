from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
dtype="int8"
@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv55: T.handle, lv11: T.Buffer((T.int64(2560), T.int64(10240)), "int8"), lv12: T.Buffer((T.int64(2560),), "int8"), p_output0: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            n = T.int64()
            lv55 = T.match_buffer(p_lv55, (T.int64(1), n, T.int64(10240)), "int8")
            var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "int8")
            # with T.block("root"):
            var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "int8")
            for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(2560), T.int64(10240)):
                with T.block("NT_matmul"):
                    v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                    T.reads(lv55[v_i0, v_i1, v_k], lv11[v_i2, v_k])
                    T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                    with T.init():
                        var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.int8(0)
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv55[v_i0, v_i1, v_k] * lv11[v_i2, v_k]
            for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
                with T.block("T_add"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], lv12[v_ax2])
                    T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                    var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + lv12[v_ax2]
    # @T.prim_func(private=True)
    # def main(p_lv55: T.handle, lv11: T.Buffer((T.int64(10240), T.int64(2560)), "int8"), lv12: T.Buffer((T.int64(10240),), "int8"), p_output0: T.handle):
    #         T.func_attr({"tir.noalias": T.bool(True)})
    #         n = T.int64()
    #         lv55 = T.match_buffer(p_lv55, (T.int64(1), n, T.int64(2560)), "int8")
    #         var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(10240)), "int8")
    #         # with T.block("root"):
    #         var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(10240)), "int8")
    #         for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(10240), T.int64(2560)):
    #             with T.block("NT_matmul"):
    #                 v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
    #                 T.reads(lv55[v_i0, v_i1, v_k], lv11[v_i2, v_k])
    #                 T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
    #                 with T.init():
    #                     var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.int8(0)
    #                 var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv55[v_i0, v_i1, v_k] * lv11[v_i2, v_k]
    #         for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(10240)):
    #             with T.block("T_add"):
    #                 v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
    #                 T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], lv12[v_ax2])
    #                 T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
    #                 var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + lv12[v_ax2]
    
    # @T.prim_func(private=True)
    # def main(p_lv9: T.handle, lv3: T.Buffer((T.int64(7680), T.int64(2560)), "int8"), lv4: T.Buffer((T.int64(7680),), "int8"), p_output0: T.handle):
    #     T.func_attr({"tir.noalias": T.bool(True)})
    #     n = T.int64()
    #     lv9 = T.match_buffer(p_lv9, (T.int64(1), n, T.int64(2560)), "int8")
    #     var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(7680)), "int8")
    #     # with T.block("root"):
    #     var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(7680)), "int8")
    #     for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(7680), T.int64(2560)):
    #         with T.block("NT_matmul"):
    #             v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
    #             T.reads(lv9[v_i0, v_i1, v_k], lv3[v_i2, v_k])
    #             T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
    #             with T.init():
    #                 var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.int8(0)
    #             var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv9[v_i0, v_i1, v_k] * lv3[v_i2, v_k]
    #     for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(7680)):
    #         with T.block("T_add"):
    #             v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
    #             T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], lv4[v_ax2])
    #             T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
    #             var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + lv4[v_ax2]
                    

mod=Module
import tvm
from tvm import dlight as dl
import numpy as np
import tvm.relax as rx
# mod=rx.get_pipeline()(mod)
target = tvm.target.Target("nvidia/geforce-rtx-3090")
dtype="int8"
dev=tvm.cuda(7)
with target:
    mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
                    dl.gpu.Matmul(),
                    dl.gpu.GEMV(),
                    dl.gpu.Reduction(),
                    dl.gpu.GeneralReduction(),
                    dl.gpu.Fallback(),
                )(mod)

mod=rx.transform.AttachGlobalSymbol()(mod)
print(mod)
with tvm.transform.PassContext(config={ "tir.use_async_copy": True ,}):
    lib=tvm.build(mod,target=target)

hidden_size = 2560
result = {}
timer = lib.time_evaluator("main", dev, number=5)

# m=[128,300,512,768,1024,2048,3584,4096]
for i in range(1, 4097):
    x = tvm.nd.array(np.random.uniform(0, 1, (1, i, 4*hidden_size)).astype(dtype), dev)
    weight = tvm.nd.array(
        np.random.uniform(0, 1, (hidden_size, 4*hidden_size)).astype(dtype), dev
    )
    bias = tvm.nd.array(np.random.uniform(0, 1, (hidden_size,)).astype(dtype), dev)
    z = tvm.nd.array(np.random.uniform(0, 1, (1, i, hidden_size)).astype(dtype), dev)
    result[i] = timer(x, weight, bias, z).mean*1e6
    # print(vm.profile("MyT",a))
    del x, weight, bias

def save_to_json(profile, filename):
    import json
    with open(filename, "w") as f:
        f.write(json.dumps(profile,indent=4))
        
save_to_json(result,"../data/dlight.json")