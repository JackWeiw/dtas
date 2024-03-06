from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
dtype="float16"   
@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv6: T.handle, lv1: T.Buffer((T.int64(2560),), "float32"), lv2: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(2560)))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        A_red_temp_v0_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        A_red_temp_v1_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        for ax0_fused in T.thread_binding(n, thread="blockIdx.x"):
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6[T.int64(0), v0, v1] * lv6[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(288) + ax1_1)
                        T.where(ax1_0 * T.int64(288) + ax1_1 < T.int64(2560))
                        T.reads(lv6[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], lv1[v1], lv2[v1])
                        T.writes(var_compute_intermediate[T.int64(0), v0, v1])
                        var_compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * lv1[v1] + lv2[v1])
                        
                        
                        
    @R.function
    def WT_test(A: R.Tensor((1, "n", 2560), dtype="float32"),lv1: R.Tensor((2560,), "float32"), lv2: R.Tensor((2560,), "float32")) -> R.Tensor((1, "n", 2560), dtype="float16"):
                n = T.int64()
                cls = Module
                with R.dataflow():
                    lv = R.call_tir(cls.main, (A, lv1, lv2), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
                    gv: R.Tensor((1, n, 2560), dtype="float16") = lv
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
dev=tvm.cuda(2)
exe=rx.build(mod, target=target)
vm = rx.VirtualMachine(exe, dev,profile=True)

hidden_size=2560
result={}

# n = 1000
import json
for n in range(1, 4097):
    x = tvm.nd.array(np.random.uniform(0, 1, (1,  n, hidden_size)).astype("float32"), dev)
    a = tvm.nd.array(np.random.uniform(0, 1, ( hidden_size,)).astype("float32"), dev)
    b = tvm.nd.array(np.random.uniform(0, 1, ( hidden_size,)).astype("float32"), dev)
    vm["WT_test"](x, a, b)
    s = vm.module["profile"]("WT_test", x, a, b)
    dic = json.loads(s)
    for j in range(len(dic["calls"])):
        if dic["calls"][j]["Name"]["string"] == "main":
            result[n] = dic["calls"][j]["Duration (us)"]["microseconds"]
    # print(vm.profile("MyT",a))
    del x, a, b
# vm["WT_test"](x, a, b)
# print(vm.profile("WT_test",x, a, b))
def save_to_json(profile, filename):
    import json

    with open(filename, "w") as f:
        f.write(json.dumps(profile, indent=4))


save_to_json(
    result,
    "/home/weitao/XIAG8XX/profile/testIR/layernorm/data/288.json",
)                           