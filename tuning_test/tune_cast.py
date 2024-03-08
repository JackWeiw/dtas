from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R

dtype = "float16"

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def cast(var_A: T.handle, var_compute: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
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
        
    @R.function
    def WT_test(A: R.Tensor((1, "n", 2560), dtype="float16")) -> R.Tensor((1, "n", 2560), dtype="float32"):
                n = T.int64()
                R.func_attr({"tir_var_upper_bound": {"m": 512, "n": 4096}})
                cls = Module
                with R.dataflow():
                    lv = R.call_tir(cls.cast, (A, ), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
                    gv: R.Tensor((1, n, 2560), dtype="float32") = lv
                    R.output(gv)
                return gv
            
                
from dtas.engine.tuner import Engine
from dtas.arch import RTX3090

mod = Module
engine = Engine(1, parallel_build=True, work_dir="/home/weitao/XIAG8XX/profile/dtas_tuned/cast/top1_256/")
import tvm
import time
start_time = time.time()
rt_mod = engine.tune_module(mod, arch=RTX3090(7), range_div_factor = 256)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"tune cast : {elapsed_time} 秒")
# timer_cast=rt_mod.time_evaluator(func_name="cast", dev=tvm.cuda(7), number=5,  min_repeat_ms=50 )
# timer_layernorm = rt_mod.time_evaluator(func_name="fused_layer_norm_cast1", dev=tvm.cuda(7), number=5,  min_repeat_ms=50 )
# import numpy as np
# hidden_size = 2560
# dtype = "float16"
# dev = tvm.cuda(7)
# x = tvm.nd.array(np.random.uniform(0, 1, (1, 100, hidden_size)).astype(dtype), dev)
# y = tvm.nd.array(np.random.uniform(0, 1, (1, 100, hidden_size)).astype(dtype), dev)
# beta = tvm.nd.array(np.random.uniform(0, 1, (hidden_size,)).astype("float32"), dev)
# gama = tvm.nd.array(np.random.uniform(0, 1, (hidden_size,)).astype("float32"), dev)
# rt_mod["cast"](x, y)
# print(timer_cast(x,y))
# print(timer_layernorm(x,beta, gama, y))