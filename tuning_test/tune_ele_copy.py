import numpy as np
dtype="float32"
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R

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

    @R.function
    def WT_test(A: R.Tensor((1, "n", 2560), dtype="float32")) -> R.Tensor((1, "n", 2560), dtype="float32"):
        n = T.int64()
        cls = Module
        R.func_attr({"tir_var_upper_bound": {"m": 512, "n": 4096}})
        with R.dataflow():
            gv = R.call_tir(cls.copy, (A ), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            R.output(gv)
        return gv
    
from dtas.engine.tuner import Engine
from dtas.arch import RTX3090

mod = Module
engine = Engine(1, parallel_build=True, work_dir="/home/weitao/XIAG8XX/profile/dtas_tuned/elementwise/copy/float32/top1_256/")
import tvm
import time
start_time = time.time()
rt_mod = engine.tune_module(mod, arch=RTX3090(7), range_div_factor = 256)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"tune copy : {elapsed_time} 秒")