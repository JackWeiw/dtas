import numpy as np
dtype="int8"
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)), "int8")
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)), "int8")
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)), "int8")
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax1, v_ax2], B[v_ax0, v_ax1, v_ax2])
                T.writes(T_add[v_ax0, v_ax1, v_ax2])
                T_add[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2] + B[v_ax0, v_ax1, v_ax2]

    @R.function
    def WT_test(A: R.Tensor((1, "n", 2560), dtype="int8"), B: R.Tensor((1, "n", 2560), dtype="int8")) -> R.Tensor((1, "n", 2560), dtype="int8"):
        n = T.int64()
        cls = Module
        R.func_attr({"tir_var_upper_bound": {"m": 512, "n": 4096}})
        with R.dataflow():
            gv = R.call_tir(cls.add, (A, B), out_sinfo=R.Tensor((1, n, 2560), dtype="int8"))
            R.output(gv)
        return gv

# from dtas.engine.tuner import Engine
# from dtas.arch import RTX3090

mod = Module
# engine = Engine(10, parallel_build=True, work_dir="/home/weitao/XIAG8XX/profile/dtas_tuned/elementwise/add/int8/top10_256/")
import tvm
# import time
# start_time = time.time()
# rt_mod = engine.tune_module(mod, arch=RTX3090(7), range_div_factor = 256)
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"tune cast : {elapsed_time} 秒")
print(tvm.lower(mod["add"]))