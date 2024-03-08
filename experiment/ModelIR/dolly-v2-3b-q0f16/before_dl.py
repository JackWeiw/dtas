from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def NT_matmul5(A: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), B: T.Buffer((T.int64(50280), T.int64(2560)), "float32"), NT_matmul: T.Buffer((T.int64(1), T.int64(1), T.int64(50280)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(50280), T.int64(2560)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_k], B[v_i2, v_k])
                T.writes(NT_matmul[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul[v_i0, v_i1, v_i2] = T.float32(0)
                NT_matmul[v_i0, v_i1, v_i2] = NT_matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i2, v_k]

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

    @T.prim_func(private=True)
    def cast4(A: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), compute: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(A[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = A[v_i0, v_i1, v_i2]

    @T.prim_func(private=True)
    def cast5(A: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), compute: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(A[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.Cast("float32", A[v_i0, v_i1, v_i2])

    @T.prim_func(private=True)
    def divide2(A: T.Buffer((T.int64(1), T.int64(1), T.int64(50280)), "float32"), B: T.Buffer((), "float32"), T_divide: T.Buffer((T.int64(1), T.int64(1), T.int64(50280)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(50280)):
            with T.block("T_divide"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax1, v_ax2], B[()])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                T_divide[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2] / B[()]

    @T.prim_func(private=True)
    def extend_te(var_A: T.handle, var_concat_te: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(1), n, n), "float16")
        m = T.int64()
        concat_te = T.match_buffer(var_concat_te, (T.int64(1), T.int64(1), n, m), "float16")
        # with T.block("root"):
        for b, _, i, j in T.grid(T.int64(1), T.int64(1), n, m):
            with T.block("concat_te"):
                v_b, v__, v_i, v_j = T.axis.remap("SSSS", [b, _, i, j])
                T.reads(A[v_b, v__, v_i, v_j + n - m])
                T.writes(concat_te[v_b, v__, v_i, v_j])
                concat_te[v_b, v__, v_i, v_j] = T.if_then_else(v_j < m - n, T.float16(65504), A[v_b, v__, v_i, v_j + n - m])

    @T.prim_func(private=True)
    def full(var_T_full: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        m = T.int64()
        T_full = T.match_buffer(var_T_full, (T.int64(1), T.int64(1), T.int64(1), m), "float16")
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(1), m):
            with T.block("T_full"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads()
                T.writes(T_full[v_ax0, v_ax1, v_ax2, v_ax3])
                T_full[v_ax0, v_ax1, v_ax2, v_ax3] = T.float16(65504)

    @T.prim_func(private=True)
    def fused_NT_matmul10_add5(lv1825: T.Buffer((T.int64(1), T.int64(1), T.int64(10240)), "float16"), param_1110: T.Buffer((T.int64(2560), T.int64(10240)), "float16"), param_1210: T.Buffer((T.int64(2560),), "float16"), T_add_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(2560), T.int64(10240)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv1825[v_i0, v_i1, v_k], param_1110[v_i2, v_k])
                T.writes(NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                NT_matmul_intermediate[v_i0, v_i1, v_i2] = NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv1825[v_i0, v_i1, v_k] * param_1110[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], param_1210[v_ax2])
                T.writes(T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T_add_intermediate[v_ax0, v_ax1, v_ax2] = NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + param_1210[v_ax2]

    @T.prim_func(private=True)
    def fused_NT_matmul1_divide_maximum_minimum_cast2(p_lv30: T.handle, p_lv31: T.handle, p_lv5: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv30 = T.match_buffer(p_lv30, (T.int64(1), T.int64(32), n, T.int64(80)), "float16")
        m = T.int64()
        lv31 = T.match_buffer(p_lv31, (T.int64(1), T.int64(32), m, T.int64(80)), "float16")
        lv5 = T.match_buffer(p_lv5, (T.int64(1), T.int64(1), n, m), "float16")
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m))
        # with T.block("root"):
        NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m), "float16")
        T_divide_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m), "float16")
        T_maximum_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m), "float16")
        T_minimum_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m), "float16")
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), n, m, T.int64(80)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(lv30[v_i0, v_i1, v_i2, v_k], lv31[v_i0, v_i1, v_i3, v_k])
                T.writes(NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + lv30[v_i0, v_i1, v_i2, v_k] * lv31[v_i0, v_i1, v_i3, v_k]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, m):
            with T.block("T_divide"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * T.float16(0.11179039301310044)
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, m):
            with T.block("T_maximum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float16(-65504))
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, m):
            with T.block("T_minimum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv5[v_ax0, T.int64(0), v_ax2, v_ax3])
                T.writes(T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.min(T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv5[v_ax0, T.int64(0), v_ax2, v_ax3])
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
            with T.block("compute"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_minimum_intermediate[v_i0, v_i1, v_i2, v_i3])
                T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Cast("float32", T_minimum_intermediate[v_i0, v_i1, v_i2, v_i3])

    @T.prim_func(private=True)
    def fused_NT_matmul2_add1_add3_add3(p_lv43: T.handle, param_7: T.Buffer((T.int64(2560), T.int64(2560)), "float16"), param_8: T.Buffer((T.int64(2560),), "float16"), p_lv58: T.handle, p_lv2: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv43 = T.match_buffer(p_lv43, (T.int64(1), n, T.int64(2560)), "float16")
        lv58 = T.match_buffer(p_lv58, (T.int64(1), n, T.int64(2560)), "float16")
        lv2 = T.match_buffer(p_lv2, (T.int64(1), n, T.int64(2560)), "float16")
        T_add_intermediate_1_2 = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
        T_add_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
        T_add_intermediate_1 = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(2560), T.int64(2560)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv43[v_i0, v_i1, v_k], param_7[v_i2, v_k])
                T.writes(NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                NT_matmul_intermediate[v_i0, v_i1, v_i2] = NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv43[v_i0, v_i1, v_k] * param_7[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], param_8[v_ax2])
                T.writes(T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T_add_intermediate[v_ax0, v_ax1, v_ax2] = NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + param_8[v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv58[v_ax0, v_ax1, v_ax2], T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
                T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = lv58[v_ax0, v_ax1, v_ax2] + T_add_intermediate[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_add_2"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_add_intermediate_1[v_ax0, v_ax1, v_ax2], lv2[v_ax0, v_ax1, v_ax2])
                T.writes(T_add_intermediate_1_2[v_ax0, v_ax1, v_ax2])
                T_add_intermediate_1_2[v_ax0, v_ax1, v_ax2] = T_add_intermediate_1[v_ax0, v_ax1, v_ax2] + lv2[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_NT_matmul2_add1_add3_add3_cast(p_lv1748: T.handle, param_379: T.Buffer((T.int64(2560), T.int64(2560)), "float16"), param_380: T.Buffer((T.int64(2560),), "float16"), p_lv1763: T.handle, p_lv1710: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv1748 = T.match_buffer(p_lv1748, (T.int64(1), n, T.int64(2560)), "float16")
        lv1763 = T.match_buffer(p_lv1763, (T.int64(1), n, T.int64(2560)), "float16")
        lv1710 = T.match_buffer(p_lv1710, (T.int64(1), n, T.int64(2560)), "float16")
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
        T_add_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
        T_add_intermediate_1 = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
        T_add_intermediate_1_2 = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(2560), T.int64(2560)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv1748[v_i0, v_i1, v_k], param_379[v_i2, v_k])
                T.writes(NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                NT_matmul_intermediate[v_i0, v_i1, v_i2] = NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv1748[v_i0, v_i1, v_k] * param_379[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], param_380[v_ax2])
                T.writes(T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T_add_intermediate[v_ax0, v_ax1, v_ax2] = NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + param_380[v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv1763[v_ax0, v_ax1, v_ax2], T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
                T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = lv1763[v_ax0, v_ax1, v_ax2] + T_add_intermediate[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_add_2"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_add_intermediate_1[v_ax0, v_ax1, v_ax2], lv1710[v_ax0, v_ax1, v_ax2])
                T.writes(T_add_intermediate_1_2[v_ax0, v_ax1, v_ax2])
                T_add_intermediate_1_2[v_ax0, v_ax1, v_ax2] = T_add_intermediate_1[v_ax0, v_ax1, v_ax2] + lv1710[v_ax0, v_ax1, v_ax2]
        for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_add_intermediate_1_2[v_i0, v_i1, v_i2])
                T.writes(compute_intermediate[v_i0, v_i1, v_i2])
                compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float32", T_add_intermediate_1_2[v_i0, v_i1, v_i2])

    @T.prim_func(private=True)
    def fused_NT_matmul3_add2_gelu(p_lv50: T.handle, param_9: T.Buffer((T.int64(10240), T.int64(2560)), "float16"), param_10: T.Buffer((T.int64(10240),), "float16"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv50 = T.match_buffer(p_lv50, (T.int64(1), n, T.int64(2560)), "float16")
        T_multiply_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(10240)), "float16")
        # with T.block("root"):
        NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(10240)), "float16")
        T_add_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(10240)), "float16")
        T_multiply = T.alloc_buffer((T.int64(1), n, T.int64(10240)), "float16")
        compute = T.alloc_buffer((T.int64(1), n, T.int64(10240)))
        compute_1 = T.alloc_buffer((T.int64(1), n, T.int64(10240)))
        compute_2 = T.alloc_buffer((T.int64(1), n, T.int64(10240)), "float16")
        T_multiply_1 = T.alloc_buffer((T.int64(1), n, T.int64(10240)), "float16")
        T_add = T.alloc_buffer((T.int64(1), n, T.int64(10240)), "float16")
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(10240), T.int64(2560)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv50[v_i0, v_i1, v_k], param_9[v_i2, v_k])
                T.writes(NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                NT_matmul_intermediate[v_i0, v_i1, v_i2] = NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv50[v_i0, v_i1, v_k] * param_9[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(10240)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], param_10[v_ax2])
                T.writes(T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T_add_intermediate[v_ax0, v_ax1, v_ax2] = NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + param_10[v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(10240)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                T_multiply[v_ax0, v_ax1, v_ax2] = T_add_intermediate[v_ax0, v_ax1, v_ax2] * T.float16(0.70710678118654757)
        for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(10240)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_multiply[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.Cast("float32", T_multiply[v_i0, v_i1, v_i2])
        for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(10240)):
            with T.block("compute_1"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(compute[v_i0, v_i1, v_i2])
                T.writes(compute_1[v_i0, v_i1, v_i2])
                compute_1[v_i0, v_i1, v_i2] = T.erf(compute[v_i0, v_i1, v_i2])
        for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(10240)):
            with T.block("compute_2"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(compute_1[v_i0, v_i1, v_i2])
                T.writes(compute_2[v_i0, v_i1, v_i2])
                compute_2[v_i0, v_i1, v_i2] = T.Cast("float16", compute_1[v_i0, v_i1, v_i2])
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(10240)):
            with T.block("T_multiply_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(compute_2[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply_1[v_ax0, v_ax1, v_ax2])
                T_multiply_1[v_ax0, v_ax1, v_ax2] = compute_2[v_ax0, v_ax1, v_ax2] * T.float16(0.5)
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(10240)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_multiply_1[v_ax0, v_ax1, v_ax2])
                T.writes(T_add[v_ax0, v_ax1, v_ax2])
                T_add[v_ax0, v_ax1, v_ax2] = T.float16(0.5) + T_multiply_1[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(10240)):
            with T.block("T_multiply_2"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_add_intermediate[v_ax0, v_ax1, v_ax2], T_add[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
                T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = T_add_intermediate[v_ax0, v_ax1, v_ax2] * T_add[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_NT_matmul4_add1(p_lv55: T.handle, param_11: T.Buffer((T.int64(2560), T.int64(10240)), "float16"), param_12: T.Buffer((T.int64(2560),), "float16"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv55 = T.match_buffer(p_lv55, (T.int64(1), n, T.int64(10240)), "float16")
        T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(2560), T.int64(10240)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv55[v_i0, v_i1, v_k], param_11[v_i2, v_k])
                T.writes(NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                NT_matmul_intermediate[v_i0, v_i1, v_i2] = NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv55[v_i0, v_i1, v_k] * param_11[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], param_12[v_ax2])
                T.writes(T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T_add_intermediate[v_ax0, v_ax1, v_ax2] = NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + param_12[v_ax2]

    @T.prim_func(private=True)
    def fused_NT_matmul6_add4(lv1779: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), param_510: T.Buffer((T.int64(7680), T.int64(2560)), "float16"), param_610: T.Buffer((T.int64(7680),), "float16"), T_add_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(7680)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(7680)), "float16")
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(7680), T.int64(2560)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv1779[v_i0, v_i1, v_k], param_510[v_i2, v_k])
                T.writes(NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                NT_matmul_intermediate[v_i0, v_i1, v_i2] = NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv1779[v_i0, v_i1, v_k] * param_510[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(7680)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], param_610[v_ax2])
                T.writes(T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T_add_intermediate[v_ax0, v_ax1, v_ax2] = NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + param_610[v_ax2]

    @T.prim_func(private=True)
    def fused_NT_matmul7_divide1_maximum1_minimum1_cast7(lv1800: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(80)), "float16"), p_lv1801: T.handle, p_lv1775: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        m = T.int64()
        lv1801 = T.match_buffer(p_lv1801, (T.int64(1), T.int64(32), m, T.int64(80)), "float16")
        lv1775 = T.match_buffer(p_lv1775, (T.int64(1), T.int64(1), T.int64(1), m), "float16")
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), T.int64(1), m))
        # with T.block("root"):
        NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), m), "float16")
        T_divide_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), m), "float16")
        T_maximum_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), m), "float16")
        T_minimum_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), m), "float16")
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), m, T.int64(80)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(lv1800[v_i0, v_i1, v_i2, v_k], lv1801[v_i0, v_i1, v_i3, v_k])
                T.writes(NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + lv1800[v_i0, v_i1, v_i2, v_k] * lv1801[v_i0, v_i1, v_i3, v_k]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), m):
            with T.block("T_divide"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * T.float16(0.11179039301310044)
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), m):
            with T.block("T_maximum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float16(-65504))
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), m):
            with T.block("T_minimum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv1775[v_ax0, T.int64(0), v_ax2, v_ax3])
                T.writes(T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.min(T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv1775[v_ax0, T.int64(0), v_ax2, v_ax3])
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(1), m):
            with T.block("compute"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_minimum_intermediate[v_i0, v_i1, v_i2, v_i3])
                T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Cast("float32", T_minimum_intermediate[v_i0, v_i1, v_i2, v_i3])

    @T.prim_func(private=True)
    def fused_NT_matmul8_add5_add7_add7(lv1813: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), param_710: T.Buffer((T.int64(2560), T.int64(2560)), "float16"), param_810: T.Buffer((T.int64(2560),), "float16"), lv1828: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), lv1774: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), T_add_intermediate_1_2: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")
        T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")
        T_add_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(2560), T.int64(2560)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv1813[v_i0, v_i1, v_k], param_710[v_i2, v_k])
                T.writes(NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                NT_matmul_intermediate[v_i0, v_i1, v_i2] = NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv1813[v_i0, v_i1, v_k] * param_710[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], param_810[v_ax2])
                T.writes(T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T_add_intermediate[v_ax0, v_ax1, v_ax2] = NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + param_810[v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv1828[v_ax0, v_ax1, v_ax2], T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
                T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = lv1828[v_ax0, v_ax1, v_ax2] + T_add_intermediate[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("T_add_2"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_add_intermediate_1[v_ax0, v_ax1, v_ax2], lv1774[v_ax0, v_ax1, v_ax2])
                T.writes(T_add_intermediate_1_2[v_ax0, v_ax1, v_ax2])
                T_add_intermediate_1_2[v_ax0, v_ax1, v_ax2] = T_add_intermediate_1[v_ax0, v_ax1, v_ax2] + lv1774[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_NT_matmul8_add5_add7_add7_cast5(lv3518: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), param_3791: T.Buffer((T.int64(2560), T.int64(2560)), "float16"), param_3801: T.Buffer((T.int64(2560),), "float16"), lv3533: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), lv3480: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), compute_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")
        T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")
        T_add_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")
        T_add_intermediate_1_2 = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(2560), T.int64(2560)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv3518[v_i0, v_i1, v_k], param_3791[v_i2, v_k])
                T.writes(NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                NT_matmul_intermediate[v_i0, v_i1, v_i2] = NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv3518[v_i0, v_i1, v_k] * param_3791[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], param_3801[v_ax2])
                T.writes(T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T_add_intermediate[v_ax0, v_ax1, v_ax2] = NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + param_3801[v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv3533[v_ax0, v_ax1, v_ax2], T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
                T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = lv3533[v_ax0, v_ax1, v_ax2] + T_add_intermediate[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("T_add_2"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_add_intermediate_1[v_ax0, v_ax1, v_ax2], lv3480[v_ax0, v_ax1, v_ax2])
                T.writes(T_add_intermediate_1_2[v_ax0, v_ax1, v_ax2])
                T_add_intermediate_1_2[v_ax0, v_ax1, v_ax2] = T_add_intermediate_1[v_ax0, v_ax1, v_ax2] + lv3480[v_ax0, v_ax1, v_ax2]
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_add_intermediate_1_2[v_i0, v_i1, v_i2])
                T.writes(compute_intermediate[v_i0, v_i1, v_i2])
                compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float32", T_add_intermediate_1_2[v_i0, v_i1, v_i2])

    @T.prim_func(private=True)
    def fused_NT_matmul9_add6_gelu1(lv1820: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), param_910: T.Buffer((T.int64(10240), T.int64(2560)), "float16"), param_1010: T.Buffer((T.int64(10240),), "float16"), T_multiply_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(10240)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)), "float16")
        T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)), "float16")
        T_multiply = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)), "float16")
        compute = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)))
        compute_1 = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)))
        compute_2 = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)), "float16")
        T_multiply_1 = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)), "float16")
        T_add = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)), "float16")
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(10240), T.int64(2560)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv1820[v_i0, v_i1, v_k], param_910[v_i2, v_k])
                T.writes(NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                NT_matmul_intermediate[v_i0, v_i1, v_i2] = NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv1820[v_i0, v_i1, v_k] * param_910[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(10240)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], param_1010[v_ax2])
                T.writes(T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T_add_intermediate[v_ax0, v_ax1, v_ax2] = NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + param_1010[v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(10240)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                T_multiply[v_ax0, v_ax1, v_ax2] = T_add_intermediate[v_ax0, v_ax1, v_ax2] * T.float16(0.70710678118654757)
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(10240)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_multiply[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.Cast("float32", T_multiply[v_i0, v_i1, v_i2])
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(10240)):
            with T.block("compute_1"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(compute[v_i0, v_i1, v_i2])
                T.writes(compute_1[v_i0, v_i1, v_i2])
                compute_1[v_i0, v_i1, v_i2] = T.erf(compute[v_i0, v_i1, v_i2])
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(10240)):
            with T.block("compute_2"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(compute_1[v_i0, v_i1, v_i2])
                T.writes(compute_2[v_i0, v_i1, v_i2])
                compute_2[v_i0, v_i1, v_i2] = T.Cast("float16", compute_1[v_i0, v_i1, v_i2])
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(10240)):
            with T.block("T_multiply_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(compute_2[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply_1[v_ax0, v_ax1, v_ax2])
                T_multiply_1[v_ax0, v_ax1, v_ax2] = compute_2[v_ax0, v_ax1, v_ax2] * T.float16(0.5)
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(10240)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_multiply_1[v_ax0, v_ax1, v_ax2])
                T.writes(T_add[v_ax0, v_ax1, v_ax2])
                T_add[v_ax0, v_ax1, v_ax2] = T.float16(0.5) + T_multiply_1[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(10240)):
            with T.block("T_multiply_2"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_add_intermediate[v_ax0, v_ax1, v_ax2], T_add[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
                T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = T_add_intermediate[v_ax0, v_ax1, v_ax2] * T_add[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_NT_matmul_add(p_lv9: T.handle, param_5: T.Buffer((T.int64(7680), T.int64(2560)), "float16"), param_6: T.Buffer((T.int64(7680),), "float16"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv9 = T.match_buffer(p_lv9, (T.int64(1), n, T.int64(2560)), "float16")
        T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(7680)), "float16")
        # with T.block("root"):
        NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(7680)), "float16")
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(7680), T.int64(2560)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv9[v_i0, v_i1, v_k], param_5[v_i2, v_k])
                T.writes(NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                NT_matmul_intermediate[v_i0, v_i1, v_i2] = NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv9[v_i0, v_i1, v_k] * param_5[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(7680)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], param_6[v_ax2])
                T.writes(T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T_add_intermediate[v_ax0, v_ax1, v_ax2] = NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + param_6[v_ax2]

    @T.prim_func(private=True)
    def fused_layer_norm1_cast6(lv1776: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), param_1100: T.Buffer((T.int64(2560),), "float32"), param_2100: T.Buffer((T.int64(2560),), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_red_temp_v0 = T.alloc_buffer((T.int64(1), T.int64(1)))
        A_red_temp_v1 = T.alloc_buffer((T.int64(1), T.int64(1)))
        T_layer_norm_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)))
        for ax0, ax1, k2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("A_red_temp"):
                v_ax0, v_ax1, v_k2 = T.axis.remap("SSR", [ax0, ax1, k2])
                T.reads(lv1776[v_ax0, v_ax1, v_k2])
                T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
                with T.init():
                    A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                    A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
                v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + lv1776[v_ax0, v_ax1, v_k2]
                v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + lv1776[v_ax0, v_ax1, v_k2] * lv1776[v_ax0, v_ax1, v_k2]
                A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
                A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("T_layer_norm"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv1776[v_ax0, v_ax1, v_ax2], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], param_1100[v_ax2], param_2100[v_ax2])
                T.writes(T_layer_norm_intermediate[v_ax0, v_ax1, v_ax2])
                T_layer_norm_intermediate[v_ax0, v_ax1, v_ax2] = (lv1776[v_ax0, v_ax1, v_ax2] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(0.00039062500000000002) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1100[v_ax2] + param_2100[v_ax2]
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_layer_norm_intermediate[v_i0, v_i1, v_i2])
                T.writes(compute_intermediate[v_i0, v_i1, v_i2])
                compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", T_layer_norm_intermediate[v_i0, v_i1, v_i2])

    @T.prim_func(private=True)
    def fused_layer_norm_cast1(p_lv6: T.handle, param_1: T.Buffer((T.int64(2560),), "float32"), param_2: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
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

    @T.prim_func(private=True)
    def fused_min_max_triu_te_broadcast_to(p_output0: T.handle, n: T.int64):
        T.func_attr({"tir.noalias": T.bool(True)})
        T_broadcast_to_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(1), n, n), "float16")
        # with T.block("root"):
        make_diag_mask_te_intermediate = T.alloc_buffer((n, n), "float16")
        for i, j in T.grid(n, n):
            with T.block("make_diag_mask_te"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads()
                T.writes(make_diag_mask_te_intermediate[v_i, v_j])
                make_diag_mask_te_intermediate[v_i, v_j] = T.Select(v_i < v_j, T.float16(-65504), T.float16(65504))
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), n, n):
            with T.block("T_broadcast_to"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(make_diag_mask_te_intermediate[v_ax2, v_ax3])
                T.writes(T_broadcast_to_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T_broadcast_to_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = make_diag_mask_te_intermediate[v_ax2, v_ax3]

    @T.prim_func(private=True)
    def fused_reshape7_split1(lv1782: T.Buffer((T.int64(1), T.int64(1), T.int64(7680)), "float16"), T_split_sections_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float16"), T_split_sections_intermediate_1: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float16"), T_split_sections_intermediate_1_2: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_reshape_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(240)), "float16")
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(240)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv1782[T.int64(0), T.int64(0), (v_ax2 * T.int64(240) + v_ax3) % T.int64(7680)])
                T.writes(T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv1782[T.int64(0), T.int64(0), (v_ax2 * T.int64(240) + v_ax3) % T.int64(7680)]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(80)):
            with T.block("T_split_sections"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_split_sections_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T_split_sections_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(80)):
            with T.block("T_split_sections_1"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3 + T.int64(80)])
                T.writes(T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
                T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3 + T.int64(80)]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(80)):
            with T.block("T_split_sections_2"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3 + T.int64(160)])
                T.writes(T_split_sections_intermediate_1_2[v_ax0, v_ax1, v_ax2, v_ax3])
                T_split_sections_intermediate_1_2[v_ax0, v_ax1, v_ax2, v_ax3] = T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3 + T.int64(160)]

    @T.prim_func(private=True)
    def fused_slice1_cast4(lv3537: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        slice_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)))
        for i, _, k in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("slice"):
                v_i, v__, v_k = T.axis.remap("SSS", [i, _, k])
                T.reads(lv3537[v_i, T.int64(0), v_k])
                T.writes(slice_intermediate[v_i, v__, v_k])
                slice_intermediate[v_i, v__, v_k] = lv3537[v_i, T.int64(0), v_k]
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(slice_intermediate[v_i0, v_i1, v_i2])
                T.writes(compute_intermediate[v_i0, v_i1, v_i2])
                compute_intermediate[v_i0, v_i1, v_i2] = slice_intermediate[v_i0, v_i1, v_i2]

    @T.prim_func(private=True)
    def fused_softmax1_cast8(p_lv1808: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        m = T.int64()
        lv1808 = T.match_buffer(p_lv1808, (T.int64(1), T.int64(32), T.int64(1), m))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), T.int64(1), m), "float16")
        # with T.block("root"):
        T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)))
        T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), m))
        T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)))
        T_softmax_norm_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), m))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), T.int64(1), m):
            with T.block("T_softmax_maxelem"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv1808[v_i0, v_i1, v_i2, v_k])
                T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
                with T.init():
                    T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], lv1808[v_i0, v_i1, v_i2, v_k])
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(1), m):
            with T.block("T_softmax_exp"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(lv1808[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
                T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
                T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(lv1808[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), T.int64(1), m):
            with T.block("T_softmax_expsum"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
                T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
                with T.init():
                    T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0)
                T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(1), m):
            with T.block("T_softmax_norm"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
                T.writes(T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])
                T.block_attr({"axis": 3})
                T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(1), m):
            with T.block("compute"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])
                T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Cast("float16", T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])

    @T.prim_func(private=True)
    def fused_softmax_cast3(p_lv38: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv38 = T.match_buffer(p_lv38, (T.int64(1), T.int64(32), n, m))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
        # with T.block("root"):
        T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(32), n))
        T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(32), n, m))
        T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(32), n))
        T_softmax_norm_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), n, m):
            with T.block("T_softmax_maxelem"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv38[v_i0, v_i1, v_i2, v_k])
                T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
                with T.init():
                    T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], lv38[v_i0, v_i1, v_i2, v_k])
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
            with T.block("T_softmax_exp"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(lv38[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
                T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
                T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(lv38[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), n, m):
            with T.block("T_softmax_expsum"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
                T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
                with T.init():
                    T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0)
                T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
            with T.block("T_softmax_norm"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
                T.writes(T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])
                T.block_attr({"axis": 3})
                T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
            with T.block("compute"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])
                T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Cast("float16", T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])

    @T.prim_func(private=True)
    def fused_squeeze(p_lv14_2: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv14_2 = T.match_buffer(p_lv14_2, (T.int64(1), n, T.int64(32), T.int64(80)), "float16")
        T_squeeze_intermediate = T.match_buffer(p_output0, (n, T.int64(32), T.int64(80)), "float16")
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(n, T.int64(32), T.int64(80)):
            with T.block("T_squeeze"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv14_2[T.int64(0), v_ax0, v_ax1, v_ax2])
                T.writes(T_squeeze_intermediate[v_ax0, v_ax1, v_ax2])
                T_squeeze_intermediate[v_ax0, v_ax1, v_ax2] = lv14_2[T.int64(0), v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_squeeze1(lv1784_2: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float16"), T_squeeze_intermediate: T.Buffer((T.int64(1), T.int64(32), T.int64(80)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(32), T.int64(80)):
            with T.block("T_squeeze"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv1784_2[T.int64(0), v_ax0, v_ax1, v_ax2])
                T.writes(T_squeeze_intermediate[v_ax0, v_ax1, v_ax2])
                T_squeeze_intermediate[v_ax0, v_ax1, v_ax2] = lv1784_2[T.int64(0), v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_transpose9_reshape8(lv1811: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(80)), "float16"), T_reshape_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_transpose_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float16")
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(80)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv1811[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv1811[v_ax0, v_ax2, v_ax1, v_ax3]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_transpose_intermediate[T.int64(0), T.int64(0), v_ax2 % T.int64(2560) // T.int64(80), v_ax2 % T.int64(80)])
                T.writes(T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
                T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = T_transpose_intermediate[T.int64(0), T.int64(0), v_ax2 % T.int64(2560) // T.int64(80), v_ax2 % T.int64(80)]

    @T.prim_func(private=True)
    def layer_norm(var_A: T.handle, B: T.Buffer((T.int64(2560),), "float32"), C: T.Buffer((T.int64(2560),), "float32"), var_T_layer_norm: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        T_layer_norm = T.match_buffer(var_T_layer_norm, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        A_red_temp_v0 = T.alloc_buffer((T.int64(1), n))
        A_red_temp_v1 = T.alloc_buffer((T.int64(1), n))
        for ax0, ax1, k2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("A_red_temp"):
                v_ax0, v_ax1, v_k2 = T.axis.remap("SSR", [ax0, ax1, k2])
                T.reads(A[v_ax0, v_ax1, v_k2])
                T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
                with T.init():
                    A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                    A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
                v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + A[v_ax0, v_ax1, v_k2]
                v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + A[v_ax0, v_ax1, v_k2] * A[v_ax0, v_ax1, v_k2]
                A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
                A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_layer_norm"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax1, v_ax2], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], B[v_ax2], C[v_ax2])
                T.writes(T_layer_norm[v_ax0, v_ax1, v_ax2])
                T_layer_norm[v_ax0, v_ax1, v_ax2] = (A[v_ax0, v_ax1, v_ax2] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(0.00039062500000000002) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * B[v_ax2] + C[v_ax2]

    @T.prim_func(private=True)
    def layer_norm1(A: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), B: T.Buffer((T.int64(2560),), "float32"), C: T.Buffer((T.int64(2560),), "float32"), T_layer_norm: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_red_temp_v0 = T.alloc_buffer((T.int64(1), T.int64(1)))
        A_red_temp_v1 = T.alloc_buffer((T.int64(1), T.int64(1)))
        for ax0, ax1, k2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("A_red_temp"):
                v_ax0, v_ax1, v_k2 = T.axis.remap("SSR", [ax0, ax1, k2])
                T.reads(A[v_ax0, v_ax1, v_k2])
                T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
                with T.init():
                    A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                    A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
                v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + A[v_ax0, v_ax1, v_k2]
                v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + A[v_ax0, v_ax1, v_k2] * A[v_ax0, v_ax1, v_k2]
                A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
                A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("T_layer_norm"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax1, v_ax2], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], B[v_ax2], C[v_ax2])
                T.writes(T_layer_norm[v_ax0, v_ax1, v_ax2])
                T_layer_norm[v_ax0, v_ax1, v_ax2] = (A[v_ax0, v_ax1, v_ax2] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(0.00039062500000000002) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * B[v_ax2] + C[v_ax2]

    @T.prim_func(private=True)
    def matmul10(var_A: T.handle, var_B: T.handle, matmul: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(80)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        m = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(32), T.int64(1), m), "float16")
        B = T.match_buffer(var_B, (T.int64(1), T.int64(32), m, T.int64(80)), "float16")
        # with T.block("root"):
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(80), m):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    matmul[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]

    @T.prim_func(private=True)
    def matmul8(var_A: T.handle, var_B: T.handle, var_matmul: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(32), n, m), "float16")
        B = T.match_buffer(var_B, (T.int64(1), T.int64(32), m, T.int64(80)), "float16")
        matmul = T.match_buffer(var_matmul, (T.int64(1), T.int64(32), n, T.int64(80)), "float16")
        # with T.block("root"):
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), n, T.int64(80), m):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    matmul[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]

    @T.prim_func(private=True)
    def reshape(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n), "int32")
        T_reshape = T.match_buffer(var_T_reshape, (n,), "int32")
        # with T.block("root"):
        for ax0 in range(n):
            with T.block("T_reshape"):
                v_ax0 = T.axis.spatial(n, ax0)
                T.reads(A[T.int64(0), v_ax0 % n])
                T.writes(T_reshape[v_ax0])
                T_reshape[v_ax0] = A[T.int64(0), v_ax0 % n]

    @T.prim_func(private=True)
    def reshape1(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (n, T.int64(2560)), "float16")
        T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[(v_ax2 // T.int64(2560) + v_ax0 * n + v_ax1) % n, v_ax2 % T.int64(2560)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = A[(v_ax2 // T.int64(2560) + v_ax0 * n + v_ax1) % n, v_ax2 % T.int64(2560)]

    @T.prim_func(private=True)
    def reshape2(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(7680)), "float16")
        T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), n, T.int64(32), T.int64(240)), "float16")
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), n, T.int64(32), T.int64(240)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[T.int64(0), ((v_ax2 * T.int64(240) + v_ax3) // T.int64(7680) + v_ax0 * n + v_ax1) % n, (v_ax2 * T.int64(240) + v_ax3) % T.int64(7680)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[T.int64(0), ((v_ax2 * T.int64(240) + v_ax3) // T.int64(7680) + v_ax0 * n + v_ax1) % n, (v_ax2 * T.int64(240) + v_ax3) % T.int64(7680)]

    @T.prim_func(private=True)
    def reshape3(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        m = T.int64()
        A = T.match_buffer(var_A, (m, T.int64(32), T.int64(80)), "float16")
        T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), m, T.int64(32), T.int64(80)), "float16")
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), m, T.int64(32), T.int64(80)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[((v_ax3 // T.int64(80) + v_ax2) // T.int64(32) + v_ax0 * m + v_ax1) % m, (v_ax3 // T.int64(80) + v_ax2) % T.int64(32), v_ax3 % T.int64(80)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[((v_ax3 // T.int64(80) + v_ax2) // T.int64(32) + v_ax0 * m + v_ax1) % m, (v_ax3 // T.int64(80) + v_ax2) % T.int64(32), v_ax3 % T.int64(80)]

    @T.prim_func(private=True)
    def reshape4(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(80)), "float16")
        T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[T.int64(0), (v_ax2 // T.int64(2560) + v_ax0 * n + v_ax1) % n, v_ax2 % T.int64(2560) // T.int64(80), v_ax2 % T.int64(80)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = A[T.int64(0), (v_ax2 // T.int64(2560) + v_ax0 * n + v_ax1) % n, v_ax2 % T.int64(2560) // T.int64(80), v_ax2 % T.int64(80)]

    @T.prim_func(private=True)
    def reshape5(A: T.Buffer((T.int64(1), T.int64(1)), "int32"), T_reshape: T.Buffer((T.int64(1),), "int32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0 in range(T.int64(1)):
            with T.block("T_reshape"):
                v_ax0 = T.axis.spatial(T.int64(1), ax0)
                T.reads(A[T.int64(0), T.int64(0)])
                T.writes(T_reshape[v_ax0])
                T_reshape[v_ax0] = A[T.int64(0), T.int64(0)]

    @T.prim_func(private=True)
    def reshape6(A: T.Buffer((T.int64(1), T.int64(2560)), "float16"), T_reshape: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[T.int64(0), v_ax2 % T.int64(2560)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = A[T.int64(0), v_ax2 % T.int64(2560)]

    @T.prim_func(private=True)
    def rotary_embedding(var_A: T.handle, B: T.Buffer((T.int64(4096), T.int64(20)), "float16"), C: T.Buffer((T.int64(4096), T.int64(20)), "float16"), var_rotary: T.handle, m: T.int64):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(80)), "float16")
        rotary = T.match_buffer(var_rotary, (T.int64(1), n, T.int64(32), T.int64(80)), "float16")
        # with T.block("root"):
        for i_batch_size, i_seq_len, i_num_heads, i_head_dim in T.grid(T.int64(1), n, T.int64(32), T.int64(80)):
            with T.block("rotary"):
                v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim = T.axis.remap("SSSS", [i_batch_size, i_seq_len, i_num_heads, i_head_dim])
                T.reads(B[m + v_i_seq_len - n, v_i_head_dim], A[v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim - T.int64(10):v_i_head_dim - T.int64(10) + T.int64(21)], C[m + v_i_seq_len - n, v_i_head_dim])
                T.writes(rotary[v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim])
                rotary[v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim] = T.Select(v_i_head_dim < T.int64(20), B[m + v_i_seq_len - n, v_i_head_dim] * A[v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim] + C[m + v_i_seq_len - n, v_i_head_dim] * T.Select(v_i_head_dim < T.int64(10), A[v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim + T.int64(10)] * T.float16(-1), A[v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim - T.int64(10)]), A[v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim])

    @T.prim_func(private=True)
    def rotary_embedding1(A: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float16"), B: T.Buffer((T.int64(4096), T.int64(20)), "float16"), C: T.Buffer((T.int64(4096), T.int64(20)), "float16"), rotary: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float16"), m: T.int64):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i_batch_size, i_seq_len, i_num_heads, i_head_dim in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(80)):
            with T.block("rotary"):
                v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim = T.axis.remap("SSSS", [i_batch_size, i_seq_len, i_num_heads, i_head_dim])
                T.reads(B[m + v_i_seq_len - T.int64(1), v_i_head_dim], A[v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim - T.int64(10):v_i_head_dim - T.int64(10) + T.int64(21)], C[m + v_i_seq_len - T.int64(1), v_i_head_dim])
                T.writes(rotary[v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim])
                rotary[v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim] = T.Select(v_i_head_dim < T.int64(20), B[m + v_i_seq_len - T.int64(1), v_i_head_dim] * A[v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim] + C[m + v_i_seq_len - T.int64(1), v_i_head_dim] * T.Select(v_i_head_dim < T.int64(10), A[v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim + T.int64(10)] * T.float16(-1), A[v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim - T.int64(10)]), A[v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim])

    @T.prim_func(private=True)
    def slice(var_A: T.handle, slice: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for i, _, k in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("slice"):
                v_i, v__, v_k = T.axis.remap("SSS", [i, _, k])
                T.reads(A[v_i, n - T.int64(1), v_k])
                T.writes(slice[v_i, v__, v_k])
                slice[v_i, v__, v_k] = A[v_i, n - T.int64(1), v_k]

    @T.prim_func(private=True)
    def softmax2(A: T.Buffer((T.int64(1), T.int64(1), T.int64(50280)), "float32"), T_softmax_norm: T.Buffer((T.int64(1), T.int64(1), T.int64(50280)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(1)))
        T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(50280)))
        T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(1)))
        for i0, i1, k in T.grid(T.int64(1), T.int64(1), T.int64(50280)):
            with T.block("T_softmax_maxelem"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(A[v_i0, v_i1, v_k])
                T.writes(T_softmax_maxelem[v_i0, v_i1])
                with T.init():
                    T_softmax_maxelem[v_i0, v_i1] = T.float32(-3.4028234663852886e+38)
                T_softmax_maxelem[v_i0, v_i1] = T.max(T_softmax_maxelem[v_i0, v_i1], A[v_i0, v_i1, v_k])
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(50280)):
            with T.block("T_softmax_exp"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(A[v_i0, v_i1, v_i2], T_softmax_maxelem[v_i0, v_i1])
                T.writes(T_softmax_exp[v_i0, v_i1, v_i2])
                T_softmax_exp[v_i0, v_i1, v_i2] = T.exp(A[v_i0, v_i1, v_i2] - T_softmax_maxelem[v_i0, v_i1])
        for i0, i1, k in T.grid(T.int64(1), T.int64(1), T.int64(50280)):
            with T.block("T_softmax_expsum"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(T_softmax_exp[v_i0, v_i1, v_k])
                T.writes(T_softmax_expsum[v_i0, v_i1])
                with T.init():
                    T_softmax_expsum[v_i0, v_i1] = T.float32(0)
                T_softmax_expsum[v_i0, v_i1] = T_softmax_expsum[v_i0, v_i1] + T_softmax_exp[v_i0, v_i1, v_k]
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(50280)):
            with T.block("T_softmax_norm"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2], T_softmax_expsum[v_i0, v_i1])
                T.writes(T_softmax_norm[v_i0, v_i1, v_i2])
                T.block_attr({"axis": 2})
                T_softmax_norm[v_i0, v_i1, v_i2] = T_softmax_exp[v_i0, v_i1, v_i2] / T_softmax_expsum[v_i0, v_i1]

    @T.prim_func(private=True)
    def split(var_A: T.handle, var_T_split_sections: T.handle, var_T_split_sections_1: T.handle, var_T_split_sections_2: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(240)), "float16")
        T_split_sections = T.match_buffer(var_T_split_sections, (T.int64(1), n, T.int64(32), T.int64(80)), "float16")
        T_split_sections_1 = T.match_buffer(var_T_split_sections_1, (T.int64(1), n, T.int64(32), T.int64(80)), "float16")
        T_split_sections_2 = T.match_buffer(var_T_split_sections_2, (T.int64(1), n, T.int64(32), T.int64(80)), "float16")
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), n, T.int64(32), T.int64(80)):
            with T.block("T_split_sections"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_split_sections[v_ax0, v_ax1, v_ax2, v_ax3])
                T_split_sections[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), n, T.int64(32), T.int64(80)):
            with T.block("T_split_sections_1"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3 + T.int64(80)])
                T.writes(T_split_sections_1[v_ax0, v_ax1, v_ax2, v_ax3])
                T_split_sections_1[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3 + T.int64(80)]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), n, T.int64(32), T.int64(80)):
            with T.block("T_split_sections_2"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3 + T.int64(160)])
                T.writes(T_split_sections_2[v_ax0, v_ax1, v_ax2, v_ax3])
                T_split_sections_2[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3 + T.int64(160)]

    @T.prim_func(private=True)
    def squeeze(var_A: T.handle, var_T_squeeze: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(80)), "float16")
        T_squeeze = T.match_buffer(var_T_squeeze, (n, T.int64(32), T.int64(80)), "float16")
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(n, T.int64(32), T.int64(80)):
            with T.block("T_squeeze"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[T.int64(0), v_ax0, v_ax1, v_ax2])
                T.writes(T_squeeze[v_ax0, v_ax1, v_ax2])
                T_squeeze[v_ax0, v_ax1, v_ax2] = A[T.int64(0), v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def squeeze1(A: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float16"), T_squeeze: T.Buffer((T.int64(1), T.int64(32), T.int64(80)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(32), T.int64(80)):
            with T.block("T_squeeze"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[T.int64(0), v_ax0, v_ax1, v_ax2])
                T.writes(T_squeeze[v_ax0, v_ax1, v_ax2])
                T_squeeze[v_ax0, v_ax1, v_ax2] = A[T.int64(0), v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def take(A: T.Buffer((T.int64(50280), T.int64(2560)), "float16"), var_B: T.handle, var_T_take: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        B = T.match_buffer(var_B, (n,), "int32")
        T_take = T.match_buffer(var_T_take, (n, T.int64(2560)), "float16")
        # with T.block("root"):
        for ax0, ax1 in T.grid(n, T.int64(2560)):
            with T.block("T_take"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[B[v_ax0], v_ax1], B[v_ax0])
                T.writes(T_take[v_ax0, v_ax1])
                T_take[v_ax0, v_ax1] = A[B[v_ax0], v_ax1]

    @T.prim_func(private=True)
    def take1(A: T.Buffer((T.int64(50280), T.int64(2560)), "float16"), B: T.Buffer((T.int64(1),), "int32"), T_take: T.Buffer((T.int64(1), T.int64(2560)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(2560)):
            with T.block("T_take"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[B[v_ax0], v_ax1], B[v_ax0])
                T.writes(T_take[v_ax0, v_ax1])
                T_take[v_ax0, v_ax1] = A[B[v_ax0], v_ax1]

    @T.prim_func(private=True)
    def transpose6(var_A: T.handle, var_T_transpose: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(80)), "float16")
        T_transpose = T.match_buffer(var_T_transpose, (T.int64(1), T.int64(32), n, T.int64(80)), "float16")
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, T.int64(80)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax2, v_ax1, v_ax3]

    @T.prim_func(private=True)
    def transpose7(var_A: T.handle, var_T_transpose: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(32), n, T.int64(80)), "float16")
        T_transpose = T.match_buffer(var_T_transpose, (T.int64(1), n, T.int64(32), T.int64(80)), "float16")
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), n, T.int64(32), T.int64(80)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax2, v_ax1, v_ax3]

    @T.prim_func(private=True)
    def transpose8(A: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float16"), T_transpose: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(80)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(80)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax2, v_ax1, v_ax3]

    @R.function
    def create_kv_cache() -> R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object):
        R.func_attr({"tir_var_upper_bound": {"m": 4096, "n": 4096}})
        with R.dataflow():
            lv3543: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3544: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3545: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3546: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3547: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3548: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3549: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3550: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3551: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3552: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3553: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3554: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3555: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3556: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3557: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3558: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3559: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3560: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3561: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3562: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3563: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3564: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3565: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3566: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3567: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3568: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3569: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3570: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3571: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3572: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3573: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3574: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3575: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3576: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3577: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3578: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3579: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3580: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3581: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3582: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3583: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3584: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3585: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3586: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3587: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3588: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3589: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3590: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3591: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3592: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3593: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3594: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3595: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3596: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3597: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3598: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3599: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3600: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3601: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3602: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3603: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3604: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3605: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3606: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([4096, 32, 80]), R.prim_value(0), sinfo_args=(R.Object,))
            gv2: R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object) = lv3543, lv3544, lv3545, lv3546, lv3547, lv3548, lv3549, lv3550, lv3551, lv3552, lv3553, lv3554, lv3555, lv3556, lv3557, lv3558, lv3559, lv3560, lv3561, lv3562, lv3563, lv3564, lv3565, lv3566, lv3567, lv3568, lv3569, lv3570, lv3571, lv3572, lv3573, lv3574, lv3575, lv3576, lv3577, lv3578, lv3579, lv3580, lv3581, lv3582, lv3583, lv3584, lv3585, lv3586, lv3587, lv3588, lv3589, lv3590, lv3591, lv3592, lv3593, lv3594, lv3595, lv3596, lv3597, lv3598, lv3599, lv3600, lv3601, lv3602, lv3603, lv3604, lv3605, lv3606
            R.output(gv2)
        return gv2

    @R.function
    def decode(input_ids1: R.Tensor((1, 1), dtype="int32"), all_seq_len: R.Shape(["m"]), kv_cache: R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object), model_params: R.Tuple(R.Tensor((50280, 2560), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((50280, 2560), dtype="float32"))) -> R.Tuple(R.Tensor((1, 1, 50280), dtype="float32"), R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object)):
        m = T.int64()
        R.func_attr({"num_input": 3, "tir_var_upper_bound": {"m": 4096, "n": 4096}})
        cls = Module
        with R.dataflow():
            lv1772 = R.call_tir(cls.reshape5, (input_ids1,), out_sinfo=R.Tensor((1,), dtype="int32"))
            param_01: R.Tensor((50280, 2560), dtype="float16") = model_params[0]
            lv1773 = R.call_tir(cls.take1, (param_01, lv1772), out_sinfo=R.Tensor((1, 2560), dtype="float16"))
            lv1774 = R.call_tir(cls.reshape6, (lv1773,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv1775 = R.call_tir(cls.full, R.tuple(), out_sinfo=R.Tensor((1, 1, 1, m), dtype="float16"))
            lv1776 = R.call_tir(cls.cast5, (lv1774,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_1100: R.Tensor((2560,), dtype="float32") = model_params[1]
            param_2100: R.Tensor((2560,), dtype="float32") = model_params[2]
            lv321 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1776, param_1100, param_2100), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_510: R.Tensor((7680, 2560), dtype="float16") = model_params[5]
            param_610: R.Tensor((7680,), dtype="float16") = model_params[6]
            lv322 = R.call_tir(cls.fused_NT_matmul6_add4, (lv321, param_510, param_610), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv323 = R.call_tir(cls.fused_reshape7_split1, (lv322,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv1785: R.Tensor((1, 1, 32, 80), dtype="float16") = lv323[0]
            lv1786 = R.call_tir(cls.rotary_embedding1, (lv1785, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1787: R.Tensor((1, 1, 32, 80), dtype="float16") = lv323[1]
            lv1788 = R.call_tir(cls.rotary_embedding1, (lv1787, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1789: R.Object = kv_cache[0]
            lv1790 = R.call_tir(cls.squeeze1, (lv1788,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv1791: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1789, lv1790, sinfo_args=(R.Object,))
            lv1792: R.Object = kv_cache[1]
            lv324: R.Tensor((1, 1, 32, 80), dtype="float16") = lv323[2]
            lv325 = R.call_tir(cls.fused_squeeze1, (lv324,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv1795: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1792, lv325, sinfo_args=(R.Object,))
            lv1796: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1791, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1797: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1795, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1798 = R.call_tir(cls.reshape3, (lv1796,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1799 = R.call_tir(cls.reshape3, (lv1797,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1800 = R.call_tir(cls.transpose8, (lv1786,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv1801 = R.call_tir(cls.transpose6, (lv1798,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1802 = R.call_tir(cls.transpose6, (lv1799,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv326 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv1800, lv1801, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv327 = R.call_tir(cls.fused_softmax1_cast8, (lv326,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv1811 = R.call_tir(cls.matmul10, (lv327, lv1802), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv328 = R.call_tir(cls.fused_transpose9_reshape8, (lv1811,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_710: R.Tensor((2560, 2560), dtype="float16") = model_params[7]
            param_810: R.Tensor((2560,), dtype="float16") = model_params[8]
            lv1817 = R.call_tir(cls.cast5, (lv1774,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_388: R.Tensor((2560,), dtype="float32") = model_params[3]
            param_410: R.Tensor((2560,), dtype="float32") = model_params[4]
            lv329 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1817, param_388, param_410), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_910: R.Tensor((10240, 2560), dtype="float16") = model_params[9]
            param_1010: R.Tensor((10240,), dtype="float16") = model_params[10]
            lv330 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv329, param_910, param_1010), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_1110: R.Tensor((2560, 10240), dtype="float16") = model_params[11]
            param_1210: R.Tensor((2560,), dtype="float16") = model_params[12]
            lv331 = R.call_tir(cls.fused_NT_matmul10_add5, (lv330, param_1110, param_1210), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv332 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv328, param_710, param_810, lv331, lv1774), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv1831 = R.call_tir(cls.cast5, (lv332,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_1310: R.Tensor((2560,), dtype="float32") = model_params[13]
            param_1410: R.Tensor((2560,), dtype="float32") = model_params[14]
            lv333 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1831, param_1310, param_1410), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1710: R.Tensor((7680, 2560), dtype="float16") = model_params[17]
            param_1810: R.Tensor((7680,), dtype="float16") = model_params[18]
            lv334 = R.call_tir(cls.fused_NT_matmul6_add4, (lv333, param_1710, param_1810), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv335 = R.call_tir(cls.fused_reshape7_split1, (lv334,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv1840: R.Tensor((1, 1, 32, 80), dtype="float16") = lv335[0]
            lv1841 = R.call_tir(cls.rotary_embedding1, (lv1840, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1842: R.Tensor((1, 1, 32, 80), dtype="float16") = lv335[1]
            lv1843 = R.call_tir(cls.rotary_embedding1, (lv1842, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1844: R.Object = kv_cache[2]
            lv1845 = R.call_tir(cls.squeeze1, (lv1843,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv1846: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1844, lv1845, sinfo_args=(R.Object,))
            lv1847: R.Object = kv_cache[3]
            lv336: R.Tensor((1, 1, 32, 80), dtype="float16") = lv335[2]
            lv337 = R.call_tir(cls.fused_squeeze1, (lv336,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv1850: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1847, lv337, sinfo_args=(R.Object,))
            lv1851: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1846, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1852: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1850, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1853 = R.call_tir(cls.reshape3, (lv1851,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1854 = R.call_tir(cls.reshape3, (lv1852,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1855 = R.call_tir(cls.transpose8, (lv1841,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv1856 = R.call_tir(cls.transpose6, (lv1853,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1857 = R.call_tir(cls.transpose6, (lv1854,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv338 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv1855, lv1856, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv339 = R.call_tir(cls.fused_softmax1_cast8, (lv338,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv1866 = R.call_tir(cls.matmul10, (lv339, lv1857), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv340 = R.call_tir(cls.fused_transpose9_reshape8, (lv1866,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1910: R.Tensor((2560, 2560), dtype="float16") = model_params[19]
            param_2010: R.Tensor((2560,), dtype="float16") = model_params[20]
            lv1872 = R.call_tir(cls.cast5, (lv332,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_1510: R.Tensor((2560,), dtype="float32") = model_params[15]
            param_1610: R.Tensor((2560,), dtype="float32") = model_params[16]
            lv341 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1872, param_1510, param_1610), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2110: R.Tensor((10240, 2560), dtype="float16") = model_params[21]
            param_2210: R.Tensor((10240,), dtype="float16") = model_params[22]
            lv342 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv341, param_2110, param_2210), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_2310: R.Tensor((2560, 10240), dtype="float16") = model_params[23]
            param_2410: R.Tensor((2560,), dtype="float16") = model_params[24]
            lv343 = R.call_tir(cls.fused_NT_matmul10_add5, (lv342, param_2310, param_2410), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv344 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv340, param_1910, param_2010, lv343, lv332), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv1886 = R.call_tir(cls.cast5, (lv344,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_2510: R.Tensor((2560,), dtype="float32") = model_params[25]
            param_2610: R.Tensor((2560,), dtype="float32") = model_params[26]
            lv345 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1886, param_2510, param_2610), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2910: R.Tensor((7680, 2560), dtype="float16") = model_params[29]
            param_3010: R.Tensor((7680,), dtype="float16") = model_params[30]
            lv346 = R.call_tir(cls.fused_NT_matmul6_add4, (lv345, param_2910, param_3010), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv347 = R.call_tir(cls.fused_reshape7_split1, (lv346,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv1895: R.Tensor((1, 1, 32, 80), dtype="float16") = lv347[0]
            lv1896 = R.call_tir(cls.rotary_embedding1, (lv1895, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1897: R.Tensor((1, 1, 32, 80), dtype="float16") = lv347[1]
            lv1898 = R.call_tir(cls.rotary_embedding1, (lv1897, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1899: R.Object = kv_cache[4]
            lv1900 = R.call_tir(cls.squeeze1, (lv1898,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv1901: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1899, lv1900, sinfo_args=(R.Object,))
            lv1902: R.Object = kv_cache[5]
            lv348: R.Tensor((1, 1, 32, 80), dtype="float16") = lv347[2]
            lv349 = R.call_tir(cls.fused_squeeze1, (lv348,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv1905: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1902, lv349, sinfo_args=(R.Object,))
            lv1906: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1901, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1907: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1905, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1908 = R.call_tir(cls.reshape3, (lv1906,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1909 = R.call_tir(cls.reshape3, (lv1907,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1910 = R.call_tir(cls.transpose8, (lv1896,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv1911 = R.call_tir(cls.transpose6, (lv1908,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1912 = R.call_tir(cls.transpose6, (lv1909,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv350 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv1910, lv1911, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv351 = R.call_tir(cls.fused_softmax1_cast8, (lv350,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv1921 = R.call_tir(cls.matmul10, (lv351, lv1912), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv352 = R.call_tir(cls.fused_transpose9_reshape8, (lv1921,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_3110: R.Tensor((2560, 2560), dtype="float16") = model_params[31]
            param_3210: R.Tensor((2560,), dtype="float16") = model_params[32]
            lv1927 = R.call_tir(cls.cast5, (lv344,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_2710: R.Tensor((2560,), dtype="float32") = model_params[27]
            param_2810: R.Tensor((2560,), dtype="float32") = model_params[28]
            lv353 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1927, param_2710, param_2810), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_3310: R.Tensor((10240, 2560), dtype="float16") = model_params[33]
            param_3410: R.Tensor((10240,), dtype="float16") = model_params[34]
            lv354 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv353, param_3310, param_3410), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_3510: R.Tensor((2560, 10240), dtype="float16") = model_params[35]
            param_3610: R.Tensor((2560,), dtype="float16") = model_params[36]
            lv355 = R.call_tir(cls.fused_NT_matmul10_add5, (lv354, param_3510, param_3610), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv356 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv352, param_3110, param_3210, lv355, lv344), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv1941 = R.call_tir(cls.cast5, (lv356,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_3710: R.Tensor((2560,), dtype="float32") = model_params[37]
            param_389: R.Tensor((2560,), dtype="float32") = model_params[38]
            lv357 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1941, param_3710, param_389), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_411: R.Tensor((7680, 2560), dtype="float16") = model_params[41]
            param_421: R.Tensor((7680,), dtype="float16") = model_params[42]
            lv358 = R.call_tir(cls.fused_NT_matmul6_add4, (lv357, param_411, param_421), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv359 = R.call_tir(cls.fused_reshape7_split1, (lv358,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv1950: R.Tensor((1, 1, 32, 80), dtype="float16") = lv359[0]
            lv1951 = R.call_tir(cls.rotary_embedding1, (lv1950, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1952: R.Tensor((1, 1, 32, 80), dtype="float16") = lv359[1]
            lv1953 = R.call_tir(cls.rotary_embedding1, (lv1952, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1954: R.Object = kv_cache[6]
            lv1955 = R.call_tir(cls.squeeze1, (lv1953,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv1956: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1954, lv1955, sinfo_args=(R.Object,))
            lv1957: R.Object = kv_cache[7]
            lv360: R.Tensor((1, 1, 32, 80), dtype="float16") = lv359[2]
            lv361 = R.call_tir(cls.fused_squeeze1, (lv360,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv1960: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1957, lv361, sinfo_args=(R.Object,))
            lv1961: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1956, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1962: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1960, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1963 = R.call_tir(cls.reshape3, (lv1961,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1964 = R.call_tir(cls.reshape3, (lv1962,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1965 = R.call_tir(cls.transpose8, (lv1951,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv1966 = R.call_tir(cls.transpose6, (lv1963,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1967 = R.call_tir(cls.transpose6, (lv1964,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv362 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv1965, lv1966, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv363 = R.call_tir(cls.fused_softmax1_cast8, (lv362,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv1976 = R.call_tir(cls.matmul10, (lv363, lv1967), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv364 = R.call_tir(cls.fused_transpose9_reshape8, (lv1976,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_431: R.Tensor((2560, 2560), dtype="float16") = model_params[43]
            param_441: R.Tensor((2560,), dtype="float16") = model_params[44]
            lv1982 = R.call_tir(cls.cast5, (lv356,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_391: R.Tensor((2560,), dtype="float32") = model_params[39]
            param_401: R.Tensor((2560,), dtype="float32") = model_params[40]
            lv365 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1982, param_391, param_401), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_451: R.Tensor((10240, 2560), dtype="float16") = model_params[45]
            param_461: R.Tensor((10240,), dtype="float16") = model_params[46]
            lv366 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv365, param_451, param_461), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_471: R.Tensor((2560, 10240), dtype="float16") = model_params[47]
            param_481: R.Tensor((2560,), dtype="float16") = model_params[48]
            lv367 = R.call_tir(cls.fused_NT_matmul10_add5, (lv366, param_471, param_481), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv368 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv364, param_431, param_441, lv367, lv356), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv1996 = R.call_tir(cls.cast5, (lv368,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_491: R.Tensor((2560,), dtype="float32") = model_params[49]
            param_501: R.Tensor((2560,), dtype="float32") = model_params[50]
            lv369 = R.call_tir(cls.fused_layer_norm1_cast6, (lv1996, param_491, param_501), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_531: R.Tensor((7680, 2560), dtype="float16") = model_params[53]
            param_541: R.Tensor((7680,), dtype="float16") = model_params[54]
            lv370 = R.call_tir(cls.fused_NT_matmul6_add4, (lv369, param_531, param_541), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv371 = R.call_tir(cls.fused_reshape7_split1, (lv370,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2005: R.Tensor((1, 1, 32, 80), dtype="float16") = lv371[0]
            lv2006 = R.call_tir(cls.rotary_embedding1, (lv2005, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2007: R.Tensor((1, 1, 32, 80), dtype="float16") = lv371[1]
            lv2008 = R.call_tir(cls.rotary_embedding1, (lv2007, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2009: R.Object = kv_cache[8]
            lv2010 = R.call_tir(cls.squeeze1, (lv2008,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2011: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2009, lv2010, sinfo_args=(R.Object,))
            lv2012: R.Object = kv_cache[9]
            lv372: R.Tensor((1, 1, 32, 80), dtype="float16") = lv371[2]
            lv373 = R.call_tir(cls.fused_squeeze1, (lv372,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2015: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2012, lv373, sinfo_args=(R.Object,))
            lv2016: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2011, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2017: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2015, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2018 = R.call_tir(cls.reshape3, (lv2016,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2019 = R.call_tir(cls.reshape3, (lv2017,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2020 = R.call_tir(cls.transpose8, (lv2006,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2021 = R.call_tir(cls.transpose6, (lv2018,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2022 = R.call_tir(cls.transpose6, (lv2019,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv374 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv2020, lv2021, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv375 = R.call_tir(cls.fused_softmax1_cast8, (lv374,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2031 = R.call_tir(cls.matmul10, (lv375, lv2022), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv376 = R.call_tir(cls.fused_transpose9_reshape8, (lv2031,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_551: R.Tensor((2560, 2560), dtype="float16") = model_params[55]
            param_561: R.Tensor((2560,), dtype="float16") = model_params[56]
            lv2037 = R.call_tir(cls.cast5, (lv368,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_511: R.Tensor((2560,), dtype="float32") = model_params[51]
            param_521: R.Tensor((2560,), dtype="float32") = model_params[52]
            lv377 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2037, param_511, param_521), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_571: R.Tensor((10240, 2560), dtype="float16") = model_params[57]
            param_581: R.Tensor((10240,), dtype="float16") = model_params[58]
            lv378 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv377, param_571, param_581), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_591: R.Tensor((2560, 10240), dtype="float16") = model_params[59]
            param_601: R.Tensor((2560,), dtype="float16") = model_params[60]
            lv379 = R.call_tir(cls.fused_NT_matmul10_add5, (lv378, param_591, param_601), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv380 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv376, param_551, param_561, lv379, lv368), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2051 = R.call_tir(cls.cast5, (lv380,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_611: R.Tensor((2560,), dtype="float32") = model_params[61]
            param_621: R.Tensor((2560,), dtype="float32") = model_params[62]
            lv381 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2051, param_611, param_621), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_651: R.Tensor((7680, 2560), dtype="float16") = model_params[65]
            param_661: R.Tensor((7680,), dtype="float16") = model_params[66]
            lv382 = R.call_tir(cls.fused_NT_matmul6_add4, (lv381, param_651, param_661), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv383 = R.call_tir(cls.fused_reshape7_split1, (lv382,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2060: R.Tensor((1, 1, 32, 80), dtype="float16") = lv383[0]
            lv2061 = R.call_tir(cls.rotary_embedding1, (lv2060, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2062: R.Tensor((1, 1, 32, 80), dtype="float16") = lv383[1]
            lv2063 = R.call_tir(cls.rotary_embedding1, (lv2062, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2064: R.Object = kv_cache[10]
            lv2065 = R.call_tir(cls.squeeze1, (lv2063,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2066: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2064, lv2065, sinfo_args=(R.Object,))
            lv2067: R.Object = kv_cache[11]
            lv384: R.Tensor((1, 1, 32, 80), dtype="float16") = lv383[2]
            lv385 = R.call_tir(cls.fused_squeeze1, (lv384,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2070: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2067, lv385, sinfo_args=(R.Object,))
            lv2071: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2066, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2072: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2070, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2073 = R.call_tir(cls.reshape3, (lv2071,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2074 = R.call_tir(cls.reshape3, (lv2072,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2075 = R.call_tir(cls.transpose8, (lv2061,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2076 = R.call_tir(cls.transpose6, (lv2073,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2077 = R.call_tir(cls.transpose6, (lv2074,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv386 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv2075, lv2076, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv387 = R.call_tir(cls.fused_softmax1_cast8, (lv386,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2086 = R.call_tir(cls.matmul10, (lv387, lv2077), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv388 = R.call_tir(cls.fused_transpose9_reshape8, (lv2086,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_671: R.Tensor((2560, 2560), dtype="float16") = model_params[67]
            param_681: R.Tensor((2560,), dtype="float16") = model_params[68]
            lv2092 = R.call_tir(cls.cast5, (lv380,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_631: R.Tensor((2560,), dtype="float32") = model_params[63]
            param_641: R.Tensor((2560,), dtype="float32") = model_params[64]
            lv389 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2092, param_631, param_641), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_691: R.Tensor((10240, 2560), dtype="float16") = model_params[69]
            param_701: R.Tensor((10240,), dtype="float16") = model_params[70]
            lv390 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv389, param_691, param_701), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_711: R.Tensor((2560, 10240), dtype="float16") = model_params[71]
            param_721: R.Tensor((2560,), dtype="float16") = model_params[72]
            lv391 = R.call_tir(cls.fused_NT_matmul10_add5, (lv390, param_711, param_721), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv392 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv388, param_671, param_681, lv391, lv380), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2106 = R.call_tir(cls.cast5, (lv392,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_731: R.Tensor((2560,), dtype="float32") = model_params[73]
            param_741: R.Tensor((2560,), dtype="float32") = model_params[74]
            lv393 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2106, param_731, param_741), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_771: R.Tensor((7680, 2560), dtype="float16") = model_params[77]
            param_781: R.Tensor((7680,), dtype="float16") = model_params[78]
            lv394 = R.call_tir(cls.fused_NT_matmul6_add4, (lv393, param_771, param_781), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv395 = R.call_tir(cls.fused_reshape7_split1, (lv394,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2115: R.Tensor((1, 1, 32, 80), dtype="float16") = lv395[0]
            lv2116 = R.call_tir(cls.rotary_embedding1, (lv2115, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2117: R.Tensor((1, 1, 32, 80), dtype="float16") = lv395[1]
            lv2118 = R.call_tir(cls.rotary_embedding1, (lv2117, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2119: R.Object = kv_cache[12]
            lv2120 = R.call_tir(cls.squeeze1, (lv2118,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2121: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2119, lv2120, sinfo_args=(R.Object,))
            lv2122: R.Object = kv_cache[13]
            lv396: R.Tensor((1, 1, 32, 80), dtype="float16") = lv395[2]
            lv397 = R.call_tir(cls.fused_squeeze1, (lv396,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2125: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2122, lv397, sinfo_args=(R.Object,))
            lv2126: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2121, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2127: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2125, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2128 = R.call_tir(cls.reshape3, (lv2126,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2129 = R.call_tir(cls.reshape3, (lv2127,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2130 = R.call_tir(cls.transpose8, (lv2116,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2131 = R.call_tir(cls.transpose6, (lv2128,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2132 = R.call_tir(cls.transpose6, (lv2129,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv398 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv2130, lv2131, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv399 = R.call_tir(cls.fused_softmax1_cast8, (lv398,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2141 = R.call_tir(cls.matmul10, (lv399, lv2132), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv400 = R.call_tir(cls.fused_transpose9_reshape8, (lv2141,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_791: R.Tensor((2560, 2560), dtype="float16") = model_params[79]
            param_801: R.Tensor((2560,), dtype="float16") = model_params[80]
            lv2147 = R.call_tir(cls.cast5, (lv392,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_751: R.Tensor((2560,), dtype="float32") = model_params[75]
            param_761: R.Tensor((2560,), dtype="float32") = model_params[76]
            lv401 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2147, param_751, param_761), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_811: R.Tensor((10240, 2560), dtype="float16") = model_params[81]
            param_821: R.Tensor((10240,), dtype="float16") = model_params[82]
            lv402 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv401, param_811, param_821), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_831: R.Tensor((2560, 10240), dtype="float16") = model_params[83]
            param_841: R.Tensor((2560,), dtype="float16") = model_params[84]
            lv403 = R.call_tir(cls.fused_NT_matmul10_add5, (lv402, param_831, param_841), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv404 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv400, param_791, param_801, lv403, lv392), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2161 = R.call_tir(cls.cast5, (lv404,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_851: R.Tensor((2560,), dtype="float32") = model_params[85]
            param_861: R.Tensor((2560,), dtype="float32") = model_params[86]
            lv405 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2161, param_851, param_861), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_891: R.Tensor((7680, 2560), dtype="float16") = model_params[89]
            param_901: R.Tensor((7680,), dtype="float16") = model_params[90]
            lv406 = R.call_tir(cls.fused_NT_matmul6_add4, (lv405, param_891, param_901), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv407 = R.call_tir(cls.fused_reshape7_split1, (lv406,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2170: R.Tensor((1, 1, 32, 80), dtype="float16") = lv407[0]
            lv2171 = R.call_tir(cls.rotary_embedding1, (lv2170, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2172: R.Tensor((1, 1, 32, 80), dtype="float16") = lv407[1]
            lv2173 = R.call_tir(cls.rotary_embedding1, (lv2172, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2174: R.Object = kv_cache[14]
            lv2175 = R.call_tir(cls.squeeze1, (lv2173,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2176: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2174, lv2175, sinfo_args=(R.Object,))
            lv2177: R.Object = kv_cache[15]
            lv408: R.Tensor((1, 1, 32, 80), dtype="float16") = lv407[2]
            lv409 = R.call_tir(cls.fused_squeeze1, (lv408,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2180: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2177, lv409, sinfo_args=(R.Object,))
            lv2181: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2176, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2182: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2180, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2183 = R.call_tir(cls.reshape3, (lv2181,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2184 = R.call_tir(cls.reshape3, (lv2182,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2185 = R.call_tir(cls.transpose8, (lv2171,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2186 = R.call_tir(cls.transpose6, (lv2183,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2187 = R.call_tir(cls.transpose6, (lv2184,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv410 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv2185, lv2186, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv411 = R.call_tir(cls.fused_softmax1_cast8, (lv410,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2196 = R.call_tir(cls.matmul10, (lv411, lv2187), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv412 = R.call_tir(cls.fused_transpose9_reshape8, (lv2196,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_911: R.Tensor((2560, 2560), dtype="float16") = model_params[91]
            param_921: R.Tensor((2560,), dtype="float16") = model_params[92]
            lv2202 = R.call_tir(cls.cast5, (lv404,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_871: R.Tensor((2560,), dtype="float32") = model_params[87]
            param_881: R.Tensor((2560,), dtype="float32") = model_params[88]
            lv413 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2202, param_871, param_881), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_931: R.Tensor((10240, 2560), dtype="float16") = model_params[93]
            param_941: R.Tensor((10240,), dtype="float16") = model_params[94]
            lv414 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv413, param_931, param_941), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_951: R.Tensor((2560, 10240), dtype="float16") = model_params[95]
            param_961: R.Tensor((2560,), dtype="float16") = model_params[96]
            lv415 = R.call_tir(cls.fused_NT_matmul10_add5, (lv414, param_951, param_961), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv416 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv412, param_911, param_921, lv415, lv404), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2216 = R.call_tir(cls.cast5, (lv416,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_971: R.Tensor((2560,), dtype="float32") = model_params[97]
            param_981: R.Tensor((2560,), dtype="float32") = model_params[98]
            lv417 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2216, param_971, param_981), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1011: R.Tensor((7680, 2560), dtype="float16") = model_params[101]
            param_1021: R.Tensor((7680,), dtype="float16") = model_params[102]
            lv418 = R.call_tir(cls.fused_NT_matmul6_add4, (lv417, param_1011, param_1021), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv419 = R.call_tir(cls.fused_reshape7_split1, (lv418,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2225: R.Tensor((1, 1, 32, 80), dtype="float16") = lv419[0]
            lv2226 = R.call_tir(cls.rotary_embedding1, (lv2225, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2227: R.Tensor((1, 1, 32, 80), dtype="float16") = lv419[1]
            lv2228 = R.call_tir(cls.rotary_embedding1, (lv2227, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2229: R.Object = kv_cache[16]
            lv2230 = R.call_tir(cls.squeeze1, (lv2228,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2231: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2229, lv2230, sinfo_args=(R.Object,))
            lv2232: R.Object = kv_cache[17]
            lv420: R.Tensor((1, 1, 32, 80), dtype="float16") = lv419[2]
            lv421 = R.call_tir(cls.fused_squeeze1, (lv420,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2235: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2232, lv421, sinfo_args=(R.Object,))
            lv2236: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2231, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2237: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2235, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2238 = R.call_tir(cls.reshape3, (lv2236,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2239 = R.call_tir(cls.reshape3, (lv2237,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2240 = R.call_tir(cls.transpose8, (lv2226,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2241 = R.call_tir(cls.transpose6, (lv2238,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2242 = R.call_tir(cls.transpose6, (lv2239,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv422 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv2240, lv2241, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv423 = R.call_tir(cls.fused_softmax1_cast8, (lv422,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2251 = R.call_tir(cls.matmul10, (lv423, lv2242), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv424 = R.call_tir(cls.fused_transpose9_reshape8, (lv2251,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1031: R.Tensor((2560, 2560), dtype="float16") = model_params[103]
            param_1041: R.Tensor((2560,), dtype="float16") = model_params[104]
            lv2257 = R.call_tir(cls.cast5, (lv416,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_991: R.Tensor((2560,), dtype="float32") = model_params[99]
            param_1001: R.Tensor((2560,), dtype="float32") = model_params[100]
            lv425 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2257, param_991, param_1001), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1051: R.Tensor((10240, 2560), dtype="float16") = model_params[105]
            param_1061: R.Tensor((10240,), dtype="float16") = model_params[106]
            lv426 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv425, param_1051, param_1061), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_1071: R.Tensor((2560, 10240), dtype="float16") = model_params[107]
            param_1081: R.Tensor((2560,), dtype="float16") = model_params[108]
            lv427 = R.call_tir(cls.fused_NT_matmul10_add5, (lv426, param_1071, param_1081), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv428 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv424, param_1031, param_1041, lv427, lv416), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2271 = R.call_tir(cls.cast5, (lv428,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_1091: R.Tensor((2560,), dtype="float32") = model_params[109]
            param_1101: R.Tensor((2560,), dtype="float32") = model_params[110]
            lv429 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2271, param_1091, param_1101), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1131: R.Tensor((7680, 2560), dtype="float16") = model_params[113]
            param_1141: R.Tensor((7680,), dtype="float16") = model_params[114]
            lv430 = R.call_tir(cls.fused_NT_matmul6_add4, (lv429, param_1131, param_1141), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv431 = R.call_tir(cls.fused_reshape7_split1, (lv430,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2280: R.Tensor((1, 1, 32, 80), dtype="float16") = lv431[0]
            lv2281 = R.call_tir(cls.rotary_embedding1, (lv2280, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2282: R.Tensor((1, 1, 32, 80), dtype="float16") = lv431[1]
            lv2283 = R.call_tir(cls.rotary_embedding1, (lv2282, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2284: R.Object = kv_cache[18]
            lv2285 = R.call_tir(cls.squeeze1, (lv2283,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2286: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2284, lv2285, sinfo_args=(R.Object,))
            lv2287: R.Object = kv_cache[19]
            lv432: R.Tensor((1, 1, 32, 80), dtype="float16") = lv431[2]
            lv433 = R.call_tir(cls.fused_squeeze1, (lv432,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2290: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2287, lv433, sinfo_args=(R.Object,))
            lv2291: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2286, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2292: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2290, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2293 = R.call_tir(cls.reshape3, (lv2291,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2294 = R.call_tir(cls.reshape3, (lv2292,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2295 = R.call_tir(cls.transpose8, (lv2281,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2296 = R.call_tir(cls.transpose6, (lv2293,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2297 = R.call_tir(cls.transpose6, (lv2294,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv434 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv2295, lv2296, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv435 = R.call_tir(cls.fused_softmax1_cast8, (lv434,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2306 = R.call_tir(cls.matmul10, (lv435, lv2297), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv436 = R.call_tir(cls.fused_transpose9_reshape8, (lv2306,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1151: R.Tensor((2560, 2560), dtype="float16") = model_params[115]
            param_1161: R.Tensor((2560,), dtype="float16") = model_params[116]
            lv2312 = R.call_tir(cls.cast5, (lv428,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_1111: R.Tensor((2560,), dtype="float32") = model_params[111]
            param_1121: R.Tensor((2560,), dtype="float32") = model_params[112]
            lv437 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2312, param_1111, param_1121), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1171: R.Tensor((10240, 2560), dtype="float16") = model_params[117]
            param_1181: R.Tensor((10240,), dtype="float16") = model_params[118]
            lv438 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv437, param_1171, param_1181), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_1191: R.Tensor((2560, 10240), dtype="float16") = model_params[119]
            param_1201: R.Tensor((2560,), dtype="float16") = model_params[120]
            lv439 = R.call_tir(cls.fused_NT_matmul10_add5, (lv438, param_1191, param_1201), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv440 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv436, param_1151, param_1161, lv439, lv428), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2326 = R.call_tir(cls.cast5, (lv440,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_1211: R.Tensor((2560,), dtype="float32") = model_params[121]
            param_1221: R.Tensor((2560,), dtype="float32") = model_params[122]
            lv441 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2326, param_1211, param_1221), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1251: R.Tensor((7680, 2560), dtype="float16") = model_params[125]
            param_1261: R.Tensor((7680,), dtype="float16") = model_params[126]
            lv442 = R.call_tir(cls.fused_NT_matmul6_add4, (lv441, param_1251, param_1261), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv443 = R.call_tir(cls.fused_reshape7_split1, (lv442,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2335: R.Tensor((1, 1, 32, 80), dtype="float16") = lv443[0]
            lv2336 = R.call_tir(cls.rotary_embedding1, (lv2335, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2337: R.Tensor((1, 1, 32, 80), dtype="float16") = lv443[1]
            lv2338 = R.call_tir(cls.rotary_embedding1, (lv2337, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2339: R.Object = kv_cache[20]
            lv2340 = R.call_tir(cls.squeeze1, (lv2338,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2341: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2339, lv2340, sinfo_args=(R.Object,))
            lv2342: R.Object = kv_cache[21]
            lv444: R.Tensor((1, 1, 32, 80), dtype="float16") = lv443[2]
            lv445 = R.call_tir(cls.fused_squeeze1, (lv444,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2345: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2342, lv445, sinfo_args=(R.Object,))
            lv2346: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2341, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2347: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2345, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2348 = R.call_tir(cls.reshape3, (lv2346,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2349 = R.call_tir(cls.reshape3, (lv2347,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2350 = R.call_tir(cls.transpose8, (lv2336,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2351 = R.call_tir(cls.transpose6, (lv2348,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2352 = R.call_tir(cls.transpose6, (lv2349,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv446 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv2350, lv2351, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv447 = R.call_tir(cls.fused_softmax1_cast8, (lv446,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2361 = R.call_tir(cls.matmul10, (lv447, lv2352), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv448 = R.call_tir(cls.fused_transpose9_reshape8, (lv2361,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1271: R.Tensor((2560, 2560), dtype="float16") = model_params[127]
            param_1281: R.Tensor((2560,), dtype="float16") = model_params[128]
            lv2367 = R.call_tir(cls.cast5, (lv440,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_1231: R.Tensor((2560,), dtype="float32") = model_params[123]
            param_1241: R.Tensor((2560,), dtype="float32") = model_params[124]
            lv449 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2367, param_1231, param_1241), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1291: R.Tensor((10240, 2560), dtype="float16") = model_params[129]
            param_1301: R.Tensor((10240,), dtype="float16") = model_params[130]
            lv450 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv449, param_1291, param_1301), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_1311: R.Tensor((2560, 10240), dtype="float16") = model_params[131]
            param_1321: R.Tensor((2560,), dtype="float16") = model_params[132]
            lv451 = R.call_tir(cls.fused_NT_matmul10_add5, (lv450, param_1311, param_1321), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv452 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv448, param_1271, param_1281, lv451, lv440), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2381 = R.call_tir(cls.cast5, (lv452,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_1331: R.Tensor((2560,), dtype="float32") = model_params[133]
            param_1341: R.Tensor((2560,), dtype="float32") = model_params[134]
            lv453 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2381, param_1331, param_1341), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1371: R.Tensor((7680, 2560), dtype="float16") = model_params[137]
            param_1381: R.Tensor((7680,), dtype="float16") = model_params[138]
            lv454 = R.call_tir(cls.fused_NT_matmul6_add4, (lv453, param_1371, param_1381), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv455 = R.call_tir(cls.fused_reshape7_split1, (lv454,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2390: R.Tensor((1, 1, 32, 80), dtype="float16") = lv455[0]
            lv2391 = R.call_tir(cls.rotary_embedding1, (lv2390, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2392: R.Tensor((1, 1, 32, 80), dtype="float16") = lv455[1]
            lv2393 = R.call_tir(cls.rotary_embedding1, (lv2392, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2394: R.Object = kv_cache[22]
            lv2395 = R.call_tir(cls.squeeze1, (lv2393,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2396: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2394, lv2395, sinfo_args=(R.Object,))
            lv2397: R.Object = kv_cache[23]
            lv456: R.Tensor((1, 1, 32, 80), dtype="float16") = lv455[2]
            lv457 = R.call_tir(cls.fused_squeeze1, (lv456,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2400: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2397, lv457, sinfo_args=(R.Object,))
            lv2401: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2396, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2402: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2400, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2403 = R.call_tir(cls.reshape3, (lv2401,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2404 = R.call_tir(cls.reshape3, (lv2402,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2405 = R.call_tir(cls.transpose8, (lv2391,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2406 = R.call_tir(cls.transpose6, (lv2403,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2407 = R.call_tir(cls.transpose6, (lv2404,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv458 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv2405, lv2406, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv459 = R.call_tir(cls.fused_softmax1_cast8, (lv458,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2416 = R.call_tir(cls.matmul10, (lv459, lv2407), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv460 = R.call_tir(cls.fused_transpose9_reshape8, (lv2416,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1391: R.Tensor((2560, 2560), dtype="float16") = model_params[139]
            param_1401: R.Tensor((2560,), dtype="float16") = model_params[140]
            lv2422 = R.call_tir(cls.cast5, (lv452,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_1351: R.Tensor((2560,), dtype="float32") = model_params[135]
            param_1361: R.Tensor((2560,), dtype="float32") = model_params[136]
            lv461 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2422, param_1351, param_1361), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1411: R.Tensor((10240, 2560), dtype="float16") = model_params[141]
            param_1421: R.Tensor((10240,), dtype="float16") = model_params[142]
            lv462 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv461, param_1411, param_1421), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_1431: R.Tensor((2560, 10240), dtype="float16") = model_params[143]
            param_1441: R.Tensor((2560,), dtype="float16") = model_params[144]
            lv463 = R.call_tir(cls.fused_NT_matmul10_add5, (lv462, param_1431, param_1441), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv464 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv460, param_1391, param_1401, lv463, lv452), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2436 = R.call_tir(cls.cast5, (lv464,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_1451: R.Tensor((2560,), dtype="float32") = model_params[145]
            param_1461: R.Tensor((2560,), dtype="float32") = model_params[146]
            lv465 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2436, param_1451, param_1461), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1491: R.Tensor((7680, 2560), dtype="float16") = model_params[149]
            param_1501: R.Tensor((7680,), dtype="float16") = model_params[150]
            lv466 = R.call_tir(cls.fused_NT_matmul6_add4, (lv465, param_1491, param_1501), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv467 = R.call_tir(cls.fused_reshape7_split1, (lv466,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2445: R.Tensor((1, 1, 32, 80), dtype="float16") = lv467[0]
            lv2446 = R.call_tir(cls.rotary_embedding1, (lv2445, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2447: R.Tensor((1, 1, 32, 80), dtype="float16") = lv467[1]
            lv2448 = R.call_tir(cls.rotary_embedding1, (lv2447, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2449: R.Object = kv_cache[24]
            lv2450 = R.call_tir(cls.squeeze1, (lv2448,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2451: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2449, lv2450, sinfo_args=(R.Object,))
            lv2452: R.Object = kv_cache[25]
            lv468: R.Tensor((1, 1, 32, 80), dtype="float16") = lv467[2]
            lv469 = R.call_tir(cls.fused_squeeze1, (lv468,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2455: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2452, lv469, sinfo_args=(R.Object,))
            lv2456: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2451, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2457: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2455, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2458 = R.call_tir(cls.reshape3, (lv2456,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2459 = R.call_tir(cls.reshape3, (lv2457,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2460 = R.call_tir(cls.transpose8, (lv2446,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2461 = R.call_tir(cls.transpose6, (lv2458,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2462 = R.call_tir(cls.transpose6, (lv2459,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv470 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv2460, lv2461, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv471 = R.call_tir(cls.fused_softmax1_cast8, (lv470,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2471 = R.call_tir(cls.matmul10, (lv471, lv2462), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv472 = R.call_tir(cls.fused_transpose9_reshape8, (lv2471,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1511: R.Tensor((2560, 2560), dtype="float16") = model_params[151]
            param_1521: R.Tensor((2560,), dtype="float16") = model_params[152]
            lv2477 = R.call_tir(cls.cast5, (lv464,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_1471: R.Tensor((2560,), dtype="float32") = model_params[147]
            param_1481: R.Tensor((2560,), dtype="float32") = model_params[148]
            lv473 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2477, param_1471, param_1481), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1531: R.Tensor((10240, 2560), dtype="float16") = model_params[153]
            param_1541: R.Tensor((10240,), dtype="float16") = model_params[154]
            lv474 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv473, param_1531, param_1541), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_1551: R.Tensor((2560, 10240), dtype="float16") = model_params[155]
            param_1561: R.Tensor((2560,), dtype="float16") = model_params[156]
            lv475 = R.call_tir(cls.fused_NT_matmul10_add5, (lv474, param_1551, param_1561), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv476 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv472, param_1511, param_1521, lv475, lv464), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2491 = R.call_tir(cls.cast5, (lv476,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_1571: R.Tensor((2560,), dtype="float32") = model_params[157]
            param_1581: R.Tensor((2560,), dtype="float32") = model_params[158]
            lv477 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2491, param_1571, param_1581), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1611: R.Tensor((7680, 2560), dtype="float16") = model_params[161]
            param_1621: R.Tensor((7680,), dtype="float16") = model_params[162]
            lv478 = R.call_tir(cls.fused_NT_matmul6_add4, (lv477, param_1611, param_1621), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv479 = R.call_tir(cls.fused_reshape7_split1, (lv478,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2500: R.Tensor((1, 1, 32, 80), dtype="float16") = lv479[0]
            lv2501 = R.call_tir(cls.rotary_embedding1, (lv2500, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2502: R.Tensor((1, 1, 32, 80), dtype="float16") = lv479[1]
            lv2503 = R.call_tir(cls.rotary_embedding1, (lv2502, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2504: R.Object = kv_cache[26]
            lv2505 = R.call_tir(cls.squeeze1, (lv2503,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2506: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2504, lv2505, sinfo_args=(R.Object,))
            lv2507: R.Object = kv_cache[27]
            lv480: R.Tensor((1, 1, 32, 80), dtype="float16") = lv479[2]
            lv481 = R.call_tir(cls.fused_squeeze1, (lv480,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2510: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2507, lv481, sinfo_args=(R.Object,))
            lv2511: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2506, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2512: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2510, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2513 = R.call_tir(cls.reshape3, (lv2511,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2514 = R.call_tir(cls.reshape3, (lv2512,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2515 = R.call_tir(cls.transpose8, (lv2501,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2516 = R.call_tir(cls.transpose6, (lv2513,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2517 = R.call_tir(cls.transpose6, (lv2514,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv482 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv2515, lv2516, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv483 = R.call_tir(cls.fused_softmax1_cast8, (lv482,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2526 = R.call_tir(cls.matmul10, (lv483, lv2517), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv484 = R.call_tir(cls.fused_transpose9_reshape8, (lv2526,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1631: R.Tensor((2560, 2560), dtype="float16") = model_params[163]
            param_1641: R.Tensor((2560,), dtype="float16") = model_params[164]
            lv2532 = R.call_tir(cls.cast5, (lv476,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_1591: R.Tensor((2560,), dtype="float32") = model_params[159]
            param_1601: R.Tensor((2560,), dtype="float32") = model_params[160]
            lv485 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2532, param_1591, param_1601), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1651: R.Tensor((10240, 2560), dtype="float16") = model_params[165]
            param_1661: R.Tensor((10240,), dtype="float16") = model_params[166]
            lv486 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv485, param_1651, param_1661), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_1671: R.Tensor((2560, 10240), dtype="float16") = model_params[167]
            param_1681: R.Tensor((2560,), dtype="float16") = model_params[168]
            lv487 = R.call_tir(cls.fused_NT_matmul10_add5, (lv486, param_1671, param_1681), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv488 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv484, param_1631, param_1641, lv487, lv476), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2546 = R.call_tir(cls.cast5, (lv488,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_1691: R.Tensor((2560,), dtype="float32") = model_params[169]
            param_1701: R.Tensor((2560,), dtype="float32") = model_params[170]
            lv489 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2546, param_1691, param_1701), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1731: R.Tensor((7680, 2560), dtype="float16") = model_params[173]
            param_1741: R.Tensor((7680,), dtype="float16") = model_params[174]
            lv490 = R.call_tir(cls.fused_NT_matmul6_add4, (lv489, param_1731, param_1741), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv491 = R.call_tir(cls.fused_reshape7_split1, (lv490,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2555: R.Tensor((1, 1, 32, 80), dtype="float16") = lv491[0]
            lv2556 = R.call_tir(cls.rotary_embedding1, (lv2555, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2557: R.Tensor((1, 1, 32, 80), dtype="float16") = lv491[1]
            lv2558 = R.call_tir(cls.rotary_embedding1, (lv2557, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2559: R.Object = kv_cache[28]
            lv2560 = R.call_tir(cls.squeeze1, (lv2558,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2561: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2559, lv2560, sinfo_args=(R.Object,))
            lv2562: R.Object = kv_cache[29]
            lv492: R.Tensor((1, 1, 32, 80), dtype="float16") = lv491[2]
            lv493 = R.call_tir(cls.fused_squeeze1, (lv492,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2565: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2562, lv493, sinfo_args=(R.Object,))
            lv2566: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2561, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2567: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2565, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2568 = R.call_tir(cls.reshape3, (lv2566,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2569 = R.call_tir(cls.reshape3, (lv2567,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2570 = R.call_tir(cls.transpose8, (lv2556,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2571 = R.call_tir(cls.transpose6, (lv2568,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2572 = R.call_tir(cls.transpose6, (lv2569,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv494 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv2570, lv2571, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv495 = R.call_tir(cls.fused_softmax1_cast8, (lv494,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2581 = R.call_tir(cls.matmul10, (lv495, lv2572), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv496 = R.call_tir(cls.fused_transpose9_reshape8, (lv2581,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1751: R.Tensor((2560, 2560), dtype="float16") = model_params[175]
            param_1761: R.Tensor((2560,), dtype="float16") = model_params[176]
            lv2587 = R.call_tir(cls.cast5, (lv488,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_1711: R.Tensor((2560,), dtype="float32") = model_params[171]
            param_1721: R.Tensor((2560,), dtype="float32") = model_params[172]
            lv497 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2587, param_1711, param_1721), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1771: R.Tensor((10240, 2560), dtype="float16") = model_params[177]
            param_1781: R.Tensor((10240,), dtype="float16") = model_params[178]
            lv498 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv497, param_1771, param_1781), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_1791: R.Tensor((2560, 10240), dtype="float16") = model_params[179]
            param_1801: R.Tensor((2560,), dtype="float16") = model_params[180]
            lv499 = R.call_tir(cls.fused_NT_matmul10_add5, (lv498, param_1791, param_1801), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv500 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv496, param_1751, param_1761, lv499, lv488), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2601 = R.call_tir(cls.cast5, (lv500,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_1811: R.Tensor((2560,), dtype="float32") = model_params[181]
            param_1821: R.Tensor((2560,), dtype="float32") = model_params[182]
            lv501 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2601, param_1811, param_1821), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1851: R.Tensor((7680, 2560), dtype="float16") = model_params[185]
            param_1861: R.Tensor((7680,), dtype="float16") = model_params[186]
            lv502 = R.call_tir(cls.fused_NT_matmul6_add4, (lv501, param_1851, param_1861), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv503 = R.call_tir(cls.fused_reshape7_split1, (lv502,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2610: R.Tensor((1, 1, 32, 80), dtype="float16") = lv503[0]
            lv2611 = R.call_tir(cls.rotary_embedding1, (lv2610, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2612: R.Tensor((1, 1, 32, 80), dtype="float16") = lv503[1]
            lv2613 = R.call_tir(cls.rotary_embedding1, (lv2612, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2614: R.Object = kv_cache[30]
            lv2615 = R.call_tir(cls.squeeze1, (lv2613,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2616: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2614, lv2615, sinfo_args=(R.Object,))
            lv2617: R.Object = kv_cache[31]
            lv504: R.Tensor((1, 1, 32, 80), dtype="float16") = lv503[2]
            lv505 = R.call_tir(cls.fused_squeeze1, (lv504,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2620: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2617, lv505, sinfo_args=(R.Object,))
            lv2621: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2616, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2622: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2620, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2623 = R.call_tir(cls.reshape3, (lv2621,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2624 = R.call_tir(cls.reshape3, (lv2622,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2625 = R.call_tir(cls.transpose8, (lv2611,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2626 = R.call_tir(cls.transpose6, (lv2623,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2627 = R.call_tir(cls.transpose6, (lv2624,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv506 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv2625, lv2626, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv507 = R.call_tir(cls.fused_softmax1_cast8, (lv506,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2636 = R.call_tir(cls.matmul10, (lv507, lv2627), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv508 = R.call_tir(cls.fused_transpose9_reshape8, (lv2636,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1871: R.Tensor((2560, 2560), dtype="float16") = model_params[187]
            param_1881: R.Tensor((2560,), dtype="float16") = model_params[188]
            lv2642 = R.call_tir(cls.cast5, (lv500,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_1831: R.Tensor((2560,), dtype="float32") = model_params[183]
            param_1841: R.Tensor((2560,), dtype="float32") = model_params[184]
            lv509 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2642, param_1831, param_1841), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1891: R.Tensor((10240, 2560), dtype="float16") = model_params[189]
            param_1901: R.Tensor((10240,), dtype="float16") = model_params[190]
            lv510 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv509, param_1891, param_1901), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_1911: R.Tensor((2560, 10240), dtype="float16") = model_params[191]
            param_1921: R.Tensor((2560,), dtype="float16") = model_params[192]
            lv511 = R.call_tir(cls.fused_NT_matmul10_add5, (lv510, param_1911, param_1921), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv512 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv508, param_1871, param_1881, lv511, lv500), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2656 = R.call_tir(cls.cast5, (lv512,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_1931: R.Tensor((2560,), dtype="float32") = model_params[193]
            param_1941: R.Tensor((2560,), dtype="float32") = model_params[194]
            lv513 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2656, param_1931, param_1941), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1971: R.Tensor((7680, 2560), dtype="float16") = model_params[197]
            param_1981: R.Tensor((7680,), dtype="float16") = model_params[198]
            lv514 = R.call_tir(cls.fused_NT_matmul6_add4, (lv513, param_1971, param_1981), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv515 = R.call_tir(cls.fused_reshape7_split1, (lv514,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2665: R.Tensor((1, 1, 32, 80), dtype="float16") = lv515[0]
            lv2666 = R.call_tir(cls.rotary_embedding1, (lv2665, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2667: R.Tensor((1, 1, 32, 80), dtype="float16") = lv515[1]
            lv2668 = R.call_tir(cls.rotary_embedding1, (lv2667, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2669: R.Object = kv_cache[32]
            lv2670 = R.call_tir(cls.squeeze1, (lv2668,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2671: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2669, lv2670, sinfo_args=(R.Object,))
            lv2672: R.Object = kv_cache[33]
            lv516: R.Tensor((1, 1, 32, 80), dtype="float16") = lv515[2]
            lv517 = R.call_tir(cls.fused_squeeze1, (lv516,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2675: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2672, lv517, sinfo_args=(R.Object,))
            lv2676: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2671, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2677: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2675, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2678 = R.call_tir(cls.reshape3, (lv2676,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2679 = R.call_tir(cls.reshape3, (lv2677,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2680 = R.call_tir(cls.transpose8, (lv2666,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2681 = R.call_tir(cls.transpose6, (lv2678,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2682 = R.call_tir(cls.transpose6, (lv2679,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv518 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv2680, lv2681, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv519 = R.call_tir(cls.fused_softmax1_cast8, (lv518,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2691 = R.call_tir(cls.matmul10, (lv519, lv2682), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv520 = R.call_tir(cls.fused_transpose9_reshape8, (lv2691,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_1991: R.Tensor((2560, 2560), dtype="float16") = model_params[199]
            param_2001: R.Tensor((2560,), dtype="float16") = model_params[200]
            lv2697 = R.call_tir(cls.cast5, (lv512,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_1951: R.Tensor((2560,), dtype="float32") = model_params[195]
            param_1961: R.Tensor((2560,), dtype="float32") = model_params[196]
            lv521 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2697, param_1951, param_1961), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2011: R.Tensor((10240, 2560), dtype="float16") = model_params[201]
            param_2021: R.Tensor((10240,), dtype="float16") = model_params[202]
            lv522 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv521, param_2011, param_2021), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_2031: R.Tensor((2560, 10240), dtype="float16") = model_params[203]
            param_2041: R.Tensor((2560,), dtype="float16") = model_params[204]
            lv523 = R.call_tir(cls.fused_NT_matmul10_add5, (lv522, param_2031, param_2041), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv524 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv520, param_1991, param_2001, lv523, lv512), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2711 = R.call_tir(cls.cast5, (lv524,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_2051: R.Tensor((2560,), dtype="float32") = model_params[205]
            param_2061: R.Tensor((2560,), dtype="float32") = model_params[206]
            lv525 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2711, param_2051, param_2061), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2091: R.Tensor((7680, 2560), dtype="float16") = model_params[209]
            param_2101: R.Tensor((7680,), dtype="float16") = model_params[210]
            lv526 = R.call_tir(cls.fused_NT_matmul6_add4, (lv525, param_2091, param_2101), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv527 = R.call_tir(cls.fused_reshape7_split1, (lv526,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2720: R.Tensor((1, 1, 32, 80), dtype="float16") = lv527[0]
            lv2721 = R.call_tir(cls.rotary_embedding1, (lv2720, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2722: R.Tensor((1, 1, 32, 80), dtype="float16") = lv527[1]
            lv2723 = R.call_tir(cls.rotary_embedding1, (lv2722, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2724: R.Object = kv_cache[34]
            lv2725 = R.call_tir(cls.squeeze1, (lv2723,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2726: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2724, lv2725, sinfo_args=(R.Object,))
            lv2727: R.Object = kv_cache[35]
            lv528: R.Tensor((1, 1, 32, 80), dtype="float16") = lv527[2]
            lv529 = R.call_tir(cls.fused_squeeze1, (lv528,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2730: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2727, lv529, sinfo_args=(R.Object,))
            lv2731: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2726, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2732: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2730, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2733 = R.call_tir(cls.reshape3, (lv2731,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2734 = R.call_tir(cls.reshape3, (lv2732,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2735 = R.call_tir(cls.transpose8, (lv2721,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2736 = R.call_tir(cls.transpose6, (lv2733,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2737 = R.call_tir(cls.transpose6, (lv2734,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv530 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv2735, lv2736, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv531 = R.call_tir(cls.fused_softmax1_cast8, (lv530,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2746 = R.call_tir(cls.matmul10, (lv531, lv2737), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv532 = R.call_tir(cls.fused_transpose9_reshape8, (lv2746,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2111: R.Tensor((2560, 2560), dtype="float16") = model_params[211]
            param_2121: R.Tensor((2560,), dtype="float16") = model_params[212]
            lv2752 = R.call_tir(cls.cast5, (lv524,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_2071: R.Tensor((2560,), dtype="float32") = model_params[207]
            param_2081: R.Tensor((2560,), dtype="float32") = model_params[208]
            lv533 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2752, param_2071, param_2081), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2131: R.Tensor((10240, 2560), dtype="float16") = model_params[213]
            param_2141: R.Tensor((10240,), dtype="float16") = model_params[214]
            lv534 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv533, param_2131, param_2141), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_2151: R.Tensor((2560, 10240), dtype="float16") = model_params[215]
            param_2161: R.Tensor((2560,), dtype="float16") = model_params[216]
            lv535 = R.call_tir(cls.fused_NT_matmul10_add5, (lv534, param_2151, param_2161), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv536 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv532, param_2111, param_2121, lv535, lv524), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2766 = R.call_tir(cls.cast5, (lv536,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_2171: R.Tensor((2560,), dtype="float32") = model_params[217]
            param_2181: R.Tensor((2560,), dtype="float32") = model_params[218]
            lv537 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2766, param_2171, param_2181), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2211: R.Tensor((7680, 2560), dtype="float16") = model_params[221]
            param_2221: R.Tensor((7680,), dtype="float16") = model_params[222]
            lv538 = R.call_tir(cls.fused_NT_matmul6_add4, (lv537, param_2211, param_2221), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv539 = R.call_tir(cls.fused_reshape7_split1, (lv538,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2775: R.Tensor((1, 1, 32, 80), dtype="float16") = lv539[0]
            lv2776 = R.call_tir(cls.rotary_embedding1, (lv2775, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2777: R.Tensor((1, 1, 32, 80), dtype="float16") = lv539[1]
            lv2778 = R.call_tir(cls.rotary_embedding1, (lv2777, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2779: R.Object = kv_cache[36]
            lv2780 = R.call_tir(cls.squeeze1, (lv2778,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2781: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2779, lv2780, sinfo_args=(R.Object,))
            lv2782: R.Object = kv_cache[37]
            lv540: R.Tensor((1, 1, 32, 80), dtype="float16") = lv539[2]
            lv541 = R.call_tir(cls.fused_squeeze1, (lv540,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2785: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2782, lv541, sinfo_args=(R.Object,))
            lv2786: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2781, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2787: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2785, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2788 = R.call_tir(cls.reshape3, (lv2786,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2789 = R.call_tir(cls.reshape3, (lv2787,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2790 = R.call_tir(cls.transpose8, (lv2776,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2791 = R.call_tir(cls.transpose6, (lv2788,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2792 = R.call_tir(cls.transpose6, (lv2789,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv542 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv2790, lv2791, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv543 = R.call_tir(cls.fused_softmax1_cast8, (lv542,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2801 = R.call_tir(cls.matmul10, (lv543, lv2792), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv544 = R.call_tir(cls.fused_transpose9_reshape8, (lv2801,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2231: R.Tensor((2560, 2560), dtype="float16") = model_params[223]
            param_2241: R.Tensor((2560,), dtype="float16") = model_params[224]
            lv2807 = R.call_tir(cls.cast5, (lv536,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_2191: R.Tensor((2560,), dtype="float32") = model_params[219]
            param_2201: R.Tensor((2560,), dtype="float32") = model_params[220]
            lv545 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2807, param_2191, param_2201), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2251: R.Tensor((10240, 2560), dtype="float16") = model_params[225]
            param_2261: R.Tensor((10240,), dtype="float16") = model_params[226]
            lv546 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv545, param_2251, param_2261), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_2271: R.Tensor((2560, 10240), dtype="float16") = model_params[227]
            param_2281: R.Tensor((2560,), dtype="float16") = model_params[228]
            lv547 = R.call_tir(cls.fused_NT_matmul10_add5, (lv546, param_2271, param_2281), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv548 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv544, param_2231, param_2241, lv547, lv536), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2821 = R.call_tir(cls.cast5, (lv548,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_2291: R.Tensor((2560,), dtype="float32") = model_params[229]
            param_2301: R.Tensor((2560,), dtype="float32") = model_params[230]
            lv549 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2821, param_2291, param_2301), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2331: R.Tensor((7680, 2560), dtype="float16") = model_params[233]
            param_2341: R.Tensor((7680,), dtype="float16") = model_params[234]
            lv550 = R.call_tir(cls.fused_NT_matmul6_add4, (lv549, param_2331, param_2341), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv551 = R.call_tir(cls.fused_reshape7_split1, (lv550,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2830: R.Tensor((1, 1, 32, 80), dtype="float16") = lv551[0]
            lv2831 = R.call_tir(cls.rotary_embedding1, (lv2830, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2832: R.Tensor((1, 1, 32, 80), dtype="float16") = lv551[1]
            lv2833 = R.call_tir(cls.rotary_embedding1, (lv2832, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2834: R.Object = kv_cache[38]
            lv2835 = R.call_tir(cls.squeeze1, (lv2833,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2836: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2834, lv2835, sinfo_args=(R.Object,))
            lv2837: R.Object = kv_cache[39]
            lv552: R.Tensor((1, 1, 32, 80), dtype="float16") = lv551[2]
            lv553 = R.call_tir(cls.fused_squeeze1, (lv552,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2840: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2837, lv553, sinfo_args=(R.Object,))
            lv2841: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2836, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2842: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2840, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2843 = R.call_tir(cls.reshape3, (lv2841,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2844 = R.call_tir(cls.reshape3, (lv2842,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2845 = R.call_tir(cls.transpose8, (lv2831,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2846 = R.call_tir(cls.transpose6, (lv2843,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2847 = R.call_tir(cls.transpose6, (lv2844,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv554 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv2845, lv2846, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv555 = R.call_tir(cls.fused_softmax1_cast8, (lv554,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2856 = R.call_tir(cls.matmul10, (lv555, lv2847), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv556 = R.call_tir(cls.fused_transpose9_reshape8, (lv2856,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2351: R.Tensor((2560, 2560), dtype="float16") = model_params[235]
            param_2361: R.Tensor((2560,), dtype="float16") = model_params[236]
            lv2862 = R.call_tir(cls.cast5, (lv548,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_2311: R.Tensor((2560,), dtype="float32") = model_params[231]
            param_2321: R.Tensor((2560,), dtype="float32") = model_params[232]
            lv557 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2862, param_2311, param_2321), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2371: R.Tensor((10240, 2560), dtype="float16") = model_params[237]
            param_2381: R.Tensor((10240,), dtype="float16") = model_params[238]
            lv558 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv557, param_2371, param_2381), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_2391: R.Tensor((2560, 10240), dtype="float16") = model_params[239]
            param_2401: R.Tensor((2560,), dtype="float16") = model_params[240]
            lv559 = R.call_tir(cls.fused_NT_matmul10_add5, (lv558, param_2391, param_2401), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv560 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv556, param_2351, param_2361, lv559, lv548), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2876 = R.call_tir(cls.cast5, (lv560,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_2411: R.Tensor((2560,), dtype="float32") = model_params[241]
            param_2421: R.Tensor((2560,), dtype="float32") = model_params[242]
            lv561 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2876, param_2411, param_2421), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2451: R.Tensor((7680, 2560), dtype="float16") = model_params[245]
            param_2461: R.Tensor((7680,), dtype="float16") = model_params[246]
            lv562 = R.call_tir(cls.fused_NT_matmul6_add4, (lv561, param_2451, param_2461), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv563 = R.call_tir(cls.fused_reshape7_split1, (lv562,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2885: R.Tensor((1, 1, 32, 80), dtype="float16") = lv563[0]
            lv2886 = R.call_tir(cls.rotary_embedding1, (lv2885, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2887: R.Tensor((1, 1, 32, 80), dtype="float16") = lv563[1]
            lv2888 = R.call_tir(cls.rotary_embedding1, (lv2887, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2889: R.Object = kv_cache[40]
            lv2890 = R.call_tir(cls.squeeze1, (lv2888,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2891: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2889, lv2890, sinfo_args=(R.Object,))
            lv2892: R.Object = kv_cache[41]
            lv564: R.Tensor((1, 1, 32, 80), dtype="float16") = lv563[2]
            lv565 = R.call_tir(cls.fused_squeeze1, (lv564,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2895: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2892, lv565, sinfo_args=(R.Object,))
            lv2896: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2891, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2897: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2895, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2898 = R.call_tir(cls.reshape3, (lv2896,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2899 = R.call_tir(cls.reshape3, (lv2897,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2900 = R.call_tir(cls.transpose8, (lv2886,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2901 = R.call_tir(cls.transpose6, (lv2898,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2902 = R.call_tir(cls.transpose6, (lv2899,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv566 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv2900, lv2901, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv567 = R.call_tir(cls.fused_softmax1_cast8, (lv566,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2911 = R.call_tir(cls.matmul10, (lv567, lv2902), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv568 = R.call_tir(cls.fused_transpose9_reshape8, (lv2911,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2471: R.Tensor((2560, 2560), dtype="float16") = model_params[247]
            param_2481: R.Tensor((2560,), dtype="float16") = model_params[248]
            lv2917 = R.call_tir(cls.cast5, (lv560,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_2431: R.Tensor((2560,), dtype="float32") = model_params[243]
            param_2441: R.Tensor((2560,), dtype="float32") = model_params[244]
            lv569 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2917, param_2431, param_2441), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2491: R.Tensor((10240, 2560), dtype="float16") = model_params[249]
            param_2501: R.Tensor((10240,), dtype="float16") = model_params[250]
            lv570 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv569, param_2491, param_2501), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_2511: R.Tensor((2560, 10240), dtype="float16") = model_params[251]
            param_2521: R.Tensor((2560,), dtype="float16") = model_params[252]
            lv571 = R.call_tir(cls.fused_NT_matmul10_add5, (lv570, param_2511, param_2521), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv572 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv568, param_2471, param_2481, lv571, lv560), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2931 = R.call_tir(cls.cast5, (lv572,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_2531: R.Tensor((2560,), dtype="float32") = model_params[253]
            param_2541: R.Tensor((2560,), dtype="float32") = model_params[254]
            lv573 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2931, param_2531, param_2541), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2571: R.Tensor((7680, 2560), dtype="float16") = model_params[257]
            param_2581: R.Tensor((7680,), dtype="float16") = model_params[258]
            lv574 = R.call_tir(cls.fused_NT_matmul6_add4, (lv573, param_2571, param_2581), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv575 = R.call_tir(cls.fused_reshape7_split1, (lv574,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2940: R.Tensor((1, 1, 32, 80), dtype="float16") = lv575[0]
            lv2941 = R.call_tir(cls.rotary_embedding1, (lv2940, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2942: R.Tensor((1, 1, 32, 80), dtype="float16") = lv575[1]
            lv2943 = R.call_tir(cls.rotary_embedding1, (lv2942, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2944: R.Object = kv_cache[42]
            lv2945 = R.call_tir(cls.squeeze1, (lv2943,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2946: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2944, lv2945, sinfo_args=(R.Object,))
            lv2947: R.Object = kv_cache[43]
            lv576: R.Tensor((1, 1, 32, 80), dtype="float16") = lv575[2]
            lv577 = R.call_tir(cls.fused_squeeze1, (lv576,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv2950: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2947, lv577, sinfo_args=(R.Object,))
            lv2951: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2946, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2952: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv2950, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv2953 = R.call_tir(cls.reshape3, (lv2951,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2954 = R.call_tir(cls.reshape3, (lv2952,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv2955 = R.call_tir(cls.transpose8, (lv2941,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv2956 = R.call_tir(cls.transpose6, (lv2953,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv2957 = R.call_tir(cls.transpose6, (lv2954,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv578 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv2955, lv2956, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv579 = R.call_tir(cls.fused_softmax1_cast8, (lv578,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv2966 = R.call_tir(cls.matmul10, (lv579, lv2957), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv580 = R.call_tir(cls.fused_transpose9_reshape8, (lv2966,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2591: R.Tensor((2560, 2560), dtype="float16") = model_params[259]
            param_2601: R.Tensor((2560,), dtype="float16") = model_params[260]
            lv2972 = R.call_tir(cls.cast5, (lv572,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_2551: R.Tensor((2560,), dtype="float32") = model_params[255]
            param_2561: R.Tensor((2560,), dtype="float32") = model_params[256]
            lv581 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2972, param_2551, param_2561), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2611: R.Tensor((10240, 2560), dtype="float16") = model_params[261]
            param_2621: R.Tensor((10240,), dtype="float16") = model_params[262]
            lv582 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv581, param_2611, param_2621), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_2631: R.Tensor((2560, 10240), dtype="float16") = model_params[263]
            param_2641: R.Tensor((2560,), dtype="float16") = model_params[264]
            lv583 = R.call_tir(cls.fused_NT_matmul10_add5, (lv582, param_2631, param_2641), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv584 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv580, param_2591, param_2601, lv583, lv572), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv2986 = R.call_tir(cls.cast5, (lv584,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_2651: R.Tensor((2560,), dtype="float32") = model_params[265]
            param_2661: R.Tensor((2560,), dtype="float32") = model_params[266]
            lv585 = R.call_tir(cls.fused_layer_norm1_cast6, (lv2986, param_2651, param_2661), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2691: R.Tensor((7680, 2560), dtype="float16") = model_params[269]
            param_2701: R.Tensor((7680,), dtype="float16") = model_params[270]
            lv586 = R.call_tir(cls.fused_NT_matmul6_add4, (lv585, param_2691, param_2701), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv587 = R.call_tir(cls.fused_reshape7_split1, (lv586,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv2995: R.Tensor((1, 1, 32, 80), dtype="float16") = lv587[0]
            lv2996 = R.call_tir(cls.rotary_embedding1, (lv2995, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2997: R.Tensor((1, 1, 32, 80), dtype="float16") = lv587[1]
            lv2998 = R.call_tir(cls.rotary_embedding1, (lv2997, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv2999: R.Object = kv_cache[44]
            lv3000 = R.call_tir(cls.squeeze1, (lv2998,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3001: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv2999, lv3000, sinfo_args=(R.Object,))
            lv3002: R.Object = kv_cache[45]
            lv588: R.Tensor((1, 1, 32, 80), dtype="float16") = lv587[2]
            lv589 = R.call_tir(cls.fused_squeeze1, (lv588,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3005: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3002, lv589, sinfo_args=(R.Object,))
            lv3006: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3001, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3007: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3005, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3008 = R.call_tir(cls.reshape3, (lv3006,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3009 = R.call_tir(cls.reshape3, (lv3007,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3010 = R.call_tir(cls.transpose8, (lv2996,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv3011 = R.call_tir(cls.transpose6, (lv3008,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv3012 = R.call_tir(cls.transpose6, (lv3009,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv590 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv3010, lv3011, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv591 = R.call_tir(cls.fused_softmax1_cast8, (lv590,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv3021 = R.call_tir(cls.matmul10, (lv591, lv3012), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv592 = R.call_tir(cls.fused_transpose9_reshape8, (lv3021,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2711: R.Tensor((2560, 2560), dtype="float16") = model_params[271]
            param_2721: R.Tensor((2560,), dtype="float16") = model_params[272]
            lv3027 = R.call_tir(cls.cast5, (lv584,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_2671: R.Tensor((2560,), dtype="float32") = model_params[267]
            param_2681: R.Tensor((2560,), dtype="float32") = model_params[268]
            lv593 = R.call_tir(cls.fused_layer_norm1_cast6, (lv3027, param_2671, param_2681), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2731: R.Tensor((10240, 2560), dtype="float16") = model_params[273]
            param_2741: R.Tensor((10240,), dtype="float16") = model_params[274]
            lv594 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv593, param_2731, param_2741), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_2751: R.Tensor((2560, 10240), dtype="float16") = model_params[275]
            param_2761: R.Tensor((2560,), dtype="float16") = model_params[276]
            lv595 = R.call_tir(cls.fused_NT_matmul10_add5, (lv594, param_2751, param_2761), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv596 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv592, param_2711, param_2721, lv595, lv584), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv3041 = R.call_tir(cls.cast5, (lv596,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_2771: R.Tensor((2560,), dtype="float32") = model_params[277]
            param_2781: R.Tensor((2560,), dtype="float32") = model_params[278]
            lv597 = R.call_tir(cls.fused_layer_norm1_cast6, (lv3041, param_2771, param_2781), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2811: R.Tensor((7680, 2560), dtype="float16") = model_params[281]
            param_2821: R.Tensor((7680,), dtype="float16") = model_params[282]
            lv598 = R.call_tir(cls.fused_NT_matmul6_add4, (lv597, param_2811, param_2821), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv599 = R.call_tir(cls.fused_reshape7_split1, (lv598,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv3050: R.Tensor((1, 1, 32, 80), dtype="float16") = lv599[0]
            lv3051 = R.call_tir(cls.rotary_embedding1, (lv3050, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3052: R.Tensor((1, 1, 32, 80), dtype="float16") = lv599[1]
            lv3053 = R.call_tir(cls.rotary_embedding1, (lv3052, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3054: R.Object = kv_cache[46]
            lv3055 = R.call_tir(cls.squeeze1, (lv3053,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3056: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3054, lv3055, sinfo_args=(R.Object,))
            lv3057: R.Object = kv_cache[47]
            lv600: R.Tensor((1, 1, 32, 80), dtype="float16") = lv599[2]
            lv601 = R.call_tir(cls.fused_squeeze1, (lv600,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3060: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3057, lv601, sinfo_args=(R.Object,))
            lv3061: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3056, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3062: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3060, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3063 = R.call_tir(cls.reshape3, (lv3061,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3064 = R.call_tir(cls.reshape3, (lv3062,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3065 = R.call_tir(cls.transpose8, (lv3051,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv3066 = R.call_tir(cls.transpose6, (lv3063,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv3067 = R.call_tir(cls.transpose6, (lv3064,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv602 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv3065, lv3066, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv603 = R.call_tir(cls.fused_softmax1_cast8, (lv602,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv3076 = R.call_tir(cls.matmul10, (lv603, lv3067), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv604 = R.call_tir(cls.fused_transpose9_reshape8, (lv3076,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2831: R.Tensor((2560, 2560), dtype="float16") = model_params[283]
            param_2841: R.Tensor((2560,), dtype="float16") = model_params[284]
            lv3082 = R.call_tir(cls.cast5, (lv596,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_2791: R.Tensor((2560,), dtype="float32") = model_params[279]
            param_2801: R.Tensor((2560,), dtype="float32") = model_params[280]
            lv605 = R.call_tir(cls.fused_layer_norm1_cast6, (lv3082, param_2791, param_2801), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2851: R.Tensor((10240, 2560), dtype="float16") = model_params[285]
            param_2861: R.Tensor((10240,), dtype="float16") = model_params[286]
            lv606 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv605, param_2851, param_2861), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_2871: R.Tensor((2560, 10240), dtype="float16") = model_params[287]
            param_2881: R.Tensor((2560,), dtype="float16") = model_params[288]
            lv607 = R.call_tir(cls.fused_NT_matmul10_add5, (lv606, param_2871, param_2881), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv608 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv604, param_2831, param_2841, lv607, lv596), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv3096 = R.call_tir(cls.cast5, (lv608,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_2891: R.Tensor((2560,), dtype="float32") = model_params[289]
            param_2901: R.Tensor((2560,), dtype="float32") = model_params[290]
            lv609 = R.call_tir(cls.fused_layer_norm1_cast6, (lv3096, param_2891, param_2901), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2931: R.Tensor((7680, 2560), dtype="float16") = model_params[293]
            param_2941: R.Tensor((7680,), dtype="float16") = model_params[294]
            lv610 = R.call_tir(cls.fused_NT_matmul6_add4, (lv609, param_2931, param_2941), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv611 = R.call_tir(cls.fused_reshape7_split1, (lv610,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv3105: R.Tensor((1, 1, 32, 80), dtype="float16") = lv611[0]
            lv3106 = R.call_tir(cls.rotary_embedding1, (lv3105, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3107: R.Tensor((1, 1, 32, 80), dtype="float16") = lv611[1]
            lv3108 = R.call_tir(cls.rotary_embedding1, (lv3107, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3109: R.Object = kv_cache[48]
            lv3110 = R.call_tir(cls.squeeze1, (lv3108,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3111: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3109, lv3110, sinfo_args=(R.Object,))
            lv3112: R.Object = kv_cache[49]
            lv612: R.Tensor((1, 1, 32, 80), dtype="float16") = lv611[2]
            lv613 = R.call_tir(cls.fused_squeeze1, (lv612,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3115: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3112, lv613, sinfo_args=(R.Object,))
            lv3116: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3111, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3117: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3115, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3118 = R.call_tir(cls.reshape3, (lv3116,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3119 = R.call_tir(cls.reshape3, (lv3117,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3120 = R.call_tir(cls.transpose8, (lv3106,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv3121 = R.call_tir(cls.transpose6, (lv3118,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv3122 = R.call_tir(cls.transpose6, (lv3119,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv614 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv3120, lv3121, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv615 = R.call_tir(cls.fused_softmax1_cast8, (lv614,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv3131 = R.call_tir(cls.matmul10, (lv615, lv3122), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv616 = R.call_tir(cls.fused_transpose9_reshape8, (lv3131,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2951: R.Tensor((2560, 2560), dtype="float16") = model_params[295]
            param_2961: R.Tensor((2560,), dtype="float16") = model_params[296]
            lv3137 = R.call_tir(cls.cast5, (lv608,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_2911: R.Tensor((2560,), dtype="float32") = model_params[291]
            param_2921: R.Tensor((2560,), dtype="float32") = model_params[292]
            lv617 = R.call_tir(cls.fused_layer_norm1_cast6, (lv3137, param_2911, param_2921), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_2971: R.Tensor((10240, 2560), dtype="float16") = model_params[297]
            param_2981: R.Tensor((10240,), dtype="float16") = model_params[298]
            lv618 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv617, param_2971, param_2981), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_2991: R.Tensor((2560, 10240), dtype="float16") = model_params[299]
            param_3001: R.Tensor((2560,), dtype="float16") = model_params[300]
            lv619 = R.call_tir(cls.fused_NT_matmul10_add5, (lv618, param_2991, param_3001), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv620 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv616, param_2951, param_2961, lv619, lv608), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv3151 = R.call_tir(cls.cast5, (lv620,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_3011: R.Tensor((2560,), dtype="float32") = model_params[301]
            param_3021: R.Tensor((2560,), dtype="float32") = model_params[302]
            lv621 = R.call_tir(cls.fused_layer_norm1_cast6, (lv3151, param_3011, param_3021), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_3051: R.Tensor((7680, 2560), dtype="float16") = model_params[305]
            param_3061: R.Tensor((7680,), dtype="float16") = model_params[306]
            lv622 = R.call_tir(cls.fused_NT_matmul6_add4, (lv621, param_3051, param_3061), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv623 = R.call_tir(cls.fused_reshape7_split1, (lv622,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv3160: R.Tensor((1, 1, 32, 80), dtype="float16") = lv623[0]
            lv3161 = R.call_tir(cls.rotary_embedding1, (lv3160, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3162: R.Tensor((1, 1, 32, 80), dtype="float16") = lv623[1]
            lv3163 = R.call_tir(cls.rotary_embedding1, (lv3162, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3164: R.Object = kv_cache[50]
            lv3165 = R.call_tir(cls.squeeze1, (lv3163,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3166: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3164, lv3165, sinfo_args=(R.Object,))
            lv3167: R.Object = kv_cache[51]
            lv624: R.Tensor((1, 1, 32, 80), dtype="float16") = lv623[2]
            lv625 = R.call_tir(cls.fused_squeeze1, (lv624,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3170: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3167, lv625, sinfo_args=(R.Object,))
            lv3171: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3166, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3172: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3170, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3173 = R.call_tir(cls.reshape3, (lv3171,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3174 = R.call_tir(cls.reshape3, (lv3172,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3175 = R.call_tir(cls.transpose8, (lv3161,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv3176 = R.call_tir(cls.transpose6, (lv3173,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv3177 = R.call_tir(cls.transpose6, (lv3174,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv626 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv3175, lv3176, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv627 = R.call_tir(cls.fused_softmax1_cast8, (lv626,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv3186 = R.call_tir(cls.matmul10, (lv627, lv3177), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv628 = R.call_tir(cls.fused_transpose9_reshape8, (lv3186,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_3071: R.Tensor((2560, 2560), dtype="float16") = model_params[307]
            param_3081: R.Tensor((2560,), dtype="float16") = model_params[308]
            lv3192 = R.call_tir(cls.cast5, (lv620,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_3031: R.Tensor((2560,), dtype="float32") = model_params[303]
            param_3041: R.Tensor((2560,), dtype="float32") = model_params[304]
            lv629 = R.call_tir(cls.fused_layer_norm1_cast6, (lv3192, param_3031, param_3041), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_3091: R.Tensor((10240, 2560), dtype="float16") = model_params[309]
            param_3101: R.Tensor((10240,), dtype="float16") = model_params[310]
            lv630 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv629, param_3091, param_3101), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_3111: R.Tensor((2560, 10240), dtype="float16") = model_params[311]
            param_3121: R.Tensor((2560,), dtype="float16") = model_params[312]
            lv631 = R.call_tir(cls.fused_NT_matmul10_add5, (lv630, param_3111, param_3121), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv632 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv628, param_3071, param_3081, lv631, lv620), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv3206 = R.call_tir(cls.cast5, (lv632,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_3131: R.Tensor((2560,), dtype="float32") = model_params[313]
            param_3141: R.Tensor((2560,), dtype="float32") = model_params[314]
            lv633 = R.call_tir(cls.fused_layer_norm1_cast6, (lv3206, param_3131, param_3141), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_3171: R.Tensor((7680, 2560), dtype="float16") = model_params[317]
            param_3181: R.Tensor((7680,), dtype="float16") = model_params[318]
            lv634 = R.call_tir(cls.fused_NT_matmul6_add4, (lv633, param_3171, param_3181), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv635 = R.call_tir(cls.fused_reshape7_split1, (lv634,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv3215: R.Tensor((1, 1, 32, 80), dtype="float16") = lv635[0]
            lv3216 = R.call_tir(cls.rotary_embedding1, (lv3215, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3217: R.Tensor((1, 1, 32, 80), dtype="float16") = lv635[1]
            lv3218 = R.call_tir(cls.rotary_embedding1, (lv3217, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3219: R.Object = kv_cache[52]
            lv3220 = R.call_tir(cls.squeeze1, (lv3218,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3221: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3219, lv3220, sinfo_args=(R.Object,))
            lv3222: R.Object = kv_cache[53]
            lv636: R.Tensor((1, 1, 32, 80), dtype="float16") = lv635[2]
            lv637 = R.call_tir(cls.fused_squeeze1, (lv636,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3225: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3222, lv637, sinfo_args=(R.Object,))
            lv3226: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3221, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3227: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3225, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3228 = R.call_tir(cls.reshape3, (lv3226,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3229 = R.call_tir(cls.reshape3, (lv3227,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3230 = R.call_tir(cls.transpose8, (lv3216,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv3231 = R.call_tir(cls.transpose6, (lv3228,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv3232 = R.call_tir(cls.transpose6, (lv3229,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv638 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv3230, lv3231, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv639 = R.call_tir(cls.fused_softmax1_cast8, (lv638,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv3241 = R.call_tir(cls.matmul10, (lv639, lv3232), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv640 = R.call_tir(cls.fused_transpose9_reshape8, (lv3241,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_3191: R.Tensor((2560, 2560), dtype="float16") = model_params[319]
            param_3201: R.Tensor((2560,), dtype="float16") = model_params[320]
            lv3247 = R.call_tir(cls.cast5, (lv632,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_3151: R.Tensor((2560,), dtype="float32") = model_params[315]
            param_3161: R.Tensor((2560,), dtype="float32") = model_params[316]
            lv641 = R.call_tir(cls.fused_layer_norm1_cast6, (lv3247, param_3151, param_3161), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_3211: R.Tensor((10240, 2560), dtype="float16") = model_params[321]
            param_3221: R.Tensor((10240,), dtype="float16") = model_params[322]
            lv642 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv641, param_3211, param_3221), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_3231: R.Tensor((2560, 10240), dtype="float16") = model_params[323]
            param_3241: R.Tensor((2560,), dtype="float16") = model_params[324]
            lv643 = R.call_tir(cls.fused_NT_matmul10_add5, (lv642, param_3231, param_3241), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv644 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv640, param_3191, param_3201, lv643, lv632), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv3261 = R.call_tir(cls.cast5, (lv644,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_3251: R.Tensor((2560,), dtype="float32") = model_params[325]
            param_3261: R.Tensor((2560,), dtype="float32") = model_params[326]
            lv645 = R.call_tir(cls.fused_layer_norm1_cast6, (lv3261, param_3251, param_3261), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_3291: R.Tensor((7680, 2560), dtype="float16") = model_params[329]
            param_3301: R.Tensor((7680,), dtype="float16") = model_params[330]
            lv646 = R.call_tir(cls.fused_NT_matmul6_add4, (lv645, param_3291, param_3301), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv647 = R.call_tir(cls.fused_reshape7_split1, (lv646,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv3270: R.Tensor((1, 1, 32, 80), dtype="float16") = lv647[0]
            lv3271 = R.call_tir(cls.rotary_embedding1, (lv3270, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3272: R.Tensor((1, 1, 32, 80), dtype="float16") = lv647[1]
            lv3273 = R.call_tir(cls.rotary_embedding1, (lv3272, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3274: R.Object = kv_cache[54]
            lv3275 = R.call_tir(cls.squeeze1, (lv3273,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3276: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3274, lv3275, sinfo_args=(R.Object,))
            lv3277: R.Object = kv_cache[55]
            lv648: R.Tensor((1, 1, 32, 80), dtype="float16") = lv647[2]
            lv649 = R.call_tir(cls.fused_squeeze1, (lv648,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3280: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3277, lv649, sinfo_args=(R.Object,))
            lv3281: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3276, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3282: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3280, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3283 = R.call_tir(cls.reshape3, (lv3281,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3284 = R.call_tir(cls.reshape3, (lv3282,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3285 = R.call_tir(cls.transpose8, (lv3271,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv3286 = R.call_tir(cls.transpose6, (lv3283,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv3287 = R.call_tir(cls.transpose6, (lv3284,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv650 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv3285, lv3286, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv651 = R.call_tir(cls.fused_softmax1_cast8, (lv650,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv3296 = R.call_tir(cls.matmul10, (lv651, lv3287), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv652 = R.call_tir(cls.fused_transpose9_reshape8, (lv3296,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_3311: R.Tensor((2560, 2560), dtype="float16") = model_params[331]
            param_3321: R.Tensor((2560,), dtype="float16") = model_params[332]
            lv3302 = R.call_tir(cls.cast5, (lv644,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_3271: R.Tensor((2560,), dtype="float32") = model_params[327]
            param_3281: R.Tensor((2560,), dtype="float32") = model_params[328]
            lv653 = R.call_tir(cls.fused_layer_norm1_cast6, (lv3302, param_3271, param_3281), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_3331: R.Tensor((10240, 2560), dtype="float16") = model_params[333]
            param_3341: R.Tensor((10240,), dtype="float16") = model_params[334]
            lv654 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv653, param_3331, param_3341), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_3351: R.Tensor((2560, 10240), dtype="float16") = model_params[335]
            param_3361: R.Tensor((2560,), dtype="float16") = model_params[336]
            lv655 = R.call_tir(cls.fused_NT_matmul10_add5, (lv654, param_3351, param_3361), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv656 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv652, param_3311, param_3321, lv655, lv644), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv3316 = R.call_tir(cls.cast5, (lv656,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_3371: R.Tensor((2560,), dtype="float32") = model_params[337]
            param_3381: R.Tensor((2560,), dtype="float32") = model_params[338]
            lv657 = R.call_tir(cls.fused_layer_norm1_cast6, (lv3316, param_3371, param_3381), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_3411: R.Tensor((7680, 2560), dtype="float16") = model_params[341]
            param_3421: R.Tensor((7680,), dtype="float16") = model_params[342]
            lv658 = R.call_tir(cls.fused_NT_matmul6_add4, (lv657, param_3411, param_3421), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv659 = R.call_tir(cls.fused_reshape7_split1, (lv658,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv3325: R.Tensor((1, 1, 32, 80), dtype="float16") = lv659[0]
            lv3326 = R.call_tir(cls.rotary_embedding1, (lv3325, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3327: R.Tensor((1, 1, 32, 80), dtype="float16") = lv659[1]
            lv3328 = R.call_tir(cls.rotary_embedding1, (lv3327, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3329: R.Object = kv_cache[56]
            lv3330 = R.call_tir(cls.squeeze1, (lv3328,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3331: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3329, lv3330, sinfo_args=(R.Object,))
            lv3332: R.Object = kv_cache[57]
            lv660: R.Tensor((1, 1, 32, 80), dtype="float16") = lv659[2]
            lv661 = R.call_tir(cls.fused_squeeze1, (lv660,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3335: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3332, lv661, sinfo_args=(R.Object,))
            lv3336: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3331, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3337: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3335, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3338 = R.call_tir(cls.reshape3, (lv3336,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3339 = R.call_tir(cls.reshape3, (lv3337,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3340 = R.call_tir(cls.transpose8, (lv3326,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv3341 = R.call_tir(cls.transpose6, (lv3338,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv3342 = R.call_tir(cls.transpose6, (lv3339,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv662 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv3340, lv3341, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv663 = R.call_tir(cls.fused_softmax1_cast8, (lv662,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv3351 = R.call_tir(cls.matmul10, (lv663, lv3342), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv664 = R.call_tir(cls.fused_transpose9_reshape8, (lv3351,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_3431: R.Tensor((2560, 2560), dtype="float16") = model_params[343]
            param_3441: R.Tensor((2560,), dtype="float16") = model_params[344]
            lv3357 = R.call_tir(cls.cast5, (lv656,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_3391: R.Tensor((2560,), dtype="float32") = model_params[339]
            param_3401: R.Tensor((2560,), dtype="float32") = model_params[340]
            lv665 = R.call_tir(cls.fused_layer_norm1_cast6, (lv3357, param_3391, param_3401), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_3451: R.Tensor((10240, 2560), dtype="float16") = model_params[345]
            param_3461: R.Tensor((10240,), dtype="float16") = model_params[346]
            lv666 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv665, param_3451, param_3461), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_3471: R.Tensor((2560, 10240), dtype="float16") = model_params[347]
            param_3481: R.Tensor((2560,), dtype="float16") = model_params[348]
            lv667 = R.call_tir(cls.fused_NT_matmul10_add5, (lv666, param_3471, param_3481), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv668 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv664, param_3431, param_3441, lv667, lv656), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv3371 = R.call_tir(cls.cast5, (lv668,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_3491: R.Tensor((2560,), dtype="float32") = model_params[349]
            param_3501: R.Tensor((2560,), dtype="float32") = model_params[350]
            lv669 = R.call_tir(cls.fused_layer_norm1_cast6, (lv3371, param_3491, param_3501), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_3531: R.Tensor((7680, 2560), dtype="float16") = model_params[353]
            param_3541: R.Tensor((7680,), dtype="float16") = model_params[354]
            lv670 = R.call_tir(cls.fused_NT_matmul6_add4, (lv669, param_3531, param_3541), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv671 = R.call_tir(cls.fused_reshape7_split1, (lv670,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv3380: R.Tensor((1, 1, 32, 80), dtype="float16") = lv671[0]
            lv3381 = R.call_tir(cls.rotary_embedding1, (lv3380, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3382: R.Tensor((1, 1, 32, 80), dtype="float16") = lv671[1]
            lv3383 = R.call_tir(cls.rotary_embedding1, (lv3382, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3384: R.Object = kv_cache[58]
            lv3385 = R.call_tir(cls.squeeze1, (lv3383,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3386: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3384, lv3385, sinfo_args=(R.Object,))
            lv3387: R.Object = kv_cache[59]
            lv672: R.Tensor((1, 1, 32, 80), dtype="float16") = lv671[2]
            lv673 = R.call_tir(cls.fused_squeeze1, (lv672,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3390: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3387, lv673, sinfo_args=(R.Object,))
            lv3391: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3386, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3392: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3390, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3393 = R.call_tir(cls.reshape3, (lv3391,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3394 = R.call_tir(cls.reshape3, (lv3392,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3395 = R.call_tir(cls.transpose8, (lv3381,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv3396 = R.call_tir(cls.transpose6, (lv3393,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv3397 = R.call_tir(cls.transpose6, (lv3394,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv674 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv3395, lv3396, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv675 = R.call_tir(cls.fused_softmax1_cast8, (lv674,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv3406 = R.call_tir(cls.matmul10, (lv675, lv3397), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv676 = R.call_tir(cls.fused_transpose9_reshape8, (lv3406,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_3551: R.Tensor((2560, 2560), dtype="float16") = model_params[355]
            param_3561: R.Tensor((2560,), dtype="float16") = model_params[356]
            lv3412 = R.call_tir(cls.cast5, (lv668,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_3511: R.Tensor((2560,), dtype="float32") = model_params[351]
            param_3521: R.Tensor((2560,), dtype="float32") = model_params[352]
            lv677 = R.call_tir(cls.fused_layer_norm1_cast6, (lv3412, param_3511, param_3521), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_3571: R.Tensor((10240, 2560), dtype="float16") = model_params[357]
            param_3581: R.Tensor((10240,), dtype="float16") = model_params[358]
            lv678 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv677, param_3571, param_3581), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_3591: R.Tensor((2560, 10240), dtype="float16") = model_params[359]
            param_3601: R.Tensor((2560,), dtype="float16") = model_params[360]
            lv679 = R.call_tir(cls.fused_NT_matmul10_add5, (lv678, param_3591, param_3601), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv680 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv676, param_3551, param_3561, lv679, lv668), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv3426 = R.call_tir(cls.cast5, (lv680,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_3611: R.Tensor((2560,), dtype="float32") = model_params[361]
            param_3621: R.Tensor((2560,), dtype="float32") = model_params[362]
            lv681 = R.call_tir(cls.fused_layer_norm1_cast6, (lv3426, param_3611, param_3621), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_3651: R.Tensor((7680, 2560), dtype="float16") = model_params[365]
            param_3661: R.Tensor((7680,), dtype="float16") = model_params[366]
            lv682 = R.call_tir(cls.fused_NT_matmul6_add4, (lv681, param_3651, param_3661), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv683 = R.call_tir(cls.fused_reshape7_split1, (lv682,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv3435: R.Tensor((1, 1, 32, 80), dtype="float16") = lv683[0]
            lv3436 = R.call_tir(cls.rotary_embedding1, (lv3435, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3437: R.Tensor((1, 1, 32, 80), dtype="float16") = lv683[1]
            lv3438 = R.call_tir(cls.rotary_embedding1, (lv3437, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3439: R.Object = kv_cache[60]
            lv3440 = R.call_tir(cls.squeeze1, (lv3438,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3441: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3439, lv3440, sinfo_args=(R.Object,))
            lv3442: R.Object = kv_cache[61]
            lv684: R.Tensor((1, 1, 32, 80), dtype="float16") = lv683[2]
            lv685 = R.call_tir(cls.fused_squeeze1, (lv684,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3445: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3442, lv685, sinfo_args=(R.Object,))
            lv3446: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3441, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3447: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3445, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3448 = R.call_tir(cls.reshape3, (lv3446,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3449 = R.call_tir(cls.reshape3, (lv3447,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3450 = R.call_tir(cls.transpose8, (lv3436,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv3451 = R.call_tir(cls.transpose6, (lv3448,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv3452 = R.call_tir(cls.transpose6, (lv3449,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv686 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv3450, lv3451, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv687 = R.call_tir(cls.fused_softmax1_cast8, (lv686,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv3461 = R.call_tir(cls.matmul10, (lv687, lv3452), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv688 = R.call_tir(cls.fused_transpose9_reshape8, (lv3461,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_3671: R.Tensor((2560, 2560), dtype="float16") = model_params[367]
            param_3681: R.Tensor((2560,), dtype="float16") = model_params[368]
            lv3467 = R.call_tir(cls.cast5, (lv680,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_3631: R.Tensor((2560,), dtype="float32") = model_params[363]
            param_3641: R.Tensor((2560,), dtype="float32") = model_params[364]
            lv689 = R.call_tir(cls.fused_layer_norm1_cast6, (lv3467, param_3631, param_3641), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_3691: R.Tensor((10240, 2560), dtype="float16") = model_params[369]
            param_3701: R.Tensor((10240,), dtype="float16") = model_params[370]
            lv690 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv689, param_3691, param_3701), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_3711: R.Tensor((2560, 10240), dtype="float16") = model_params[371]
            param_3721: R.Tensor((2560,), dtype="float16") = model_params[372]
            lv691 = R.call_tir(cls.fused_NT_matmul10_add5, (lv690, param_3711, param_3721), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv692 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7, (lv688, param_3671, param_3681, lv691, lv680), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv3481 = R.call_tir(cls.cast5, (lv692,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_3731: R.Tensor((2560,), dtype="float32") = model_params[373]
            param_3741: R.Tensor((2560,), dtype="float32") = model_params[374]
            lv693 = R.call_tir(cls.fused_layer_norm1_cast6, (lv3481, param_3731, param_3741), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_3771: R.Tensor((7680, 2560), dtype="float16") = model_params[377]
            param_3781: R.Tensor((7680,), dtype="float16") = model_params[378]
            lv694 = R.call_tir(cls.fused_NT_matmul6_add4, (lv693, param_3771, param_3781), out_sinfo=R.Tensor((1, 1, 7680), dtype="float16"))
            lv695 = R.call_tir(cls.fused_reshape7_split1, (lv694,), out_sinfo=[R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16"), R.Tensor((1, 1, 32, 80), dtype="float16")])
            lv3490: R.Tensor((1, 1, 32, 80), dtype="float16") = lv695[0]
            lv3491 = R.call_tir(cls.rotary_embedding1, (lv3490, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3492: R.Tensor((1, 1, 32, 80), dtype="float16") = lv695[1]
            lv3493 = R.call_tir(cls.rotary_embedding1, (lv3492, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 1, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv3494: R.Object = kv_cache[62]
            lv3495 = R.call_tir(cls.squeeze1, (lv3493,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3496: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3494, lv3495, sinfo_args=(R.Object,))
            lv3497: R.Object = kv_cache[63]
            lv696: R.Tensor((1, 1, 32, 80), dtype="float16") = lv695[2]
            lv697 = R.call_tir(cls.fused_squeeze1, (lv696,), out_sinfo=R.Tensor((1, 32, 80), dtype="float16"))
            lv3500: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv3497, lv697, sinfo_args=(R.Object,))
            lv3501: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3496, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3502: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv3500, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv3503 = R.call_tir(cls.reshape3, (lv3501,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3504 = R.call_tir(cls.reshape3, (lv3502,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv3505 = R.call_tir(cls.transpose8, (lv3491,), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv3506 = R.call_tir(cls.transpose6, (lv3503,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv3507 = R.call_tir(cls.transpose6, (lv3504,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv698 = R.call_tir(cls.fused_NT_matmul7_divide1_maximum1_minimum1_cast7, (lv3505, lv3506, lv1775), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float32"))
            lv699 = R.call_tir(cls.fused_softmax1_cast8, (lv698,), out_sinfo=R.Tensor((1, 32, 1, m), dtype="float16"))
            lv3516 = R.call_tir(cls.matmul10, (lv699, lv3507), out_sinfo=R.Tensor((1, 32, 1, 80), dtype="float16"))
            lv700 = R.call_tir(cls.fused_transpose9_reshape8, (lv3516,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_3791: R.Tensor((2560, 2560), dtype="float16") = model_params[379]
            param_3801: R.Tensor((2560,), dtype="float16") = model_params[380]
            lv3522 = R.call_tir(cls.cast5, (lv692,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_3751: R.Tensor((2560,), dtype="float32") = model_params[375]
            param_3761: R.Tensor((2560,), dtype="float32") = model_params[376]
            lv701 = R.call_tir(cls.fused_layer_norm1_cast6, (lv3522, param_3751, param_3761), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            param_3811: R.Tensor((10240, 2560), dtype="float16") = model_params[381]
            param_3821: R.Tensor((10240,), dtype="float16") = model_params[382]
            lv702 = R.call_tir(cls.fused_NT_matmul9_add6_gelu1, (lv701, param_3811, param_3821), out_sinfo=R.Tensor((1, 1, 10240), dtype="float16"))
            param_3831: R.Tensor((2560, 10240), dtype="float16") = model_params[383]
            param_3841: R.Tensor((2560,), dtype="float16") = model_params[384]
            lv703 = R.call_tir(cls.fused_NT_matmul10_add5, (lv702, param_3831, param_3841), out_sinfo=R.Tensor((1, 1, 2560), dtype="float16"))
            lv704 = R.call_tir(cls.fused_NT_matmul8_add5_add7_add7_cast5, (lv700, param_3791, param_3801, lv703, lv692), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_3851: R.Tensor((2560,), dtype="float32") = model_params[385]
            param_3861: R.Tensor((2560,), dtype="float32") = model_params[386]
            lv3537 = R.call_tir(cls.layer_norm1, (lv704, param_3851, param_3861), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            lv705 = R.call_tir(cls.fused_slice1_cast4, (lv3537,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_3871: R.Tensor((50280, 2560), dtype="float32") = model_params[387]
            lv321_1 = R.call_tir(cls.NT_matmul5, (lv705, param_3871), out_sinfo=R.Tensor((1, 1, 50280), dtype="float32"))
            gv1: R.Tuple(R.Tensor((1, 1, 50280), dtype="float32"), R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object)) = lv321_1, (lv1791, lv1795, lv1846, lv1850, lv1901, lv1905, lv1956, lv1960, lv2011, lv2015, lv2066, lv2070, lv2121, lv2125, lv2176, lv2180, lv2231, lv2235, lv2286, lv2290, lv2341, lv2345, lv2396, lv2400, lv2451, lv2455, lv2506, lv2510, lv2561, lv2565, lv2616, lv2620, lv2671, lv2675, lv2726, lv2730, lv2781, lv2785, lv2836, lv2840, lv2891, lv2895, lv2946, lv2950, lv3001, lv3005, lv3056, lv3060, lv3111, lv3115, lv3166, lv3170, lv3221, lv3225, lv3276, lv3280, lv3331, lv3335, lv3386, lv3390, lv3441, lv3445, lv3496, lv3500)
            R.output(gv1)
        return gv1

    @R.function
    def get_metadata() -> R.Object:
        R.func_attr({"tir_var_upper_bound": {"m": 4096, "n": 4096}})
        return R.str("{\"model_name\": \"dolly-v2-3b\", \"max_window_size\": 4096, \"stop_tokens\": [2], \"add_prefix_space\": false, \"prefill_chunk_size\": -1, \"sliding_window\": -1}")

    @R.function
    def prefill(input_ids: R.Tensor((1, "n"), dtype="int32"), all_seq_len: R.Shape(["m"]), kv_cache: R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object), model_params: R.Tuple(R.Tensor((50280, 2560), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((7680, 2560), dtype="float16"), R.Tensor((7680,), dtype="float16"), R.Tensor((2560, 2560), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((10240, 2560), dtype="float16"), R.Tensor((10240,), dtype="float16"), R.Tensor((2560, 10240), dtype="float16"), R.Tensor((2560,), dtype="float16"), R.Tensor((2560,), dtype="float32"), R.Tensor((2560,), dtype="float32"), R.Tensor((50280, 2560), dtype="float32"))) -> R.Tuple(R.Tensor((1, 1, 50280), dtype="float32"), R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object)):
        n = T.int64()
        m = T.int64()
        R.func_attr({"num_input": 3, "tir_var_upper_bound": {"m": 4096, "n": 4096}})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.reshape, (input_ids,), out_sinfo=R.Tensor((n,), dtype="int32"))
            param_0: R.Tensor((50280, 2560), dtype="float16") = model_params[0]
            lv1 = R.call_tir(cls.take, (param_0, lv), out_sinfo=R.Tensor((n, 2560), dtype="float16"))
            lv2 = R.call_tir(cls.reshape1, (lv1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv_1 = R.call_tir(cls.fused_min_max_triu_te_broadcast_to, R.tuple(), out_sinfo=R.Tensor((1, 1, n, n), dtype="float16"), tir_vars=R.shape([n]))
            lv5 = R.call_tir(cls.extend_te, (lv_1,), out_sinfo=R.Tensor((1, 1, n, m), dtype="float16"))
            lv6 = R.call_tir(cls.cast, (lv2,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_1: R.Tensor((2560,), dtype="float32") = model_params[1]
            param_2: R.Tensor((2560,), dtype="float32") = model_params[2]
            lv1_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv6, param_1, param_2), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_5: R.Tensor((7680, 2560), dtype="float16") = model_params[5]
            param_6: R.Tensor((7680,), dtype="float16") = model_params[6]
            lv2_1 = R.call_tir(cls.fused_NT_matmul_add, (lv1_1, param_5, param_6), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv13 = R.call_tir(cls.reshape2, (lv2_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv14 = R.call_tir(cls.split, (lv13,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv15: R.Tensor((1, n, 32, 80), dtype="float16") = lv14[0]
            lv16 = R.call_tir(cls.rotary_embedding, (lv15, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv17: R.Tensor((1, n, 32, 80), dtype="float16") = lv14[1]
            lv18 = R.call_tir(cls.rotary_embedding, (lv17, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv19: R.Object = kv_cache[0]
            lv20 = R.call_tir(cls.squeeze, (lv18,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv21: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv19, lv20, sinfo_args=(R.Object,))
            lv22: R.Object = kv_cache[1]
            lv3: R.Tensor((1, n, 32, 80), dtype="float16") = lv14[2]
            lv4 = R.call_tir(cls.fused_squeeze, (lv3,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv25: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv22, lv4, sinfo_args=(R.Object,))
            lv26: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv21, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv27: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv25, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv28 = R.call_tir(cls.reshape3, (lv26,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv29 = R.call_tir(cls.reshape3, (lv27,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv30 = R.call_tir(cls.transpose6, (lv16,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv31 = R.call_tir(cls.transpose6, (lv28,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv32 = R.call_tir(cls.transpose6, (lv29,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv5_1 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv30, lv31, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv6_1 = R.call_tir(cls.fused_softmax_cast3, (lv5_1,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv41 = R.call_tir(cls.matmul8, (lv6_1, lv32), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv42 = R.call_tir(cls.transpose7, (lv41,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv43 = R.call_tir(cls.reshape4, (lv42,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_7: R.Tensor((2560, 2560), dtype="float16") = model_params[7]
            param_8: R.Tensor((2560,), dtype="float16") = model_params[8]
            lv47 = R.call_tir(cls.cast, (lv2,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_3: R.Tensor((2560,), dtype="float32") = model_params[3]
            param_4: R.Tensor((2560,), dtype="float32") = model_params[4]
            lv7 = R.call_tir(cls.fused_layer_norm_cast1, (lv47, param_3, param_4), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_9: R.Tensor((10240, 2560), dtype="float16") = model_params[9]
            param_10: R.Tensor((10240,), dtype="float16") = model_params[10]
            lv8 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv7, param_9, param_10), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_11: R.Tensor((2560, 10240), dtype="float16") = model_params[11]
            param_12: R.Tensor((2560,), dtype="float16") = model_params[12]
            lv9 = R.call_tir(cls.fused_NT_matmul4_add1, (lv8, param_11, param_12), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv10 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv43, param_7, param_8, lv9, lv2), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv61 = R.call_tir(cls.cast, (lv10,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_13: R.Tensor((2560,), dtype="float32") = model_params[13]
            param_14: R.Tensor((2560,), dtype="float32") = model_params[14]
            lv11 = R.call_tir(cls.fused_layer_norm_cast1, (lv61, param_13, param_14), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_17: R.Tensor((7680, 2560), dtype="float16") = model_params[17]
            param_18: R.Tensor((7680,), dtype="float16") = model_params[18]
            lv12 = R.call_tir(cls.fused_NT_matmul_add, (lv11, param_17, param_18), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv68 = R.call_tir(cls.reshape2, (lv12,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv69 = R.call_tir(cls.split, (lv68,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv70: R.Tensor((1, n, 32, 80), dtype="float16") = lv69[0]
            lv71 = R.call_tir(cls.rotary_embedding, (lv70, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv72: R.Tensor((1, n, 32, 80), dtype="float16") = lv69[1]
            lv73 = R.call_tir(cls.rotary_embedding, (lv72, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv74: R.Object = kv_cache[2]
            lv75 = R.call_tir(cls.squeeze, (lv73,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv76: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv74, lv75, sinfo_args=(R.Object,))
            lv77: R.Object = kv_cache[3]
            lv13_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv69[2]
            lv14_1 = R.call_tir(cls.fused_squeeze, (lv13_1,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv80: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv77, lv14_1, sinfo_args=(R.Object,))
            lv81: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv76, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv82: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv80, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv83 = R.call_tir(cls.reshape3, (lv81,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv84 = R.call_tir(cls.reshape3, (lv82,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv85 = R.call_tir(cls.transpose6, (lv71,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv86 = R.call_tir(cls.transpose6, (lv83,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv87 = R.call_tir(cls.transpose6, (lv84,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv15_1 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv85, lv86, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv16_1 = R.call_tir(cls.fused_softmax_cast3, (lv15_1,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv96 = R.call_tir(cls.matmul8, (lv16_1, lv87), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv97 = R.call_tir(cls.transpose7, (lv96,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv98 = R.call_tir(cls.reshape4, (lv97,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_19: R.Tensor((2560, 2560), dtype="float16") = model_params[19]
            param_20: R.Tensor((2560,), dtype="float16") = model_params[20]
            lv102 = R.call_tir(cls.cast, (lv10,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_15: R.Tensor((2560,), dtype="float32") = model_params[15]
            param_16: R.Tensor((2560,), dtype="float32") = model_params[16]
            lv17_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv102, param_15, param_16), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_21: R.Tensor((10240, 2560), dtype="float16") = model_params[21]
            param_22: R.Tensor((10240,), dtype="float16") = model_params[22]
            lv18_1 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv17_1, param_21, param_22), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_23: R.Tensor((2560, 10240), dtype="float16") = model_params[23]
            param_24: R.Tensor((2560,), dtype="float16") = model_params[24]
            lv19_1 = R.call_tir(cls.fused_NT_matmul4_add1, (lv18_1, param_23, param_24), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv20_1 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv98, param_19, param_20, lv19_1, lv10), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv116 = R.call_tir(cls.cast, (lv20_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_25: R.Tensor((2560,), dtype="float32") = model_params[25]
            param_26: R.Tensor((2560,), dtype="float32") = model_params[26]
            lv21_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv116, param_25, param_26), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_29: R.Tensor((7680, 2560), dtype="float16") = model_params[29]
            param_30: R.Tensor((7680,), dtype="float16") = model_params[30]
            lv22_1 = R.call_tir(cls.fused_NT_matmul_add, (lv21_1, param_29, param_30), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv123 = R.call_tir(cls.reshape2, (lv22_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv124 = R.call_tir(cls.split, (lv123,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv125: R.Tensor((1, n, 32, 80), dtype="float16") = lv124[0]
            lv126 = R.call_tir(cls.rotary_embedding, (lv125, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv127: R.Tensor((1, n, 32, 80), dtype="float16") = lv124[1]
            lv128 = R.call_tir(cls.rotary_embedding, (lv127, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv129: R.Object = kv_cache[4]
            lv130 = R.call_tir(cls.squeeze, (lv128,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv131: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv129, lv130, sinfo_args=(R.Object,))
            lv132: R.Object = kv_cache[5]
            lv23: R.Tensor((1, n, 32, 80), dtype="float16") = lv124[2]
            lv24 = R.call_tir(cls.fused_squeeze, (lv23,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv135: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv132, lv24, sinfo_args=(R.Object,))
            lv136: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv131, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv137: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv135, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv138 = R.call_tir(cls.reshape3, (lv136,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv139 = R.call_tir(cls.reshape3, (lv137,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv140 = R.call_tir(cls.transpose6, (lv126,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv141 = R.call_tir(cls.transpose6, (lv138,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv142 = R.call_tir(cls.transpose6, (lv139,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv25_1 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv140, lv141, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv26_1 = R.call_tir(cls.fused_softmax_cast3, (lv25_1,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv151 = R.call_tir(cls.matmul8, (lv26_1, lv142), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv152 = R.call_tir(cls.transpose7, (lv151,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv153 = R.call_tir(cls.reshape4, (lv152,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_31: R.Tensor((2560, 2560), dtype="float16") = model_params[31]
            param_32: R.Tensor((2560,), dtype="float16") = model_params[32]
            lv157 = R.call_tir(cls.cast, (lv20_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_27: R.Tensor((2560,), dtype="float32") = model_params[27]
            param_28: R.Tensor((2560,), dtype="float32") = model_params[28]
            lv27_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv157, param_27, param_28), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_33: R.Tensor((10240, 2560), dtype="float16") = model_params[33]
            param_34: R.Tensor((10240,), dtype="float16") = model_params[34]
            lv28_1 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv27_1, param_33, param_34), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_35: R.Tensor((2560, 10240), dtype="float16") = model_params[35]
            param_36: R.Tensor((2560,), dtype="float16") = model_params[36]
            lv29_1 = R.call_tir(cls.fused_NT_matmul4_add1, (lv28_1, param_35, param_36), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv30_1 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv153, param_31, param_32, lv29_1, lv20_1), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv171 = R.call_tir(cls.cast, (lv30_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_37: R.Tensor((2560,), dtype="float32") = model_params[37]
            param_38: R.Tensor((2560,), dtype="float32") = model_params[38]
            lv31_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv171, param_37, param_38), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_41: R.Tensor((7680, 2560), dtype="float16") = model_params[41]
            param_42: R.Tensor((7680,), dtype="float16") = model_params[42]
            lv32_1 = R.call_tir(cls.fused_NT_matmul_add, (lv31_1, param_41, param_42), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv178 = R.call_tir(cls.reshape2, (lv32_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv179 = R.call_tir(cls.split, (lv178,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv180: R.Tensor((1, n, 32, 80), dtype="float16") = lv179[0]
            lv181 = R.call_tir(cls.rotary_embedding, (lv180, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv182: R.Tensor((1, n, 32, 80), dtype="float16") = lv179[1]
            lv183 = R.call_tir(cls.rotary_embedding, (lv182, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv184: R.Object = kv_cache[6]
            lv185 = R.call_tir(cls.squeeze, (lv183,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv186: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv184, lv185, sinfo_args=(R.Object,))
            lv187: R.Object = kv_cache[7]
            lv33: R.Tensor((1, n, 32, 80), dtype="float16") = lv179[2]
            lv34 = R.call_tir(cls.fused_squeeze, (lv33,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv190: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv187, lv34, sinfo_args=(R.Object,))
            lv191: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv186, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv192: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv190, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv193 = R.call_tir(cls.reshape3, (lv191,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv194 = R.call_tir(cls.reshape3, (lv192,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv195 = R.call_tir(cls.transpose6, (lv181,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv196 = R.call_tir(cls.transpose6, (lv193,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv197 = R.call_tir(cls.transpose6, (lv194,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv35 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv195, lv196, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv36 = R.call_tir(cls.fused_softmax_cast3, (lv35,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv206 = R.call_tir(cls.matmul8, (lv36, lv197), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv207 = R.call_tir(cls.transpose7, (lv206,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv208 = R.call_tir(cls.reshape4, (lv207,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_43: R.Tensor((2560, 2560), dtype="float16") = model_params[43]
            param_44: R.Tensor((2560,), dtype="float16") = model_params[44]
            lv212 = R.call_tir(cls.cast, (lv30_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_39: R.Tensor((2560,), dtype="float32") = model_params[39]
            param_40: R.Tensor((2560,), dtype="float32") = model_params[40]
            lv37 = R.call_tir(cls.fused_layer_norm_cast1, (lv212, param_39, param_40), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_45: R.Tensor((10240, 2560), dtype="float16") = model_params[45]
            param_46: R.Tensor((10240,), dtype="float16") = model_params[46]
            lv38 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv37, param_45, param_46), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_47: R.Tensor((2560, 10240), dtype="float16") = model_params[47]
            param_48: R.Tensor((2560,), dtype="float16") = model_params[48]
            lv39 = R.call_tir(cls.fused_NT_matmul4_add1, (lv38, param_47, param_48), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv40 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv208, param_43, param_44, lv39, lv30_1), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv226 = R.call_tir(cls.cast, (lv40,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_49: R.Tensor((2560,), dtype="float32") = model_params[49]
            param_50: R.Tensor((2560,), dtype="float32") = model_params[50]
            lv41_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv226, param_49, param_50), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_53: R.Tensor((7680, 2560), dtype="float16") = model_params[53]
            param_54: R.Tensor((7680,), dtype="float16") = model_params[54]
            lv42_1 = R.call_tir(cls.fused_NT_matmul_add, (lv41_1, param_53, param_54), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv233 = R.call_tir(cls.reshape2, (lv42_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv234 = R.call_tir(cls.split, (lv233,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv235: R.Tensor((1, n, 32, 80), dtype="float16") = lv234[0]
            lv236 = R.call_tir(cls.rotary_embedding, (lv235, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv237: R.Tensor((1, n, 32, 80), dtype="float16") = lv234[1]
            lv238 = R.call_tir(cls.rotary_embedding, (lv237, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv239: R.Object = kv_cache[8]
            lv240 = R.call_tir(cls.squeeze, (lv238,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv241: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv239, lv240, sinfo_args=(R.Object,))
            lv242: R.Object = kv_cache[9]
            lv43_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv234[2]
            lv44 = R.call_tir(cls.fused_squeeze, (lv43_1,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv245: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv242, lv44, sinfo_args=(R.Object,))
            lv246: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv241, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv247: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv245, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv248 = R.call_tir(cls.reshape3, (lv246,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv249 = R.call_tir(cls.reshape3, (lv247,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv250 = R.call_tir(cls.transpose6, (lv236,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv251 = R.call_tir(cls.transpose6, (lv248,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv252 = R.call_tir(cls.transpose6, (lv249,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv45 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv250, lv251, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv46 = R.call_tir(cls.fused_softmax_cast3, (lv45,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv261 = R.call_tir(cls.matmul8, (lv46, lv252), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv262 = R.call_tir(cls.transpose7, (lv261,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv263 = R.call_tir(cls.reshape4, (lv262,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_55: R.Tensor((2560, 2560), dtype="float16") = model_params[55]
            param_56: R.Tensor((2560,), dtype="float16") = model_params[56]
            lv267 = R.call_tir(cls.cast, (lv40,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_51: R.Tensor((2560,), dtype="float32") = model_params[51]
            param_52: R.Tensor((2560,), dtype="float32") = model_params[52]
            lv47_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv267, param_51, param_52), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_57: R.Tensor((10240, 2560), dtype="float16") = model_params[57]
            param_58: R.Tensor((10240,), dtype="float16") = model_params[58]
            lv48 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv47_1, param_57, param_58), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_59: R.Tensor((2560, 10240), dtype="float16") = model_params[59]
            param_60: R.Tensor((2560,), dtype="float16") = model_params[60]
            lv49 = R.call_tir(cls.fused_NT_matmul4_add1, (lv48, param_59, param_60), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv50 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv263, param_55, param_56, lv49, lv40), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv281 = R.call_tir(cls.cast, (lv50,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_61: R.Tensor((2560,), dtype="float32") = model_params[61]
            param_62: R.Tensor((2560,), dtype="float32") = model_params[62]
            lv51 = R.call_tir(cls.fused_layer_norm_cast1, (lv281, param_61, param_62), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_65: R.Tensor((7680, 2560), dtype="float16") = model_params[65]
            param_66: R.Tensor((7680,), dtype="float16") = model_params[66]
            lv52 = R.call_tir(cls.fused_NT_matmul_add, (lv51, param_65, param_66), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv288 = R.call_tir(cls.reshape2, (lv52,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv289 = R.call_tir(cls.split, (lv288,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv290: R.Tensor((1, n, 32, 80), dtype="float16") = lv289[0]
            lv291 = R.call_tir(cls.rotary_embedding, (lv290, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv292: R.Tensor((1, n, 32, 80), dtype="float16") = lv289[1]
            lv293 = R.call_tir(cls.rotary_embedding, (lv292, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv294: R.Object = kv_cache[10]
            lv295 = R.call_tir(cls.squeeze, (lv293,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv296: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv294, lv295, sinfo_args=(R.Object,))
            lv297: R.Object = kv_cache[11]
            lv53: R.Tensor((1, n, 32, 80), dtype="float16") = lv289[2]
            lv54 = R.call_tir(cls.fused_squeeze, (lv53,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv300: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv297, lv54, sinfo_args=(R.Object,))
            lv301: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv296, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv302: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv300, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv303 = R.call_tir(cls.reshape3, (lv301,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv304 = R.call_tir(cls.reshape3, (lv302,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv305 = R.call_tir(cls.transpose6, (lv291,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv306 = R.call_tir(cls.transpose6, (lv303,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv307 = R.call_tir(cls.transpose6, (lv304,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv55 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv305, lv306, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv56 = R.call_tir(cls.fused_softmax_cast3, (lv55,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv316 = R.call_tir(cls.matmul8, (lv56, lv307), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv317 = R.call_tir(cls.transpose7, (lv316,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv318 = R.call_tir(cls.reshape4, (lv317,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_67: R.Tensor((2560, 2560), dtype="float16") = model_params[67]
            param_68: R.Tensor((2560,), dtype="float16") = model_params[68]
            lv322 = R.call_tir(cls.cast, (lv50,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_63: R.Tensor((2560,), dtype="float32") = model_params[63]
            param_64: R.Tensor((2560,), dtype="float32") = model_params[64]
            lv57 = R.call_tir(cls.fused_layer_norm_cast1, (lv322, param_63, param_64), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_69: R.Tensor((10240, 2560), dtype="float16") = model_params[69]
            param_70: R.Tensor((10240,), dtype="float16") = model_params[70]
            lv58 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv57, param_69, param_70), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_71: R.Tensor((2560, 10240), dtype="float16") = model_params[71]
            param_72: R.Tensor((2560,), dtype="float16") = model_params[72]
            lv59 = R.call_tir(cls.fused_NT_matmul4_add1, (lv58, param_71, param_72), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv60 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv318, param_67, param_68, lv59, lv50), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv336 = R.call_tir(cls.cast, (lv60,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_73: R.Tensor((2560,), dtype="float32") = model_params[73]
            param_74: R.Tensor((2560,), dtype="float32") = model_params[74]
            lv61_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv336, param_73, param_74), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_77: R.Tensor((7680, 2560), dtype="float16") = model_params[77]
            param_78: R.Tensor((7680,), dtype="float16") = model_params[78]
            lv62 = R.call_tir(cls.fused_NT_matmul_add, (lv61_1, param_77, param_78), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv343 = R.call_tir(cls.reshape2, (lv62,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv344 = R.call_tir(cls.split, (lv343,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv345: R.Tensor((1, n, 32, 80), dtype="float16") = lv344[0]
            lv346 = R.call_tir(cls.rotary_embedding, (lv345, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv347: R.Tensor((1, n, 32, 80), dtype="float16") = lv344[1]
            lv348 = R.call_tir(cls.rotary_embedding, (lv347, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv349: R.Object = kv_cache[12]
            lv350 = R.call_tir(cls.squeeze, (lv348,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv351: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv349, lv350, sinfo_args=(R.Object,))
            lv352: R.Object = kv_cache[13]
            lv63: R.Tensor((1, n, 32, 80), dtype="float16") = lv344[2]
            lv64 = R.call_tir(cls.fused_squeeze, (lv63,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv355: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv352, lv64, sinfo_args=(R.Object,))
            lv356: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv351, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv357: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv355, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv358 = R.call_tir(cls.reshape3, (lv356,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv359 = R.call_tir(cls.reshape3, (lv357,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv360 = R.call_tir(cls.transpose6, (lv346,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv361 = R.call_tir(cls.transpose6, (lv358,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv362 = R.call_tir(cls.transpose6, (lv359,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv65 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv360, lv361, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv66 = R.call_tir(cls.fused_softmax_cast3, (lv65,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv371 = R.call_tir(cls.matmul8, (lv66, lv362), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv372 = R.call_tir(cls.transpose7, (lv371,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv373 = R.call_tir(cls.reshape4, (lv372,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_79: R.Tensor((2560, 2560), dtype="float16") = model_params[79]
            param_80: R.Tensor((2560,), dtype="float16") = model_params[80]
            lv377 = R.call_tir(cls.cast, (lv60,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_75: R.Tensor((2560,), dtype="float32") = model_params[75]
            param_76: R.Tensor((2560,), dtype="float32") = model_params[76]
            lv67 = R.call_tir(cls.fused_layer_norm_cast1, (lv377, param_75, param_76), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_81: R.Tensor((10240, 2560), dtype="float16") = model_params[81]
            param_82: R.Tensor((10240,), dtype="float16") = model_params[82]
            lv68_1 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv67, param_81, param_82), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_83: R.Tensor((2560, 10240), dtype="float16") = model_params[83]
            param_84: R.Tensor((2560,), dtype="float16") = model_params[84]
            lv69_1 = R.call_tir(cls.fused_NT_matmul4_add1, (lv68_1, param_83, param_84), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv70_1 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv373, param_79, param_80, lv69_1, lv60), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv391 = R.call_tir(cls.cast, (lv70_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_85: R.Tensor((2560,), dtype="float32") = model_params[85]
            param_86: R.Tensor((2560,), dtype="float32") = model_params[86]
            lv71_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv391, param_85, param_86), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_89: R.Tensor((7680, 2560), dtype="float16") = model_params[89]
            param_90: R.Tensor((7680,), dtype="float16") = model_params[90]
            lv72_1 = R.call_tir(cls.fused_NT_matmul_add, (lv71_1, param_89, param_90), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv398 = R.call_tir(cls.reshape2, (lv72_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv399 = R.call_tir(cls.split, (lv398,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv400: R.Tensor((1, n, 32, 80), dtype="float16") = lv399[0]
            lv401 = R.call_tir(cls.rotary_embedding, (lv400, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv402: R.Tensor((1, n, 32, 80), dtype="float16") = lv399[1]
            lv403 = R.call_tir(cls.rotary_embedding, (lv402, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv404: R.Object = kv_cache[14]
            lv405 = R.call_tir(cls.squeeze, (lv403,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv406: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv404, lv405, sinfo_args=(R.Object,))
            lv407: R.Object = kv_cache[15]
            lv73_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv399[2]
            lv74_1 = R.call_tir(cls.fused_squeeze, (lv73_1,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv410: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv407, lv74_1, sinfo_args=(R.Object,))
            lv411: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv406, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv412: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv410, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv413 = R.call_tir(cls.reshape3, (lv411,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv414 = R.call_tir(cls.reshape3, (lv412,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv415 = R.call_tir(cls.transpose6, (lv401,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv416 = R.call_tir(cls.transpose6, (lv413,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv417 = R.call_tir(cls.transpose6, (lv414,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv75_1 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv415, lv416, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv76_1 = R.call_tir(cls.fused_softmax_cast3, (lv75_1,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv426 = R.call_tir(cls.matmul8, (lv76_1, lv417), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv427 = R.call_tir(cls.transpose7, (lv426,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv428 = R.call_tir(cls.reshape4, (lv427,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_91: R.Tensor((2560, 2560), dtype="float16") = model_params[91]
            param_92: R.Tensor((2560,), dtype="float16") = model_params[92]
            lv432 = R.call_tir(cls.cast, (lv70_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_87: R.Tensor((2560,), dtype="float32") = model_params[87]
            param_88: R.Tensor((2560,), dtype="float32") = model_params[88]
            lv77_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv432, param_87, param_88), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_93: R.Tensor((10240, 2560), dtype="float16") = model_params[93]
            param_94: R.Tensor((10240,), dtype="float16") = model_params[94]
            lv78 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv77_1, param_93, param_94), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_95: R.Tensor((2560, 10240), dtype="float16") = model_params[95]
            param_96: R.Tensor((2560,), dtype="float16") = model_params[96]
            lv79 = R.call_tir(cls.fused_NT_matmul4_add1, (lv78, param_95, param_96), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv80_1 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv428, param_91, param_92, lv79, lv70_1), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv446 = R.call_tir(cls.cast, (lv80_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_97: R.Tensor((2560,), dtype="float32") = model_params[97]
            param_98: R.Tensor((2560,), dtype="float32") = model_params[98]
            lv81_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv446, param_97, param_98), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_101: R.Tensor((7680, 2560), dtype="float16") = model_params[101]
            param_102: R.Tensor((7680,), dtype="float16") = model_params[102]
            lv82_1 = R.call_tir(cls.fused_NT_matmul_add, (lv81_1, param_101, param_102), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv453 = R.call_tir(cls.reshape2, (lv82_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv454 = R.call_tir(cls.split, (lv453,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv455: R.Tensor((1, n, 32, 80), dtype="float16") = lv454[0]
            lv456 = R.call_tir(cls.rotary_embedding, (lv455, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv457: R.Tensor((1, n, 32, 80), dtype="float16") = lv454[1]
            lv458 = R.call_tir(cls.rotary_embedding, (lv457, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv459: R.Object = kv_cache[16]
            lv460 = R.call_tir(cls.squeeze, (lv458,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv461: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv459, lv460, sinfo_args=(R.Object,))
            lv462: R.Object = kv_cache[17]
            lv83_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv454[2]
            lv84_1 = R.call_tir(cls.fused_squeeze, (lv83_1,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv465: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv462, lv84_1, sinfo_args=(R.Object,))
            lv466: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv461, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv467: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv465, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv468 = R.call_tir(cls.reshape3, (lv466,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv469 = R.call_tir(cls.reshape3, (lv467,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv470 = R.call_tir(cls.transpose6, (lv456,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv471 = R.call_tir(cls.transpose6, (lv468,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv472 = R.call_tir(cls.transpose6, (lv469,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv85_1 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv470, lv471, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv86_1 = R.call_tir(cls.fused_softmax_cast3, (lv85_1,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv481 = R.call_tir(cls.matmul8, (lv86_1, lv472), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv482 = R.call_tir(cls.transpose7, (lv481,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv483 = R.call_tir(cls.reshape4, (lv482,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_103: R.Tensor((2560, 2560), dtype="float16") = model_params[103]
            param_104: R.Tensor((2560,), dtype="float16") = model_params[104]
            lv487 = R.call_tir(cls.cast, (lv80_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_99: R.Tensor((2560,), dtype="float32") = model_params[99]
            param_100: R.Tensor((2560,), dtype="float32") = model_params[100]
            lv87_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv487, param_99, param_100), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_105: R.Tensor((10240, 2560), dtype="float16") = model_params[105]
            param_106: R.Tensor((10240,), dtype="float16") = model_params[106]
            lv88 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv87_1, param_105, param_106), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_107: R.Tensor((2560, 10240), dtype="float16") = model_params[107]
            param_108: R.Tensor((2560,), dtype="float16") = model_params[108]
            lv89 = R.call_tir(cls.fused_NT_matmul4_add1, (lv88, param_107, param_108), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv90 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv483, param_103, param_104, lv89, lv80_1), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv501 = R.call_tir(cls.cast, (lv90,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_109: R.Tensor((2560,), dtype="float32") = model_params[109]
            param_110: R.Tensor((2560,), dtype="float32") = model_params[110]
            lv91 = R.call_tir(cls.fused_layer_norm_cast1, (lv501, param_109, param_110), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_113: R.Tensor((7680, 2560), dtype="float16") = model_params[113]
            param_114: R.Tensor((7680,), dtype="float16") = model_params[114]
            lv92 = R.call_tir(cls.fused_NT_matmul_add, (lv91, param_113, param_114), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv508 = R.call_tir(cls.reshape2, (lv92,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv509 = R.call_tir(cls.split, (lv508,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv510: R.Tensor((1, n, 32, 80), dtype="float16") = lv509[0]
            lv511 = R.call_tir(cls.rotary_embedding, (lv510, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv512: R.Tensor((1, n, 32, 80), dtype="float16") = lv509[1]
            lv513 = R.call_tir(cls.rotary_embedding, (lv512, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv514: R.Object = kv_cache[18]
            lv515 = R.call_tir(cls.squeeze, (lv513,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv516: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv514, lv515, sinfo_args=(R.Object,))
            lv517: R.Object = kv_cache[19]
            lv93: R.Tensor((1, n, 32, 80), dtype="float16") = lv509[2]
            lv94 = R.call_tir(cls.fused_squeeze, (lv93,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv520: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv517, lv94, sinfo_args=(R.Object,))
            lv521: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv516, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv522: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv520, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv523 = R.call_tir(cls.reshape3, (lv521,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv524 = R.call_tir(cls.reshape3, (lv522,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv525 = R.call_tir(cls.transpose6, (lv511,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv526 = R.call_tir(cls.transpose6, (lv523,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv527 = R.call_tir(cls.transpose6, (lv524,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv95 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv525, lv526, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv96_1 = R.call_tir(cls.fused_softmax_cast3, (lv95,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv536 = R.call_tir(cls.matmul8, (lv96_1, lv527), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv537 = R.call_tir(cls.transpose7, (lv536,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv538 = R.call_tir(cls.reshape4, (lv537,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_115: R.Tensor((2560, 2560), dtype="float16") = model_params[115]
            param_116: R.Tensor((2560,), dtype="float16") = model_params[116]
            lv542 = R.call_tir(cls.cast, (lv90,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_111: R.Tensor((2560,), dtype="float32") = model_params[111]
            param_112: R.Tensor((2560,), dtype="float32") = model_params[112]
            lv97_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv542, param_111, param_112), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_117: R.Tensor((10240, 2560), dtype="float16") = model_params[117]
            param_118: R.Tensor((10240,), dtype="float16") = model_params[118]
            lv98_1 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv97_1, param_117, param_118), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_119: R.Tensor((2560, 10240), dtype="float16") = model_params[119]
            param_120: R.Tensor((2560,), dtype="float16") = model_params[120]
            lv99 = R.call_tir(cls.fused_NT_matmul4_add1, (lv98_1, param_119, param_120), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv100 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv538, param_115, param_116, lv99, lv90), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv556 = R.call_tir(cls.cast, (lv100,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_121: R.Tensor((2560,), dtype="float32") = model_params[121]
            param_122: R.Tensor((2560,), dtype="float32") = model_params[122]
            lv101 = R.call_tir(cls.fused_layer_norm_cast1, (lv556, param_121, param_122), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_125: R.Tensor((7680, 2560), dtype="float16") = model_params[125]
            param_126: R.Tensor((7680,), dtype="float16") = model_params[126]
            lv102_1 = R.call_tir(cls.fused_NT_matmul_add, (lv101, param_125, param_126), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv563 = R.call_tir(cls.reshape2, (lv102_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv564 = R.call_tir(cls.split, (lv563,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv565: R.Tensor((1, n, 32, 80), dtype="float16") = lv564[0]
            lv566 = R.call_tir(cls.rotary_embedding, (lv565, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv567: R.Tensor((1, n, 32, 80), dtype="float16") = lv564[1]
            lv568 = R.call_tir(cls.rotary_embedding, (lv567, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv569: R.Object = kv_cache[20]
            lv570 = R.call_tir(cls.squeeze, (lv568,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv571: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv569, lv570, sinfo_args=(R.Object,))
            lv572: R.Object = kv_cache[21]
            lv103: R.Tensor((1, n, 32, 80), dtype="float16") = lv564[2]
            lv104 = R.call_tir(cls.fused_squeeze, (lv103,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv575: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv572, lv104, sinfo_args=(R.Object,))
            lv576: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv571, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv577: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv575, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv578 = R.call_tir(cls.reshape3, (lv576,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv579 = R.call_tir(cls.reshape3, (lv577,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv580 = R.call_tir(cls.transpose6, (lv566,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv581 = R.call_tir(cls.transpose6, (lv578,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv582 = R.call_tir(cls.transpose6, (lv579,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv105 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv580, lv581, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv106 = R.call_tir(cls.fused_softmax_cast3, (lv105,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv591 = R.call_tir(cls.matmul8, (lv106, lv582), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv592 = R.call_tir(cls.transpose7, (lv591,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv593 = R.call_tir(cls.reshape4, (lv592,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_127: R.Tensor((2560, 2560), dtype="float16") = model_params[127]
            param_128: R.Tensor((2560,), dtype="float16") = model_params[128]
            lv597 = R.call_tir(cls.cast, (lv100,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_123: R.Tensor((2560,), dtype="float32") = model_params[123]
            param_124: R.Tensor((2560,), dtype="float32") = model_params[124]
            lv107 = R.call_tir(cls.fused_layer_norm_cast1, (lv597, param_123, param_124), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_129: R.Tensor((10240, 2560), dtype="float16") = model_params[129]
            param_130: R.Tensor((10240,), dtype="float16") = model_params[130]
            lv108 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv107, param_129, param_130), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_131: R.Tensor((2560, 10240), dtype="float16") = model_params[131]
            param_132: R.Tensor((2560,), dtype="float16") = model_params[132]
            lv109 = R.call_tir(cls.fused_NT_matmul4_add1, (lv108, param_131, param_132), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv110 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv593, param_127, param_128, lv109, lv100), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv611 = R.call_tir(cls.cast, (lv110,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_133: R.Tensor((2560,), dtype="float32") = model_params[133]
            param_134: R.Tensor((2560,), dtype="float32") = model_params[134]
            lv111 = R.call_tir(cls.fused_layer_norm_cast1, (lv611, param_133, param_134), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_137: R.Tensor((7680, 2560), dtype="float16") = model_params[137]
            param_138: R.Tensor((7680,), dtype="float16") = model_params[138]
            lv112 = R.call_tir(cls.fused_NT_matmul_add, (lv111, param_137, param_138), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv618 = R.call_tir(cls.reshape2, (lv112,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv619 = R.call_tir(cls.split, (lv618,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv620: R.Tensor((1, n, 32, 80), dtype="float16") = lv619[0]
            lv621 = R.call_tir(cls.rotary_embedding, (lv620, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv622: R.Tensor((1, n, 32, 80), dtype="float16") = lv619[1]
            lv623 = R.call_tir(cls.rotary_embedding, (lv622, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv624: R.Object = kv_cache[22]
            lv625 = R.call_tir(cls.squeeze, (lv623,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv626: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv624, lv625, sinfo_args=(R.Object,))
            lv627: R.Object = kv_cache[23]
            lv113: R.Tensor((1, n, 32, 80), dtype="float16") = lv619[2]
            lv114 = R.call_tir(cls.fused_squeeze, (lv113,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv630: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv627, lv114, sinfo_args=(R.Object,))
            lv631: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv626, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv632: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv630, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv633 = R.call_tir(cls.reshape3, (lv631,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv634 = R.call_tir(cls.reshape3, (lv632,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv635 = R.call_tir(cls.transpose6, (lv621,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv636 = R.call_tir(cls.transpose6, (lv633,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv637 = R.call_tir(cls.transpose6, (lv634,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv115 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv635, lv636, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv116_1 = R.call_tir(cls.fused_softmax_cast3, (lv115,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv646 = R.call_tir(cls.matmul8, (lv116_1, lv637), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv647 = R.call_tir(cls.transpose7, (lv646,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv648 = R.call_tir(cls.reshape4, (lv647,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_139: R.Tensor((2560, 2560), dtype="float16") = model_params[139]
            param_140: R.Tensor((2560,), dtype="float16") = model_params[140]
            lv652 = R.call_tir(cls.cast, (lv110,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_135: R.Tensor((2560,), dtype="float32") = model_params[135]
            param_136: R.Tensor((2560,), dtype="float32") = model_params[136]
            lv117 = R.call_tir(cls.fused_layer_norm_cast1, (lv652, param_135, param_136), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_141: R.Tensor((10240, 2560), dtype="float16") = model_params[141]
            param_142: R.Tensor((10240,), dtype="float16") = model_params[142]
            lv118 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv117, param_141, param_142), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_143: R.Tensor((2560, 10240), dtype="float16") = model_params[143]
            param_144: R.Tensor((2560,), dtype="float16") = model_params[144]
            lv119 = R.call_tir(cls.fused_NT_matmul4_add1, (lv118, param_143, param_144), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv120 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv648, param_139, param_140, lv119, lv110), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv666 = R.call_tir(cls.cast, (lv120,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_145: R.Tensor((2560,), dtype="float32") = model_params[145]
            param_146: R.Tensor((2560,), dtype="float32") = model_params[146]
            lv121 = R.call_tir(cls.fused_layer_norm_cast1, (lv666, param_145, param_146), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_149: R.Tensor((7680, 2560), dtype="float16") = model_params[149]
            param_150: R.Tensor((7680,), dtype="float16") = model_params[150]
            lv122 = R.call_tir(cls.fused_NT_matmul_add, (lv121, param_149, param_150), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv673 = R.call_tir(cls.reshape2, (lv122,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv674 = R.call_tir(cls.split, (lv673,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv675: R.Tensor((1, n, 32, 80), dtype="float16") = lv674[0]
            lv676 = R.call_tir(cls.rotary_embedding, (lv675, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv677: R.Tensor((1, n, 32, 80), dtype="float16") = lv674[1]
            lv678 = R.call_tir(cls.rotary_embedding, (lv677, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv679: R.Object = kv_cache[24]
            lv680 = R.call_tir(cls.squeeze, (lv678,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv681: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv679, lv680, sinfo_args=(R.Object,))
            lv682: R.Object = kv_cache[25]
            lv123_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv674[2]
            lv124_1 = R.call_tir(cls.fused_squeeze, (lv123_1,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv685: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv682, lv124_1, sinfo_args=(R.Object,))
            lv686: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv681, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv687: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv685, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv688 = R.call_tir(cls.reshape3, (lv686,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv689 = R.call_tir(cls.reshape3, (lv687,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv690 = R.call_tir(cls.transpose6, (lv676,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv691 = R.call_tir(cls.transpose6, (lv688,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv692 = R.call_tir(cls.transpose6, (lv689,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv125_1 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv690, lv691, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv126_1 = R.call_tir(cls.fused_softmax_cast3, (lv125_1,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv701 = R.call_tir(cls.matmul8, (lv126_1, lv692), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv702 = R.call_tir(cls.transpose7, (lv701,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv703 = R.call_tir(cls.reshape4, (lv702,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_151: R.Tensor((2560, 2560), dtype="float16") = model_params[151]
            param_152: R.Tensor((2560,), dtype="float16") = model_params[152]
            lv707 = R.call_tir(cls.cast, (lv120,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_147: R.Tensor((2560,), dtype="float32") = model_params[147]
            param_148: R.Tensor((2560,), dtype="float32") = model_params[148]
            lv127_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv707, param_147, param_148), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_153: R.Tensor((10240, 2560), dtype="float16") = model_params[153]
            param_154: R.Tensor((10240,), dtype="float16") = model_params[154]
            lv128_1 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv127_1, param_153, param_154), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_155: R.Tensor((2560, 10240), dtype="float16") = model_params[155]
            param_156: R.Tensor((2560,), dtype="float16") = model_params[156]
            lv129_1 = R.call_tir(cls.fused_NT_matmul4_add1, (lv128_1, param_155, param_156), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv130_1 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv703, param_151, param_152, lv129_1, lv120), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv721 = R.call_tir(cls.cast, (lv130_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_157: R.Tensor((2560,), dtype="float32") = model_params[157]
            param_158: R.Tensor((2560,), dtype="float32") = model_params[158]
            lv131_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv721, param_157, param_158), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_161: R.Tensor((7680, 2560), dtype="float16") = model_params[161]
            param_162: R.Tensor((7680,), dtype="float16") = model_params[162]
            lv132_1 = R.call_tir(cls.fused_NT_matmul_add, (lv131_1, param_161, param_162), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv728 = R.call_tir(cls.reshape2, (lv132_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv729 = R.call_tir(cls.split, (lv728,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv730: R.Tensor((1, n, 32, 80), dtype="float16") = lv729[0]
            lv731 = R.call_tir(cls.rotary_embedding, (lv730, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv732: R.Tensor((1, n, 32, 80), dtype="float16") = lv729[1]
            lv733 = R.call_tir(cls.rotary_embedding, (lv732, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv734: R.Object = kv_cache[26]
            lv735 = R.call_tir(cls.squeeze, (lv733,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv736: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv734, lv735, sinfo_args=(R.Object,))
            lv737: R.Object = kv_cache[27]
            lv133: R.Tensor((1, n, 32, 80), dtype="float16") = lv729[2]
            lv134 = R.call_tir(cls.fused_squeeze, (lv133,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv740: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv737, lv134, sinfo_args=(R.Object,))
            lv741: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv736, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv742: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv740, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv743 = R.call_tir(cls.reshape3, (lv741,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv744 = R.call_tir(cls.reshape3, (lv742,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv745 = R.call_tir(cls.transpose6, (lv731,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv746 = R.call_tir(cls.transpose6, (lv743,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv747 = R.call_tir(cls.transpose6, (lv744,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv135_1 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv745, lv746, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv136_1 = R.call_tir(cls.fused_softmax_cast3, (lv135_1,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv756 = R.call_tir(cls.matmul8, (lv136_1, lv747), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv757 = R.call_tir(cls.transpose7, (lv756,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv758 = R.call_tir(cls.reshape4, (lv757,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_163: R.Tensor((2560, 2560), dtype="float16") = model_params[163]
            param_164: R.Tensor((2560,), dtype="float16") = model_params[164]
            lv762 = R.call_tir(cls.cast, (lv130_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_159: R.Tensor((2560,), dtype="float32") = model_params[159]
            param_160: R.Tensor((2560,), dtype="float32") = model_params[160]
            lv137_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv762, param_159, param_160), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_165: R.Tensor((10240, 2560), dtype="float16") = model_params[165]
            param_166: R.Tensor((10240,), dtype="float16") = model_params[166]
            lv138_1 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv137_1, param_165, param_166), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_167: R.Tensor((2560, 10240), dtype="float16") = model_params[167]
            param_168: R.Tensor((2560,), dtype="float16") = model_params[168]
            lv139_1 = R.call_tir(cls.fused_NT_matmul4_add1, (lv138_1, param_167, param_168), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv140_1 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv758, param_163, param_164, lv139_1, lv130_1), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv776 = R.call_tir(cls.cast, (lv140_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_169: R.Tensor((2560,), dtype="float32") = model_params[169]
            param_170: R.Tensor((2560,), dtype="float32") = model_params[170]
            lv141_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv776, param_169, param_170), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_173: R.Tensor((7680, 2560), dtype="float16") = model_params[173]
            param_174: R.Tensor((7680,), dtype="float16") = model_params[174]
            lv142_1 = R.call_tir(cls.fused_NT_matmul_add, (lv141_1, param_173, param_174), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv783 = R.call_tir(cls.reshape2, (lv142_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv784 = R.call_tir(cls.split, (lv783,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv785: R.Tensor((1, n, 32, 80), dtype="float16") = lv784[0]
            lv786 = R.call_tir(cls.rotary_embedding, (lv785, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv787: R.Tensor((1, n, 32, 80), dtype="float16") = lv784[1]
            lv788 = R.call_tir(cls.rotary_embedding, (lv787, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv789: R.Object = kv_cache[28]
            lv790 = R.call_tir(cls.squeeze, (lv788,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv791: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv789, lv790, sinfo_args=(R.Object,))
            lv792: R.Object = kv_cache[29]
            lv143: R.Tensor((1, n, 32, 80), dtype="float16") = lv784[2]
            lv144 = R.call_tir(cls.fused_squeeze, (lv143,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv795: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv792, lv144, sinfo_args=(R.Object,))
            lv796: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv791, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv797: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv795, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv798 = R.call_tir(cls.reshape3, (lv796,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv799 = R.call_tir(cls.reshape3, (lv797,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv800 = R.call_tir(cls.transpose6, (lv786,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv801 = R.call_tir(cls.transpose6, (lv798,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv802 = R.call_tir(cls.transpose6, (lv799,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv145 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv800, lv801, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv146 = R.call_tir(cls.fused_softmax_cast3, (lv145,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv811 = R.call_tir(cls.matmul8, (lv146, lv802), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv812 = R.call_tir(cls.transpose7, (lv811,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv813 = R.call_tir(cls.reshape4, (lv812,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_175: R.Tensor((2560, 2560), dtype="float16") = model_params[175]
            param_176: R.Tensor((2560,), dtype="float16") = model_params[176]
            lv817 = R.call_tir(cls.cast, (lv140_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_171: R.Tensor((2560,), dtype="float32") = model_params[171]
            param_172: R.Tensor((2560,), dtype="float32") = model_params[172]
            lv147 = R.call_tir(cls.fused_layer_norm_cast1, (lv817, param_171, param_172), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_177: R.Tensor((10240, 2560), dtype="float16") = model_params[177]
            param_178: R.Tensor((10240,), dtype="float16") = model_params[178]
            lv148 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv147, param_177, param_178), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_179: R.Tensor((2560, 10240), dtype="float16") = model_params[179]
            param_180: R.Tensor((2560,), dtype="float16") = model_params[180]
            lv149 = R.call_tir(cls.fused_NT_matmul4_add1, (lv148, param_179, param_180), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv150 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv813, param_175, param_176, lv149, lv140_1), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv831 = R.call_tir(cls.cast, (lv150,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_181: R.Tensor((2560,), dtype="float32") = model_params[181]
            param_182: R.Tensor((2560,), dtype="float32") = model_params[182]
            lv151_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv831, param_181, param_182), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_185: R.Tensor((7680, 2560), dtype="float16") = model_params[185]
            param_186: R.Tensor((7680,), dtype="float16") = model_params[186]
            lv152_1 = R.call_tir(cls.fused_NT_matmul_add, (lv151_1, param_185, param_186), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv838 = R.call_tir(cls.reshape2, (lv152_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv839 = R.call_tir(cls.split, (lv838,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv840: R.Tensor((1, n, 32, 80), dtype="float16") = lv839[0]
            lv841 = R.call_tir(cls.rotary_embedding, (lv840, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv842: R.Tensor((1, n, 32, 80), dtype="float16") = lv839[1]
            lv843 = R.call_tir(cls.rotary_embedding, (lv842, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv844: R.Object = kv_cache[30]
            lv845 = R.call_tir(cls.squeeze, (lv843,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv846: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv844, lv845, sinfo_args=(R.Object,))
            lv847: R.Object = kv_cache[31]
            lv153_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv839[2]
            lv154 = R.call_tir(cls.fused_squeeze, (lv153_1,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv850: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv847, lv154, sinfo_args=(R.Object,))
            lv851: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv846, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv852: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv850, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv853 = R.call_tir(cls.reshape3, (lv851,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv854 = R.call_tir(cls.reshape3, (lv852,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv855 = R.call_tir(cls.transpose6, (lv841,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv856 = R.call_tir(cls.transpose6, (lv853,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv857 = R.call_tir(cls.transpose6, (lv854,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv155 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv855, lv856, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv156 = R.call_tir(cls.fused_softmax_cast3, (lv155,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv866 = R.call_tir(cls.matmul8, (lv156, lv857), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv867 = R.call_tir(cls.transpose7, (lv866,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv868 = R.call_tir(cls.reshape4, (lv867,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_187: R.Tensor((2560, 2560), dtype="float16") = model_params[187]
            param_188: R.Tensor((2560,), dtype="float16") = model_params[188]
            lv872 = R.call_tir(cls.cast, (lv150,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_183: R.Tensor((2560,), dtype="float32") = model_params[183]
            param_184: R.Tensor((2560,), dtype="float32") = model_params[184]
            lv157_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv872, param_183, param_184), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_189: R.Tensor((10240, 2560), dtype="float16") = model_params[189]
            param_190: R.Tensor((10240,), dtype="float16") = model_params[190]
            lv158 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv157_1, param_189, param_190), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_191: R.Tensor((2560, 10240), dtype="float16") = model_params[191]
            param_192: R.Tensor((2560,), dtype="float16") = model_params[192]
            lv159 = R.call_tir(cls.fused_NT_matmul4_add1, (lv158, param_191, param_192), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv160 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv868, param_187, param_188, lv159, lv150), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv886 = R.call_tir(cls.cast, (lv160,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_193: R.Tensor((2560,), dtype="float32") = model_params[193]
            param_194: R.Tensor((2560,), dtype="float32") = model_params[194]
            lv161 = R.call_tir(cls.fused_layer_norm_cast1, (lv886, param_193, param_194), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_197: R.Tensor((7680, 2560), dtype="float16") = model_params[197]
            param_198: R.Tensor((7680,), dtype="float16") = model_params[198]
            lv162 = R.call_tir(cls.fused_NT_matmul_add, (lv161, param_197, param_198), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv893 = R.call_tir(cls.reshape2, (lv162,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv894 = R.call_tir(cls.split, (lv893,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv895: R.Tensor((1, n, 32, 80), dtype="float16") = lv894[0]
            lv896 = R.call_tir(cls.rotary_embedding, (lv895, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv897: R.Tensor((1, n, 32, 80), dtype="float16") = lv894[1]
            lv898 = R.call_tir(cls.rotary_embedding, (lv897, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv899: R.Object = kv_cache[32]
            lv900 = R.call_tir(cls.squeeze, (lv898,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv901: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv899, lv900, sinfo_args=(R.Object,))
            lv902: R.Object = kv_cache[33]
            lv163: R.Tensor((1, n, 32, 80), dtype="float16") = lv894[2]
            lv164 = R.call_tir(cls.fused_squeeze, (lv163,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv905: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv902, lv164, sinfo_args=(R.Object,))
            lv906: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv901, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv907: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv905, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv908 = R.call_tir(cls.reshape3, (lv906,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv909 = R.call_tir(cls.reshape3, (lv907,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv910 = R.call_tir(cls.transpose6, (lv896,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv911 = R.call_tir(cls.transpose6, (lv908,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv912 = R.call_tir(cls.transpose6, (lv909,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv165 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv910, lv911, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv166 = R.call_tir(cls.fused_softmax_cast3, (lv165,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv921 = R.call_tir(cls.matmul8, (lv166, lv912), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv922 = R.call_tir(cls.transpose7, (lv921,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv923 = R.call_tir(cls.reshape4, (lv922,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_199: R.Tensor((2560, 2560), dtype="float16") = model_params[199]
            param_200: R.Tensor((2560,), dtype="float16") = model_params[200]
            lv927 = R.call_tir(cls.cast, (lv160,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_195: R.Tensor((2560,), dtype="float32") = model_params[195]
            param_196: R.Tensor((2560,), dtype="float32") = model_params[196]
            lv167 = R.call_tir(cls.fused_layer_norm_cast1, (lv927, param_195, param_196), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_201: R.Tensor((10240, 2560), dtype="float16") = model_params[201]
            param_202: R.Tensor((10240,), dtype="float16") = model_params[202]
            lv168 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv167, param_201, param_202), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_203: R.Tensor((2560, 10240), dtype="float16") = model_params[203]
            param_204: R.Tensor((2560,), dtype="float16") = model_params[204]
            lv169 = R.call_tir(cls.fused_NT_matmul4_add1, (lv168, param_203, param_204), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv170 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv923, param_199, param_200, lv169, lv160), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv941 = R.call_tir(cls.cast, (lv170,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_205: R.Tensor((2560,), dtype="float32") = model_params[205]
            param_206: R.Tensor((2560,), dtype="float32") = model_params[206]
            lv171_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv941, param_205, param_206), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_209: R.Tensor((7680, 2560), dtype="float16") = model_params[209]
            param_210: R.Tensor((7680,), dtype="float16") = model_params[210]
            lv172 = R.call_tir(cls.fused_NT_matmul_add, (lv171_1, param_209, param_210), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv948 = R.call_tir(cls.reshape2, (lv172,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv949 = R.call_tir(cls.split, (lv948,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv950: R.Tensor((1, n, 32, 80), dtype="float16") = lv949[0]
            lv951 = R.call_tir(cls.rotary_embedding, (lv950, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv952: R.Tensor((1, n, 32, 80), dtype="float16") = lv949[1]
            lv953 = R.call_tir(cls.rotary_embedding, (lv952, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv954: R.Object = kv_cache[34]
            lv955 = R.call_tir(cls.squeeze, (lv953,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv956: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv954, lv955, sinfo_args=(R.Object,))
            lv957: R.Object = kv_cache[35]
            lv173: R.Tensor((1, n, 32, 80), dtype="float16") = lv949[2]
            lv174 = R.call_tir(cls.fused_squeeze, (lv173,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv960: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv957, lv174, sinfo_args=(R.Object,))
            lv961: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv956, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv962: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv960, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv963 = R.call_tir(cls.reshape3, (lv961,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv964 = R.call_tir(cls.reshape3, (lv962,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv965 = R.call_tir(cls.transpose6, (lv951,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv966 = R.call_tir(cls.transpose6, (lv963,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv967 = R.call_tir(cls.transpose6, (lv964,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv175 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv965, lv966, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv176 = R.call_tir(cls.fused_softmax_cast3, (lv175,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv976 = R.call_tir(cls.matmul8, (lv176, lv967), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv977 = R.call_tir(cls.transpose7, (lv976,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv978 = R.call_tir(cls.reshape4, (lv977,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_211: R.Tensor((2560, 2560), dtype="float16") = model_params[211]
            param_212: R.Tensor((2560,), dtype="float16") = model_params[212]
            lv982 = R.call_tir(cls.cast, (lv170,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_207: R.Tensor((2560,), dtype="float32") = model_params[207]
            param_208: R.Tensor((2560,), dtype="float32") = model_params[208]
            lv177 = R.call_tir(cls.fused_layer_norm_cast1, (lv982, param_207, param_208), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_213: R.Tensor((10240, 2560), dtype="float16") = model_params[213]
            param_214: R.Tensor((10240,), dtype="float16") = model_params[214]
            lv178_1 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv177, param_213, param_214), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_215: R.Tensor((2560, 10240), dtype="float16") = model_params[215]
            param_216: R.Tensor((2560,), dtype="float16") = model_params[216]
            lv179_1 = R.call_tir(cls.fused_NT_matmul4_add1, (lv178_1, param_215, param_216), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv180_1 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv978, param_211, param_212, lv179_1, lv170), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv996 = R.call_tir(cls.cast, (lv180_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_217: R.Tensor((2560,), dtype="float32") = model_params[217]
            param_218: R.Tensor((2560,), dtype="float32") = model_params[218]
            lv181_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv996, param_217, param_218), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_221: R.Tensor((7680, 2560), dtype="float16") = model_params[221]
            param_222: R.Tensor((7680,), dtype="float16") = model_params[222]
            lv182_1 = R.call_tir(cls.fused_NT_matmul_add, (lv181_1, param_221, param_222), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1003 = R.call_tir(cls.reshape2, (lv182_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1004 = R.call_tir(cls.split, (lv1003,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1005: R.Tensor((1, n, 32, 80), dtype="float16") = lv1004[0]
            lv1006 = R.call_tir(cls.rotary_embedding, (lv1005, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1007: R.Tensor((1, n, 32, 80), dtype="float16") = lv1004[1]
            lv1008 = R.call_tir(cls.rotary_embedding, (lv1007, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1009: R.Object = kv_cache[36]
            lv1010 = R.call_tir(cls.squeeze, (lv1008,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1011: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1009, lv1010, sinfo_args=(R.Object,))
            lv1012: R.Object = kv_cache[37]
            lv183_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv1004[2]
            lv184_1 = R.call_tir(cls.fused_squeeze, (lv183_1,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1015: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1012, lv184_1, sinfo_args=(R.Object,))
            lv1016: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1011, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1017: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1015, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1018 = R.call_tir(cls.reshape3, (lv1016,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1019 = R.call_tir(cls.reshape3, (lv1017,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1020 = R.call_tir(cls.transpose6, (lv1006,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1021 = R.call_tir(cls.transpose6, (lv1018,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1022 = R.call_tir(cls.transpose6, (lv1019,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv185_1 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv1020, lv1021, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv186_1 = R.call_tir(cls.fused_softmax_cast3, (lv185_1,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1031 = R.call_tir(cls.matmul8, (lv186_1, lv1022), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1032 = R.call_tir(cls.transpose7, (lv1031,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1033 = R.call_tir(cls.reshape4, (lv1032,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_223: R.Tensor((2560, 2560), dtype="float16") = model_params[223]
            param_224: R.Tensor((2560,), dtype="float16") = model_params[224]
            lv1037 = R.call_tir(cls.cast, (lv180_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_219: R.Tensor((2560,), dtype="float32") = model_params[219]
            param_220: R.Tensor((2560,), dtype="float32") = model_params[220]
            lv187_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv1037, param_219, param_220), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_225: R.Tensor((10240, 2560), dtype="float16") = model_params[225]
            param_226: R.Tensor((10240,), dtype="float16") = model_params[226]
            lv188 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv187_1, param_225, param_226), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_227: R.Tensor((2560, 10240), dtype="float16") = model_params[227]
            param_228: R.Tensor((2560,), dtype="float16") = model_params[228]
            lv189 = R.call_tir(cls.fused_NT_matmul4_add1, (lv188, param_227, param_228), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv190_1 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv1033, param_223, param_224, lv189, lv180_1), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1051 = R.call_tir(cls.cast, (lv190_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_229: R.Tensor((2560,), dtype="float32") = model_params[229]
            param_230: R.Tensor((2560,), dtype="float32") = model_params[230]
            lv191_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv1051, param_229, param_230), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_233: R.Tensor((7680, 2560), dtype="float16") = model_params[233]
            param_234: R.Tensor((7680,), dtype="float16") = model_params[234]
            lv192_1 = R.call_tir(cls.fused_NT_matmul_add, (lv191_1, param_233, param_234), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1058 = R.call_tir(cls.reshape2, (lv192_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1059 = R.call_tir(cls.split, (lv1058,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1060: R.Tensor((1, n, 32, 80), dtype="float16") = lv1059[0]
            lv1061 = R.call_tir(cls.rotary_embedding, (lv1060, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1062: R.Tensor((1, n, 32, 80), dtype="float16") = lv1059[1]
            lv1063 = R.call_tir(cls.rotary_embedding, (lv1062, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1064: R.Object = kv_cache[38]
            lv1065 = R.call_tir(cls.squeeze, (lv1063,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1066: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1064, lv1065, sinfo_args=(R.Object,))
            lv1067: R.Object = kv_cache[39]
            lv193_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv1059[2]
            lv194_1 = R.call_tir(cls.fused_squeeze, (lv193_1,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1070: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1067, lv194_1, sinfo_args=(R.Object,))
            lv1071: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1066, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1072: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1070, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1073 = R.call_tir(cls.reshape3, (lv1071,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1074 = R.call_tir(cls.reshape3, (lv1072,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1075 = R.call_tir(cls.transpose6, (lv1061,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1076 = R.call_tir(cls.transpose6, (lv1073,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1077 = R.call_tir(cls.transpose6, (lv1074,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv195_1 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv1075, lv1076, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv196_1 = R.call_tir(cls.fused_softmax_cast3, (lv195_1,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1086 = R.call_tir(cls.matmul8, (lv196_1, lv1077), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1087 = R.call_tir(cls.transpose7, (lv1086,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1088 = R.call_tir(cls.reshape4, (lv1087,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_235: R.Tensor((2560, 2560), dtype="float16") = model_params[235]
            param_236: R.Tensor((2560,), dtype="float16") = model_params[236]
            lv1092 = R.call_tir(cls.cast, (lv190_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_231: R.Tensor((2560,), dtype="float32") = model_params[231]
            param_232: R.Tensor((2560,), dtype="float32") = model_params[232]
            lv197_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv1092, param_231, param_232), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_237: R.Tensor((10240, 2560), dtype="float16") = model_params[237]
            param_238: R.Tensor((10240,), dtype="float16") = model_params[238]
            lv198 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv197_1, param_237, param_238), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_239: R.Tensor((2560, 10240), dtype="float16") = model_params[239]
            param_240: R.Tensor((2560,), dtype="float16") = model_params[240]
            lv199 = R.call_tir(cls.fused_NT_matmul4_add1, (lv198, param_239, param_240), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv200 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv1088, param_235, param_236, lv199, lv190_1), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1106 = R.call_tir(cls.cast, (lv200,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_241: R.Tensor((2560,), dtype="float32") = model_params[241]
            param_242: R.Tensor((2560,), dtype="float32") = model_params[242]
            lv201 = R.call_tir(cls.fused_layer_norm_cast1, (lv1106, param_241, param_242), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_245: R.Tensor((7680, 2560), dtype="float16") = model_params[245]
            param_246: R.Tensor((7680,), dtype="float16") = model_params[246]
            lv202 = R.call_tir(cls.fused_NT_matmul_add, (lv201, param_245, param_246), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1113 = R.call_tir(cls.reshape2, (lv202,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1114 = R.call_tir(cls.split, (lv1113,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1115: R.Tensor((1, n, 32, 80), dtype="float16") = lv1114[0]
            lv1116 = R.call_tir(cls.rotary_embedding, (lv1115, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1117: R.Tensor((1, n, 32, 80), dtype="float16") = lv1114[1]
            lv1118 = R.call_tir(cls.rotary_embedding, (lv1117, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1119: R.Object = kv_cache[40]
            lv1120 = R.call_tir(cls.squeeze, (lv1118,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1121: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1119, lv1120, sinfo_args=(R.Object,))
            lv1122: R.Object = kv_cache[41]
            lv203: R.Tensor((1, n, 32, 80), dtype="float16") = lv1114[2]
            lv204 = R.call_tir(cls.fused_squeeze, (lv203,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1125: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1122, lv204, sinfo_args=(R.Object,))
            lv1126: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1121, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1127: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1125, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1128 = R.call_tir(cls.reshape3, (lv1126,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1129 = R.call_tir(cls.reshape3, (lv1127,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1130 = R.call_tir(cls.transpose6, (lv1116,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1131 = R.call_tir(cls.transpose6, (lv1128,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1132 = R.call_tir(cls.transpose6, (lv1129,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv205 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv1130, lv1131, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv206_1 = R.call_tir(cls.fused_softmax_cast3, (lv205,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1141 = R.call_tir(cls.matmul8, (lv206_1, lv1132), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1142 = R.call_tir(cls.transpose7, (lv1141,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1143 = R.call_tir(cls.reshape4, (lv1142,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_247: R.Tensor((2560, 2560), dtype="float16") = model_params[247]
            param_248: R.Tensor((2560,), dtype="float16") = model_params[248]
            lv1147 = R.call_tir(cls.cast, (lv200,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_243: R.Tensor((2560,), dtype="float32") = model_params[243]
            param_244: R.Tensor((2560,), dtype="float32") = model_params[244]
            lv207_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv1147, param_243, param_244), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_249: R.Tensor((10240, 2560), dtype="float16") = model_params[249]
            param_250: R.Tensor((10240,), dtype="float16") = model_params[250]
            lv208_1 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv207_1, param_249, param_250), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_251: R.Tensor((2560, 10240), dtype="float16") = model_params[251]
            param_252: R.Tensor((2560,), dtype="float16") = model_params[252]
            lv209 = R.call_tir(cls.fused_NT_matmul4_add1, (lv208_1, param_251, param_252), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv210 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv1143, param_247, param_248, lv209, lv200), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1161 = R.call_tir(cls.cast, (lv210,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_253: R.Tensor((2560,), dtype="float32") = model_params[253]
            param_254: R.Tensor((2560,), dtype="float32") = model_params[254]
            lv211 = R.call_tir(cls.fused_layer_norm_cast1, (lv1161, param_253, param_254), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_257: R.Tensor((7680, 2560), dtype="float16") = model_params[257]
            param_258: R.Tensor((7680,), dtype="float16") = model_params[258]
            lv212_1 = R.call_tir(cls.fused_NT_matmul_add, (lv211, param_257, param_258), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1168 = R.call_tir(cls.reshape2, (lv212_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1169 = R.call_tir(cls.split, (lv1168,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1170: R.Tensor((1, n, 32, 80), dtype="float16") = lv1169[0]
            lv1171 = R.call_tir(cls.rotary_embedding, (lv1170, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1172: R.Tensor((1, n, 32, 80), dtype="float16") = lv1169[1]
            lv1173 = R.call_tir(cls.rotary_embedding, (lv1172, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1174: R.Object = kv_cache[42]
            lv1175 = R.call_tir(cls.squeeze, (lv1173,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1176: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1174, lv1175, sinfo_args=(R.Object,))
            lv1177: R.Object = kv_cache[43]
            lv213: R.Tensor((1, n, 32, 80), dtype="float16") = lv1169[2]
            lv214 = R.call_tir(cls.fused_squeeze, (lv213,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1180: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1177, lv214, sinfo_args=(R.Object,))
            lv1181: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1176, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1182: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1180, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1183 = R.call_tir(cls.reshape3, (lv1181,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1184 = R.call_tir(cls.reshape3, (lv1182,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1185 = R.call_tir(cls.transpose6, (lv1171,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1186 = R.call_tir(cls.transpose6, (lv1183,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1187 = R.call_tir(cls.transpose6, (lv1184,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv215 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv1185, lv1186, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv216 = R.call_tir(cls.fused_softmax_cast3, (lv215,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1196 = R.call_tir(cls.matmul8, (lv216, lv1187), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1197 = R.call_tir(cls.transpose7, (lv1196,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1198 = R.call_tir(cls.reshape4, (lv1197,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_259: R.Tensor((2560, 2560), dtype="float16") = model_params[259]
            param_260: R.Tensor((2560,), dtype="float16") = model_params[260]
            lv1202 = R.call_tir(cls.cast, (lv210,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_255: R.Tensor((2560,), dtype="float32") = model_params[255]
            param_256: R.Tensor((2560,), dtype="float32") = model_params[256]
            lv217 = R.call_tir(cls.fused_layer_norm_cast1, (lv1202, param_255, param_256), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_261: R.Tensor((10240, 2560), dtype="float16") = model_params[261]
            param_262: R.Tensor((10240,), dtype="float16") = model_params[262]
            lv218 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv217, param_261, param_262), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_263: R.Tensor((2560, 10240), dtype="float16") = model_params[263]
            param_264: R.Tensor((2560,), dtype="float16") = model_params[264]
            lv219 = R.call_tir(cls.fused_NT_matmul4_add1, (lv218, param_263, param_264), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv220 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv1198, param_259, param_260, lv219, lv210), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1216 = R.call_tir(cls.cast, (lv220,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_265: R.Tensor((2560,), dtype="float32") = model_params[265]
            param_266: R.Tensor((2560,), dtype="float32") = model_params[266]
            lv221 = R.call_tir(cls.fused_layer_norm_cast1, (lv1216, param_265, param_266), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_269: R.Tensor((7680, 2560), dtype="float16") = model_params[269]
            param_270: R.Tensor((7680,), dtype="float16") = model_params[270]
            lv222 = R.call_tir(cls.fused_NT_matmul_add, (lv221, param_269, param_270), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1223 = R.call_tir(cls.reshape2, (lv222,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1224 = R.call_tir(cls.split, (lv1223,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1225: R.Tensor((1, n, 32, 80), dtype="float16") = lv1224[0]
            lv1226 = R.call_tir(cls.rotary_embedding, (lv1225, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1227: R.Tensor((1, n, 32, 80), dtype="float16") = lv1224[1]
            lv1228 = R.call_tir(cls.rotary_embedding, (lv1227, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1229: R.Object = kv_cache[44]
            lv1230 = R.call_tir(cls.squeeze, (lv1228,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1231: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1229, lv1230, sinfo_args=(R.Object,))
            lv1232: R.Object = kv_cache[45]
            lv223: R.Tensor((1, n, 32, 80), dtype="float16") = lv1224[2]
            lv224 = R.call_tir(cls.fused_squeeze, (lv223,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1235: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1232, lv224, sinfo_args=(R.Object,))
            lv1236: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1231, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1237: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1235, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1238 = R.call_tir(cls.reshape3, (lv1236,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1239 = R.call_tir(cls.reshape3, (lv1237,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1240 = R.call_tir(cls.transpose6, (lv1226,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1241 = R.call_tir(cls.transpose6, (lv1238,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1242 = R.call_tir(cls.transpose6, (lv1239,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv225 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv1240, lv1241, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv226_1 = R.call_tir(cls.fused_softmax_cast3, (lv225,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1251 = R.call_tir(cls.matmul8, (lv226_1, lv1242), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1252 = R.call_tir(cls.transpose7, (lv1251,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1253 = R.call_tir(cls.reshape4, (lv1252,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_271: R.Tensor((2560, 2560), dtype="float16") = model_params[271]
            param_272: R.Tensor((2560,), dtype="float16") = model_params[272]
            lv1257 = R.call_tir(cls.cast, (lv220,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_267: R.Tensor((2560,), dtype="float32") = model_params[267]
            param_268: R.Tensor((2560,), dtype="float32") = model_params[268]
            lv227 = R.call_tir(cls.fused_layer_norm_cast1, (lv1257, param_267, param_268), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_273: R.Tensor((10240, 2560), dtype="float16") = model_params[273]
            param_274: R.Tensor((10240,), dtype="float16") = model_params[274]
            lv228 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv227, param_273, param_274), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_275: R.Tensor((2560, 10240), dtype="float16") = model_params[275]
            param_276: R.Tensor((2560,), dtype="float16") = model_params[276]
            lv229 = R.call_tir(cls.fused_NT_matmul4_add1, (lv228, param_275, param_276), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv230 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv1253, param_271, param_272, lv229, lv220), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1271 = R.call_tir(cls.cast, (lv230,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_277: R.Tensor((2560,), dtype="float32") = model_params[277]
            param_278: R.Tensor((2560,), dtype="float32") = model_params[278]
            lv231 = R.call_tir(cls.fused_layer_norm_cast1, (lv1271, param_277, param_278), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_281: R.Tensor((7680, 2560), dtype="float16") = model_params[281]
            param_282: R.Tensor((7680,), dtype="float16") = model_params[282]
            lv232 = R.call_tir(cls.fused_NT_matmul_add, (lv231, param_281, param_282), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1278 = R.call_tir(cls.reshape2, (lv232,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1279 = R.call_tir(cls.split, (lv1278,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1280: R.Tensor((1, n, 32, 80), dtype="float16") = lv1279[0]
            lv1281 = R.call_tir(cls.rotary_embedding, (lv1280, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1282: R.Tensor((1, n, 32, 80), dtype="float16") = lv1279[1]
            lv1283 = R.call_tir(cls.rotary_embedding, (lv1282, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1284: R.Object = kv_cache[46]
            lv1285 = R.call_tir(cls.squeeze, (lv1283,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1286: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1284, lv1285, sinfo_args=(R.Object,))
            lv1287: R.Object = kv_cache[47]
            lv233_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv1279[2]
            lv234_1 = R.call_tir(cls.fused_squeeze, (lv233_1,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1290: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1287, lv234_1, sinfo_args=(R.Object,))
            lv1291: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1286, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1292: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1290, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1293 = R.call_tir(cls.reshape3, (lv1291,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1294 = R.call_tir(cls.reshape3, (lv1292,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1295 = R.call_tir(cls.transpose6, (lv1281,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1296 = R.call_tir(cls.transpose6, (lv1293,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1297 = R.call_tir(cls.transpose6, (lv1294,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv235_1 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv1295, lv1296, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv236_1 = R.call_tir(cls.fused_softmax_cast3, (lv235_1,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1306 = R.call_tir(cls.matmul8, (lv236_1, lv1297), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1307 = R.call_tir(cls.transpose7, (lv1306,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1308 = R.call_tir(cls.reshape4, (lv1307,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_283: R.Tensor((2560, 2560), dtype="float16") = model_params[283]
            param_284: R.Tensor((2560,), dtype="float16") = model_params[284]
            lv1312 = R.call_tir(cls.cast, (lv230,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_279: R.Tensor((2560,), dtype="float32") = model_params[279]
            param_280: R.Tensor((2560,), dtype="float32") = model_params[280]
            lv237_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv1312, param_279, param_280), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_285: R.Tensor((10240, 2560), dtype="float16") = model_params[285]
            param_286: R.Tensor((10240,), dtype="float16") = model_params[286]
            lv238_1 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv237_1, param_285, param_286), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_287: R.Tensor((2560, 10240), dtype="float16") = model_params[287]
            param_288: R.Tensor((2560,), dtype="float16") = model_params[288]
            lv239_1 = R.call_tir(cls.fused_NT_matmul4_add1, (lv238_1, param_287, param_288), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv240_1 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv1308, param_283, param_284, lv239_1, lv230), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1326 = R.call_tir(cls.cast, (lv240_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_289: R.Tensor((2560,), dtype="float32") = model_params[289]
            param_290: R.Tensor((2560,), dtype="float32") = model_params[290]
            lv241_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv1326, param_289, param_290), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_293: R.Tensor((7680, 2560), dtype="float16") = model_params[293]
            param_294: R.Tensor((7680,), dtype="float16") = model_params[294]
            lv242_1 = R.call_tir(cls.fused_NT_matmul_add, (lv241_1, param_293, param_294), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1333 = R.call_tir(cls.reshape2, (lv242_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1334 = R.call_tir(cls.split, (lv1333,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1335: R.Tensor((1, n, 32, 80), dtype="float16") = lv1334[0]
            lv1336 = R.call_tir(cls.rotary_embedding, (lv1335, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1337: R.Tensor((1, n, 32, 80), dtype="float16") = lv1334[1]
            lv1338 = R.call_tir(cls.rotary_embedding, (lv1337, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1339: R.Object = kv_cache[48]
            lv1340 = R.call_tir(cls.squeeze, (lv1338,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1341: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1339, lv1340, sinfo_args=(R.Object,))
            lv1342: R.Object = kv_cache[49]
            lv243: R.Tensor((1, n, 32, 80), dtype="float16") = lv1334[2]
            lv244 = R.call_tir(cls.fused_squeeze, (lv243,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1345: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1342, lv244, sinfo_args=(R.Object,))
            lv1346: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1341, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1347: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1345, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1348 = R.call_tir(cls.reshape3, (lv1346,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1349 = R.call_tir(cls.reshape3, (lv1347,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1350 = R.call_tir(cls.transpose6, (lv1336,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1351 = R.call_tir(cls.transpose6, (lv1348,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1352 = R.call_tir(cls.transpose6, (lv1349,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv245_1 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv1350, lv1351, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv246_1 = R.call_tir(cls.fused_softmax_cast3, (lv245_1,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1361 = R.call_tir(cls.matmul8, (lv246_1, lv1352), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1362 = R.call_tir(cls.transpose7, (lv1361,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1363 = R.call_tir(cls.reshape4, (lv1362,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_295: R.Tensor((2560, 2560), dtype="float16") = model_params[295]
            param_296: R.Tensor((2560,), dtype="float16") = model_params[296]
            lv1367 = R.call_tir(cls.cast, (lv240_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_291: R.Tensor((2560,), dtype="float32") = model_params[291]
            param_292: R.Tensor((2560,), dtype="float32") = model_params[292]
            lv247_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv1367, param_291, param_292), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_297: R.Tensor((10240, 2560), dtype="float16") = model_params[297]
            param_298: R.Tensor((10240,), dtype="float16") = model_params[298]
            lv248_1 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv247_1, param_297, param_298), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_299: R.Tensor((2560, 10240), dtype="float16") = model_params[299]
            param_300: R.Tensor((2560,), dtype="float16") = model_params[300]
            lv249_1 = R.call_tir(cls.fused_NT_matmul4_add1, (lv248_1, param_299, param_300), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv250_1 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv1363, param_295, param_296, lv249_1, lv240_1), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1381 = R.call_tir(cls.cast, (lv250_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_301: R.Tensor((2560,), dtype="float32") = model_params[301]
            param_302: R.Tensor((2560,), dtype="float32") = model_params[302]
            lv251_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv1381, param_301, param_302), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_305: R.Tensor((7680, 2560), dtype="float16") = model_params[305]
            param_306: R.Tensor((7680,), dtype="float16") = model_params[306]
            lv252_1 = R.call_tir(cls.fused_NT_matmul_add, (lv251_1, param_305, param_306), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1388 = R.call_tir(cls.reshape2, (lv252_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1389 = R.call_tir(cls.split, (lv1388,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1390: R.Tensor((1, n, 32, 80), dtype="float16") = lv1389[0]
            lv1391 = R.call_tir(cls.rotary_embedding, (lv1390, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1392: R.Tensor((1, n, 32, 80), dtype="float16") = lv1389[1]
            lv1393 = R.call_tir(cls.rotary_embedding, (lv1392, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1394: R.Object = kv_cache[50]
            lv1395 = R.call_tir(cls.squeeze, (lv1393,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1396: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1394, lv1395, sinfo_args=(R.Object,))
            lv1397: R.Object = kv_cache[51]
            lv253: R.Tensor((1, n, 32, 80), dtype="float16") = lv1389[2]
            lv254 = R.call_tir(cls.fused_squeeze, (lv253,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1400: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1397, lv254, sinfo_args=(R.Object,))
            lv1401: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1396, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1402: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1400, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1403 = R.call_tir(cls.reshape3, (lv1401,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1404 = R.call_tir(cls.reshape3, (lv1402,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1405 = R.call_tir(cls.transpose6, (lv1391,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1406 = R.call_tir(cls.transpose6, (lv1403,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1407 = R.call_tir(cls.transpose6, (lv1404,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv255 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv1405, lv1406, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv256 = R.call_tir(cls.fused_softmax_cast3, (lv255,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1416 = R.call_tir(cls.matmul8, (lv256, lv1407), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1417 = R.call_tir(cls.transpose7, (lv1416,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1418 = R.call_tir(cls.reshape4, (lv1417,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_307: R.Tensor((2560, 2560), dtype="float16") = model_params[307]
            param_308: R.Tensor((2560,), dtype="float16") = model_params[308]
            lv1422 = R.call_tir(cls.cast, (lv250_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_303: R.Tensor((2560,), dtype="float32") = model_params[303]
            param_304: R.Tensor((2560,), dtype="float32") = model_params[304]
            lv257 = R.call_tir(cls.fused_layer_norm_cast1, (lv1422, param_303, param_304), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_309: R.Tensor((10240, 2560), dtype="float16") = model_params[309]
            param_310: R.Tensor((10240,), dtype="float16") = model_params[310]
            lv258 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv257, param_309, param_310), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_311: R.Tensor((2560, 10240), dtype="float16") = model_params[311]
            param_312: R.Tensor((2560,), dtype="float16") = model_params[312]
            lv259 = R.call_tir(cls.fused_NT_matmul4_add1, (lv258, param_311, param_312), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv260 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv1418, param_307, param_308, lv259, lv250_1), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1436 = R.call_tir(cls.cast, (lv260,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_313: R.Tensor((2560,), dtype="float32") = model_params[313]
            param_314: R.Tensor((2560,), dtype="float32") = model_params[314]
            lv261_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv1436, param_313, param_314), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_317: R.Tensor((7680, 2560), dtype="float16") = model_params[317]
            param_318: R.Tensor((7680,), dtype="float16") = model_params[318]
            lv262_1 = R.call_tir(cls.fused_NT_matmul_add, (lv261_1, param_317, param_318), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1443 = R.call_tir(cls.reshape2, (lv262_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1444 = R.call_tir(cls.split, (lv1443,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1445: R.Tensor((1, n, 32, 80), dtype="float16") = lv1444[0]
            lv1446 = R.call_tir(cls.rotary_embedding, (lv1445, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1447: R.Tensor((1, n, 32, 80), dtype="float16") = lv1444[1]
            lv1448 = R.call_tir(cls.rotary_embedding, (lv1447, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1449: R.Object = kv_cache[52]
            lv1450 = R.call_tir(cls.squeeze, (lv1448,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1451: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1449, lv1450, sinfo_args=(R.Object,))
            lv1452: R.Object = kv_cache[53]
            lv263_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv1444[2]
            lv264 = R.call_tir(cls.fused_squeeze, (lv263_1,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1455: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1452, lv264, sinfo_args=(R.Object,))
            lv1456: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1451, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1457: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1455, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1458 = R.call_tir(cls.reshape3, (lv1456,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1459 = R.call_tir(cls.reshape3, (lv1457,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1460 = R.call_tir(cls.transpose6, (lv1446,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1461 = R.call_tir(cls.transpose6, (lv1458,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1462 = R.call_tir(cls.transpose6, (lv1459,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv265 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv1460, lv1461, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv266 = R.call_tir(cls.fused_softmax_cast3, (lv265,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1471 = R.call_tir(cls.matmul8, (lv266, lv1462), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1472 = R.call_tir(cls.transpose7, (lv1471,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1473 = R.call_tir(cls.reshape4, (lv1472,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_319: R.Tensor((2560, 2560), dtype="float16") = model_params[319]
            param_320: R.Tensor((2560,), dtype="float16") = model_params[320]
            lv1477 = R.call_tir(cls.cast, (lv260,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_315: R.Tensor((2560,), dtype="float32") = model_params[315]
            param_316: R.Tensor((2560,), dtype="float32") = model_params[316]
            lv267_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv1477, param_315, param_316), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_321: R.Tensor((10240, 2560), dtype="float16") = model_params[321]
            param_322: R.Tensor((10240,), dtype="float16") = model_params[322]
            lv268 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv267_1, param_321, param_322), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_323: R.Tensor((2560, 10240), dtype="float16") = model_params[323]
            param_324: R.Tensor((2560,), dtype="float16") = model_params[324]
            lv269 = R.call_tir(cls.fused_NT_matmul4_add1, (lv268, param_323, param_324), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv270 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv1473, param_319, param_320, lv269, lv260), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1491 = R.call_tir(cls.cast, (lv270,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_325: R.Tensor((2560,), dtype="float32") = model_params[325]
            param_326: R.Tensor((2560,), dtype="float32") = model_params[326]
            lv271 = R.call_tir(cls.fused_layer_norm_cast1, (lv1491, param_325, param_326), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_329: R.Tensor((7680, 2560), dtype="float16") = model_params[329]
            param_330: R.Tensor((7680,), dtype="float16") = model_params[330]
            lv272 = R.call_tir(cls.fused_NT_matmul_add, (lv271, param_329, param_330), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1498 = R.call_tir(cls.reshape2, (lv272,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1499 = R.call_tir(cls.split, (lv1498,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1500: R.Tensor((1, n, 32, 80), dtype="float16") = lv1499[0]
            lv1501 = R.call_tir(cls.rotary_embedding, (lv1500, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1502: R.Tensor((1, n, 32, 80), dtype="float16") = lv1499[1]
            lv1503 = R.call_tir(cls.rotary_embedding, (lv1502, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1504: R.Object = kv_cache[54]
            lv1505 = R.call_tir(cls.squeeze, (lv1503,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1506: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1504, lv1505, sinfo_args=(R.Object,))
            lv1507: R.Object = kv_cache[55]
            lv273: R.Tensor((1, n, 32, 80), dtype="float16") = lv1499[2]
            lv274 = R.call_tir(cls.fused_squeeze, (lv273,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1510: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1507, lv274, sinfo_args=(R.Object,))
            lv1511: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1506, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1512: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1510, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1513 = R.call_tir(cls.reshape3, (lv1511,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1514 = R.call_tir(cls.reshape3, (lv1512,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1515 = R.call_tir(cls.transpose6, (lv1501,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1516 = R.call_tir(cls.transpose6, (lv1513,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1517 = R.call_tir(cls.transpose6, (lv1514,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv275 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv1515, lv1516, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv276 = R.call_tir(cls.fused_softmax_cast3, (lv275,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1526 = R.call_tir(cls.matmul8, (lv276, lv1517), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1527 = R.call_tir(cls.transpose7, (lv1526,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1528 = R.call_tir(cls.reshape4, (lv1527,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_331: R.Tensor((2560, 2560), dtype="float16") = model_params[331]
            param_332: R.Tensor((2560,), dtype="float16") = model_params[332]
            lv1532 = R.call_tir(cls.cast, (lv270,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_327: R.Tensor((2560,), dtype="float32") = model_params[327]
            param_328: R.Tensor((2560,), dtype="float32") = model_params[328]
            lv277 = R.call_tir(cls.fused_layer_norm_cast1, (lv1532, param_327, param_328), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_333: R.Tensor((10240, 2560), dtype="float16") = model_params[333]
            param_334: R.Tensor((10240,), dtype="float16") = model_params[334]
            lv278 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv277, param_333, param_334), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_335: R.Tensor((2560, 10240), dtype="float16") = model_params[335]
            param_336: R.Tensor((2560,), dtype="float16") = model_params[336]
            lv279 = R.call_tir(cls.fused_NT_matmul4_add1, (lv278, param_335, param_336), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv280 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv1528, param_331, param_332, lv279, lv270), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1546 = R.call_tir(cls.cast, (lv280,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_337: R.Tensor((2560,), dtype="float32") = model_params[337]
            param_338: R.Tensor((2560,), dtype="float32") = model_params[338]
            lv281_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv1546, param_337, param_338), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_341: R.Tensor((7680, 2560), dtype="float16") = model_params[341]
            param_342: R.Tensor((7680,), dtype="float16") = model_params[342]
            lv282 = R.call_tir(cls.fused_NT_matmul_add, (lv281_1, param_341, param_342), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1553 = R.call_tir(cls.reshape2, (lv282,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1554 = R.call_tir(cls.split, (lv1553,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1555: R.Tensor((1, n, 32, 80), dtype="float16") = lv1554[0]
            lv1556 = R.call_tir(cls.rotary_embedding, (lv1555, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1557: R.Tensor((1, n, 32, 80), dtype="float16") = lv1554[1]
            lv1558 = R.call_tir(cls.rotary_embedding, (lv1557, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1559: R.Object = kv_cache[56]
            lv1560 = R.call_tir(cls.squeeze, (lv1558,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1561: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1559, lv1560, sinfo_args=(R.Object,))
            lv1562: R.Object = kv_cache[57]
            lv283: R.Tensor((1, n, 32, 80), dtype="float16") = lv1554[2]
            lv284 = R.call_tir(cls.fused_squeeze, (lv283,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1565: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1562, lv284, sinfo_args=(R.Object,))
            lv1566: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1561, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1567: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1565, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1568 = R.call_tir(cls.reshape3, (lv1566,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1569 = R.call_tir(cls.reshape3, (lv1567,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1570 = R.call_tir(cls.transpose6, (lv1556,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1571 = R.call_tir(cls.transpose6, (lv1568,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1572 = R.call_tir(cls.transpose6, (lv1569,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv285 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv1570, lv1571, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv286 = R.call_tir(cls.fused_softmax_cast3, (lv285,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1581 = R.call_tir(cls.matmul8, (lv286, lv1572), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1582 = R.call_tir(cls.transpose7, (lv1581,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1583 = R.call_tir(cls.reshape4, (lv1582,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_343: R.Tensor((2560, 2560), dtype="float16") = model_params[343]
            param_344: R.Tensor((2560,), dtype="float16") = model_params[344]
            lv1587 = R.call_tir(cls.cast, (lv280,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_339: R.Tensor((2560,), dtype="float32") = model_params[339]
            param_340: R.Tensor((2560,), dtype="float32") = model_params[340]
            lv287 = R.call_tir(cls.fused_layer_norm_cast1, (lv1587, param_339, param_340), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_345: R.Tensor((10240, 2560), dtype="float16") = model_params[345]
            param_346: R.Tensor((10240,), dtype="float16") = model_params[346]
            lv288_1 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv287, param_345, param_346), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_347: R.Tensor((2560, 10240), dtype="float16") = model_params[347]
            param_348: R.Tensor((2560,), dtype="float16") = model_params[348]
            lv289_1 = R.call_tir(cls.fused_NT_matmul4_add1, (lv288_1, param_347, param_348), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv290_1 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv1583, param_343, param_344, lv289_1, lv280), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1601 = R.call_tir(cls.cast, (lv290_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_349: R.Tensor((2560,), dtype="float32") = model_params[349]
            param_350: R.Tensor((2560,), dtype="float32") = model_params[350]
            lv291_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv1601, param_349, param_350), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_353: R.Tensor((7680, 2560), dtype="float16") = model_params[353]
            param_354: R.Tensor((7680,), dtype="float16") = model_params[354]
            lv292_1 = R.call_tir(cls.fused_NT_matmul_add, (lv291_1, param_353, param_354), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1608 = R.call_tir(cls.reshape2, (lv292_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1609 = R.call_tir(cls.split, (lv1608,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1610: R.Tensor((1, n, 32, 80), dtype="float16") = lv1609[0]
            lv1611 = R.call_tir(cls.rotary_embedding, (lv1610, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1612: R.Tensor((1, n, 32, 80), dtype="float16") = lv1609[1]
            lv1613 = R.call_tir(cls.rotary_embedding, (lv1612, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1614: R.Object = kv_cache[58]
            lv1615 = R.call_tir(cls.squeeze, (lv1613,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1616: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1614, lv1615, sinfo_args=(R.Object,))
            lv1617: R.Object = kv_cache[59]
            lv293_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv1609[2]
            lv294_1 = R.call_tir(cls.fused_squeeze, (lv293_1,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1620: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1617, lv294_1, sinfo_args=(R.Object,))
            lv1621: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1616, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1622: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1620, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1623 = R.call_tir(cls.reshape3, (lv1621,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1624 = R.call_tir(cls.reshape3, (lv1622,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1625 = R.call_tir(cls.transpose6, (lv1611,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1626 = R.call_tir(cls.transpose6, (lv1623,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1627 = R.call_tir(cls.transpose6, (lv1624,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv295_1 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv1625, lv1626, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv296_1 = R.call_tir(cls.fused_softmax_cast3, (lv295_1,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1636 = R.call_tir(cls.matmul8, (lv296_1, lv1627), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1637 = R.call_tir(cls.transpose7, (lv1636,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1638 = R.call_tir(cls.reshape4, (lv1637,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_355: R.Tensor((2560, 2560), dtype="float16") = model_params[355]
            param_356: R.Tensor((2560,), dtype="float16") = model_params[356]
            lv1642 = R.call_tir(cls.cast, (lv290_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_351: R.Tensor((2560,), dtype="float32") = model_params[351]
            param_352: R.Tensor((2560,), dtype="float32") = model_params[352]
            lv297_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv1642, param_351, param_352), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_357: R.Tensor((10240, 2560), dtype="float16") = model_params[357]
            param_358: R.Tensor((10240,), dtype="float16") = model_params[358]
            lv298 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv297_1, param_357, param_358), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_359: R.Tensor((2560, 10240), dtype="float16") = model_params[359]
            param_360: R.Tensor((2560,), dtype="float16") = model_params[360]
            lv299 = R.call_tir(cls.fused_NT_matmul4_add1, (lv298, param_359, param_360), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv300_1 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv1638, param_355, param_356, lv299, lv290_1), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1656 = R.call_tir(cls.cast, (lv300_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_361: R.Tensor((2560,), dtype="float32") = model_params[361]
            param_362: R.Tensor((2560,), dtype="float32") = model_params[362]
            lv301_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv1656, param_361, param_362), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_365: R.Tensor((7680, 2560), dtype="float16") = model_params[365]
            param_366: R.Tensor((7680,), dtype="float16") = model_params[366]
            lv302_1 = R.call_tir(cls.fused_NT_matmul_add, (lv301_1, param_365, param_366), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1663 = R.call_tir(cls.reshape2, (lv302_1,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1664 = R.call_tir(cls.split, (lv1663,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1665: R.Tensor((1, n, 32, 80), dtype="float16") = lv1664[0]
            lv1666 = R.call_tir(cls.rotary_embedding, (lv1665, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1667: R.Tensor((1, n, 32, 80), dtype="float16") = lv1664[1]
            lv1668 = R.call_tir(cls.rotary_embedding, (lv1667, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1669: R.Object = kv_cache[60]
            lv1670 = R.call_tir(cls.squeeze, (lv1668,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1671: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1669, lv1670, sinfo_args=(R.Object,))
            lv1672: R.Object = kv_cache[61]
            lv303_1: R.Tensor((1, n, 32, 80), dtype="float16") = lv1664[2]
            lv304_1 = R.call_tir(cls.fused_squeeze, (lv303_1,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1675: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1672, lv304_1, sinfo_args=(R.Object,))
            lv1676: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1671, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1677: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1675, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1678 = R.call_tir(cls.reshape3, (lv1676,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1679 = R.call_tir(cls.reshape3, (lv1677,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1680 = R.call_tir(cls.transpose6, (lv1666,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1681 = R.call_tir(cls.transpose6, (lv1678,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1682 = R.call_tir(cls.transpose6, (lv1679,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv305_1 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv1680, lv1681, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv306_1 = R.call_tir(cls.fused_softmax_cast3, (lv305_1,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1691 = R.call_tir(cls.matmul8, (lv306_1, lv1682), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1692 = R.call_tir(cls.transpose7, (lv1691,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1693 = R.call_tir(cls.reshape4, (lv1692,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_367: R.Tensor((2560, 2560), dtype="float16") = model_params[367]
            param_368: R.Tensor((2560,), dtype="float16") = model_params[368]
            lv1697 = R.call_tir(cls.cast, (lv300_1,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_363: R.Tensor((2560,), dtype="float32") = model_params[363]
            param_364: R.Tensor((2560,), dtype="float32") = model_params[364]
            lv307_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv1697, param_363, param_364), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_369: R.Tensor((10240, 2560), dtype="float16") = model_params[369]
            param_370: R.Tensor((10240,), dtype="float16") = model_params[370]
            lv308 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv307_1, param_369, param_370), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_371: R.Tensor((2560, 10240), dtype="float16") = model_params[371]
            param_372: R.Tensor((2560,), dtype="float16") = model_params[372]
            lv309 = R.call_tir(cls.fused_NT_matmul4_add1, (lv308, param_371, param_372), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv310 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3, (lv1693, param_367, param_368, lv309, lv300_1), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv1711 = R.call_tir(cls.cast, (lv310,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_373: R.Tensor((2560,), dtype="float32") = model_params[373]
            param_374: R.Tensor((2560,), dtype="float32") = model_params[374]
            lv311 = R.call_tir(cls.fused_layer_norm_cast1, (lv1711, param_373, param_374), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_377: R.Tensor((7680, 2560), dtype="float16") = model_params[377]
            param_378: R.Tensor((7680,), dtype="float16") = model_params[378]
            lv312 = R.call_tir(cls.fused_NT_matmul_add, (lv311, param_377, param_378), out_sinfo=R.Tensor((1, n, 7680), dtype="float16"))
            lv1718 = R.call_tir(cls.reshape2, (lv312,), out_sinfo=R.Tensor((1, n, 32, 240), dtype="float16"))
            lv1719 = R.call_tir(cls.split, (lv1718,), out_sinfo=[R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16"), R.Tensor((1, n, 32, 80), dtype="float16")])
            lv1720: R.Tensor((1, n, 32, 80), dtype="float16") = lv1719[0]
            lv1721 = R.call_tir(cls.rotary_embedding, (lv1720, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1722: R.Tensor((1, n, 32, 80), dtype="float16") = lv1719[1]
            lv1723 = R.call_tir(cls.rotary_embedding, (lv1722, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"), tir_vars=R.shape([m]))
            lv1724: R.Object = kv_cache[62]
            lv1725 = R.call_tir(cls.squeeze, (lv1723,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1726: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1724, lv1725, sinfo_args=(R.Object,))
            lv1727: R.Object = kv_cache[63]
            lv313: R.Tensor((1, n, 32, 80), dtype="float16") = lv1719[2]
            lv314 = R.call_tir(cls.fused_squeeze, (lv313,), out_sinfo=R.Tensor((n, 32, 80), dtype="float16"))
            lv1730: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv1727, lv314, sinfo_args=(R.Object,))
            lv1731: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1726, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1732: R.Tensor((m, 32, 80), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv1730, R.shape([m, 32, 80]), sinfo_args=(R.Tensor((m, 32, 80), dtype="float16"),))
            lv1733 = R.call_tir(cls.reshape3, (lv1731,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1734 = R.call_tir(cls.reshape3, (lv1732,), out_sinfo=R.Tensor((1, m, 32, 80), dtype="float16"))
            lv1735 = R.call_tir(cls.transpose6, (lv1721,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1736 = R.call_tir(cls.transpose6, (lv1733,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv1737 = R.call_tir(cls.transpose6, (lv1734,), out_sinfo=R.Tensor((1, 32, m, 80), dtype="float16"))
            lv315 = R.call_tir(cls.fused_NT_matmul1_divide_maximum_minimum_cast2, (lv1735, lv1736, lv5), out_sinfo=R.Tensor((1, 32, n, m), dtype="float32"))
            lv316_1 = R.call_tir(cls.fused_softmax_cast3, (lv315,), out_sinfo=R.Tensor((1, 32, n, m), dtype="float16"))
            lv1746 = R.call_tir(cls.matmul8, (lv316_1, lv1737), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
            lv1747 = R.call_tir(cls.transpose7, (lv1746,), out_sinfo=R.Tensor((1, n, 32, 80), dtype="float16"))
            lv1748 = R.call_tir(cls.reshape4, (lv1747,), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_379: R.Tensor((2560, 2560), dtype="float16") = model_params[379]
            param_380: R.Tensor((2560,), dtype="float16") = model_params[380]
            lv1752 = R.call_tir(cls.cast, (lv310,), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_375: R.Tensor((2560,), dtype="float32") = model_params[375]
            param_376: R.Tensor((2560,), dtype="float32") = model_params[376]
            lv317_1 = R.call_tir(cls.fused_layer_norm_cast1, (lv1752, param_375, param_376), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            param_381: R.Tensor((10240, 2560), dtype="float16") = model_params[381]
            param_382: R.Tensor((10240,), dtype="float16") = model_params[382]
            lv318_1 = R.call_tir(cls.fused_NT_matmul3_add2_gelu, (lv317_1, param_381, param_382), out_sinfo=R.Tensor((1, n, 10240), dtype="float16"))
            param_383: R.Tensor((2560, 10240), dtype="float16") = model_params[383]
            param_384: R.Tensor((2560,), dtype="float16") = model_params[384]
            lv319 = R.call_tir(cls.fused_NT_matmul4_add1, (lv318_1, param_383, param_384), out_sinfo=R.Tensor((1, n, 2560), dtype="float16"))
            lv320 = R.call_tir(cls.fused_NT_matmul2_add1_add3_add3_cast, (lv1748, param_379, param_380, lv319, lv310), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            param_385: R.Tensor((2560,), dtype="float32") = model_params[385]
            param_386: R.Tensor((2560,), dtype="float32") = model_params[386]
            lv1767 = R.call_tir(cls.layer_norm, (lv320, param_385, param_386), out_sinfo=R.Tensor((1, n, 2560), dtype="float32"))
            lv1768 = R.call_tir(cls.slice, (lv1767,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            lv1769 = R.call_tir(cls.cast4, (lv1768,), out_sinfo=R.Tensor((1, 1, 2560), dtype="float32"))
            param_387: R.Tensor((50280, 2560), dtype="float32") = model_params[387]
            lv160_1 = R.call_tir(cls.NT_matmul5, (lv1769, param_387), out_sinfo=R.Tensor((1, 1, 50280), dtype="float32"))
            gv: R.Tuple(R.Tensor((1, 1, 50280), dtype="float32"), R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object)) = lv160_1, (lv21, lv25, lv76, lv80, lv131, lv135, lv186, lv190, lv241, lv245, lv296, lv300, lv351, lv355, lv406, lv410, lv461, lv465, lv516, lv520, lv571, lv575, lv626, lv630, lv681, lv685, lv736, lv740, lv791, lv795, lv846, lv850, lv901, lv905, lv956, lv960, lv1011, lv1015, lv1066, lv1070, lv1121, lv1125, lv1176, lv1180, lv1231, lv1235, lv1286, lv1290, lv1341, lv1345, lv1396, lv1400, lv1451, lv1455, lv1506, lv1510, lv1561, lv1565, lv1616, lv1620, lv1671, lv1675, lv1726, lv1730)
            R.output(gv)
        return gv

    @R.function
    def softmax_with_temperature(logits: R.Tensor((1, 1, 50280), dtype="float32"), temperature: R.Tensor((), dtype="float32")) -> R.Tensor((1, 1, 50280), dtype="float32"):
        R.func_attr({"tir_var_upper_bound": {"m": 4096, "n": 4096}})
        cls = Module
        with R.dataflow():
            lv3607 = R.call_tir(cls.divide2, (logits, temperature), out_sinfo=R.Tensor((1, 1, 50280), dtype="float32"))
            gv3 = R.call_tir(cls.softmax2, (lv3607,), out_sinfo=R.Tensor((1, 1, 50280), dtype="float32"))
            R.output(gv3)
        return gv3

# Metadata omitted. Use show_meta=True in script() method to show it.