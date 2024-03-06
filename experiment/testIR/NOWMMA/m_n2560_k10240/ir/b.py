from tvm.script import ir as I
from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv9: T.handle, lv3: T.Buffer((T.int64(2560), T.int64(10240)), "int8"), lv4: T.Buffer((T.int64(2560),), "int8"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv9 = T.match_buffer(p_lv9, (T.int64(1), n, T.int64(10240)), "int8")
        var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "int8")
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "int8")
        lv9_reindex = T.alloc_buffer((T.int64(1), n, T.int64(10240)), "int8")
        lv3_reindex = T.alloc_buffer((T.int64(1), T.int64(2560), T.int64(10240)), "int8")
        var_NT_matmul_intermediate_reindex = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "int8")
        for ax0, ax1 in T.grid(n, T.int64(10240)):
            with T.block("lv9_reindex_reindex"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv9[T.int64(0), v0, v1])
                T.writes(lv9_reindex[T.int64(0), v0, v1])
                lv9_reindex[T.int64(0), v0, v1] = lv9[T.int64(0), v0, v1]
        for ax0, ax1 in T.grid(T.int64(2560), T.int64(10240)):
            with T.block("lv3_reindex_reindex"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv3[v0, v1])
                T.writes(lv3_reindex[T.int64(0), v0, v1])
                lv3_reindex[T.int64(0), v0, v1] = lv3[v0, v1]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), n, T.int64(2560), T.int64(10240)):
            with T.block("NT_matmul"):
                v0, v1, v2, v3 = T.axis.remap("SSSR", [ax0, ax1, ax2, ax3])
                T.reads(lv9_reindex[T.int64(0), v1, v3], lv3_reindex[T.int64(0), v2, v3])
                T.writes(var_NT_matmul_intermediate_reindex[T.int64(0), v1, v2])
                with T.init():
                    var_NT_matmul_intermediate_reindex[T.int64(0), v1, v2] = T.int8(0)
                var_NT_matmul_intermediate_reindex[T.int64(0), v1, v2] = var_NT_matmul_intermediate_reindex[T.int64(0), v1, v2] + lv9_reindex[T.int64(0), v1, v3] * lv3_reindex[T.int64(0), v2, v3]
        for ax0, ax1 in T.grid(n, T.int64(2560)):
            with T.block("var_NT_matmul_intermediate_reindex"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(var_NT_matmul_intermediate_reindex[T.int64(0), v0, v1])
                T.writes(var_NT_matmul_intermediate[T.int64(0), v0, v1])
                var_NT_matmul_intermediate[T.int64(0), v0, v1] = var_NT_matmul_intermediate_reindex[T.int64(0), v0, v1]
        for ax0, ax1 in T.grid(n, T.int64(2560)):
            with T.block("T_add"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(var_NT_matmul_intermediate[T.int64(0), v0, v1], lv4[v1])
                T.writes(var_T_add_intermediate[T.int64(0), v0, v1])
                var_T_add_intermediate[T.int64(0), v0, v1] = var_NT_matmul_intermediate[T.int64(0), v0, v1] + lv4[v1]