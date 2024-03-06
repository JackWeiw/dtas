# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_compute: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)), "float16")
        compute = T.match_buffer(var_compute, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(15744), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(1)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) % T.int64(2560))
                            T.where((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])