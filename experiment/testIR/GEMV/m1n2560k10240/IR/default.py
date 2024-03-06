from tvm.script import ir as I
from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(lv9: T.Buffer((1, 1, 10240), "float16"), lv3: T.Buffer((2560, 10240), "float16"), lv4: T.Buffer((2560,), "float16"), var_T_add_intermediate: T.Buffer((1, 1, 2560), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate_local = T.alloc_buffer((1, 1, 2560), "float16", scope="local")
        var_NT_matmul_intermediate_rf_local = T.alloc_buffer((128, 1, 1, 2560), "float16", scope="local")
        var_NT_matmul_intermediate_rf_local_1 = T.alloc_buffer((32, 1, 1, 2560), "float16", scope="local")
        lv3_local = T.alloc_buffer((2560, 10240), "float16", scope="local")
        lv9_shared = T.alloc_buffer((1, 1, 10240), "float16", scope="shared")
        for u_fused_ax0_fused_fused_0 in T.thread_binding(160, thread="blockIdx.x"):
            for u_fused_ax0_fused_fused_1 in T.thread_binding(16, thread="threadIdx.y"):
                for ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 in T.thread_binding(32, thread="threadIdx.x"):
                    for ax0, ax1 in T.grid(1, 1):
                        for ax2_0 in T.serial(5, annotations={"pragma_unroll_explicit": 256, "pragma_vectorize": 1}):
                            for ax2_1 in T.thread_binding(16, thread="threadIdx.y"):
                                for ax2_2 in T.thread_binding(32, thread="threadIdx.x"):
                                    for ax2_3 in T.vectorized(4):
                                        with T.block("lv9_shared"):
                                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                                            v2 = T.axis.spatial(10240, ax2_0 * 2048 + ax2_1 * 128 + ax2_2 * 4 + ax2_3)
                                            T.reads(lv9[v0, v1, v2])
                                            T.writes(lv9_shared[v0, v1, v2])
                                            lv9_shared[v0, v1, v2] = lv9[v0, v1, v2]
                    for u_fused_ax0_fused_fused_2_init in range(1):
                        for ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1_init in T.vectorized(4):
                            with T.block("NT_matmul_rf_init"):
                                vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused = T.axis.spatial(128, ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 * 4 + ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1_init)
                                v0 = T.axis.spatial(2560, u_fused_ax0_fused_fused_0 * 16 + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2_init)
                                T.reads()
                                T.writes(var_NT_matmul_intermediate_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, 0, 0, v0])
                                var_NT_matmul_intermediate_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, 0, 0, v0] = T.float16(0)
                    for ax1_fused_u_fused_0 in T.serial(40, annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        for ax0_0, ax1 in T.grid(1, 8):
                            for ax0_1 in T.vectorized(1):
                                with T.block("lv3_local"):
                                    v0 = T.axis.spatial(2560, u_fused_ax0_fused_fused_0 * 16 + u_fused_ax0_fused_fused_1 + ax0_0 + ax0_1)
                                    v1 = T.axis.spatial(10240, ax1_fused_u_fused_0 * 256 + ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 * 8 + ax1)
                                    T.reads(lv3[v0, v1])
                                    T.writes(lv3_local[v0, v1])
                                    lv3_local[v0, v1] = lv3[v0, v1]
                        for u_fused_ax0_fused_fused_2, ax1_fused_u_fused_2 in T.grid(1, 2):
                            for ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1 in T.vectorized(4):
                                with T.block("NT_matmul_rf_update"):
                                    vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused = T.axis.spatial(128, ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 * 4 + ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1)
                                    v0 = T.axis.spatial(2560, u_fused_ax0_fused_fused_0 * 16 + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2)
                                    vax1_fused_u_fused_0, vax1_fused_u_fused_2 = T.axis.remap("RR", [ax1_fused_u_fused_0, ax1_fused_u_fused_2])
                                    T.reads(var_NT_matmul_intermediate_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, 0, 0, v0], lv9_shared[0, 0, vax1_fused_u_fused_0 * 256 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused // 4 * 8 + vax1_fused_u_fused_2 * 4 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused % 4], lv3_local[v0, vax1_fused_u_fused_0 * 256 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused // 4 * 8 + vax1_fused_u_fused_2 * 4 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused % 4])
                                    T.writes(var_NT_matmul_intermediate_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, 0, 0, v0])
                                    var_NT_matmul_intermediate_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, 0, 0, v0] = var_NT_matmul_intermediate_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, 0, 0, v0] + lv9_shared[0, 0, vax1_fused_u_fused_0 * 256 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused // 4 * 8 + vax1_fused_u_fused_2 * 4 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused % 4] * lv3_local[v0, vax1_fused_u_fused_0 * 256 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused // 4 * 8 + vax1_fused_u_fused_2 * 4 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused % 4]
            for ax2_fused_0 in T.thread_binding(16, thread="threadIdx.y"):
                for ax0 in T.thread_binding(32, thread="threadIdx.x"):
                    for ax2_fused_1_0 in T.serial(1, annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        for ax2_fused_1_1 in T.vectorized(1):
                            with T.block("NT_matmul_rf_init"):
                                vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 = T.axis.spatial(32, ax0)
                                v0 = T.axis.spatial(2560, u_fused_ax0_fused_fused_0 * 16 + ax2_fused_0 + ax2_fused_1_0 + ax2_fused_1_1)
                                T.reads()
                                T.writes(var_NT_matmul_intermediate_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, 0, v0])
                                var_NT_matmul_intermediate_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, 0, v0] = T.float16(0)
                            for ax1 in range(4):
                                with T.block("NT_matmul_rf_update"):
                                    vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1 = T.axis.remap("SR", [ax0, ax1])
                                    v0 = T.axis.spatial(2560, u_fused_ax0_fused_fused_0 * 16 + ax2_fused_0 + ax2_fused_1_0 + ax2_fused_1_1)
                                    T.reads(var_NT_matmul_intermediate_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, 0, v0], var_NT_matmul_intermediate_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 * 4 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1, 0, 0, v0])
                                    T.writes(var_NT_matmul_intermediate_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, 0, v0])
                                    var_NT_matmul_intermediate_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, 0, v0] = var_NT_matmul_intermediate_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, 0, v0] + var_NT_matmul_intermediate_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 * 4 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1, 0, 0, v0]
            for ax1_fused_1 in range(1):
                for ax1_fused_0 in T.thread_binding(16, thread="threadIdx.y"):
                    for ax0 in T.thread_binding(32, thread="threadIdx.x"):
                        with T.block("NT_matmul"):
                            vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 = T.axis.reduce(32, ax0)
                            v0 = T.axis.spatial(2560, u_fused_ax0_fused_fused_0 * 16 + ax1_fused_0 + ax1_fused_1)
                            T.reads(var_NT_matmul_intermediate_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, 0, v0])
                            T.writes(var_NT_matmul_intermediate_local[0, 0, v0])
                            with T.init():
                                var_NT_matmul_intermediate_local[0, 0, v0] = T.float16(0)
                            var_NT_matmul_intermediate_local[0, 0, v0] = var_NT_matmul_intermediate_local[0, 0, v0] + var_NT_matmul_intermediate_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, 0, v0]
            for ax0_fused_0 in T.thread_binding(16, thread="threadIdx.y"):
                for ax0_fused_1 in range(1):
                    with T.block("T_add"):
                        v0 = T.axis.spatial(2560, u_fused_ax0_fused_fused_0 * 16 + ax0_fused_0 + ax0_fused_1)
                        T.reads(var_NT_matmul_intermediate_local[0, 0, v0], lv4[v0])
                        T.writes(var_T_add_intermediate[0, 0, v0])
                        var_T_add_intermediate[0, 0, v0] = var_NT_matmul_intermediate_local[0, 0, v0] + lv4[v0]