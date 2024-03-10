#name: fused_layer_norm_cast1
#range: (<n: Range(1, 256)>,)  latency(ms):    0.0043    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn', 'max_active_blocks_per_sm': 10}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv6: T.handle, param_1: T.Buffer((T.int64(2560),), "float32"), param_2: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(2560)))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        A_red_temp_v0_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        A_red_temp_v1_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        lv6_shared_dyn = T.alloc_buffer((T.int64(1), n, T.int64(2560)), scope="shared.dyn")
        for ax0_fused in T.thread_binding(n, thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_0 in range(T.int64(5)):
                        for ax2_2 in T.vectorized(T.int64(4)):
                            with T.block("lv6_shared.dyn"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(n, ax0_fused + ax1)
                                v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                                T.reads(lv6[v0, v1, v2])
                                T.writes(lv6_shared_dyn[v0, v1, v2])
                                T.block_attr({"tir.manifest_shared_memory_local_stage": 1})
                                lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(128) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(257, 512)>,)  latency(ms):    0.0078    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn', 'max_active_blocks_per_sm': 10}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv6: T.handle, param_1: T.Buffer((T.int64(2560),), "float32"), param_2: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(2560)))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        A_red_temp_v0_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        A_red_temp_v1_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        lv6_shared_dyn = T.alloc_buffer((T.int64(1), n, T.int64(2560)), scope="shared.dyn")
        for ax0_fused in T.thread_binding(n, thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_0 in range(T.int64(5)):
                        for ax2_2 in T.vectorized(T.int64(4)):
                            with T.block("lv6_shared.dyn"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(n, ax0_fused + ax1)
                                v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                                T.reads(lv6[v0, v1, v2])
                                T.writes(lv6_shared_dyn[v0, v1, v2])
                                T.block_attr({"tir.manifest_shared_memory_local_stage": 1})
                                lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(128) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(513, 768)>,)  latency(ms):    0.0154    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn', 'max_active_blocks_per_sm': 10}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv6: T.handle, param_1: T.Buffer((T.int64(2560),), "float32"), param_2: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(2560)))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        A_red_temp_v0_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        A_red_temp_v1_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        lv6_shared_dyn = T.alloc_buffer((T.int64(1), n, T.int64(2560)), scope="shared.dyn")
        for ax0_fused in T.thread_binding(n, thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_0 in range(T.int64(5)):
                        for ax2_2 in T.vectorized(T.int64(4)):
                            with T.block("lv6_shared.dyn"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(n, ax0_fused + ax1)
                                v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                                T.reads(lv6[v0, v1, v2])
                                T.writes(lv6_shared_dyn[v0, v1, v2])
                                T.block_attr({"tir.manifest_shared_memory_local_stage": 1})
                                lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(128) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.0209    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn', 'max_active_blocks_per_sm': 10}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv6: T.handle, param_1: T.Buffer((T.int64(2560),), "float32"), param_2: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(2560)))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        A_red_temp_v0_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        A_red_temp_v1_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        lv6_shared_dyn = T.alloc_buffer((T.int64(1), n, T.int64(2560)), scope="shared.dyn")
        for ax0_fused in T.thread_binding(n, thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_0 in range(T.int64(5)):
                        for ax2_2 in T.vectorized(T.int64(4)):
                            with T.block("lv6_shared.dyn"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(n, ax0_fused + ax1)
                                v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                                T.reads(lv6[v0, v1, v2])
                                T.writes(lv6_shared_dyn[v0, v1, v2])
                                T.block_attr({"tir.manifest_shared_memory_local_stage": 1})
                                lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(128) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.0256    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn', 'max_active_blocks_per_sm': 10}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv6: T.handle, param_1: T.Buffer((T.int64(2560),), "float32"), param_2: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(2560)))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        A_red_temp_v0_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        A_red_temp_v1_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        lv6_shared_dyn = T.alloc_buffer((T.int64(1), n, T.int64(2560)), scope="shared.dyn")
        for ax0_fused in T.thread_binding(n, thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_0 in range(T.int64(5)):
                        for ax2_2 in T.vectorized(T.int64(4)):
                            with T.block("lv6_shared.dyn"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(n, ax0_fused + ax1)
                                v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                                T.reads(lv6[v0, v1, v2])
                                T.writes(lv6_shared_dyn[v0, v1, v2])
                                T.block_attr({"tir.manifest_shared_memory_local_stage": 1})
                                lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(128) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.0298    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn', 'max_active_blocks_per_sm': 10}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv6: T.handle, param_1: T.Buffer((T.int64(2560),), "float32"), param_2: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(2560)))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        A_red_temp_v0_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        A_red_temp_v1_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        lv6_shared_dyn = T.alloc_buffer((T.int64(1), n, T.int64(2560)), scope="shared.dyn")
        for ax0_fused in T.thread_binding(n, thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_0 in range(T.int64(5)):
                        for ax2_2 in T.vectorized(T.int64(4)):
                            with T.block("lv6_shared.dyn"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(n, ax0_fused + ax1)
                                v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                                T.reads(lv6[v0, v1, v2])
                                T.writes(lv6_shared_dyn[v0, v1, v2])
                                T.block_attr({"tir.manifest_shared_memory_local_stage": 1})
                                lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(128) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.0341    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn', 'max_active_blocks_per_sm': 10}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv6: T.handle, param_1: T.Buffer((T.int64(2560),), "float32"), param_2: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(2560)))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        A_red_temp_v0_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        A_red_temp_v1_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        lv6_shared_dyn = T.alloc_buffer((T.int64(1), n, T.int64(2560)), scope="shared.dyn")
        for ax0_fused in T.thread_binding(n, thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_0 in range(T.int64(5)):
                        for ax2_2 in T.vectorized(T.int64(4)):
                            with T.block("lv6_shared.dyn"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(n, ax0_fused + ax1)
                                v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                                T.reads(lv6[v0, v1, v2])
                                T.writes(lv6_shared_dyn[v0, v1, v2])
                                T.block_attr({"tir.manifest_shared_memory_local_stage": 1})
                                lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(128) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.0386    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn', 'max_active_blocks_per_sm': 10}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv6: T.handle, param_1: T.Buffer((T.int64(2560),), "float32"), param_2: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(2560)))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        A_red_temp_v0_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        A_red_temp_v1_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        lv6_shared_dyn = T.alloc_buffer((T.int64(1), n, T.int64(2560)), scope="shared.dyn")
        for ax0_fused in T.thread_binding(n, thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_0 in range(T.int64(5)):
                        for ax2_2 in T.vectorized(T.int64(4)):
                            with T.block("lv6_shared.dyn"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(n, ax0_fused + ax1)
                                v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                                T.reads(lv6[v0, v1, v2])
                                T.writes(lv6_shared_dyn[v0, v1, v2])
                                T.block_attr({"tir.manifest_shared_memory_local_stage": 1})
                                lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(128) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.0436    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn', 'max_active_blocks_per_sm': 10}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv6: T.handle, param_1: T.Buffer((T.int64(2560),), "float32"), param_2: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(2560)))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        A_red_temp_v0_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        A_red_temp_v1_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        lv6_shared_dyn = T.alloc_buffer((T.int64(1), n, T.int64(2560)), scope="shared.dyn")
        for ax0_fused in T.thread_binding(n, thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_0 in range(T.int64(5)):
                        for ax2_2 in T.vectorized(T.int64(4)):
                            with T.block("lv6_shared.dyn"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(n, ax0_fused + ax1)
                                v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                                T.reads(lv6[v0, v1, v2])
                                T.writes(lv6_shared_dyn[v0, v1, v2])
                                T.block_attr({"tir.manifest_shared_memory_local_stage": 1})
                                lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(128) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.0485    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn', 'max_active_blocks_per_sm': 10}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv6: T.handle, param_1: T.Buffer((T.int64(2560),), "float32"), param_2: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(2560)))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        A_red_temp_v0_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        A_red_temp_v1_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        lv6_shared_dyn = T.alloc_buffer((T.int64(1), n, T.int64(2560)), scope="shared.dyn")
        for ax0_fused in T.thread_binding(n, thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_0 in range(T.int64(5)):
                        for ax2_2 in T.vectorized(T.int64(4)):
                            with T.block("lv6_shared.dyn"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(n, ax0_fused + ax1)
                                v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                                T.reads(lv6[v0, v1, v2])
                                T.writes(lv6_shared_dyn[v0, v1, v2])
                                T.block_attr({"tir.manifest_shared_memory_local_stage": 1})
                                lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(128) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.0531    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn', 'max_active_blocks_per_sm': 10}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv6: T.handle, param_1: T.Buffer((T.int64(2560),), "float32"), param_2: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(2560)))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        A_red_temp_v0_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        A_red_temp_v1_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        lv6_shared_dyn = T.alloc_buffer((T.int64(1), n, T.int64(2560)), scope="shared.dyn")
        for ax0_fused in T.thread_binding(n, thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_0 in range(T.int64(5)):
                        for ax2_2 in T.vectorized(T.int64(4)):
                            with T.block("lv6_shared.dyn"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(n, ax0_fused + ax1)
                                v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                                T.reads(lv6[v0, v1, v2])
                                T.writes(lv6_shared_dyn[v0, v1, v2])
                                T.block_attr({"tir.manifest_shared_memory_local_stage": 1})
                                lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(128) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.0577    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn', 'max_active_blocks_per_sm': 10}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv6: T.handle, param_1: T.Buffer((T.int64(2560),), "float32"), param_2: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(2560)))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        A_red_temp_v0_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        A_red_temp_v1_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        lv6_shared_dyn = T.alloc_buffer((T.int64(1), n, T.int64(2560)), scope="shared.dyn")
        for ax0_fused in T.thread_binding(n, thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_0 in range(T.int64(5)):
                        for ax2_2 in T.vectorized(T.int64(4)):
                            with T.block("lv6_shared.dyn"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(n, ax0_fused + ax1)
                                v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                                T.reads(lv6[v0, v1, v2])
                                T.writes(lv6_shared_dyn[v0, v1, v2])
                                T.block_attr({"tir.manifest_shared_memory_local_stage": 1})
                                lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(128) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.0623    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn', 'max_active_blocks_per_sm': 10}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv6: T.handle, param_1: T.Buffer((T.int64(2560),), "float32"), param_2: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(2560)))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        A_red_temp_v0_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        A_red_temp_v1_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        lv6_shared_dyn = T.alloc_buffer((T.int64(1), n, T.int64(2560)), scope="shared.dyn")
        for ax0_fused in T.thread_binding(n, thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_0 in range(T.int64(5)):
                        for ax2_2 in T.vectorized(T.int64(4)):
                            with T.block("lv6_shared.dyn"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(n, ax0_fused + ax1)
                                v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                                T.reads(lv6[v0, v1, v2])
                                T.writes(lv6_shared_dyn[v0, v1, v2])
                                T.block_attr({"tir.manifest_shared_memory_local_stage": 1})
                                lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(128) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.0670    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn', 'max_active_blocks_per_sm': 10}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv6: T.handle, param_1: T.Buffer((T.int64(2560),), "float32"), param_2: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(2560)))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        A_red_temp_v0_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        A_red_temp_v1_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        lv6_shared_dyn = T.alloc_buffer((T.int64(1), n, T.int64(2560)), scope="shared.dyn")
        for ax0_fused in T.thread_binding(n, thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_0 in range(T.int64(5)):
                        for ax2_2 in T.vectorized(T.int64(4)):
                            with T.block("lv6_shared.dyn"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(n, ax0_fused + ax1)
                                v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                                T.reads(lv6[v0, v1, v2])
                                T.writes(lv6_shared_dyn[v0, v1, v2])
                                T.block_attr({"tir.manifest_shared_memory_local_stage": 1})
                                lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(128) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.0716    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn', 'max_active_blocks_per_sm': 10}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv6: T.handle, param_1: T.Buffer((T.int64(2560),), "float32"), param_2: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(2560)))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        A_red_temp_v0_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        A_red_temp_v1_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        lv6_shared_dyn = T.alloc_buffer((T.int64(1), n, T.int64(2560)), scope="shared.dyn")
        for ax0_fused in T.thread_binding(n, thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_0 in range(T.int64(5)):
                        for ax2_2 in T.vectorized(T.int64(4)):
                            with T.block("lv6_shared.dyn"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(n, ax0_fused + ax1)
                                v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                                T.reads(lv6[v0, v1, v2])
                                T.writes(lv6_shared_dyn[v0, v1, v2])
                                T.block_attr({"tir.manifest_shared_memory_local_stage": 1})
                                lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(128) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.0763    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn', 'max_active_blocks_per_sm': 10}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(p_lv6: T.handle, param_1: T.Buffer((T.int64(2560),), "float32"), param_2: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(2560)))
        compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        A_red_temp_v0_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        A_red_temp_v1_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
        lv6_shared_dyn = T.alloc_buffer((T.int64(1), n, T.int64(2560)), scope="shared.dyn")
        for ax0_fused in T.thread_binding(n, thread="blockIdx.x"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_0 in range(T.int64(5)):
                        for ax2_2 in T.vectorized(T.int64(4)):
                            with T.block("lv6_shared.dyn"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(n, ax0_fused + ax1)
                                v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                                T.reads(lv6[v0, v1, v2])
                                T.writes(lv6_shared_dyn[v0, v1, v2])
                                T.block_attr({"tir.manifest_shared_memory_local_stage": 1})
                                lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(128) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(20), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(128) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
