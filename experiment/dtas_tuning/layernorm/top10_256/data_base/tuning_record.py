#name: fused_layer_norm_cast1
#range: (<n: Range(1, 256)>,)  latency(ms):    0.0044    
#config: ReductionConfig: {'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1408) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(352) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(352) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(352) + ax1_1)
                        T.where(ax1_0 * T.int64(352) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1, 256)>,)  latency(ms):    0.0041    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(5)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
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
#range: (<n: Range(1, 256)>,)  latency(ms):    0.0042    
#config: ReductionConfig: {'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(640) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(160) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1, 256)>,)  latency(ms):    0.0042    
#config: ReductionConfig: {'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(768) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(192) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(192) + ax1_1)
                        T.where(ax1_0 * T.int64(192) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1, 256)>,)  latency(ms):    0.0045    
#config: ReductionConfig: {'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(2048) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(512) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(512) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1, 256)>,)  latency(ms):    0.0041    
#config: ReductionConfig: {'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1024) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(256) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(256) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1, 256)>,)  latency(ms):    0.0042    
#config: ReductionConfig: {'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1280) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(320) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1, 256)>,)  latency(ms):    0.0043    
#config: ReductionConfig: {'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(896) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(224) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(224) + ax1_1)
                        T.where(ax1_0 * T.int64(224) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1, 256)>,)  latency(ms):    0.0043    
#config: ReductionConfig: {'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1152) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(288) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(288) + ax1_1)
                        T.where(ax1_0 * T.int64(288) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1, 256)>,)  latency(ms):    0.0044    
#config: ReductionConfig: {'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1536) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(384) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(384) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(384) + ax1_1)
                        T.where(ax1_0 * T.int64(384) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(257, 512)>,)  latency(ms):    0.0084    
#config: ReductionConfig: {'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(896) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(224) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(224) + ax1_1)
                        T.where(ax1_0 * T.int64(224) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(257, 512)>,)  latency(ms):    0.0084    
#config: ReductionConfig: {'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1024) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(256) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(256) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(257, 512)>,)  latency(ms):    0.0088    
#config: ReductionConfig: {'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(2048) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(512) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(512) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(257, 512)>,)  latency(ms):    0.0079    
#config: ReductionConfig: {'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1280) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(320) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(257, 512)>,)  latency(ms):    0.0084    
#config: ReductionConfig: {'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1152) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(288) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(288) + ax1_1)
                        T.where(ax1_0 * T.int64(288) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(257, 512)>,)  latency(ms):    0.0083    
#config: ReductionConfig: {'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1408) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(352) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(352) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(352) + ax1_1)
                        T.where(ax1_0 * T.int64(352) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(257, 512)>,)  latency(ms):    0.0079    
#config: ReductionConfig: {'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(640) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(160) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(257, 512)>,)  latency(ms):    0.0080    
#config: ReductionConfig: {'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(768) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(192) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(192) + ax1_1)
                        T.where(ax1_0 * T.int64(192) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(257, 512)>,)  latency(ms):    0.0084    
#config: ReductionConfig: {'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1536) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(384) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(384) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(384) + ax1_1)
                        T.where(ax1_0 * T.int64(384) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(257, 512)>,)  latency(ms):    0.0078    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(5)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
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
#range: (<n: Range(513, 768)>,)  latency(ms):    0.0168    
#config: ReductionConfig: {'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(896) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(224) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(224) + ax1_1)
                        T.where(ax1_0 * T.int64(224) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(513, 768)>,)  latency(ms):    0.0150    
#config: ReductionConfig: {'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1408) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(352) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(352) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(352) + ax1_1)
                        T.where(ax1_0 * T.int64(352) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(513, 768)>,)  latency(ms):    0.0160    
#config: ReductionConfig: {'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1152) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(288) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(288) + ax1_1)
                        T.where(ax1_0 * T.int64(288) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(513, 768)>,)  latency(ms):    0.0149    
#config: ReductionConfig: {'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1536) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(384) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(384) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(384) + ax1_1)
                        T.where(ax1_0 * T.int64(384) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(513, 768)>,)  latency(ms):    0.0146    
#config: ReductionConfig: {'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1280) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(320) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(513, 768)>,)  latency(ms):    0.0155    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(5)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
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
#range: (<n: Range(513, 768)>,)  latency(ms):    0.0158    
#config: ReductionConfig: {'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(640) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(160) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(513, 768)>,)  latency(ms):    0.0162    
#config: ReductionConfig: {'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(768) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(192) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(192) + ax1_1)
                        T.where(ax1_0 * T.int64(192) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(513, 768)>,)  latency(ms):    0.0166    
#config: ReductionConfig: {'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1024) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(256) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(256) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(513, 768)>,)  latency(ms):    0.0150    
#config: ReductionConfig: {'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(2048) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(512) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(512) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.0209    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(5)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
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
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.0213    
#config: ReductionConfig: {'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(896) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(224) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(224) + ax1_1)
                        T.where(ax1_0 * T.int64(224) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.0210    
#config: ReductionConfig: {'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(768) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(192) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(192) + ax1_1)
                        T.where(ax1_0 * T.int64(192) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.0193    
#config: ReductionConfig: {'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1280) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(320) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.0210    
#config: ReductionConfig: {'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1024) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(256) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(256) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.0197    
#config: ReductionConfig: {'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1408) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(352) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(352) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(352) + ax1_1)
                        T.where(ax1_0 * T.int64(352) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.0198    
#config: ReductionConfig: {'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1536) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(384) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(384) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(384) + ax1_1)
                        T.where(ax1_0 * T.int64(384) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.0197    
#config: ReductionConfig: {'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(2048) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(512) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(512) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.0204    
#config: ReductionConfig: {'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1152) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(288) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(288) + ax1_1)
                        T.where(ax1_0 * T.int64(288) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.0209    
#config: ReductionConfig: {'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(640) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(160) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.0238    
#config: ReductionConfig: {'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1280) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(320) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.0242    
#config: ReductionConfig: {'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1408) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(352) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(352) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(352) + ax1_1)
                        T.where(ax1_0 * T.int64(352) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.0262    
#config: ReductionConfig: {'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(896) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(224) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(224) + ax1_1)
                        T.where(ax1_0 * T.int64(224) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.0253    
#config: ReductionConfig: {'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(768) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(192) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(192) + ax1_1)
                        T.where(ax1_0 * T.int64(192) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.0243    
#config: ReductionConfig: {'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(2048) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(512) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(512) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.0259    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(5)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
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
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.0258    
#config: ReductionConfig: {'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1024) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(256) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(256) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.0246    
#config: ReductionConfig: {'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1152) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(288) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(288) + ax1_1)
                        T.where(ax1_0 * T.int64(288) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.0242    
#config: ReductionConfig: {'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1536) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(384) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(384) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(384) + ax1_1)
                        T.where(ax1_0 * T.int64(384) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.0255    
#config: ReductionConfig: {'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(640) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(160) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.0291    
#config: ReductionConfig: {'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1280) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(320) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.0293    
#config: ReductionConfig: {'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1536) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(384) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(384) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(384) + ax1_1)
                        T.where(ax1_0 * T.int64(384) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.0297    
#config: ReductionConfig: {'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(640) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(160) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.0294    
#config: ReductionConfig: {'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1408) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(352) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(352) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(352) + ax1_1)
                        T.where(ax1_0 * T.int64(352) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.0300    
#config: ReductionConfig: {'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(768) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(192) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(192) + ax1_1)
                        T.where(ax1_0 * T.int64(192) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.0301    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(5)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
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
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.0307    
#config: ReductionConfig: {'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(896) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(224) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(224) + ax1_1)
                        T.where(ax1_0 * T.int64(224) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.0305    
#config: ReductionConfig: {'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1024) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(256) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(256) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.0289    
#config: ReductionConfig: {'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(2048) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(512) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(512) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.0297    
#config: ReductionConfig: {'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1152) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(288) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(288) + ax1_1)
                        T.where(ax1_0 * T.int64(288) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.0343    
#config: ReductionConfig: {'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1536) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(384) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(384) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(384) + ax1_1)
                        T.where(ax1_0 * T.int64(384) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.0344    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(5)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
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
#config: ReductionConfig: {'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1280) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(320) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.0345    
#config: ReductionConfig: {'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1408) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(352) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(352) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(352) + ax1_1)
                        T.where(ax1_0 * T.int64(352) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.0335    
#config: ReductionConfig: {'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(2048) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(512) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(512) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.0340    
#config: ReductionConfig: {'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(640) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(160) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.0343    
#config: ReductionConfig: {'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(768) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(192) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(192) + ax1_1)
                        T.where(ax1_0 * T.int64(192) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.0347    
#config: ReductionConfig: {'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1152) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(288) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(288) + ax1_1)
                        T.where(ax1_0 * T.int64(288) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.0348    
#config: ReductionConfig: {'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1024) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(256) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(256) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.0350    
#config: ReductionConfig: {'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(896) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(224) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(224) + ax1_1)
                        T.where(ax1_0 * T.int64(224) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.0384    
#config: ReductionConfig: {'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1280) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(320) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.0395    
#config: ReductionConfig: {'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1024) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(256) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(256) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.0390    
#config: ReductionConfig: {'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1536) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(384) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(384) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(384) + ax1_1)
                        T.where(ax1_0 * T.int64(384) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.0397    
#config: ReductionConfig: {'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(896) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(224) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(224) + ax1_1)
                        T.where(ax1_0 * T.int64(224) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.0386    
#config: ReductionConfig: {'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(640) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(160) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.0393    
#config: ReductionConfig: {'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1152) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(288) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(288) + ax1_1)
                        T.where(ax1_0 * T.int64(288) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.0390    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(5)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
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
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.0390    
#config: ReductionConfig: {'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1408) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(352) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(352) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(352) + ax1_1)
                        T.where(ax1_0 * T.int64(352) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.0391    
#config: ReductionConfig: {'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(768) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(192) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(192) + ax1_1)
                        T.where(ax1_0 * T.int64(192) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.0382    
#config: ReductionConfig: {'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(2048) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(512) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(512) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.0433    
#config: ReductionConfig: {'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(2048) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(512) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(512) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.0437    
#config: ReductionConfig: {'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1408) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(352) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(352) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(352) + ax1_1)
                        T.where(ax1_0 * T.int64(352) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.0437    
#config: ReductionConfig: {'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1536) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(384) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(384) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(384) + ax1_1)
                        T.where(ax1_0 * T.int64(384) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.0435    
#config: ReductionConfig: {'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(640) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(160) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.0440    
#config: ReductionConfig: {'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1152) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(288) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(288) + ax1_1)
                        T.where(ax1_0 * T.int64(288) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.0442    
#config: ReductionConfig: {'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(896) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(224) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(224) + ax1_1)
                        T.where(ax1_0 * T.int64(224) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.0438    
#config: ReductionConfig: {'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1024) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(256) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(256) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.0437    
#config: ReductionConfig: {'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(768) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(192) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(192) + ax1_1)
                        T.where(ax1_0 * T.int64(192) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.0433    
#config: ReductionConfig: {'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1280) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(320) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.0438    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(5)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
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
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.0482    
#config: ReductionConfig: {'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1408) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(352) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(352) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(352) + ax1_1)
                        T.where(ax1_0 * T.int64(352) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.0488    
#config: ReductionConfig: {'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1152) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(288) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(288) + ax1_1)
                        T.where(ax1_0 * T.int64(288) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.0482    
#config: ReductionConfig: {'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(640) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(160) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.0483    
#config: ReductionConfig: {'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1536) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(384) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(384) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(384) + ax1_1)
                        T.where(ax1_0 * T.int64(384) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.0489    
#config: ReductionConfig: {'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(896) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(224) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(224) + ax1_1)
                        T.where(ax1_0 * T.int64(224) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.0486    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(5)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
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
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.0479    
#config: ReductionConfig: {'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1280) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(320) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.0487    
#config: ReductionConfig: {'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1024) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(256) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(256) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.0477    
#config: ReductionConfig: {'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(2048) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(512) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(512) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.0485    
#config: ReductionConfig: {'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(768) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(192) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(192) + ax1_1)
                        T.where(ax1_0 * T.int64(192) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.0528    
#config: ReductionConfig: {'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1408) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(352) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(352) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(352) + ax1_1)
                        T.where(ax1_0 * T.int64(352) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.0534    
#config: ReductionConfig: {'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1152) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(288) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(288) + ax1_1)
                        T.where(ax1_0 * T.int64(288) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.0536    
#config: ReductionConfig: {'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(896) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(224) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(224) + ax1_1)
                        T.where(ax1_0 * T.int64(224) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.0525    
#config: ReductionConfig: {'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(2048) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(512) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(512) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.0525    
#config: ReductionConfig: {'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1280) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(320) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.0527    
#config: ReductionConfig: {'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(640) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(160) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.0532    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(5)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
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
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.0529    
#config: ReductionConfig: {'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1536) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(384) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(384) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(384) + ax1_1)
                        T.where(ax1_0 * T.int64(384) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.0529    
#config: ReductionConfig: {'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1024) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(256) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(256) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.0528    
#config: ReductionConfig: {'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(768) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(192) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(192) + ax1_1)
                        T.where(ax1_0 * T.int64(192) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.0575    
#config: ReductionConfig: {'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1024) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(256) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(256) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.0582    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(5)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
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
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.0579    
#config: ReductionConfig: {'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(768) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(192) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(192) + ax1_1)
                        T.where(ax1_0 * T.int64(192) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.0576    
#config: ReductionConfig: {'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1536) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(384) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(384) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(384) + ax1_1)
                        T.where(ax1_0 * T.int64(384) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.0570    
#config: ReductionConfig: {'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(2048) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(512) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(512) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.0582    
#config: ReductionConfig: {'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(896) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(224) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(224) + ax1_1)
                        T.where(ax1_0 * T.int64(224) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.0574    
#config: ReductionConfig: {'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(640) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(160) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.0577    
#config: ReductionConfig: {'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1408) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(352) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(352) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(352) + ax1_1)
                        T.where(ax1_0 * T.int64(352) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.0581    
#config: ReductionConfig: {'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1152) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(288) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(288) + ax1_1)
                        T.where(ax1_0 * T.int64(288) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.0573    
#config: ReductionConfig: {'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1280) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(320) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.0621    
#config: ReductionConfig: {'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1536) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(384) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(384) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(384) + ax1_1)
                        T.where(ax1_0 * T.int64(384) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.0627    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(5)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
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
#config: ReductionConfig: {'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1408) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(352) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(352) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(352) + ax1_1)
                        T.where(ax1_0 * T.int64(352) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.0620    
#config: ReductionConfig: {'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(640) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(160) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.0626    
#config: ReductionConfig: {'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1152) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(288) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(288) + ax1_1)
                        T.where(ax1_0 * T.int64(288) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.0623    
#config: ReductionConfig: {'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1024) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(256) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(256) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.0615    
#config: ReductionConfig: {'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(2048) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(512) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(512) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.0620    
#config: ReductionConfig: {'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1280) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(320) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.0621    
#config: ReductionConfig: {'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(768) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(192) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(192) + ax1_1)
                        T.where(ax1_0 * T.int64(192) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.0625    
#config: ReductionConfig: {'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(896) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(224) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(224) + ax1_1)
                        T.where(ax1_0 * T.int64(224) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.0664    
#config: ReductionConfig: {'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(2048) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(512) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(512) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.0669    
#config: ReductionConfig: {'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1408) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(352) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(352) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(352) + ax1_1)
                        T.where(ax1_0 * T.int64(352) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.0680    
#config: ReductionConfig: {'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1024) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(256) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(256) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.0665    
#config: ReductionConfig: {'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1280) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(320) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.0668    
#config: ReductionConfig: {'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(768) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(192) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(192) + ax1_1)
                        T.where(ax1_0 * T.int64(192) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.0672    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(5)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
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
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.0666    
#config: ReductionConfig: {'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(640) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(160) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.0673    
#config: ReductionConfig: {'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1152) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(288) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(288) + ax1_1)
                        T.where(ax1_0 * T.int64(288) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.0668    
#config: ReductionConfig: {'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1536) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(384) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(384) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(384) + ax1_1)
                        T.where(ax1_0 * T.int64(384) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.0679    
#config: ReductionConfig: {'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(896) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(224) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(224) + ax1_1)
                        T.where(ax1_0 * T.int64(224) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.0714    
#config: ReductionConfig: {'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1536) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(384) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(384) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(384) + ax1_1)
                        T.where(ax1_0 * T.int64(384) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.0716    
#config: ReductionConfig: {'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(640) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(160) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.0713    
#config: ReductionConfig: {'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1280) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(320) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.0716    
#config: ReductionConfig: {'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1024) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(256) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(256) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.0709    
#config: ReductionConfig: {'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(2048) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(512) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(512) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.0720    
#config: ReductionConfig: {'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1152) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(288) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(288) + ax1_1)
                        T.where(ax1_0 * T.int64(288) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.0715    
#config: ReductionConfig: {'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1408) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(352) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(352) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(352) + ax1_1)
                        T.where(ax1_0 * T.int64(352) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.0714    
#config: ReductionConfig: {'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(768) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(192) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(192) + ax1_1)
                        T.where(ax1_0 * T.int64(192) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.0717    
#config: ReductionConfig: {'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(896) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(224) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(224) + ax1_1)
                        T.where(ax1_0 * T.int64(224) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.0718    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(5)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
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
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.0761    
#config: ReductionConfig: {'len_tx': 128, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(5)):
                for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(512) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
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
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.0775    
#config: ReductionConfig: {'len_tx': 256, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1024) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(256) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(256) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.0763    
#config: ReductionConfig: {'len_tx': 192, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(768) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(192) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(192) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(192) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(192), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(14), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(192) + ax1_1)
                        T.where(ax1_0 * T.int64(192) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.0758    
#config: ReductionConfig: {'len_tx': 160, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(640) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(160) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(160) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.0757    
#config: ReductionConfig: {'len_tx': 512, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(2048) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(512) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(512) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(5), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(512) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.0760    
#config: ReductionConfig: {'len_tx': 384, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1536) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(384) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(384) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(384) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(384), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(7), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(384) + ax1_1)
                        T.where(ax1_0 * T.int64(384) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.0761    
#config: ReductionConfig: {'len_tx': 352, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1408) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(352) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(352) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(352) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(352), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(352) + ax1_1)
                        T.where(ax1_0 * T.int64(352) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.0758    
#config: ReductionConfig: {'len_tx': 320, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1280) + ax2_1 * T.int64(4) + ax2_2)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(320) + ax1_fused_1)
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(320), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(8), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(320) + ax1_1)
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.0767    
#config: ReductionConfig: {'len_tx': 288, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(1152) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(288) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(288) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(288) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(288), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(9), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(288) + ax1_1)
                        T.where(ax1_0 * T.int64(288) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
#name: fused_layer_norm_cast1
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.0766    
#config: ReductionConfig: {'len_tx': 224, 'unroll_depth': 256, 'vector_size': 4, 'temp_storage': 'shared.dyn'}
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
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                for ax2_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax2_2 in T.vectorized(T.int64(4)):
                        with T.block("lv6_shared.dyn"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(n, ax0_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), ax2_0 * T.int64(896) + ax2_1 * T.int64(4) + ax2_2)
                            T.where((ax2_0 * T.int64(224) + ax2_1) * T.int64(4) + ax2_2 < T.int64(2560))
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared_dyn[v0, v1, v2])
                            lv6_shared_dyn[v0, v1, v2] = lv6[v0, v1, v2]
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("A_red_temp"):
                            v0 = T.axis.spatial(n, ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(224) + ax1_fused_1)
                            T.where(ax1_fused_0 * T.int64(224) + ax1_fused_1 < T.int64(2560))
                            T.reads(lv6_shared_dyn[T.int64(0), v0, v1])
                            T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6_shared_dyn[T.int64(0), v0, v1] * lv6_shared_dyn[T.int64(0), v0, v1]
                            A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                            A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(224), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(12), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("compute"):
                        v0 = T.axis.spatial(n, ax0_fused)
                        v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(224) + ax1_1)
                        T.where(ax1_0 * T.int64(224) + ax1_1 < T.int64(2560))
                        T.reads(lv6_shared_dyn[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], param_1[v1], param_2[v1])
                        T.writes(compute_intermediate[T.int64(0), v0, v1])
                        compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6_shared_dyn[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * param_1[v1] + param_2[v1])
 
