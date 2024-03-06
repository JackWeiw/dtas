#name: cast
#range: (<n: Range(1, 128)>,)  latency(ms):    0.0023    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 320, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(320), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(327679)) // T.int64(327680), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(327679)) // T.int64(327680) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(327679)) // T.int64(327680) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(327679)) // T.int64(327680)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1, 128)>,)  latency(ms):    0.0023    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 640, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(640), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(327679)) // T.int64(327680), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(327679)) // T.int64(327680) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(327679)) // T.int64(327680) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(327679)) // T.int64(327680)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1, 128)>,)  latency(ms):    0.0023    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 160, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(160), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(327679)) // T.int64(327680), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(327679)) // T.int64(327680) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(327679)) // T.int64(327680) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(327679)) // T.int64(327680)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(129, 256)>,)  latency(ms):    0.0032    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 1280, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1280), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(655359)) // T.int64(655360), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655359)) // T.int64(655360) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655359)) // T.int64(655360) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655359)) // T.int64(655360)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(129, 256)>,)  latency(ms):    0.0034    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 320, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(320), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(655359)) // T.int64(655360), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655359)) // T.int64(655360) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655359)) // T.int64(655360) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655359)) // T.int64(655360)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(129, 256)>,)  latency(ms):    0.0027    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 640, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(640), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(655359)) // T.int64(655360), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655359)) // T.int64(655360) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655359)) // T.int64(655360) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655359)) // T.int64(655360)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(257, 384)>,)  latency(ms):    0.0045    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 1920, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1920), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(983039)) // T.int64(983040), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(983039)) // T.int64(983040) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(983039)) // T.int64(983040) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(983039)) // T.int64(983040)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(257, 384)>,)  latency(ms):    0.0035    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 960, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(960), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(983039)) // T.int64(983040), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(983039)) // T.int64(983040) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(983039)) // T.int64(983040) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(983039)) // T.int64(983040)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(257, 384)>,)  latency(ms):    0.0052    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 480, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(480), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(983039)) // T.int64(983040), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(983039)) // T.int64(983040) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(983039)) // T.int64(983040) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(983039)) // T.int64(983040)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(385, 512)>,)  latency(ms):    0.0068    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 1280, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1280), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(385, 512)>,)  latency(ms):    0.0079    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 2560, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(2560), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(385, 512)>,)  latency(ms):    0.0084    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 640, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(640), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(513, 640)>,)  latency(ms):    0.0128    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 3200, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(3200), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(1638399)) // T.int64(1638400), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1638399)) // T.int64(1638400) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1638399)) // T.int64(1638400) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1638399)) // T.int64(1638400)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(513, 640)>,)  latency(ms):    0.0124    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 1600, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1600), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(1638399)) // T.int64(1638400), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1638399)) // T.int64(1638400) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1638399)) // T.int64(1638400) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1638399)) // T.int64(1638400)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(513, 640)>,)  latency(ms):    0.0125    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 800, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(800), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(1638399)) // T.int64(1638400), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1638399)) // T.int64(1638400) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1638399)) // T.int64(1638400) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1638399)) // T.int64(1638400)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(641, 768)>,)  latency(ms):    0.0152    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 3840, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(3840), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(641, 768)>,)  latency(ms):    0.0147    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 960, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(960), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(641, 768)>,)  latency(ms):    0.0149    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 1920, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1920), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(769, 896)>,)  latency(ms):    0.0172    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 1120, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1120), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(2293759)) // T.int64(2293760), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2293759)) // T.int64(2293760) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2293759)) // T.int64(2293760) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2293759)) // T.int64(2293760)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(769, 896)>,)  latency(ms):    0.0176    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 4480, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(4480), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(2293759)) // T.int64(2293760), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2293759)) // T.int64(2293760) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2293759)) // T.int64(2293760) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2293759)) // T.int64(2293760)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(769, 896)>,)  latency(ms):    0.0173    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 2240, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(2240), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(2293759)) // T.int64(2293760), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2293759)) // T.int64(2293760) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2293759)) // T.int64(2293760) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2293759)) // T.int64(2293760)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(897, 1024)>,)  latency(ms):    0.0198    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 5120, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(5120), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(897, 1024)>,)  latency(ms):    0.0196    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 1280, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1280), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(897, 1024)>,)  latency(ms):    0.0195    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 2560, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(2560), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1025, 1152)>,)  latency(ms):    0.0222    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 5760, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(5760), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(2949119)) // T.int64(2949120), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2949119)) // T.int64(2949120) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2949119)) // T.int64(2949120) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2949119)) // T.int64(2949120)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1025, 1152)>,)  latency(ms):    0.0220    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 1440, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1440), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(2949119)) // T.int64(2949120), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2949119)) // T.int64(2949120) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2949119)) // T.int64(2949120) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2949119)) // T.int64(2949120)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1025, 1152)>,)  latency(ms):    0.0220    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 2880, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(2880), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(2949119)) // T.int64(2949120), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2949119)) // T.int64(2949120) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2949119)) // T.int64(2949120) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2949119)) // T.int64(2949120)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1153, 1280)>,)  latency(ms):    0.0243    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 3200, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(3200), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1153, 1280)>,)  latency(ms):    0.0243    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 1600, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1600), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1153, 1280)>,)  latency(ms):    0.0246    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 6400, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(6400), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1281, 1408)>,)  latency(ms):    0.0267    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 1760, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1760), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(3604479)) // T.int64(3604480), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3604479)) // T.int64(3604480) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3604479)) // T.int64(3604480) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3604479)) // T.int64(3604480)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1281, 1408)>,)  latency(ms):    0.0267    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 3520, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(3520), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(3604479)) // T.int64(3604480), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3604479)) // T.int64(3604480) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3604479)) // T.int64(3604480) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3604479)) // T.int64(3604480)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1281, 1408)>,)  latency(ms):    0.0270    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 7040, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(7040), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(3604479)) // T.int64(3604480), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3604479)) // T.int64(3604480) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3604479)) // T.int64(3604480) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3604479)) // T.int64(3604480)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1409, 1536)>,)  latency(ms):    0.0290    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 1920, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1920), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1409, 1536)>,)  latency(ms):    0.0290    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 3840, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(3840), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1409, 1536)>,)  latency(ms):    0.0293    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 7680, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(7680), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1537, 1664)>,)  latency(ms):    0.0313    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 2080, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(2080), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4259839)) // T.int64(4259840), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4259839)) // T.int64(4259840) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4259839)) // T.int64(4259840) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4259839)) // T.int64(4259840)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1537, 1664)>,)  latency(ms):    0.0316    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 8320, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(8320), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4259839)) // T.int64(4259840), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4259839)) // T.int64(4259840) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4259839)) // T.int64(4259840) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4259839)) // T.int64(4259840)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1537, 1664)>,)  latency(ms):    0.0313    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 4160, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(4160), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4259839)) // T.int64(4259840), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4259839)) // T.int64(4259840) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4259839)) // T.int64(4259840) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4259839)) // T.int64(4259840)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1665, 1792)>,)  latency(ms):    0.0337    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 2240, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(2240), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1665, 1792)>,)  latency(ms):    0.0337    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 4480, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(4480), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1665, 1792)>,)  latency(ms):    0.0340    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 8960, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(8960), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1793, 1920)>,)  latency(ms):    0.0360    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 2400, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(2400), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4915199)) // T.int64(4915200), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4915199)) // T.int64(4915200) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4915199)) // T.int64(4915200) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4915199)) // T.int64(4915200)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1793, 1920)>,)  latency(ms):    0.0360    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 4800, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(4800), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4915199)) // T.int64(4915200), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4915199)) // T.int64(4915200) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4915199)) // T.int64(4915200) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4915199)) // T.int64(4915200)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1793, 1920)>,)  latency(ms):    0.0364    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 9600, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(9600), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4915199)) // T.int64(4915200), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4915199)) // T.int64(4915200) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4915199)) // T.int64(4915200) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4915199)) // T.int64(4915200)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1921, 2048)>,)  latency(ms):    0.0382    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 5120, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(5120), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1921, 2048)>,)  latency(ms):    0.0386    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 10240, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(10240), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(1921, 2048)>,)  latency(ms):    0.0383    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 2560, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(2560), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2049, 2176)>,)  latency(ms):    0.0406    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 5440, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(5440), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(5570559)) // T.int64(5570560), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5570559)) // T.int64(5570560) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5570559)) // T.int64(5570560) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5570559)) // T.int64(5570560)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2049, 2176)>,)  latency(ms):    0.0410    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 10880, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(10880), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(5570559)) // T.int64(5570560), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5570559)) // T.int64(5570560) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5570559)) // T.int64(5570560) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5570559)) // T.int64(5570560)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2049, 2176)>,)  latency(ms):    0.0407    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 2720, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(2720), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(5570559)) // T.int64(5570560), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5570559)) // T.int64(5570560) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5570559)) // T.int64(5570560) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5570559)) // T.int64(5570560)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2177, 2304)>,)  latency(ms):    0.0430    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 5760, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(5760), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2177, 2304)>,)  latency(ms):    0.0431    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 2880, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(2880), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2177, 2304)>,)  latency(ms):    0.0434    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 11520, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(11520), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2305, 2432)>,)  latency(ms):    0.0453    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 6080, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(6080), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(6225919)) // T.int64(6225920), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6225919)) // T.int64(6225920) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6225919)) // T.int64(6225920) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6225919)) // T.int64(6225920)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2305, 2432)>,)  latency(ms):    0.0454    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 3040, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(3040), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(6225919)) // T.int64(6225920), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6225919)) // T.int64(6225920) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6225919)) // T.int64(6225920) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6225919)) // T.int64(6225920)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2305, 2432)>,)  latency(ms):    0.0457    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 12160, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(12160), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(6225919)) // T.int64(6225920), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6225919)) // T.int64(6225920) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6225919)) // T.int64(6225920) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6225919)) // T.int64(6225920)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2433, 2560)>,)  latency(ms):    0.0477    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 3200, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(3200), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2433, 2560)>,)  latency(ms):    0.0480    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 12800, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2433, 2560)>,)  latency(ms):    0.0477    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 6400, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(6400), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2561, 2688)>,)  latency(ms):    0.0504    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 13440, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(13440), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(6881279)) // T.int64(6881280), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6881279)) // T.int64(6881280) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6881279)) // T.int64(6881280) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6881279)) // T.int64(6881280)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2561, 2688)>,)  latency(ms):    0.0500    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 3360, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(3360), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(6881279)) // T.int64(6881280), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6881279)) // T.int64(6881280) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6881279)) // T.int64(6881280) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6881279)) // T.int64(6881280)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2561, 2688)>,)  latency(ms):    0.0500    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 6720, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(6720), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(6881279)) // T.int64(6881280), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6881279)) // T.int64(6881280) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6881279)) // T.int64(6881280) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6881279)) // T.int64(6881280)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2689, 2816)>,)  latency(ms):    0.0524    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 3520, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(3520), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2689, 2816)>,)  latency(ms):    0.0527    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 14080, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(14080), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2689, 2816)>,)  latency(ms):    0.0524    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 7040, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(7040), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2817, 2944)>,)  latency(ms):    0.0547    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 3680, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(3680), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(7536639)) // T.int64(7536640), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7536639)) // T.int64(7536640) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7536639)) // T.int64(7536640) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7536639)) // T.int64(7536640)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2817, 2944)>,)  latency(ms):    0.0551    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 14720, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(14720), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(7536639)) // T.int64(7536640), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7536639)) // T.int64(7536640) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7536639)) // T.int64(7536640) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7536639)) // T.int64(7536640)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2817, 2944)>,)  latency(ms):    0.0547    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 7360, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(7360), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(7536639)) // T.int64(7536640), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7536639)) // T.int64(7536640) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7536639)) // T.int64(7536640) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7536639)) // T.int64(7536640)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2945, 3072)>,)  latency(ms):    0.0574    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 15360, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(15360), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2945, 3072)>,)  latency(ms):    0.0570    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 3840, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(3840), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(2945, 3072)>,)  latency(ms):    0.0570    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 7680, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(7680), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3073, 3200)>,)  latency(ms):    0.0644    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 15744, 'unroll_depth': 256}
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
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3073, 3200)>,)  latency(ms):    0.0593    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 8000, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(8000), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(8191999)) // T.int64(8192000), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8191999)) // T.int64(8192000) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8191999)) // T.int64(8192000) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8191999)) // T.int64(8192000)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3073, 3200)>,)  latency(ms):    0.0593    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 4000, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(4000), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(8191999)) // T.int64(8192000), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8191999)) // T.int64(8192000) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8191999)) // T.int64(8192000) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8191999)) // T.int64(8192000)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3201, 3328)>,)  latency(ms):    0.0616    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 4160, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(4160), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(8519679)) // T.int64(8519680), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8519679)) // T.int64(8519680) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8519679)) // T.int64(8519680) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8519679)) // T.int64(8519680)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3201, 3328)>,)  latency(ms):    0.0735    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 15744, 'unroll_depth': 256}
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
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3201, 3328)>,)  latency(ms):    0.0617    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 8320, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(8320), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(8519679)) // T.int64(8519680), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8519679)) // T.int64(8519680) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8519679)) // T.int64(8519680) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8519679)) // T.int64(8519680)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3329, 3456)>,)  latency(ms):    0.0752    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 15744, 'unroll_depth': 256}
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
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3329, 3456)>,)  latency(ms):    0.0640    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 4320, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(4320), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(8847359)) // T.int64(8847360), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8847359)) // T.int64(8847360) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8847359)) // T.int64(8847360) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8847359)) // T.int64(8847360)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3329, 3456)>,)  latency(ms):    0.0640    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 8640, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(8640), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(8847359)) // T.int64(8847360), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8847359)) // T.int64(8847360) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8847359)) // T.int64(8847360) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8847359)) // T.int64(8847360)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3457, 3584)>,)  latency(ms):    0.0772    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 15744, 'unroll_depth': 256}
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
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3457, 3584)>,)  latency(ms):    0.0663    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 8960, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(8960), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(9175039)) // T.int64(9175040), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9175039)) // T.int64(9175040) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9175039)) // T.int64(9175040) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9175039)) // T.int64(9175040)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3457, 3584)>,)  latency(ms):    0.0663    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 4480, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(4480), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(9175039)) // T.int64(9175040), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9175039)) // T.int64(9175040) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9175039)) // T.int64(9175040) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9175039)) // T.int64(9175040)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3585, 3712)>,)  latency(ms):    0.0686    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 4640, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(4640), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(9502719)) // T.int64(9502720), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9502719)) // T.int64(9502720) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9502719)) // T.int64(9502720) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9502719)) // T.int64(9502720)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3585, 3712)>,)  latency(ms):    0.0787    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 15744, 'unroll_depth': 256}
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
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3585, 3712)>,)  latency(ms):    0.0687    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 9280, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(9280), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(9502719)) // T.int64(9502720), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9502719)) // T.int64(9502720) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9502719)) // T.int64(9502720) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9502719)) // T.int64(9502720)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3713, 3840)>,)  latency(ms):    0.0806    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 15744, 'unroll_depth': 256}
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
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3713, 3840)>,)  latency(ms):    0.0710    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 9600, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(9600), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(9830399)) // T.int64(9830400), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9830399)) // T.int64(9830400) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9830399)) // T.int64(9830400) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9830399)) // T.int64(9830400)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3713, 3840)>,)  latency(ms):    0.0709    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 4800, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(4800), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(9830399)) // T.int64(9830400), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9830399)) // T.int64(9830400) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9830399)) // T.int64(9830400) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9830399)) // T.int64(9830400)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3841, 3968)>,)  latency(ms):    0.0823    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 15744, 'unroll_depth': 256}
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
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3841, 3968)>,)  latency(ms):    0.0733    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 9920, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(9920), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(10158079)) // T.int64(10158080), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(10158079)) // T.int64(10158080) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(10158079)) // T.int64(10158080) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(10158079)) // T.int64(10158080)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3841, 3968)>,)  latency(ms):    0.0732    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 4960, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(4960), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(10158079)) // T.int64(10158080), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(10158079)) // T.int64(10158080) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(10158079)) // T.int64(10158080) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(10158079)) // T.int64(10158080)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3969, 4096)>,)  latency(ms):    0.0755    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 8, 'grid_size': 5120, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(5120), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(10485759)) // T.int64(10485760), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(10485759)) // T.int64(10485760) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(10485759)) // T.int64(10485760) * T.int64(2048)) + ax0_ax1_fused_1 * T.int64(2048) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(10485759)) // T.int64(10485760)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(8) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3969, 4096)>,)  latency(ms):    0.0755    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 10240, 'unroll_depth': 256}
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
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(10240), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(10485759)) // T.int64(10485760), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(10485759)) // T.int64(10485760) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(10485759)) // T.int64(10485760) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(10485759)) // T.int64(10485760)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
#name: cast
#range: (<n: Range(3969, 4096)>,)  latency(ms):    0.0840    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 15744, 'unroll_depth': 256}
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
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1])
                            T.writes(compute[T.int64(0), v0, v1])
                            compute[T.int64(0), v0, v1] = T.Cast("float32", A[T.int64(0), v0, v1])
 
