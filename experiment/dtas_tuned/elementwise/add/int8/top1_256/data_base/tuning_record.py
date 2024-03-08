#name: add
#range: (<n: Range(1, 256)>,)  latency(ms):    0.0022    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 16, 'grid_size': 160, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)), "int8")
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)), "int8")
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)), "int8")
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(160), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(655359)) // T.int64(655360), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(16)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655359)) // T.int64(655360) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655359)) // T.int64(655360) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655359)) // T.int64(655360)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(16) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(257, 512)>,)  latency(ms):    0.0026    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 16, 'grid_size': 320, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)), "int8")
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)), "int8")
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)), "int8")
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(320), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(16)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(16) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(513, 768)>,)  latency(ms):    0.0033    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 16, 'grid_size': 480, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)), "int8")
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)), "int8")
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)), "int8")
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(480), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(16)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(16) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.0073    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 16, 'grid_size': 640, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)), "int8")
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)), "int8")
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)), "int8")
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(640), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(16)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(16) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.0114    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 16, 'grid_size': 800, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)), "int8")
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)), "int8")
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)), "int8")
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(800), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(16)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(16) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.0140    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 16, 'grid_size': 960, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)), "int8")
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)), "int8")
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)), "int8")
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(960), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(16)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(16) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.0164    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 16, 'grid_size': 1120, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)), "int8")
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)), "int8")
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)), "int8")
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1120), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(16)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(16) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.0190    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 16, 'grid_size': 1280, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)), "int8")
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)), "int8")
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)), "int8")
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1280), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(16)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(16) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.0217    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 16, 'grid_size': 1440, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)), "int8")
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)), "int8")
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)), "int8")
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1440), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(16)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(16) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.0241    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 16, 'grid_size': 1600, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)), "int8")
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)), "int8")
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)), "int8")
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1600), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(16)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(16) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.0265    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 16, 'grid_size': 1760, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)), "int8")
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)), "int8")
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)), "int8")
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1760), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(16)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(16) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.0288    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 16, 'grid_size': 1920, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)), "int8")
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)), "int8")
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)), "int8")
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1920), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(16)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(16) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.0311    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 16, 'grid_size': 2080, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)), "int8")
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)), "int8")
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)), "int8")
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(2080), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(8519679)) // T.int64(8519680), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(16)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8519679)) // T.int64(8519680) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8519679)) // T.int64(8519680) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8519679)) // T.int64(8519680)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(16) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.0335    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 16, 'grid_size': 2240, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)), "int8")
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)), "int8")
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)), "int8")
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(2240), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(9175039)) // T.int64(9175040), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(16)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9175039)) // T.int64(9175040) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9175039)) // T.int64(9175040) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9175039)) // T.int64(9175040)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(16) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.0359    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 16, 'grid_size': 2400, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)), "int8")
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)), "int8")
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)), "int8")
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(2400), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(9830399)) // T.int64(9830400), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(16)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9830399)) // T.int64(9830400) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9830399)) // T.int64(9830400) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9830399)) // T.int64(9830400)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(16) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.0381    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 16, 'grid_size': 2560, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)), "int8")
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)), "int8")
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)), "int8")
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(2560), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(10485759)) // T.int64(10485760), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(16)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(10485759)) // T.int64(10485760) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(10485759)) // T.int64(10485760) * T.int64(4096)) + ax0_ax1_fused_1 * T.int64(4096) + ax0_ax1_fused_2 * T.int64(16) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(10485759)) // T.int64(10485760)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(16) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
