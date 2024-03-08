#name: add
#range: (<n: Range(1, 256)>,)  latency(ms):    0.0042    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 640, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(640), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(655359)) // T.int64(655360), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655359)) // T.int64(655360) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655359)) // T.int64(655360) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655359)) // T.int64(655360)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(1, 256)>,)  latency(ms):    0.0056    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 1, 'grid_size': 2560, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(2560), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(655359)) // T.int64(655360), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(1)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655359)) // T.int64(655360) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655359)) // T.int64(655360) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) % T.int64(2560))
                            T.where((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655359)) // T.int64(655360)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(1, 256)>,)  latency(ms):    0.0046    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 1280, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1280), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(655359)) // T.int64(655360), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655359)) // T.int64(655360) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655359)) // T.int64(655360) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655359)) // T.int64(655360)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(1, 256)>,)  latency(ms):    0.0047    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 3, 'grid_size': 854, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(854), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(655871)) // T.int64(655872), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(3)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655871)) // T.int64(655872) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655871)) // T.int64(655872) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(655871)) // T.int64(655872)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(3) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(257, 512)>,)  latency(ms):    0.0155    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 2560, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(2560), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(257, 512)>,)  latency(ms):    0.0152    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 1280, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1280), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(257, 512)>,)  latency(ms):    0.0162    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 1, 'grid_size': 5120, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(5120), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(1)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) % T.int64(2560))
                            T.where((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310719)) // T.int64(1310720)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(257, 512)>,)  latency(ms):    0.0155    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 3, 'grid_size': 1707, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1707), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(1310975)) // T.int64(1310976), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(3)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310975)) // T.int64(1310976) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310975)) // T.int64(1310976) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1310975)) // T.int64(1310976)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(3) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(513, 768)>,)  latency(ms):    0.0269    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 1, 'grid_size': 7680, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(7680), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(1)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) % T.int64(2560))
                            T.where((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(513, 768)>,)  latency(ms):    0.0253    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 1920, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1920), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(513, 768)>,)  latency(ms):    0.0255    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 3, 'grid_size': 2560, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(2560), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(3)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(3) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(513, 768)>,)  latency(ms):    0.0261    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 3840, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(3840), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(1966079)) // T.int64(1966080)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.0357    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 5120, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(5120), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.0346    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 2560, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(2560), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.0360    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 1, 'grid_size': 10240, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(10240), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(1)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) % T.int64(2560))
                            T.where((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621439)) // T.int64(2621440)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(769, 1024)>,)  latency(ms):    0.0349    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 3, 'grid_size': 3414, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(3414), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(2621951)) // T.int64(2621952), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(3)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621951)) // T.int64(2621952) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621951)) // T.int64(2621952) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(2621951)) // T.int64(2621952)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(3) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.0455    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 6400, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(6400), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.0456    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 1, 'grid_size': 12800, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(1)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) % T.int64(2560))
                            T.where((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.0440    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 3200, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(3200), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3276799)) // T.int64(3276800)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(1025, 1280)>,)  latency(ms):    0.0442    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 3, 'grid_size': 4267, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(4267), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(3277055)) // T.int64(3277056), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(3)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3277055)) // T.int64(3277056) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3277055)) // T.int64(3277056) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3277055)) // T.int64(3277056)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(3) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.0548    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 7680, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(7680), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.0533    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 3840, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(3840), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.0549    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 1, 'grid_size': 15360, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(15360), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(1)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) % T.int64(2560))
                            T.where((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(1281, 1536)>,)  latency(ms):    0.0534    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 3, 'grid_size': 5120, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(5120), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(3)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(3932159)) // T.int64(3932160)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(3) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.0627    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 3, 'grid_size': 5974, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(5974), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4588031)) // T.int64(4588032), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(3)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4588031)) // T.int64(4588032) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4588031)) // T.int64(4588032) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4588031)) // T.int64(4588032)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(3) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.0626    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 4480, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(4480), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.0640    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 8960, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(8960), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4587519)) // T.int64(4587520)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(1537, 1792)>,)  latency(ms):    0.0736    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 1, 'grid_size': 15744, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(15744), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(1)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) % T.int64(2560))
                            T.where((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.0731    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 10240, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(10240), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.0829    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 1, 'grid_size': 15744, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(15744), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(1)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) % T.int64(2560))
                            T.where((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.0720    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 3, 'grid_size': 6827, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(6827), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(5243135)) // T.int64(5243136), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(3)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5243135)) // T.int64(5243136) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5243135)) // T.int64(5243136) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5243135)) // T.int64(5243136)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(3) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(1793, 2048)>,)  latency(ms):    0.0717    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 5120, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(5120), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5242879)) // T.int64(5242880)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.0824    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 11520, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(11520), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.0812    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 3, 'grid_size': 7680, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(7680), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(3)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(3) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.0811    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 5760, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(5760), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(5898239)) // T.int64(5898240)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(2049, 2304)>,)  latency(ms):    0.0896    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 1, 'grid_size': 15744, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(15744), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(1)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) % T.int64(2560))
                            T.where((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.0906    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 3, 'grid_size': 8534, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(8534), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(6554111)) // T.int64(6554112), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(3)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6554111)) // T.int64(6554112) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6554111)) // T.int64(6554112) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6554111)) // T.int64(6554112)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(3) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.0960    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 1, 'grid_size': 15744, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(15744), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(1)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) % T.int64(2560))
                            T.where((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.0904    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 6400, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(6400), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(2305, 2560)>,)  latency(ms):    0.0915    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 12800, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(6553599)) // T.int64(6553600)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.1024    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 1, 'grid_size': 15744, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(15744), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(1)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) % T.int64(2560))
                            T.where((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.1009    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 14080, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(14080), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.0999    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 3, 'grid_size': 9387, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(9387), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(7209215)) // T.int64(7209216), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(3)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7209215)) // T.int64(7209216) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7209215)) // T.int64(7209216) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7209215)) // T.int64(7209216)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(3) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(2561, 2816)>,)  latency(ms):    0.0997    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 7040, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(7040), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7208959)) // T.int64(7208960)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.1101    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 1, 'grid_size': 15744, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(15744), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(1)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) % T.int64(2560))
                            T.where((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.1102    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 15360, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(15360), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.1090    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 3, 'grid_size': 10240, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(10240), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(3)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(3) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(2817, 3072)>,)  latency(ms):    0.1089    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 7680, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(7680), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(7864319)) // T.int64(7864320)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.1250    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 1, 'grid_size': 15744, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(15744), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(1)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) % T.int64(2560))
                            T.where((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.1183    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 3, 'grid_size': 11094, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(11094), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(8520191)) // T.int64(8520192), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(3)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8520191)) // T.int64(8520192) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8520191)) // T.int64(8520192) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8520191)) // T.int64(8520192)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(3) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.1182    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 8320, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(8320), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(8519679)) // T.int64(8519680), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8519679)) // T.int64(8519680) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8519679)) // T.int64(8519680) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8519679)) // T.int64(8519680)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(3073, 3328)>,)  latency(ms):    0.1277    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 15744, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(15744), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.1274    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 8960, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(8960), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(9175039)) // T.int64(9175040), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9175039)) // T.int64(9175040) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9175039)) // T.int64(9175040) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9175039)) // T.int64(9175040)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.1353    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 1, 'grid_size': 15744, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(15744), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(1)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) % T.int64(2560))
                            T.where((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.1395    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 15744, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(15744), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(3329, 3584)>,)  latency(ms):    0.1277    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 3, 'grid_size': 11947, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(11947), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(9175295)) // T.int64(9175296), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(3)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9175295)) // T.int64(9175296) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9175295)) // T.int64(9175296) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9175295)) // T.int64(9175296)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(3) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.1422    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 1, 'grid_size': 15744, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(15744), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(1)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) % T.int64(2560))
                            T.where((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.1367    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 9600, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(9600), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(9830399)) // T.int64(9830400), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9830399)) // T.int64(9830400) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9830399)) // T.int64(9830400) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9830399)) // T.int64(9830400)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.1476    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 15744, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(15744), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(3585, 3840)>,)  latency(ms):    0.1368    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 3, 'grid_size': 12800, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(12800), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(9830399)) // T.int64(9830400), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(3)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9830399)) // T.int64(9830400) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9830399)) // T.int64(9830400) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(9830399)) // T.int64(9830400)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(3) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.1494    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 1, 'grid_size': 15744, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(15744), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(1)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464) * T.int64(256)) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3) % T.int64(2560))
                            T.where((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(4030463)) // T.int64(4030464)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2 + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.1458    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 4, 'grid_size': 10240, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(10240), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(10485759)) // T.int64(10485760), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(10485759)) // T.int64(10485760) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(10485759)) // T.int64(10485760) * T.int64(1024)) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(10485759)) // T.int64(10485760)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(4) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.1461    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 3, 'grid_size': 13654, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(13654), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(10486271)) // T.int64(10486272), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(3)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(10486271)) // T.int64(10486272) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(10486271)) // T.int64(10486272) * T.int64(768)) + ax0_ax1_fused_1 * T.int64(768) + ax0_ax1_fused_2 * T.int64(3) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(10486271)) // T.int64(10486272)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(3) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
#name: add
#range: (<n: Range(3841, 4096)>,)  latency(ms):    0.1554    
#config: ElementwiseConfig: {'len_tx': 256, 'vector_size': 2, 'grid_size': 15744, 'unroll_depth': 256}
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        B = T.match_buffer(var_B, (T.int64(1), n, T.int64(2560)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(15744), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.serial((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(n, (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(2560))
                            v1 = T.axis.spatial(T.int64(2560), (ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928) * T.int64(512)) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(2560))
                            T.where(((ax0_ax1_fused_0 * ((n * T.int64(2560) + T.int64(8060927)) // T.int64(8060928)) + ax0_ax1_fused_1) * T.int64(256) + ax0_ax1_fused_2) * T.int64(2) + ax0_ax1_fused_3 < n * T.int64(2560))
                            T.reads(A[T.int64(0), v0, v1], B[T.int64(0), v0, v1])
                            T.writes(T_add[T.int64(0), v0, v1])
                            T_add[T.int64(0), v0, v1] = A[T.int64(0), v0, v1] + B[T.int64(0), v0, v1]
 
