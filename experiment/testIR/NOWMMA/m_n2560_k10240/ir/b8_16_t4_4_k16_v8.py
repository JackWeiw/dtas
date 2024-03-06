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
        var_NT_matmul_intermediate_reindex_pad_local = T.alloc_buffer((T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(2560)), "int8", scope="local")
        lv9_reindex_pad_shared = T.alloc_buffer((T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(10240)), "int8", scope="shared")
        lv3_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(2560), T.int64(10240)), "int8", scope="shared")
        lv9_reindex_pad_shared_local = T.alloc_buffer((T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(10240)), "int8", scope="local")
        lv3_reindex_shared_local = T.alloc_buffer((T.int64(1), T.int64(2560), T.int64(10240)), "int8", scope="local")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(40), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding((n + T.int64(31)) // T.int64(32), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax2_3_init, ax1_3_init in T.grid(T.int64(4), T.int64(4)):
                                    with T.block("NT_matmul_init"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                        v2 = T.axis.spatial(T.int64(2560), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_init)
                                        T.reads()
                                        T.writes(var_NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2])
                                        var_NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2] = T.int8(0)
                                for ax3_0 in range(T.int64(640)):
                                    for ax0, ax1_ax2_fused_0_0_0 in T.grid(T.int64(1), T.int64(1)):
                                        for ax1_ax2_fused_0_0_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                            for ax1_ax2_fused_0_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                                for ax1_ax2_fused_1 in T.vectorized(T.int64(8), annotations={"check_vector_load": 1, "remove_vector_condition": 1}):
                                                    with T.block("lv9_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), ax0)
                                                        v1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ((ax1_ax2_fused_0_0_0 * T.int64(16) + ax1_ax2_fused_0_0_1) * T.int64(64) + ax1_ax2_fused_0_1 * T.int64(8) + ax1_ax2_fused_1) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(10240), ax3_0 * T.int64(16) + ((ax1_ax2_fused_0_0_0 * T.int64(16) + ax1_ax2_fused_0_0_1) * T.int64(64) + ax1_ax2_fused_0_1 * T.int64(8) + ax1_ax2_fused_1) % T.int64(16))
                                                        T.where(ax1_ax2_fused_0_0_0 * T.int64(16) + ax1_ax2_fused_0_0_1 < T.int64(8))
                                                        T.reads(lv9[T.int64(0), v1, v2])
                                                        T.writes(lv9_reindex_pad_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 8]], "double_buffer_scope": 0, "tir.manifest_shared_memory_local_stage": 1})
                                                        lv9_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < n, lv9[T.int64(0), v1, v2], T.int8(0))
                                    for ax0, ax1_ax2_fused_0_0_0 in T.grid(T.int64(1), T.int64(1)):
                                        for ax1_ax2_fused_0_0_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                            for ax1_ax2_fused_0_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                                for ax1_ax2_fused_1 in T.vectorized(T.int64(8), annotations={"check_vector_load": 1, "remove_vector_condition": 1}):
                                                    with T.block("lv3_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), ax0)
                                                        v1 = T.axis.spatial(T.int64(2560), ax0_ax2_0_fused * T.int64(64) + (ax1_ax2_fused_0_0_0 * T.int64(1024) + ax1_ax2_fused_0_0_1 * T.int64(64) + ax1_ax2_fused_0_1 * T.int64(8) + ax1_ax2_fused_1) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(10240), ax3_0 * T.int64(16) + (ax1_ax2_fused_0_0_0 * T.int64(1024) + ax1_ax2_fused_0_0_1 * T.int64(64) + ax1_ax2_fused_0_1 * T.int64(8) + ax1_ax2_fused_1) % T.int64(16))
                                                        T.reads(lv3[v1, v2])
                                                        T.writes(lv3_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 8]], "double_buffer_scope": 0, "tir.manifest_shared_memory_local_stage": 1})
                                                        lv3_reindex_shared[v0, v1, v2] = lv3[v1, v2]
                                    for ax3_1 in range(T.int64(16)):
                                        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(4), T.int64(1)):
                                            with T.block("lv9_reindex_pad_shared_local"):
                                                v0 = T.axis.spatial(T.int64(1), ax0)
                                                v1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                                v2 = T.axis.spatial(T.int64(10240), ax3_0 * T.int64(16) + ax3_1 + ax2)
                                                T.reads(lv9_reindex_pad_shared[v0, v1, v2])
                                                T.writes(lv9_reindex_pad_shared_local[v0, v1, v2])
                                                lv9_reindex_pad_shared_local[v0, v1, v2] = lv9_reindex_pad_shared[v0, v1, v2]
                                        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(4), T.int64(1)):
                                            with T.block("lv3_reindex_shared_local"):
                                                v0 = T.axis.spatial(T.int64(1), ax0)
                                                v1 = T.axis.spatial(T.int64(2560), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax1)
                                                v2 = T.axis.spatial(T.int64(10240), ax3_0 * T.int64(16) + ax3_1 + ax2)
                                                T.reads(lv3_reindex_shared[v0, v1, v2])
                                                T.writes(lv3_reindex_shared_local[v0, v1, v2])
                                                lv3_reindex_shared_local[v0, v1, v2] = lv3_reindex_shared[v0, v1, v2]
                                        for ax2_3, ax1_3 in T.grid(T.int64(4), T.int64(4)):
                                            with T.block("NT_matmul_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(2560), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3)
                                                v3 = T.axis.reduce(T.int64(10240), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(var_NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2], lv9_reindex_pad_shared_local[T.int64(0), v1, v3], lv3_reindex_shared_local[T.int64(0), v2, v3])
                                                T.writes(var_NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2])
                                                var_NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2] = var_NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2] + lv9_reindex_pad_shared_local[T.int64(0), v1, v3] * lv3_reindex_shared_local[T.int64(0), v2, v3]
                                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(4), T.int64(4)):
                                    with T.block("var_NT_matmul_intermediate_reindex_pad_local"):
                                        v0 = T.axis.spatial(T.int64(1), ax0)
                                        v1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                        v2 = T.axis.spatial(T.int64(2560), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2)
                                        T.reads(var_NT_matmul_intermediate_reindex_pad_local[v0, v1, v2], lv4[v2])
                                        T.writes(var_T_add_intermediate[T.int64(0), v1, v2])
                                        if v1 < n:
                                            var_T_add_intermediate[T.int64(0), v1, v2] = var_NT_matmul_intermediate_reindex_pad_local[v0, v1, v2] + lv4[v2]
                                            
                                            
                                            
import tvm
from tvm._ffi import get_global_func
from tvm import tir
from tvm import relax
mod=Module
target = tvm.target.Target("nvidia/geforce-rtx-3090",host="c")   
mod=relax.transform.AttachGlobalSymbol()(mod)
# mod =tvm.lower(mod)
pass_list=[]
pass_list.append(tir.transform.InjectPrefetch())
pass_list.append(tir.transform.TextureFlatten())
pass_list.append(tir.transform.StorageFlatten(64, False))
pass_list.append(tir.transform.LowerCrossThreadReduction())
pass_list.append(tir.transform.LowerInitBlock())
pass_list.append(tir.transform.PlanAndUpdateBufferAllocationLocation())
pass_list.append(tir.transform.ConvertBlocksToOpaque())
pass_list.append(tir.transform.LiftThreadBinding())
pass_list.append(tir.transform.ManifestSharedMemoryLocalStage())
pass_list.append(tir.transform.CompactBufferAllocation())
# pass_list.append(tir.transform.LowerAutoCopy())
# pass_list.append(tir.transform.UnifyThreadBinding())
# pass_list.append(tir.transform.LowerMatchBuffer())
# pass_list.append(tir.transform.Simplify())
# pass_list.append(tir.transform.InjectPermutedLayout())
# pass_list.append(tir.transform.Simplify())
# pass_list.append(tir.transform.InjectSoftwarePipeline())
# pass_list.append(tir.transform.TransformMmaBufferLayout())
# pass_list.append(tir.transform.LowerOpaqueBlock())
# pass_list.append(tir.transform.FlattenBuffer())
# pass_list.append(tir.transform.NarrowDataType(32))
# pass_list.append(tir.transform.Simplify())
# pass_list.append(tir.transform.LoopPartition())
# pass_list.append(tir.transform.VectorizeLoop(True))
# pass_list.append(tir.transform.StorageRewrite())
device_pass=get_global_func("driver.device_mod_passes")
seq = tvm.transform.Sequential(
    pass_list
    )
mod =seq(mod)

def save_to_file(filename,mod):
    with open(filename,"w") as f:
        f.write(mod.script())
save_to_file("/home/weitao/XIAG8XX/profile/testIR/NOWMMA/TEMP/1compact3.py",mod)                                             