from typing import Tuple, Optional, List, Dict
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

import tvm
from tvm import tir
from tvm.target import Target
from tvm._ffi import get_global_func
from tvm.runtime import Module

from ..common.config import (
    Config,
    Range,
)
from ..common.analisys import FuncInfo
from ..common.utils import auto_inline_producers, auto_inline_consumers
from ..IRpass import *
from ..reference import get_ref_tensor
from ..codegen.compile_result import CompileResult
from ..arch import Arch
from ..logging import get_log_level, debug_info
from ..engine.database import DataBase

TVM_DEFAULT_NAME = "default_function_kernel0"


class TIRSchedulerBase:
    def __init__(
        self,
        func_info: FuncInfo,
    ) -> None:
        self.func_info = func_info
        self.block_size = [1, 1, 1]  # blockDim.xyz
        self.passes = self.make_passes()
        
    def cooperative_fetch(
        self,
        sch: tir.Schedule,
        SS: tir.Block,
        config: Config,
        dim_offset: int,
        is_a:bool = True,
        is_transpose:bool = False,
    ):
        loops = sch.get_loops(SS)
        if len(loops) == dim_offset:
            # handle fetching only one element
            loops.append(sch.add_unit_loop(SS))
        assert len(loops) > dim_offset
        axes = loops[-dim_offset:] 
        if (is_a and is_transpose) or (not is_a and not is_transpose):
            # specifical handle transpose read (for NN matmul or TT matmul)
            v0, v1 = sch.get_loops(SS)[-2:]
            sch.reorder(v1, v0)
            sch.transform_layout(SS, ("write", 0), lambda b, i, j: (b, j, i))
    
        ax = sch.fuse(*axes)
        if is_a:
            if config.in_vec_len_a > 1:
                ax, tv = sch.split(ax, factors=[None, config.in_vec_len_a])
                sch.vectorize(tv)
                sch.annotate(tv, "check_vector_load", True)
                sch.annotate(tv, "remove_vector_condition", True)
        else:
            if config.in_vec_len_b > 1:
                ax, tv = sch.split(ax, factors=[None, config.in_vec_len_b])
                sch.vectorize(tv)
                sch.annotate(tv, "check_vector_load", True)
                sch.annotate(tv, "remove_vector_condition", True)  
        if config.block_size[2] > 1:
            ax, tx = sch.split(ax, factors=[None, config.block_size[2]])
            sch.bind(tx, "threadIdx.x")
        if config.block_size[1] > 1:
            # if config.block_size[1] > 1 and sch.get(ax).extent.value > config.block_size[1]:
            ax, ty = sch.split(ax, factors=[None, config.block_size[1]])
            sch.bind(ty, "threadIdx.y")
        if config.block_size[0] > 1:
            # if config.block_size[0] > 1 and sch.get(ax).extent.value > config.block_size[0]:
            ax, tz = sch.split(ax, factors=[None, config.block_size[0]])
            sch.bind(tz, "threadIdx.z")
        # double buffer 得要对齐16KB
        # align 为wmma_k的倍数 block_size[2] always be multiple of 32, pad_factor to make sure always bank conflict free
        # offset: indtype float16时为8, int8时为16
        if config.swizzle:
            # Apply Swizzling
            sch.annotate(SS, ann_key="permuted_layout", ann_val=True)
        else:
            # if not, apply padding to alleviate bank conflict
            sch.storage_align(SS, 0, axis=-dim_offset, factor=16, offset=config.in_pad)
        if config.use_double_buffer:
            sch.annotate(SS, "double_buffer_scope", 0)
        if config.manifest_shared_memory_local_stage:
            sch.annotate(SS, "tir.manifest_shared_memory_local_stage", 1)
        auto_inline_producers(sch, SS)
        # if config.unroll:
        #     sch.annotate(ax, "pragma_unroll_explicit", ann_val=1)

    def make_passes(self)->List[tvm.transform.Pass]: 
        passes = []
        passes.append(AddAssert(self.func_info.dynamic_args))
        # self.passes.append(CheckVectorLoadPass().get_pass())
        # self.passes.append(RemoveConditionInVectorizePass().get_pass())
        return passes
    """
    new
    """

    def generate_tensors(self, range_tuple, device, sample_num=10) -> None:
        """

        real time profile有问题, generate tensors for profiling

        """
        shape_lists = [
            self.func_info.func.buffer_map[param].shape
            for param in self.func_info.func.params
        ]
        dtype_lists = [
            self.func_info.func.buffer_map[param].dtype
            for param in self.func_info.func.params
        ]
        sample_dict = {}
        for r in range_tuple:
            """
            均匀间隔取范围内sample_num个值, tensor的形状得为Int
            """
            sample_dict[r.var] = np.linspace(r.start, r.end, sample_num, dtype="int64")

            # sample_dict[r.var] = [
            #     random.randint(r.start, r.end) for _ in range(sample_num)
            # ]
        arg_array_lists = []
        for i in range(sample_num):
            arg_array_list = []
            for shape, dtype in zip(shape_lists, dtype_lists):
                new_shape = []
                for dim in shape:
                    if isinstance(dim, tir.expr.Var):
                        new_shape.append(sample_dict[dim][i])
                    else:
                        new_shape.append(dim.value)
                arg_array_list.append(get_ref_tensor(new_shape, device, dtype))
            arg_array_lists.append(arg_array_list)
        return arg_array_lists

    def apply_config(self, config: Config) -> Optional[tir.Schedule]:
        raise NotImplementedError()

    def apply_and_build(
        self, config: Config, arch: Arch, name
    ) -> Tuple[Config, Optional[tir.Schedule], Optional[Module]]:
        try:
            sch = self.apply_config(config)
            if sch is None:
                if get_log_level() >= 2:
                    debug_info(f"fail to apply config:\n{config}")
                return config, sch, None
            mod = sch.mod
            mod = tvm.transform.Sequential(self.passes)(mod)
            # debug_info(mod)
            with tvm.transform.PassContext(
                disabled_pass=["tir.AnnotateEntryFunc"],
                config={"tir.use_async_copy": True,},
            ):
                target = arch.target
                rt_mod = tvm.build(mod["main"], target=target, name=name)
                if get_log_level() >= 2:
                    debug_info(f"build success!")
        except Exception as e:
            if get_log_level() >= 1:
                debug_info(f"apply_and_build failed: {e}")
            sch = None
            rt_mod = None
        return (
            config,
            sch,
            rt_mod,
        )

    def select_best_parallel(
        self,
        range_tuple: Tuple[Range],
        config_list: List[Config],
        arch: Arch,
        name=TVM_DEFAULT_NAME,
        database: Optional[DataBase] = None,
    ):
        cpresults = []
        profile_tensors = []
        profile_tensors = self.generate_tensors(range_tuple, arch.device, sample_num=10)
        
        num_procs = min(len(config_list), os.cpu_count(), 10)
        if get_log_level() >=1: debug_info(f"begin to tuning func:{name} range:{range_tuple} with {len(config_list)} configs parallel with {num_procs} processes")
        with ThreadPoolExecutor(max_workers=num_procs) as executor:
            all_tasks = []
            for config in config_list:
                t = executor.submit(self.apply_and_build, config, arch, name)
                all_tasks.append(t)
        if get_log_level() >=1: debug_info(f"complete build task, begin to set profile")
        for future in as_completed(all_tasks):
            config, sch, rt_mod = future.result()
            if sch is None:
                continue
            if rt_mod is None:
                continue
            cpresult = CompileResult(name, config, sch, rt_mod)
            timer_cuda_mod = rt_mod.time_evaluator(
                name, arch.device, number=5, min_repeat_ms=50
            )
            cpresult.profile_tensors = profile_tensors
            cpresult.time_evaluator = timer_cuda_mod
            cpresults.append(cpresult)
        debug_info(f"len of cpresults:{len(cpresults)}")
        best = None
        best_latency = 1e9
        if get_log_level() >=1: debug_info(f"begin to profile")
        for cpresult in cpresults:
            config = cpresult.config
            try:
                latency = cpresult.profile()
            except Exception as e_mesg:
                if get_log_level() >= 1:debug_info(f"[WTM] Evaluation with config failed: {e_mesg}")
                continue
            if get_log_level() >= 2:
                debug_info("[WTM] Evaluation with config: {config} ")
                debug_info(
                    "[WTM] Time cost of this config: {:.3f} ms".format(latency)
                )
            cpresult.latency = latency
            if database is not None:
                database.commit_tuning_record(range_tuple, cpresult)
            if latency < best_latency:
                best_latency = latency
                best = cpresult
        if get_log_level() >= 1:
            debug_info("[WTM] best config: {:.3f} ms".format(best_latency ))
        if database is not None and best is not None:
            database.commit_best_record(range_tuple, best)
        del profile_tensors
        return cpresults, best

    def select_best(
        self,
        range_tuple: Tuple[Range],
        config_list: List[Config],
        arch: Arch,
        name=TVM_DEFAULT_NAME,
        database: Optional[DataBase] = None,
        parallel_build=False,
    ) -> Tuple[List[CompileResult], CompileResult]:
        if parallel_build:
            return self.select_best_parallel(range_tuple, config_list, arch, name, database)
        if get_log_level() >=1: debug_info(f"begin to tuning func:{name} range:{range_tuple} with {len(config_list)} configs")
        profile_tensors = self.generate_tensors(range_tuple, arch.device, sample_num=64)
        cpresults = []
        for config in config_list:
            if get_log_level() >= 2:
                debug_info("apply_and_buld")
            config, sch, rt_mod = self.apply_and_build(config, arch=arch, name=name)
            if rt_mod:
                cpresult = CompileResult(name, config, sch, rt_mod)
                if get_log_level() >= 2:
                    debug_info("setting time_evaluator......")
                timer_cuda_mod = rt_mod.time_evaluator(
                    func_name=name, dev=arch.device, number=5, min_repeat_ms=50
                )
                cpresult.profile_tensors = profile_tensors
                cpresult.time_evaluator = timer_cuda_mod
                cpresults.append(cpresult)

        best = None
        best_latency = 1e9
        for cpresult in cpresults:
            config = cpresult.config
            try:
                latency = cpresult.profile()

            except Exception as e_mesg:
                if get_log_level()>=1:debug_info(f"[WTM] Evaluation with config failed:{e_mesg} ")
                continue
            if get_log_level() >= 2:
                # debug_info("[WTM] Evaluation with config: {config} ")
                debug_info(
                    "[WTM] Time cost of this config: {:.3f} ms".format(latency)
                )
            cpresult.latency = latency
            if database is not None:
                    database.commit_tuning_record(range_tuple, cpresult)
            if latency < best_latency:
                best_latency = latency
                best = cpresult
        if database is not None and best is not None:
            database.commit_best_record(range_tuple, best)
        if get_log_level() >= 1:
            debug_info("[WTM] best config: {:.3f} ms".format(best_latency))
        del profile_tensors
        return cpresults, best


def _detect_local_cuda():
    dev = tvm.cuda()
    if not dev.exist:
        return None
    return tvm.target.Target(
        {
            "kind": "cuda",
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": dev.warp_size,
            "registers_per_block": 65536,
            "arch": "sm_" + dev.compute_version.replace(".", ""),
        }
    )
