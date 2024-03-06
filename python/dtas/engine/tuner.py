from typing import Optional, Dict, Tuple, List
from itertools import product
import os
import tempfile
import numpy as np

import tvm
from tvm import tir
from tvm.ir import IRModule
from tvm.target import Target

from ..arch import Arch
from ..schedule.schedule import get_config_emiter_template, get_scheduler_template
from ..common.analisys import FuncInfo
from ..common.config import Range, generate_range_list
from ..codegen.compile_result import CompileResult
from ..logging import get_log_level, debug_info
from ..codegen.code_gen import CodeGenerator
from ..codegen.runtime_packer import RuntimePacker
from .database import DataBase

class Engine:  # pylint: disable=too-few-public-methods
    """A IRModule pass that applies a list of ScheduleRules to all PrimFuncs in the module."""

    def __init__(
        self,
        topk: int = 10,
        range_div_factor: int = 256,
        parallel_build: bool = True,
        work_dir: str = None,
    ):
        """Construct a new ApplyFastTuning pass.

        Parameters
        ----------
        meta_database : str
            The path of database.
        """
        self.topk = topk
        self.parallel_build = parallel_build
        self.upper_bound = {}
        self.range_div_factor = range_div_factor
        if work_dir is None:
            work_dir = tempfile.TemporaryDirectory().name
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        database_dir = os.path.join(work_dir, 'data_base')
        os.makedirs(database_dir, exist_ok=True)
        best_record_path = os.path.join(database_dir, "best_record.json")
        tuning_record_path = os.path.join(database_dir,"tuning_record.py")
        self.database = DataBase(best_record_path, tuning_record_path)
        if get_log_level()>=1:debug_info(f"[WTM] Using database dir {database_dir}")

    def tune_module(  # pylint: disable=missing-function-docstring
        self,
        mod: IRModule,
        arch: Arch,
        range_div_factor: int = 256,
    ) -> tvm.runtime.Module:
        use_tc = True if arch.compute_capability >= "70" else False
        can_async_copy = True if arch.compute_capability >= "75" else False
        use_fp16 = False
        for g_var, func in mod.functions_items():
            if "tir_var_upper_bound" in func.attrs:
                for var, upper_bound in func.attrs["tir_var_upper_bound"].items():
                    self.upper_bound[var] = upper_bound

        if get_log_level() >= 2:
            debug_info(f"[WTM] Tuning module, upper_bound: {self.upper_bound}")
        cg = CodeGenerator(arch.target)
        for g_var, func in mod.functions_items():
            if isinstance(func, tir.PrimFunc):
                ranges_best_result, fp16, len_dynamic_args = self.tune_func(
                    func=func,
                    name=g_var.name_hint,
                    arch=arch,
                    use_tc=use_tc,
                    range_div_factor=range_div_factor,
                )
                if fp16:
                    use_fp16 = True
                index_table, index_stmt = self.get_index_table(ranges_best_result)
                if get_log_level() >= 2:
                    debug_info(f"index_table: {index_table}, index_stmt: {index_stmt}")
                cg.gen_code_for_func(
                    ranges_best_result, index_table, index_stmt, len(func.params) + len_dynamic_args
                )

        rt_packer = RuntimePacker(
            cg.kernel_handles,
            cg.host_forward_declares,
            cg.host_main_bodys,
            cg.device_forward_declares,
            cg.device_main_bodys,
            cg.kernel_info_dic,
            cg.func_names,
            work_dir=self.work_dir,
            use_fp16=use_fp16,
            target_kind="cuda",
        )
        mod = rt_packer.pack_to_tvm_runtime()
        return mod

    def tune_func(
        self,
        func: tir.PrimFunc,
        name,
        arch: Arch,
        use_tc = False,
        range_div_factor: int = 256,
    ):
        assert isinstance(func, tir.PrimFunc), "tune_func can only accept primfunc"
        if get_log_level() >= 2:
            debug_info(f"tuning function {name}")
        func_info = FuncInfo(func, name)
        use_fp16 =func_info.use_fp16
        scheduler = get_scheduler_template(func_info.kind, use_tc, arch.compute_capability)(func_info)
        config_emiter_template = get_config_emiter_template(func_info.kind, use_tc)
        range_to_best_result = {}
        range_tuples = self.gen_range_tuples(func_info, range_div_factor)
        
        for range_tuple in range_tuples:
            configs = config_emiter_template(func_info, arch).emit_config(
                range_tuple, self.topk
            )
            cpresults, best = scheduler.select_best(
                range_tuple, configs, arch, name, self.database, self.parallel_build
            )
            range_to_best_result[range_tuple] = best
        if get_log_level() >= 1:
            for k in range_to_best_result.keys():
                debug_info(f"before merge {k}")      
        merged_range_to_best_result = self.merge_same_configs(range_to_best_result)
        return merged_range_to_best_result, use_fp16, len(func_info.dynamic_args)

    def gen_range_tuples(self, func_info: FuncInfo, range_div_factor: int):
        if get_log_level()>=1: debug_info("getting range tuple")
        if len(func_info.dynamic_args) == 0:
            if get_log_level() >= 1:
                debug_info(" generating range_tuples, no dynamic_arg")
            return [()]
        else:
            range_lists = []
            for dyn_arg in func_info.dynamic_args:
                if dyn_arg.name in self.upper_bound:
                    upper_bound = self.upper_bound[dyn_arg.name]
                else:
                    self.upper_bound[dyn_arg.name] = 2048
                    upper_bound = 2048
                range_list = generate_range_list(
                    dyn_arg, 1, upper_bound, range_div_factor
                )
                range_lists.append(range_list)
            range_tuples = list(product(*range_lists))
        return range_tuples

    def get_index_table(
        self, ranges_best_result: Dict[Tuple[Range], CompileResult]
    ) -> Tuple[list, str]:
        if get_log_level()>=1:debug_info("getting index table .....")
        index_table = []
        index_stmt = "  int64_t index = "
        
        ranges_best_result = list(ranges_best_result.items())
        if len(ranges_best_result[0][0]) == 0:
            return index_table, index_stmt
        elif len(ranges_best_result[0][0]) == 1:
            # case only one dynamic arg
            dyn_arg_name = ranges_best_result[0][0][0].var.name
            upper_bound = self.upper_bound[dyn_arg_name]
            # index_len means the length of index_table from 0
            index_len = (upper_bound + self.range_div_factor - 1) // self.range_div_factor -1
            index_stmt += (
                f"({dyn_arg_name}/{str(self.range_div_factor)}) > {str(index_len)} ? {str(index_len)} : {dyn_arg_name}/{str(self.range_div_factor)};\n"
            )
            if get_log_level()>=1: debug_info(f"index_stmt: {index_stmt}")
            i = 0
            for range_tuple, best_cp_result in ranges_best_result:
                for _ in range(
                    range_tuple[0].start, range_tuple[0].end + 1, self.range_div_factor
                ):
                    index_table.append(i)
                i += 1
        else:
            dyn_arg_name_0 = ranges_best_result[0][0][0].var.name
            upper_bound_0 = self.upper_bound[dyn_arg_name_0]
            dyn_arg_name_1 = ranges_best_result[0][0][1].var.name
            upper_bound_1 = self.upper_bound[dyn_arg_name_1]
            # case there are two dynamic args
            index_table_size_0 = (
                upper_bound_0
                + self.range_div_factor
                - 1
            ) // self.range_div_factor
            index_len = ((upper_bound_0 + self.range_div_factor - 1) // self.range_div_factor) * ((upper_bound_1 + self.range_div_factor - 1) // self.range_div_factor) - 1
            if get_log_level()>=1: debug_info(f"idnex: {index_table_size_0}" )
            index_stmt += f"({dyn_arg_name_0}/{str(self.range_div_factor)} * {str(index_table_size_0)} + {dyn_arg_name_1}/{str(self.range_div_factor)}) > {str(index_len)} ? {str(index_len)} : ({dyn_arg_name_0}/{str(self.range_div_factor)} * {str(index_table_size_0)} + {dyn_arg_name_1}/{str(self.range_div_factor)});\n"
            i = 0
            for range_tuple, best_cp_result in ranges_best_result:
                for _ in range(
                    range_tuple[0].start, range_tuple[0].end + 1, self.range_div_factor
                ):
                    for _ in range(
                        range_tuple[1].start,
                        range_tuple[1].end + 1,
                        self.range_div_factor,
                    ):
                        index_table.append(i)
                i += 1
        return index_table, index_stmt

    def merge_same_configs(
        self, best_compile_results: Dict[Tuple[Range], CompileResult]
    ) -> Dict[Tuple[Range], CompileResult]:
        def try_merge(ranges: List[Range], range_tuple: List[Range], len = 1,) -> bool:
            try:
                ranges[len-1] = ranges[len-1].merge(range_tuple[len-1])
                if get_log_level()>=1:debug_info(f"len:{len} merge {range_tuple} to {ranges}")
                return True
            except ValueError:
                if get_log_level()>=1:debug_info(f"len:{len} fail to merge {range_tuple} to {ranges}")
                return False 
        
        def merge_adjacent_ranges(ranges_cr_list: List[Tuple[List[Range],CompileResult]]):
            merged_ranges_cr_list = []
            for i, (ranges, cr) in enumerate(ranges_cr_list):
                if i == 0:
                    # 对于第一对，直接添加到结果列表中  
                    merged_ranges_cr_list.append((ranges, cr))
                else:
                    if merged_ranges_cr_list[-1][0][1] == ranges[1]:
                        if get_log_level()>=1: debug_info(f"try to merge adjacent {ranges}")
                        try:
                            merged_ranges_cr_list[-1][0][0] = merged_ranges_cr_list[-1][0][0].merge(ranges[0])
                            if get_log_level()>=1: debug_info(f" merge {ranges} to {merged_ranges_cr_list[-1][0]}")
                        except ValueError:
                            if get_log_level()>=1: debug_info(f" except append {ranges}")
                            merged_ranges_cr_list.append((ranges, cr))
                    else:
                        if get_log_level()>=1: debug_info(f"append {ranges}")
                        merged_ranges_cr_list.append((ranges, cr))
            return merged_ranges_cr_list
        
        len_range_tuple = len(list(best_compile_results.keys())[0])
        # 存储合并后的结果
        merged_results = {}
        # 用于存储每个config对应的range和compile_result
        config_to_ranges_and_results = {}
        for range_tuple, best_cr in best_compile_results.items():
            debug_info(f"{range_tuple}")
            config = best_cr.config
            # 如果config尚未存在于映射中，则初始化一个新的键值对
            if config not in config_to_ranges_and_results.keys():
                config_to_ranges_and_results[config] = [(list(range_tuple), best_cr)]
            else:
                if get_log_level()>=1: debug_info(f"trying to merge {range_tuple}")
                # 合并已存在的Range列表
                ranges_cr_list = config_to_ranges_and_results[config]
                merged = False
                for ranges, cr in ranges_cr_list:
                    if len_range_tuple <= 2:
                        merged = try_merge(ranges, list(range_tuple), len_range_tuple)
                    else:
                        if get_log_level()>=1: debug_info(f"only support two dynamic args")
                        raise ValueError
                if not merged:
                    config_to_ranges_and_results[config].append((list(range_tuple), best_cr))
                    
        if len_range_tuple == 2:
            for config , ranges_cr_list in config_to_ranges_and_results.items():
                config_to_ranges_and_results[config] = merge_adjacent_ranges(ranges_cr_list)
                    
        # 将合并后的结果转换为所需的格式
        for config, ranges_cr_list in config_to_ranges_and_results.items():
            # if get_log_level() >= 1:
            #     debug_info(f"合并后的config: {config}")
            for (ranges, cr) in ranges_cr_list:
                # 将合并后的Range与CompileResult关联并添加到新字典中
                merged_results[tuple(ranges)] = cr
                if get_log_level() >= 2:
                    debug_info(f"merge houde ranges: {ranges}")

        def _sort_key(range_cr_item):
            start_list = []
            for r in range_cr_item[0]:
                start_list.append(r.start)
            return tuple(start_list)
            
        merged_results = dict(sorted(merged_results.items(), key=_sort_key)) 
        for k in merged_results.keys():
            if get_log_level()>=1: debug_info(f"merged_results: {k}")
        return merged_results
