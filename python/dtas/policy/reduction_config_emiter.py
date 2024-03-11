from typing import Tuple

from tvm import tir
from tvm.runtime import DataType

from .base_config_emiter import BaseConfigEmiter
from ..common.config import Range, ReductionConfig
from ..logging import get_log_level, debug_info


class ReductionConfigEmiter(BaseConfigEmiter):
    def plan_vectorize(self, re_range: Range, in_bytes, is_dynamic_reduction = False ):
        def is_type_allowed(vec):
            if in_bytes == 2:
                return (
                    (vec % 2 == 0 )
                    and in_bytes * 8 * vec <= 128
                    and in_bytes * 8 * vec >= 32
                )
            elif in_bytes == 1:
                return in_bytes * 8 * vec <= 128 and in_bytes * 8 * vec >= 32
            return in_bytes* 8 * vec <= 128 and in_bytes* 8 * vec >= 32

        # vector_load_lens = [1, 2, 3, 4, 8, 16]
        vector_load_lens = [16, 8, 4, 3, 2, 1]
        valid_vec_load_lens = list(filter(is_type_allowed, vector_load_lens))
        if not is_dynamic_reduction:        
            # avoid introducing predicates when vector length is too large
            for vec in valid_vec_load_lens:
                if re_range.end % vec == 0:
                    return vec
        else:
            return valid_vec_load_lens[0]

    def get_unroll_depth(self):
        if self.arch.platform == "CUDA":
            return 256

    def get_num_blocks(self, config:ReductionConfig, max_blocks, max_smem_usage, waves = 32):
        """
        n: The number of the elements.
        sm_count: The number of the SM.
        tpm: The maximum resident threads in per multiprocessor.
        Maximum number of 32-bit registers per thread block usually be 32k or 64k, since sm70 it always be 64k
        Maximum number of 32-bit registers per thread always be 255
        so each block can have 64 * 1024 / 255 = 256 threads for elementwise
        note. this is relatively loose limit, since we assume each thread reach the most register usage
        """
        config.max_active_blocks_per_sm = min(
            self.arch.max_resident_threads_per_sm // config.len_tx,
            self.arch.max_smem_per_sm // max_smem_usage,
            self.arch.max_resident_blocks_per_sm,
                )
        config.bx = max(
            1,
            min(
                max_blocks,
                self.arch.num_sm * config.max_active_blocks_per_sm * waves,
            ),
        )

    def generate_config_candidates(self, re_range: Range, rows:int, in_bytes:int, is_dynamic_reduction = False, topk=20):
        kNumWaves = 32
        max_smem_usage = (re_range.end + 2) * in_bytes
        # debug_info(f"max_smem_usage <= self.arch.max_smem_per_block {max_smem_usage}")
        unroll_depth = self.get_unroll_depth()
        base_block_size = self.arch.warp_size * self.arch.sm_partition  # 128
        block_size = base_block_size
        config_list = []
        if re_range.end > 1024:
            temp_storage = ( "shared.dyn" if max_smem_usage <= self.arch.max_smem_per_block else "cache")
        else:
            # case reduction is dynamic and reduction_len upperbound < 1024, 
            # if use smem will introduce extra sync overhead, so we use "cache" as temp storage 
            temp_storage = "cache"
                
        if temp_storage == "shared.dyn":
            vec_candidate = self.plan_vectorize(re_range, in_bytes)
        else:
            vec_candidate = None
            
        while block_size <= self.arch.max_threads_per_block and block_size <= re_range.end:
            config = ReductionConfig()
            config.unroll_depth = unroll_depth
            config.temp_storage = temp_storage
            config.len_tx = block_size
            self.get_num_blocks(config, rows, max_smem_usage, kNumWaves)
            if vec_candidate is not None:
                config.vector_size = vec_candidate
            config_list.append(config)
            block_size += self.arch.warp_size
            
        if get_log_level() >= 2:
            debug_info("config_list:")
            for config in config_list:
                debug_info(config)

        def _score_config(config:ReductionConfig):
            if max_smem_usage <= self.arch.max_smem_per_block and config.temp_storage == "shared.dyn":
                # 优先udaOccupancyMaxActiveBlocksPerMultiprocessor,若结果相同,使用较大的 block_size
                # debug_info(f"max_smem_usage <= self.arch.max_smem_per_block {max_smem_usage}")
                # 忽略register的影响，因为比较难估算
                return config.max_active_blocks_per_sm, config.len_tx, config.vector_size
            else:
                return config.len_tx, config.vector_size

        return (
            sorted(config_list, key=_score_config, reverse=True)[:topk]
            if len(config_list) >= topk
            else config_list
        )

    def emit_config(self, range_tuple:Tuple[Range], topk=20):
        # NOTE reduction only consider reduction length, take spatial axis as blockIdx
        re_len = self.func_info.red_len        
        rows = self.func_info.rows
        is_dynamic_reduction = self.func_info.dyn_red
        re_range = None
        if is_dynamic_reduction:
            for r in range_tuple:
                if r.var.name == re_len.name:
                    re_range = r
                else:
                    rows *= r.end
            if re_range is None:
                raise ValueError("reduction_len is dynamic, but  can't find reduction range info in range_tuple")
        else:
            re_range = Range(tir.Var("re_len","int32"), re_len, re_len)
            for r in range_tuple:
                rows *= r.end

        in_bytes = (DataType(self.func_info.in_dtype).bits+7) // 8
        # if get_log_level() >= 1: debug_info(f"re_range: {re_range}, in_bytes: {in_bytes}")
        config_candidates = self.generate_config_candidates(
            re_range, rows, in_bytes, is_dynamic_reduction, topk
        )
        # def save_config_candidates(file, config_candidates):
        #     import json
        #     configs_dict_list = [cf.to_dict() for cf in config_candidates] 
        #     with open(file, 'w') as file:  
        #         json.dump(configs_dict_list, file, indent=4)
        # save_config_candidates(f"/home/weitao/XIAG8XX/configs/softmax/{range_tuple[0].to_suffix()}top{topk}.json", config_candidates)
        return config_candidates
