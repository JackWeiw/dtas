from typing import List, Tuple

from tvm.runtime import DataType

from ..common.config import ElementwiseConfig, Range
from .base_config_emiter import BaseConfigEmiter

class ElementwiseConfigEmiter(BaseConfigEmiter):
    def plan_vectorize(self, num_elements, in_dtype, is_dynamic = False):
        dtype = DataType(in_dtype)

        def is_type_allowed(vec):
            if dtype.bits == 16:
                return (
                    (vec % 2 == 0 or vec == 1)
                    and dtype.bits * vec <= 128
                    and dtype.bits * vec >= 32
                )
            elif dtype.bits == 8:
                return dtype.bits * vec <= 128 and dtype.bits * vec >= 32
            return dtype.bits * vec <= 128
        # vector_load_lens = [1, 2, 3, 4, 8, 16]
        vector_load_lens = [16, 8, 4, 3, 2, 1]
        valid_vec_load_lens = list(filter(is_type_allowed, vector_load_lens))
        if not is_dynamic:        
            vec_candidates = []
            # avoid introducing predicates when vector length is too large
            for vec in valid_vec_load_lens:
                if num_elements % vec == 0:
                    vec_candidates.append(vec)
            return vec_candidates
        else:
            return valid_vec_load_lens

    def get_num_blocks(self, n, vector_size, num_sm, tpm):
        """
        n: The number of the elements.
        sm_count: The number of the SM.
        tpm: The maximum resident threads in per multiprocessor.
        Maximum number of 32-bit registers per thread block usually be 32k or 64k, since sm70 it always be 64k
        Maximum number of 32-bit registers per thread always be 255
        so each block can have 64 * 1024 / 255 = 256 threads for elementwise
        note. this is relatively loose limit, since we assume each thread reach the most register usage
        """
        kNumWaves = 32
        kBlockSize = 256
        num_blocks = max(
            1,
            min(
                (n + kBlockSize * vector_size - 1) // (kBlockSize * vector_size),
                num_sm * tpm // kBlockSize * kNumWaves,
            ),
        )
        return num_blocks

    def get_unroll_depth(self):
        if self.arch.platform == "CUDA":
            return 256

    def emit_config(self, range_tuple:Tuple[Range], topk=20) -> List[ElementwiseConfig]:
        config_list = []
        unroll_depth = self.get_unroll_depth()
        num_elements = self.func_info.num_elements
        in_dtype = self.func_info.in_dtype
        is_dynamic = False
        if len(range_tuple) > 0:  # dynamic case
            is_dynamic = True
            for r in range_tuple:
                num_elements *= r.end
                
        vec_candidates = self.plan_vectorize(num_elements, in_dtype , is_dynamic)
        
        for vec_candidate in vec_candidates:
            config = ElementwiseConfig()
            config.grid_size = self.get_num_blocks(
                num_elements,
                vec_candidate,
                self.arch.num_sm,
                self.arch.max_resident_threads_per_sm,
            )
            config.unroll_depth = unroll_depth
            config.vector_size = vec_candidate
            config_list.append(config)
            config_list = (
                config_list[:topk] if len(config_list) >= topk else config_list
        )

        return config_list
