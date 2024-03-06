from tvm.runtime import DataType

from .base_config_emiter import BaseConfigEmiter


class GEMVConfigEmiter(BaseConfigEmiter):
    def plan_vectorize(self):
        dtype = DataType(self.func_info.in_dtype)

        def is_type_allowed(vec):
            if dtype.bits == 16:
                return (vec % 2 == 0 or vec == 1) and dtype.bits * vec <= 128
            if dtype.bits == 8:
                return (vec % 4 == 0 or vec < 4) and dtype.bits * vec <= 128
            return dtype.bits * vec <= 128

        # vector_load_lens = [1, 2, 3, 4, 8, 16]
        vector_load_lens = [16, 8, 4, 2, 3, 1]
        valid_vec_load_lens = list(filter(is_type_allowed, vector_load_lens))
        return valid_vec_load_lens

    def get_unroll_depth(self):
        if self.arch.platform == "CUDA":
            return 256

    def emit_config(self, range_tuple, topk=20):
        config_list = []
        unroll_depth = self.get_unroll_depth()

        return config_list
