from typing import Dict, List, Union
import json

from tvm import tir

from .analisys import FuncKind

TVM_DEFAULT_NAME = "default_function_kernel0"


func_kind_map = {
    FuncKind.kFunc_GEMM: "GEMM",
    FuncKind.kFunc_GEMV: "GEMV",
    FuncKind.kFunc_Reduction: "Reduction",
    FuncKind.kFunc_Elementwise: "Elementwise",
    FuncKind.kFunc_Transpose: "Transpose",
    FuncKind.KFunc_Fallback: "Fallback",
}


class Range:
    def __init__(self, var: Union[tir.Var, str], start: int, end: int) -> None:
        self.var = var if isinstance(var, tir.Var) else tir.Var(var, "int32")
        self.start = start
        self.end = end
        self.tuned = False

    def to_suffix(self):
        return (
            "_" + self.var.name + "_" + str(self.start) + "_to_" + str(self.end) + "_"
        )

    def merge(self, other: "Range") -> "Range":
        """
        used to merge ranges that have same best config
        """
        if self.var.name != other.var.name:
            raise ValueError("can not merge different var")
        if self.end == other.start - 1:
            merge_end = other.end
            merge_start = self.start
        elif self.start == other.end + 1:
            merge_start = other.start
            merge_end = self.end
        else:
            raise ValueError("can not merge non-adjacent ranges")
        return Range(self.var, merge_start, merge_end)
    
    def __repr__(self) -> str:
        return f"<{self.var.name}: Range({self.start}, {self.end})>"

    def __str__(self) -> str:
        return f"<{self.var.name}: Range({self.start}, {self.end})>"

    def __eq__(self, o: "Range") -> bool:
        if not isinstance(o, Range):
            return False
        return self.var == o.var and self.start == o.start and self.end == o.end

    def __hash__(self) -> int:
        return hash((self.var, self.start, self.end))


def generate_range_list(
    var: tir.Var, start: int, end: int, div_factor: int
) -> List[Range]:
    assert start < end
    assert div_factor > 0
    if end - start <= div_factor:
        return [Range(var, start, start + div_factor - 1)]
    assert end - start > div_factor
    ranges = []

    while start < end:
        # ranges.append(Range(start, min(start + div_factor - 1, end)))
        ranges.append(Range(var, start, start + div_factor - 1))
        start += div_factor
    return ranges

class Statistics:
    def __init__(self) -> None:
        self.tile_size_m = 0
        self.tile_size_n = 0
        self.tile_size_k = 0
        
        self.performance = 0
        self.fma_ldg_ratio = 0
        self.assign_score = 0
        self.reg_usage = 0
        self.smem_usage = 0
        self.traffic = 0
        self.max_active_block_per_SM = 0
        self.num_wave = 0


class Config:
    def __init__(self) -> None:
        self.config_kind = "Config"

    # 用以判断两个config是否相同

    def to_dict(self) -> Dict:
        pass

    def from_dict(self, dic: Dict) -> "Config":
        pass

    def __repr__(self) -> str:
        return str(self.to_dict())

    def __str__(self) -> str:
        return self.config_kind + ": " + str(self.to_dict())

    def __hash__(self):
        return hash(json.dumps(self.to_dict(), sort_keys=True).encode())

    def __eq__(self, other: "Config"):
        if not isinstance(other, Config):
            return False
        this_dic = self.to_dict()
        other_dic = other.to_dict()
        for k in this_dic.keys():
            if k not in other_dic.keys():
                return False
            if other_dic[k] != this_dic[k]:
                return False
        return True


class GEMMConfig(Config):
    def __init__(self) -> None:
        self.config_kind = "GEMMConfig"
        # ，所以 SGEMM 中 thread tile 通常取 8*8, 8*16 等数值
        # spacial axes tiling info
        self.block_x: int = 8  # warp cnt rows
        self.block_y: int = 16
        self.vthread_x: int = 1
        self.vthread_y: int = 1
        self.micro_shape_x: int = 4
        self.micro_shape_y: int = 4
        self.micro_shape_k: int = 16
        self.unroll_depth: int = 256  # 0 means no unroll
        
        self.in_vec_len = 4  # CooperativeFetch的vector size
        # 如果是4的话会CUDA: misaligned address
        self.in_pad = 8
        self.out_vec_len = 4
        self.out_pad = 4
        self.warp_size = 32
        self.vector_size: int = 8
        self.use_shared: bool = True
        ## 像android等设备不需要storage align
        self.storage_align: bool = False
        self.inner_x: bool = False
        self.manifest_shared_memory_local_stage = True
        self.use_double_buffer = True
        self.use_software_pipeline = True
        self.use_async_copy = False  ##SM80后才有
        self.grid_size = [1, 1, 1]
        self.block_size = [1, 1, 1]

    def to_dict(self) -> Dict:
        dic = {}
        dic["tile_shape"] = (
            self.micro_shape_x * self.block_x * self.vthread_x,
            self.micro_shape_y * self.block_y * self.vthread_y,
        )
        dic["block_x"], dic["block_y"] = self.block_x, self.block_y
        dic["vthread_x"], dic["vthread_y"] = self.vthread_x, self.vthread_y
        dic["micro_shape"] = (
            self.micro_shape_x,
            self.micro_shape_y,
            self.micro_shape_k,
        )
        dic["unroll"] = self.unroll
        dic["in_vec_len_pad"] = (self.in_vec_len, self.in_pad)
        dic["out_vec_len_pad"] = (self.out_vec_len, self.out_pad)
        dic["warp_size"] = self.warp_size
        return dic

    def from_dict(self, dic: Dict) -> "Config":
        self.__init__()
        self.block_x, self.block_y = dic["block_x"], dic["block_y"]
        self.vthread_x, self.vthread_y = dic["vthread_x"], dic["vthread_y"]
        self.micro_shape_x, self.micro_shape_y, self.micro_shape_k = dic["micro_shape"]
        self.in_vec_len, self.in_pad = dic["in_vec_len_pad"]
        self.out_vec_len, self.out_pad = dic["out_vec_len_pad"]
        self.warp_size = dic["warp_size"]
        self.unroll = dic["unroll"]
        return self


class WMMAConfig(Config):
    def __init__(self) -> None:
        self.config_kind = "WMMAConfig"
        self.i = 2  ##helper in wmma matmul
        self.j = 4
        self.micro_shape_x: int = 7  # warp cnt rows
        self.micro_shape_y: int = 2  # warp cnt cols
        self.micro_shape_k: int = 4

        self.wmma_m = 16
        self.wmma_n = 16
        self.wmma_k = 16

        self.in_vec_len_a = 8  # CooperativeFetch的vector size
        self.in_vec_len_b = 8 
        # 如果是4的话会CUDA: misaligned address
        self.in_pad = 8

        self.out_vec_len = 8
        self.out_pad = 4

        self.warp_size = 32
        # manifest_shared_memory_local_stage与use_async_copy不能同时存在
        self.manifest_shared_memory_local_stage = True # always True
        self.use_double_buffer = True # always True
        # self.use_software_pipeline = True # always True
        # self.use_async_copy = False  ##SM80后才有
        self.swizzle = False
        self.grid_size = [1, 1, 1]
        self.block_size = [1, 1, 1]

        self.statistics = Statistics()
        # statistics
        # 对于tile_m,tile_n的计算访存比
        self.micro_performance = 0
        self.device_ratio = 0
        self.l2_hit_rate = 0
        self.p_ldg = 0
        self.performance = 0
        self.fma_ldg_ratio = 0
        self.assign_score = 0
        self.reg_usage = 0
        self.smem_usage = 0
        self.traffic = 0
        self.max_active_block_per_SM = 0
        self.num_wave = 0
        self.tile_size_m = 0
        self.tile_size_n = 0
        self.tile_size_k = 0

    def to_dict(self) -> Dict:
        dic = {}
        dic["performance"] = self.performance
        dic["micro_performance"] = self.micro_performance
        dic["tile_shape"] = (self.tile_size_m, self.tile_size_n)
        dic["device_ratio"] = self.device_ratio
        dic["fma_ldg_ratio"] =self.fma_ldg_ratio
        dic["l2_hit_rate"] = self.l2_hit_rate
        dic["p_ldg"] = self.p_ldg
        dic["assign_score"] = self.assign_score
        dic["wmma_shape"] = (self.wmma_m, self.wmma_n, self.wmma_k)
        dic["i"], dic["j"] = self.i, self.j
        dic["micro_shape"] = (
            self.micro_shape_x,
            self.micro_shape_y,
            self.micro_shape_k,
        )
        dic["in_vec_len_a"] = self.in_vec_len_a
        dic["in_vec_len_b"] = self.in_vec_len_b
        dic["in_pad"] = self.in_pad
        # dic["out_vec_len_pad"] = (self.out_vec_len, self.out_pad)
        # dic["warp_size"] = self.warp_size
        # dic["use_software_pipeline"] = self.use_software_pipeline
        # dic["use_double_buffer"] = self.use_double_buffer
        # dic[
        #     "manifest_shared_memory_local_stage"
        # ] = self.manifest_shared_memory_local_stage
        # dic["use_async_copy"] = self.use_async_copy

        return dic

    def from_dict(self, dic: Dict) -> "Config":
        self.__init__()
        self.wmma_m, self.wmma_n, self.wmma_k = dic["wmma_shape"]
        self.i, self.j = dic["i"], dic["j"]
        self.micro_shape_x, self.micro_shape_y, self.micro_shape_k = dic["micro_shape"]
        self.in_vec_len_a = dic["in_vec_len_a"]
        self.in_vec_len_b = dic["in_vec_len_b"] 
        self.in_pad = dic["in_pad"]
        self.out_vec_len, self.out_pad = dic["out_vec_len_pad"]
        self.warp_size = dic["warp_size"]
        if "manifest_shared_memory_local_stage" in dic:
            self.manifest_shared_memory_local_stage = dic[
                "manifest_shared_memory_local_stage"
            ]
        if "use_double_buffer" in dic:
            self.use_double_buffer = dic["use_double_buffer"]
        if "use_software_pipeline" in dic:
            self.use_software_pipeline = dic["use_software_pipeline"]
        if "use_async_copy" in dic:
            self.use_async_copy = dic["use_async_copy"]
        return self


class MMAConfig(Config):
    def __init__(self) -> None:
        self.config_kind = "MMAConfig"
        # Supported data types
        # fp16, fp16, fp16: fp16 precision
        # fp16, fp16, fp32: fp16 mixed precision

        self.swizzle_factor_for_l2_m:int = 1
        self.swizzle_factor_for_l2_n:int = 1
        
        self.thread_z: int = 2
        self.thread_y: int =2
        
        self.micro_block_cnt_in_warp_m: int = 4
        self.micro_block_cnt_in_warp_n: int = 4
        self.micro_block_cnt_in_warp_k: int = 2
        
        self.micro_size_m: int = 16
        self.micro_size_n: int = 16
        self.micro_size_k: int = 16
        
        self.rstep = 1
        self.swizzle = True
        
        self.in_vec_len_a = 8  # CooperativeFetch的vector size
        self.in_vec_len_b = 8 
        # 如果是4的话会CUDA: misaligned address
        self.in_pad = 8
        self.out_vec_len = 8
        self.out_pad = 4
        self.warp_size = 32
        self.manifest_shared_memory_local_stage = True
        self.use_double_buffer = True # always True
        self.use_software_pipeline = True # always True
        self.use_async_copy = False  ##SM80后才有
        self.grid_size = [1, 1, 1]
        self.block_size = [1, 1, 1]

        self.statistics = Statistics()
        
        
class GEMVConfig(Config):
    def __init__(self) -> None:
        self.config_kind = "GEMVConfig"
        # TILE_S >= VEC_LOAD
        self.TS = 16
        self.TILE_S = 1
        self.VEC_LOAD_S = 1

        self.TR = 32
        self.TILE_R = 8
        # make sure TILE_R % VEC_C == 0
        self.VEC_C = 8
        # TAG_S,TAG_R various whether ineredution or not
        self.TAG_S, self.TAG_R = "threadIdx.y", "threadIdx.x"

        self.LOAD_V_SHARED = True
        self.LOAD_V_VEC = 8

        self.UNROLL = 256

    def to_dict(self) -> Dict:
        dic = {}
        dic["sapctial_thread"] = self.TS
        dic["sapctial_tile"] = self.TILE_S
        dic["reduction_thread"] = self.TR
        dic["reduction_tile"] = self.TILE_R
        dic["vec_c"] = self.VEC_C
        dic["tag_s"], dic["tag_r"] = self.TAG_S, self.TAG_R
        dic["vec_load"] = self.VEC_LOAD
        dic["load_v_shared"] = self.LOAD_V_SHARED
        dic["load_v_vec"] = self.LOAD_V_VEC
        dic["unroll_factor"] = self.unroll_factor
        return dic

    def from_dict(self, dic: Dict) -> "Config":
        self.__init__()
        return self


class ReductionConfig(Config):
    def __init__(self) -> None:
        self.config_kind = "ReductionConfig"
        self.bx = 1000
        self.len_tx = 128 
        self.unroll_depth = 256
        self.vector_size = 4
        self.temp_storage = "shared.dyn"
        # self.temp_storage = "local"
        self.max_active_blocks_per_sm = 0

    def to_dict(self) -> Dict:
        dic = {}
        dic["bx"] = self.bx
        dic["len_tx"] = self.len_tx
        dic["unroll_depth"] = self.unroll_depth
        dic["vector_size"] = self.vector_size
        dic["temp_storage"] = self.temp_storage
        dic["max_active_blocks_per_sm"] =self.max_active_blocks_per_sm
        return dic

    def from_dict(self, dic: Dict) -> "Config":
        self.__init__()
        self.len_tx = dic["len_tx"]
        self.unroll_depth = dic["unroll_depth"]
        self.vector_size = dic["vector_size"]
        self.temp_storage = dic["temp_storage"]
        return self


class ElementwiseConfig(Config):
    def __init__(self) -> None:
        self.config_kind = "ElementwiseConfig"
        self.len_tx = 256  # always set to 256
        self.vector_size = 4
        self.grid_size = 4  # decide according to number of elements
        self.unroll_depth = 0

    def to_dict(self) -> Dict:
        dic = {}
        dic["len_tx"] = self.len_tx
        dic["vector_size"] = self.vector_size
        dic["grid_size"] = self.grid_size
        dic["unroll_depth"] = self.unroll_depth
        return dic

    def from_dict(self, dic: Dict) -> "Config":
        self.__init__()
        self.len_tx = dic["len_tx"]
        self.vector_size = dic["vector_size"]
        self.grid_size = dic["grid_size"]
        self.unroll_depth = dic["unroll_depth"]
        return self


class FallbackConfig(Config):
    def __init__(self) -> None:
        self.config_kind = "FallbackConfig"
        self.func_kind = "fallback"
        self.TILE_S = 16
        self.TR = 16
        self.TILE_R = 16
        self.VEC_C = 4
        self.TAG_S, self.TAG_R = "threadIdx.y", "threadIdx.x"
        self.VEC_LOAD = 1
        self.LOAD_V_SHARED = True
        self.LOAD_V_VEC = 8
