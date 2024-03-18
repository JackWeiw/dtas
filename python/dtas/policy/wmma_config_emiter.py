from typing import Dict, List, Tuple, Literal
from itertools import product
import numpy as np

from tvm import tir
from tvm.runtime import DataType

from ..common.config import Range,WMMAConfig
from ..common.utils import get_all_factors
from .base_config_emiter import BaseConfigEmiter

from ..logging import get_log_level, debug_info

class WMMAConfigEmiter(BaseConfigEmiter):
    def estimate_registers_usage(self, config: WMMAConfig):
        # 由于micro_k 都取4 for float16, 所以wmma_load, wmma_sync unroll 都是4*2=8倍
        # wmma_load 需要9个 b32寄存器, wmma_load a 为micro_x, b 为micro_y
        # mma_sync 需要8个 b32寄存器, mma_sync为micro_x * micro_y
        # wmma_store 需要
        # 当x=5, y=5 register overflow
        # x=4, y=5 or x=5, y=4 register not overflow
        # 如果用softwarepipeline的话有不同的因素考虑,vectorize好像也是寄存器
        # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-wmma
        '''
            indtype .f16 A or B
            A vector expression of eight .f16x2 registers.
            outdtype .f16 C or D
            A vector expression of four .f16x2 registers.
            outdtype .f32 C or D
            A vector expression of eight .f32 registers.
        '''
        config.reg_usage = 2 * ( config.micro_shape_x * config.wmma_m  + config.micro_shape_y * config.wmma_n + config.micro_shape_x* config.micro_shape_y* config.wmma_m * config.wmma_n)  / 32

    def estimate_smem_usage(self, config: WMMAConfig, bytes, pad_offset):
        m = config.i * config.micro_shape_x * config.wmma_m
        n = config.j * config.micro_shape_y * config.wmma_n
        k = config.micro_shape_k * config.wmma_k
        m_tile_bytes = 2 * m * (k + pad_offset) * bytes
        n_tile_bytes = 2 * n * (k + pad_offset) * bytes
        out_tile_bytes = m * n * bytes
        if any([m_tile_bytes >= out_tile_bytes, n_tile_bytes >= out_tile_bytes]):
            config.smem_usage = m_tile_bytes + n_tile_bytes
        else:
            config.smem_usage = min(m_tile_bytes, n_tile_bytes) + out_tile_bytes
        # 
        # smem = sorted([m_tile_bytes, n_tile_bytes, out_tile_bytes], reverse=True)
        # config.smem_usage = smem[0] + smem[1]

    def generate_tile_candidates(
        self,
        wmma_shape,
        bytes,
        max_threads_per_block,
        warp_size,
        max_tile_size,
    ):
        # i*j < max_threads_per_block / warp_size=1024/32=32
        # i_j_candidates = [i for i in range(1, (max_threads_per_block // warp_size) + 1)]
        i_j_candidates = [i for i in range(1, 32)]
        i_j_micro_candidates = [i_micro for i_micro in range(1, 10)]
        # To ensure that the SMEM kdim has num_elemeny*dtype= 128 Bytes
        # 128为chunk_size
        k_facotr = [128 // (wmma_shape[2] * bytes)]
        candidates = list(
            product(
                i_j_candidates,
                i_j_candidates,
                i_j_micro_candidates,
                i_j_micro_candidates,
                k_facotr,
            )
        )

        # 产生block_tile and warp_tile candidates
        # 保证tile_k是128bytes，为刚好32个bank
        return list(
            filter(
                lambda item:item[0] * item[2] % 2 ==0
                and item[1] * item[3] % 2 ==0
                and item[0] * item[1] >= 4
                and item[0] * item[1] <= (max_threads_per_block // warp_size)
                and (
                    item[0] * item[2] * wmma_shape[0]
                    + item[1] * item[3] * wmma_shape[1]
                )
                <= max_tile_size,
                candidates,
            )
        )

    def plan_vectorize(self, m_tile:int, k_tile:int, bytes:int):
        def is_type_allowed(vec):
            if bytes*8 == 16:
                return (vec % 2 == 0 or vec == 1) and bytes*8 * vec <= 128 and bytes*8 * vec >=32
            if bytes*8 == 8:
                return (vec % 4 == 0) and bytes*8 * vec <= 128 and bytes*8 * vec >=32
            return bytes*8 * vec <= 128 and bytes*8 * vec >=32

        # vector_load_lens = [1, 2, 3, 4, 8, 16]
        vector_load_lens = [16, 8, 4, 3, 2, 1]
        valid_vec_load_lens = list(filter(is_type_allowed, vector_load_lens))
        for vec in valid_vec_load_lens:
            if (m_tile * k_tile) % vec == 0 :
                return vec
        return valid_vec_load_lens[0]

    def compute_config_statistics(self, gemm_extent_info, in_bytes:int, config: WMMAConfig):
        """
        需要的数据tile_size_m,tile_size_n
        tile总数,wave大小
        """
        BSZ, is_dyn_bsz = gemm_extent_info["bsz"]
        M, is_dyn_m = gemm_extent_info["m"]
        N, is_dyn_n = gemm_extent_info["n"]
        K, is_dyn_k = gemm_extent_info["k"]
        config.max_active_block_per_SM = int(
            min(
                self.arch.max_smem_per_sm // max(config.smem_usage, 1),
                self.arch.max_registers_per_sm // max(config.reg_usage, 1),
                self.arch.max_resident_threads_per_sm
                // (config.i * config.j * config.warp_size),
            )
        )

        """
        首先说明 wave 的概念:wave 表示 GPU 上同时执行的 thread block。
        例如一个 kernel 中 thread block 为 256 线程，每个线程使用了 128 个寄存器，
        那么在 GV100 上每个 SM 可同时执行 2 个 thread block,GV100 共 80 个 SM,一个 wave 就是 160 个 thread block。
        """
        config.grid_size = [
            BSZ,
            (M + config.i * config.micro_shape_x * config.wmma_m - 1)
            // (config.i * config.micro_shape_x * config.wmma_m),
            (N + config.j * config.micro_shape_y * config.wmma_n - 1)
            // (config.j * config.micro_shape_y * config.wmma_n),
        ]
        config.block_size = [1, config.i * config.j, config.warp_size]

        #一共有几个wave
        config.num_wave = (
            np.prod(config.grid_size)
            + (config.max_active_block_per_SM * self.arch.compute_max_core)
            - 1
        ) // max((config.max_active_block_per_SM * self.arch.compute_max_core), 1)

        tile_size_m = config.i * config.micro_shape_x * config.wmma_m
        tile_size_n = config.j * config.micro_shape_y * config.wmma_n
        tile_size_k = config.micro_shape_k * config.wmma_k
        config.tile_size_m = tile_size_m
        config.tile_size_n = tile_size_n
        config.tile_size_k = tile_size_k

        ab_total_size = (M + N) * K * in_bytes
        wave_gpu = config.max_active_block_per_SM * self.arch.num_sm #一个wave有多少个block
        config.fma_ldg_ratio = (tile_size_m * tile_size_n * 2) / (
                (tile_size_m + tile_size_n) * in_bytes
            )
        
        # TODO 如何考虑wave tail effect带来的影响
        if np.prod(config.grid_size) > wave_gpu and ab_total_size > self.arch.l2_cache_size:
            config.type = 0
            '''
            大矩阵，即 M, N, K 较大,A, B 矩阵无法完全放进 L2 and Tile 总数超过一个 wave 大小；
            '''
            # jisuanliang = tile_size_m * tile_size_n * K *2 (乘加)
            # fancunliang = (tile_size_m + tile_size_n) * K * (DataType(self.func_info.in_dtype).bits+7//8)
            # wave访存请求量
            a_ldg_request = wave_gpu * tile_size_m * K
            b_ldg_request = wave_gpu * tile_size_n * K
            if wave_gpu * tile_size_n > N :
                # dram实际访问量
                wave_x = (N + tile_size_n - 1) // tile_size_n
                wave_y = (wave_gpu + wave_x -1) // wave_x
                a_dram_ldg = wave_y * tile_size_m * K
                b_dram_ldg = N * K 
            else:
                wave_x = wave_gpu
                wave_y = 1
                a_dram_ldg = wave_y * tile_size_m * K
                b_dram_ldg = wave_x * tile_size_n * K
            l2_hit_rate = 1 - ((a_dram_ldg + b_dram_ldg) / (a_ldg_request + b_ldg_request))
            config.l2_hit_rate = l2_hit_rate
            config.p_ldg = self.arch.bandwidth[1] * l2_hit_rate + self.arch.bandwidth[0] * (
                1 - l2_hit_rate
            )
            config.device_ratio = self.arch.processing_power[3] * 1e12 / (config.p_ldg * (2**30))
        elif ab_total_size > self.arch.l2_cache_size and np.prod(config.grid_size) <= wave_gpu:
            config.type = 1
            # if get_log_level()>=1:debug_info(f"中矩阵：{ab_total_size/1024/1024}MB, {np.prod(config.grid_size)}, wave_gpu:{wave_gpu}")
            # 大矩阵，即 M, N, K 较大,A, B 矩阵无法完全放进 L2 and Tile 总数不超过一个 wave 大小；
            a_ldg_request = np.prod(config.grid_size) * tile_size_m * K
            b_ldg_request = np.prod(config.grid_size) * tile_size_n * K
            a_dram_ldg = M * K
            b_dram_ldg = N * K
            l2_hit_rate = 1 - ((a_dram_ldg + b_dram_ldg) / (a_ldg_request + b_ldg_request))
            config.l2_hit_rate = l2_hit_rate
            config.p_ldg = self.arch.bandwidth[1] * l2_hit_rate + self.arch.bandwidth[0] * (
                1 - l2_hit_rate
            )
            config.device_ratio = self.arch.processing_power[3] * 1e12 / (config.p_ldg * (2**30))
        elif ab_total_size <= self.arch.l2_cache_size :
            config.type = 2
            # if get_log_level()>=1:debug_info(f"小矩阵：{ab_total_size/1024/1024}MB")
            #小矩阵，即 M, N, K 较小,A, B 矩阵可以完全放进 L2 and Tile 总数超过一个 wave 大小；
            config.l2_hit_rate = 1
            config.p_ldg = self.arch.bandwidth[1]
            config.device_ratio = self.arch.processing_power[3] * 1e12 / (config.p_ldg * (2**30))
        if config.fma_ldg_ratio <= (
            self.arch.processing_power[3] * 1e12 / (config.p_ldg * (2**30))
        ):
            config.performance = config.p_ldg* (2**30) * config.fma_ldg_ratio
        else:
            config.performance = self.arch.processing_power[3] * 1e12
            
        ffma = config.micro_shape_x*config.wmma_m*config.micro_shape_y*config.wmma_n*config.micro_shape_k*config.wmma_k*2
        lds = (config.micro_shape_x*config.wmma_m+config.micro_shape_y*config.wmma_n)*(config.micro_shape_k*config.wmma_k) * in_bytes
        config.micro_performance = ffma / lds
        
        # the small the better
        config.static_shape_align = True 
        if not is_dyn_m:
            config.static_shape_align &= M % tile_size_m == 0
        if not is_dyn_n:
            config.static_shape_align &= N % tile_size_n == 0
        if not is_dyn_k:
            config.static_shape_align &= K % tile_size_k == 0
        config.assign_score = 0
        if M % tile_size_m != 0:
            config.assign_score += tile_size_m - M % tile_size_m
        if N % tile_size_n != 0:
            config.assign_score += tile_size_n - N % tile_size_n
        if K % tile_size_k != 0:
            config.assign_score += tile_size_k - K % tile_size_k

    def check_config_valid(self, config: WMMAConfig, bytes, pad_offset) -> bool:
        self.estimate_smem_usage(config, bytes, pad_offset)
        if config.smem_usage > self.arch.max_smem_per_block:
            if get_log_level()>=1:debug_info(f"smem usage {config.smem_usage} exceeds {self.arch.max_smem_per_block}")
            return False
        self.estimate_registers_usage(config)
        if config.reg_usage > self.arch.max_registers_per_thread:
            if get_log_level()>=1:debug_info(f"reg usage {config.reg_usage} exceeds {self.arch.max_registers_per_thread}")
            return False
        return True

    def score_config(self, config: WMMAConfig):
        return (config.static_shape_align, config.performance, config.micro_performance, -config.assign_score)
        # return (config.static_shape_align, config.micro_performance, config.performance, -config.assign_score)
        
    def get_max_tile_size(self, bytes, in_pad, stages: int):
        # return upperbound of tile_m+tile_n
        # note shared memory usage is equal to (tile_m + tile_n)*(tile_k + in_pad) * bytes * 2 (if use double buffer,stages=2)
        max_seme_usage = self.arch.max_smem_per_block
        tile_k = 128 // bytes
        return max_seme_usage // stages // bytes // (tile_k + in_pad)

    def get_extent(self,range_tuple, gemm_extent_map, kind: Literal["m", "n", "k", "s"]):
        if kind not in ["m", "n", "k", "s"]:
            raise ValueError(f"kind {kind} is not supported")
        if isinstance(gemm_extent_map[kind], tir.IntImm):
            return (gemm_extent_map[kind].value, False)
        else:
            for r in range_tuple:
                if r.var.name == gemm_extent_map[kind].name:
                    return (r.end, True)
    
    def generate_config_candidates(
        self, wmma_shape, range_tuple, topk=10
    ) -> List[WMMAConfig]:
        config_candidates = []
        in_bytes = (DataType(self.func_info.in_dtype).bits + 7) // 8
        in_pad = 8 if in_bytes == 2 else 16  # 8 for fp16, 16 for int8
        extent_map = self.func_info.gemm_extent_map
        
        gemm_extent_info = {}
        gemm_extent_info["bsz"] = self.get_extent(range_tuple, extent_map, "s")
        gemm_extent_info["m"] = self.get_extent(range_tuple, extent_map, "m")
        gemm_extent_info["n"] = self.get_extent(range_tuple, extent_map, "n")
        gemm_extent_info["k"] = self.get_extent(range_tuple, extent_map, "k")
        BSZ, is_dyn_bsz = self.get_extent(range_tuple, extent_map, "s")
        M, is_dyn_m = self.get_extent(range_tuple, extent_map, "m")
        N, is_dyn_n = self.get_extent(range_tuple, extent_map, "n")
        K, is_dyn_k = self.get_extent(range_tuple, extent_map, "k")
             
        # TODO currently set out pad_offset = 4
        out_pad = 4
        # 因为TensorCore算力与Bandwith的比值较大，所以永远使用double buffer隐藏延迟
        max_tile_size = self.get_max_tile_size(in_bytes, in_pad, 2)
        tile_candidates = self.generate_tile_candidates(
            wmma_shape,
            (DataType(self.func_info.in_dtype).bits + 7) // 8,
            self.arch.max_threads_per_block,
            self.arch.warp_size,
            max_tile_size,
        )
        if get_log_level() >= 2: debug_info(f"tile_candidates: {len(tile_candidates)}") 
           
        for tile_candidate in tile_candidates:
            config = WMMAConfig()
            config.wmma_m = wmma_shape[0]
            config.wmma_n = wmma_shape[1]
            config.wmma_k = wmma_shape[2]
            config.in_pad = in_pad
            config.i = tile_candidate[0]
            config.j = tile_candidate[1]
            config.micro_shape_x = tile_candidate[2]
            config.micro_shape_y = tile_candidate[3]
            config.micro_shape_k = tile_candidate[4]
            if self.check_config_valid(config, in_bytes, in_pad):
                self.compute_config_statistics(gemm_extent_info, in_bytes, config)
                config.in_vec_len_a = self.plan_vectorize(config.tile_size_m, config.tile_size_k, in_bytes)
                config.in_vec_len_b = self.plan_vectorize(config.tile_size_n, config.tile_size_k, in_bytes)
                config.out_vec_len = self.plan_vectorize(config.tile_size_m, config.tile_size_n, in_bytes)
                config_candidates.append(config)
        # return config_candidates
        config_candidates = sorted(config_candidates, key=lambda x: self.score_config(x),reverse=True)
        if get_log_level()>=2:debug_info(f"config_candidates: {len(config_candidates)}")
        return config_candidates
        # return config_candidates[:topk] if len(config_candidates) >= topk else config_candidates

    def emit_config(self, range_tuple:Tuple[Range], topk=10) ->  List[WMMAConfig]:
        if self.arch.compute_capability < "70":
            raise ValueError("This policy only support compute capability >= 7.0")
        # elif self.arch.compute_capability >= "80":
        #     raise Warning("This policy only support CUDA platform")
        elif self.arch.platform != "CUDA":
            raise ValueError("This policy only support CUDA platform")
        config_candidates = []
        for wmma_shape in self.func_info.wmma_shape_candidats:
             config_candidates += self.generate_config_candidates(wmma_shape, range_tuple, topk=100)
        config_candidates = sorted(config_candidates, key=lambda x: self.score_config(x),reverse=True)
        if get_log_level()>=2:debug_info(f"config_candidates: {len(config_candidates)}")
        def save_config_candidates(file, config_candidates):
            import json
            configs_dict_list = [cf.to_dict() for cf in config_candidates] 
            with open(file, 'w') as file:  
                json.dump(configs_dict_list, file, indent=4)
        config_candidates = config_candidates[:topk] if len(config_candidates) >= topk else config_candidates
        return config_candidates[:topk] if len(config_candidates) >= topk else config_candidates
