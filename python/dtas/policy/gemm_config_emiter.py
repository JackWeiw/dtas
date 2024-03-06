from typing import Dict, List, Tuple
from copy import deepcopy
from itertools import product
import numpy as np

from tvm import tir
from tvm.runtime import DataType

from ..common.config import  Range, GEMMConfig
from ..common.utils import get_all_factors
from .base_config_emiter import BaseConfigEmiter


class GEMMConfigEmiter(BaseConfigEmiter):
    def estimate_registers_usage(self, config: GEMMConfig):
        # 如果用softwarepipeline的话有不同的因素考虑,vectorize好像也是寄存器
        config.reg_usage = 1.2 * (
            config.micro_shape_x * config.wmma_m * config.wmma_k
            + config.micro_shape_y * config.wmma_n * config.wmma_k
            + config.micro_shape_x
            * config.micro_shape_y
            * config.wmma_m
            * config.wmma_n
        )

    def estimate_smem_usage(self, config: GEMMConfig, bytes, pad_offset):
        m = config.i * config.micro_shape_x * config.wmma_m
        n = config.j * config.micro_shape_y * config.wmma_n
        k = config.micro_shape_k * config.wmma_k
        if config.use_async_copy:
            config.smem_usage = 3 * (m + n) * (k + pad_offset) * bytes
        else:
            config.smem_usage = 2 * (m + n) * (k + pad_offset) * bytes

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
        i_j_micro_candidates = [i_micro for i_micro in range(1, 5)]
        # To ensure that the SMEM kdim has num_elemeny*dtype= 128 Bytes
        # 128为chunk_size
        k_facotr = 128 // (wmma_shape[2] * bytes)
        print("k_factors", k_facotr)
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
        if wmma_shape[0] == wmma_shape[2]:  # case 16x16x16
            return list(
                filter(
                    lambda item: item[0] * item[2] % (64 // wmma_shape[2]) == 0
                    and item[1] * item[3] % (64 // wmma_shape[2]) == 0
                    and item[0] * item[1] <= (max_threads_per_block // warp_size)
                    and (
                        item[0] * item[2] * wmma_shape[0]
                        + item[1] * item[3] * wmma_shape[1]
                    )
                    <= max_tile_size,
                    candidates,
                )
            )
        if wmma_shape[0] < wmma_shape[2]:  # case 8x32x16
            return list(
                filter(
                    lambda item: item[0] * item[2] % (32 // wmma_shape[0]) == 0
                    and item[1] * item[3] % (64 // wmma_shape[1]) == 0
                    and item[0] * item[1] <= (max_threads_per_block // warp_size)
                    and (
                        item[0] * item[2] * wmma_shape[0]
                        + item[1] * item[3] * wmma_shape[1]
                    )
                    <= max_tile_size,
                    candidates,
                )
            )
        if wmma_shape[1] < wmma_shape[2]:  # case 32x8x16
            return list(
                filter(
                    lambda item: item[1] * item[3] % (32 // wmma_shape[1]) == 0
                    and item[0] * item[1] <= (max_threads_per_block // warp_size)
                    and (
                        item[0] * item[2] * wmma_shape[0]
                        + item[1] * item[3] * wmma_shape[1]
                    )
                    <= max_tile_size,
                    candidates,
                )
            )

    def plan_vectorize(self):
        dtype = DataType(self.func_info.in_dtype)

        def is_type_allowed(vec):
            if dtype.bits == 16:
                return (vec % 2 == 0 or vec == 1) and dtype.bits * vec <= 128
            if dtype.bits == 8:
                return (vec % 4 == 0 or vec < 4) and dtype.bits * vec <= 128
            return dtype.bits * vec <= 128

        # vector_load_lens = [1, 2, 3, 4, 8, 16]
        vector_load_lens = [16, 8, 4, 3, 2, 1]
        valid_vec_load_lens = list(filter(is_type_allowed, vector_load_lens))
        return valid_vec_load_lens

    def compute_config_statistics(self, range_tuple, config: GEMMConfig):
        """
        需要的数据tile_size_m,tile_size_n
        tile总数,wave大小
        """
        BSZ = (
            self.func_info.gemm_extent_map["s"]
            if isinstance(self.func_info.gemm_extent_map["s"], tir.IntImm)
            else sorted(
                [
                    r.end
                    if r.var.name == self.func_info.gemm_extent_map["s"].name
                    else -1
                    for r in range_tuple
                ]
            )[-1]
        )
        M = (
            self.func_info.gemm_extent_map["m"]
            if isinstance(self.func_info.gemm_extent_map["m"], tir.IntImm)
            else sorted(
                [
                    r.end
                    if r.var.name == self.func_info.gemm_extent_map["m"].name
                    else -1
                    for r in range_tuple
                ]
            )[-1]
        )
        N = (
            self.func_info.gemm_extent_map["n"]
            if isinstance(self.func_info.gemm_extent_map["n"], tir.IntImm)
            else sorted(
                [
                    r.end
                    if r.var.name == self.func_info.gemm_extent_map["n"].name
                    else -1
                    for r in range_tuple
                ][-1]
            )
        )
        K = (
            self.func_info.gemm_extent_map["k"]
            if isinstance(self.func_info.gemm_extent_map["k"], tir.IntImm)
            else sorted(
                [
                    r.end
                    if r.var.name == self.func_info.gemm_extent_map["k"].name
                    else -1
                    for r in range_tuple
                ][-1]
            )
        )

        config.max_active_block_per_SM = int(
            min(
                self.arch.max_smem_per_sm // max(config.smem_usage, 1),
                self.arch.max_registers_per_sm // max(config.reg_usage, 1),
                self.arch.max_resident_threads_per_sm
                // (config.i * config.j * config.warp_size),
            )
        )

        tile_size_m = config.i * config.micro_shape_x * config.wmma_m
        tile_size_n = config.j * config.micro_shape_y * config.wmma_n
        tile_size_k = config.micro_shape_k * config.wmma_k

        wave_gpu = config.max_active_block_per_SM * self.arch.num_sm

        # jisuanliang = tile_size_m * tile_size_n * K *2 (乘加)
        # fancunliang = (tile_size_m + tile_size_n) * K * (DataType(self.func_info.in_dtype).bits+7//8)
        in_bytes = (DataType(self.func_info.in_dtype).bits + 7) // 8
        config.fma_ldg_ratio = (tile_size_m * tile_size_n * 2) / (
            (tile_size_m + tile_size_n) * in_bytes
        )

        a_ldg_request = wave_gpu * tile_size_m * K
        b_ldg_request = wave_gpu * tile_size_n * K

        wave_x = N // tile_size_n
        wave_y = wave_gpu // wave_x
        a_dram_ldg = (wave_y + 1) * tile_size_m * K
        b_dram_ldg = N * K

        l2_hit_rate = 1 - ((a_dram_ldg + b_dram_ldg) / (a_ldg_request + b_ldg_request))
        p_ldg = self.arch.bandwidth[1] * l2_hit_rate + self.arch.bandwidth[0] * (
            1 - l2_hit_rate
        )

        if config.fma_ldg_ratio < (
            self.arch.processing_power[3] * 1e12 / (p_ldg * 2e30)
        ):
            config.performance = p_ldg * config.fma_ldg_ratio
        else:
            config.performance = self.arch.processing_power[3] * 1e12

        config.num_wave = (
            np.prod(config.grid_size)
            + (config.max_active_block_per_SM * self.arch.compute_max_core)
            - 1
        ) // (config.max_active_block_per_SM * self.arch.compute_max_core)

        config.grid_size = [
            BSZ,
            (M + config.i * config.micro_shape_x * config.wmma_m - 1)
            // (config.i * config.micro_shape_x * config.wmma_m),
            (N + config.j * config.micro_shape_y * config.wmma_n - 1)
            // (config.j * config.micro_shape_y * config.wmma_n),
        ]
        config.block_size = [1, config.i * config.j, config.warp_size]

        """
        首先说明 wave 的概念:wave 表示 GPU 上同时执行的 thread block。
        例如一个 kernel 中 thread block 为 256 线程，每个线程使用了 128 个寄存器，
        那么在 GV100 上每个 SM 可同时执行 2 个 thread block,GV100 共 80 个 SM,一个 wave 就是 160 个 thread block。
        """
        # the small the better
        config.assign_score = 0
        if M % tile_size_m != 0:
            config.assign_score += tile_size_m - M % tile_size_m
        if N % tile_size_n != 0:
            config.assign_score += tile_size_n - N % tile_size_n
        if K % tile_size_k != 0:
            config.assign_score += tile_size_k - K % tile_size_k

    def check_config_valid(self, config: GEMMConfig, bytes, pad_offset) -> bool:
        self.estimate_smem_usage(config, bytes, pad_offset)
        if config.smem_usage >= self.arch.max_smem_per_block:
            return False
        self.estimate_registers_usage(config)
        if config.reg_usage > self.arch.max_registers_per_block:
            return False

    def score_config(self, config: GEMMConfig):
        # 先 (td.traffic + 1) * td.num_wave，再(r1, r2)
        n = np.prod(config.block_size)
        num_wrap = (n + self.arch.warp_size - 1) // self.arch.warp_size
        r1 = max(num_wrap / self.arch.sm_partition, self.arch.sm_partition / num_wrap)
        r2 = (num_wrap * self.arch.warp_size - n) / n

        return (config.assign_score, config.traffic, r1, r2)

    def get_max_tile_size(self, bytes, in_pad, stages: int):
        # return upperbound of tile_m+tile_n
        # note shared memory usage is equal to (tile_m + tile_n)*(tile_k + in_pad) * bytes * 2 (if use double buffer,stages=2)
        max_seme_usage = self.arch.max_smem_per_block
        tile_k = 128 // bytes
        return max_seme_usage // stages // bytes // (tile_k + in_pad)

    def generate_config_candidates(
        self, wmma_shape, use_async_copy=False, topk=20
    ) -> List[GEMMConfig]:
        use_tc = True
        config_candidates = []
        vec_len_candidates = self.plan_vectorize()
        in_bytes = (DataType(self.func_info.in_dtype).bits + 7) // 8
        in_pad = 8 if in_bytes == 2 else 16  # 8 for fp16, 16 for int8
        # TODO currently set out pad_offset = 4
        out_pad = 4

        use_double_buffer = True  # 因为TensorCore算力与Bandwith的比值较大，所以永远使用double buffer隐藏延迟

        if use_async_copy:
            smem_local_stage = False
            stages = 3
        else:
            smem_local_stage = True
            stages = 2

        max_tile_size = self.get_max_tile_size(in_bytes, in_pad, stages)
        tile_candidates = self.generate_tile_candidates(
            wmma_shape,
            (DataType(self.func_info.in_dtype).bits + 7) // 8,
            self.arch.max_threads_per_block,
            self.arch.warp_size,
            max_tile_size,
        )

        for tile_candidate in tile_candidates:
            for vec_len_candidate in vec_len_candidates:
                dic = {}
                dic["use_tc"] = use_tc
                dic["wmma_shape"] = wmma_shape
                dic["warp_size"] = self.arch.warp_size
                dic["i"] = tile_candidate[0]
                dic["j"] = tile_candidate[1]
                dic["micro_shape"] = tile_candidate[2:5]
                dic["in_vec_len_pad"] = (vec_len_candidate, in_pad)
                dic["out_vec_len_pad"] = (vec_len_candidate, out_pad)
                dic["manifest_shared_memory_local_stage"] = smem_local_stage
                dic["use_double_buffer"] = use_double_buffer
                dic["use_async_copy"] = use_async_copy
                config = GEMMConfig().from_dict(dic)
                if self.check_config_valid(config, in_bytes, in_pad):
                    config_candidates.append(config)

        print("changdu", len(config_candidates))
        return config_candidates

    # def emit_config(self, range_tuple:Tuple[Range], topk=20) ->  List[GEMMConfig]:
    #     if self.arch.compute_capability < "70":
    #         raise ValueError("This policy only support compute capability >= 7.0")
    #     elif self.arch.platform != "CUDA":
    #         raise ValueError("This policy only support CUDA platform")
    #     else:
    #         use_tc = True
    #         can_async_copy = False
    #         if self.arch.compute_capability > "80":
    #             can_async_copy = True
    #     for wmma_shape in self.func_info.wmma_shape_candidats:

    #     config_candidates = self.generate_config_candidates()
    #     range_config_dic = {}
