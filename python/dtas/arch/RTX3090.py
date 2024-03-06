import tvm

from .arch_base import Arch

# https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
class RTX3090(Arch):
    def __init__(self, dev_id=0):
        super().__init__()
        self.reg_cap = 65536
        # self.smem_cap = 49152
        self.smem_cap = 99 * 1024
        # self.compute_max_core = 82
        self.num_sm = 82
        self.max_resident_threads_per_sm = 1536
        self.max_resident_blocks_per_sm = 16
        self.max_registers_per_sm = 64 * 1024
        self.max_registers_per_block = 64 * 1024
        self.max_smem_per_sm = 100 * 1024
        self.max_smem_per_block = 99 * 1024
        self.max_threads_per_block = 1024
        self.warp_size = 32
        self.sm_partition = 4
        self.transaction_size = [32, 128]  # in bytes
        self.l2_cache_size = 6 * 1024
        # get from Nsight-compute
        self.bandwidth = [849, 1954, 54480]
        # https://en.wikipedia.org/wiki/GeForce_30_series
        self.processing_power = [29.28, 29.28, 0.458, 142]
        self.platform = "CUDA"
        self.compute_capability = "86"
        self.target = tvm.target.Target("nvidia/geforce-rtx-3090")

        device = tvm.runtime.cuda(dev_id)
        if not device.exist:
            raise RuntimeError(f"Cannot find cuda device {dev_id}.")
        self.device: tvm.runtime.Device = device
        self.cutlass_mma = [16, 8, 16]
