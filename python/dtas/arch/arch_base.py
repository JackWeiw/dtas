import tvm
class Arch:
    def __init__(self) -> None:
        self.reg_cap = 0
        self.smem_cap = 0
        self.compute_max_core = 0

        self.num_sm = 0
        self.max_resident_threads_per_sm = 0
        self.max_resident_blocks_per_sm = 0
        self.max_registers_per_sm = 0
        self.max_registers_per_block = 0
        self.max_smem_per_sm = 0
        self.max_smem_per_block = 0
        self.max_threads_per_block = 0
        self.warp_size = 0
        self.sm_partition: int = 0
        self.transaction_size = [0, 0]
        # half, single, double, tc in TFLOPS
        self.processing_power = [0, 0, 0, 0]
        # bandwidth in GB/S, global, L2 cache
        self.bandwidth = [0, 0]
        self.l2_cache_size = 0
        
        self.device: tvm.runtime.Device = None
        self.platform = "unknown"
        self.compute_capability = "unknown"
