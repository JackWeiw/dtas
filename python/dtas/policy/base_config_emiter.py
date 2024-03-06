from ..common.analisys import FuncInfo
from ..arch import Arch


class BaseConfigEmiter:
    def __init__(self, func_info: FuncInfo, arch: Arch) -> None:
        self.func_info = func_info
        self.arch = arch

    def estimate_smem_usage(self):
        # to eliminate BANK CONFILICT
        raise NotImplementedError

    def estimate_registers_usage(self):
        raise NotImplementedError

    def generate_tile_candidates(self):
        raise NotImplementedError

    def plan_vectorize(self):
        raise NotImplementedError

    def generate_config_candidates(self):
        raise NotImplementedError

    def emit_config(self, range_tuple, topk=20):
        raise NotImplementedError
