from typing import Tuple

from ..common.config import Config, Range
from ..common.analisys import FuncKind
from .tir_base import TIRSchedulerBase
from .tir_gemv import TIRGEMVScheduler
from .tir_matmul import TIRMatmulScheduler
from .tir_wmma import TIRWMMAScheduler
from .tir_elementwise import TIRElementwiseScheduler
from .tir_reduction import TIRReductionScheduler
from .tir_mma import TIRMMAScheduler
from ..policy import (
    BaseConfigEmiter,
    GEMVConfigEmiter,
    WMMAConfigEmiter,
    ElementwiseConfigEmiter,
    ReductionConfigEmiter,
)
from ..logging import get_log_level, debug_info

# gemm_template = []
# # ,TIRMatmulScheduler
# _scheduler_map = {
#     "GEMM": TIRWMMAScheduler,
#     "wmma": TIRWMMAScheduler,
# }

# func_kind_map = {
#     FuncKind.kFunc_GEMM: "GEMM",
#     FuncKind.kFunc_GEMV: "GEMV",
#     FuncKind.kFunc_Reduction: "Reduction",
#     FuncKind.kFunc_Elementwise: "Elementwise",
# }


def get_scheduler_template(func_kind: FuncKind, use_tc: bool, sm:str) -> TIRSchedulerBase:
    if func_kind == FuncKind.kFunc_GEMM:
        if use_tc:
            ## TODO : # For A100(sm_80) or more advanced gpu, use MMA tensorization.
            if sm >= "80":
                if get_log_level() >= 1:
                    debug_info("scheduler template: MMAScheduler")
                return TIRWMMAScheduler
            else:
                if get_log_level() >= 2:
                    debug_info("scheduler template: WMMAScheduler")
                return TIRWMMAScheduler
        else:
            ## # For other GPUs, use WMMA tensorization.
            if get_log_level() >= 2:
                debug_info("scheduler template: MatmulScheduler")
            return TIRMatmulScheduler
    elif func_kind == FuncKind.kFunc_GEMV:
        if get_log_level() >= 2:
            debug_info("scheduler template: GEMVScheduler")
        return TIRGEMVScheduler
    elif func_kind == FuncKind.kFunc_Reduction:
        if get_log_level() >= 2:
            debug_info("scheduler template: ReductionScheduler")
        return TIRReductionScheduler
    elif func_kind == FuncKind.kFunc_Elementwise:
        if get_log_level() >= 2:
            debug_info("scheduler template: ElementwiseScheduler")
        return TIRElementwiseScheduler
    else:
        return TIRSchedulerBase


def get_config_emiter_template(func_kind: FuncKind, use_tc: bool) -> BaseConfigEmiter:
    if func_kind == FuncKind.kFunc_GEMM:
        if use_tc:
            if get_log_level() >= 1:
                debug_info("config emiter template: WMMAConfigEmiter")
            return WMMAConfigEmiter
        else:
            if get_log_level() >= 1:
                debug_info("config emiter template: WMMAConfigEmiter")
            return WMMAConfigEmiter
    elif func_kind == FuncKind.kFunc_GEMV:
        if get_log_level() >= 1:
            debug_info("config emiter template: GEMVConfigEmiter")
        return GEMVConfigEmiter
    elif func_kind == FuncKind.kFunc_Reduction:
        if get_log_level() >= 1:
            debug_info("config emiter template: ReductionConfigEmiter")
        return ReductionConfigEmiter
    elif func_kind == FuncKind.kFunc_Elementwise:
        if get_log_level() >= 1:
            debug_info("config emiter template: ElementwiseConfigEmiter")
        return ElementwiseConfigEmiter
    else:
        return BaseConfigEmiter


