from typing import Dict, Tuple

from typing_extensions import Literal

from tvm import tir
from tvm.tir.tensor_intrin.cuda import *
from tvm.tir.function import TensorIntrin

MMA_store_16x16_f16_shared_dyn_INTRIN = "mma_store_16x16_f16_shared_dyn_"
TensorIntrin.register(
    MMA_store_16x16_f16_shared_dyn_INTRIN,
    *get_mma_store_intrin("float16", 8, "shared.dyn", True),
)

