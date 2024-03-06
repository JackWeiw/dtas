from typing import Dict, Tuple

from typing_extensions import Literal

from tvm import tir
from tvm.tir.tensor_intrin.cuda import *
from tvm.tir.function import TensorIntrin

# wmma_load_intrin
# shape 32*8*16
WMMA_LOAD_32x8x16_F16_A_INTRIN = "wmma_load_32x8x16_f16_a_shared"
TensorIntrin.register(
    WMMA_LOAD_32x8x16_F16_A_INTRIN,
    *get_wmma_load_intrin(32, 8, 16, "float16", "shared", False, False),
)

WMMA_LOAD_32x8x16_F16_A_DYN_INTRIN = "wmma_load_32x8x16_f16_a_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_32x8x16_F16_A_DYN_INTRIN,
    *get_wmma_load_intrin(32, 8, 16, "float16", "shared.dyn", False, False),
)

WMMA_LOAD_32x8x16_F16_B_INTRIN = "wmma_load_32x8x16_f16_b_shared"
TensorIntrin.register(
    WMMA_LOAD_32x8x16_F16_B_INTRIN,
    *get_wmma_load_intrin(32, 8, 16, "float16", "shared", True, False),
)

WMMA_LOAD_32x8x16_F16_B_DYN_INTRIN = "wmma_load_32x8x16_f16_b_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_32x8x16_F16_B_DYN_INTRIN,
    *get_wmma_load_intrin(32, 8, 16, "float16", "shared.dyn", True, False),
)

WMMA_LOAD_32x8x16_F16_A_TRANS_INTRIN = "wmma_load_32x8x16_f16_a_trans_shared"
TensorIntrin.register(
    WMMA_LOAD_32x8x16_F16_A_TRANS_INTRIN,
    *get_wmma_load_intrin(32, 8, 16, "float16", "shared", False, True),
)

WMMA_LOAD_32x8x16_F16_A_TRANS_DYN_INTRIN = "wmma_load_32x8x16_f16_a_trans_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_32x8x16_F16_A_TRANS_DYN_INTRIN,
    *get_wmma_load_intrin(32, 8, 16, "float16", "shared.dyn", False, True),
)

WMMA_LOAD_32x8x16_F16_B_TRANS_INTRIN = "wmma_load_32x8x16_f16_b_trans_shared"
TensorIntrin.register(
    WMMA_LOAD_32x8x16_F16_B_TRANS_INTRIN,
    *get_wmma_load_intrin(32, 8, 16, "float16", "shared", True, True),
)

WMMA_LOAD_32x8x16_F16_B_TRANS_DYN_INTRIN = "wmma_load_32x8x16_f16_b_trans_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_32x8x16_F16_B_TRANS_DYN_INTRIN,
    *get_wmma_load_intrin(32, 8, 16, "float16", "shared.dyn", True, True),
)

WMMA_LOAD_32x8x16_S8_A_INTRIN = "wmma_load_32x8x16_s8_a_shared"
TensorIntrin.register(
    WMMA_LOAD_32x8x16_S8_A_INTRIN,
    *get_wmma_load_intrin(32, 8, 16, "int8", "shared", False, False),
)

WMMA_LOAD_32x8x16_S8_A_DYN_INTRIN = "wmma_load_32x8x16_s8_a_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_32x8x16_S8_A_DYN_INTRIN,
    *get_wmma_load_intrin(32, 8, 16, "int8", "shared.dyn", False, False),
)

WMMA_LOAD_32x8x16_S8_B_INTRIN = "wmma_load_32x8x16_s8_b_shared"
TensorIntrin.register(
    WMMA_LOAD_32x8x16_S8_B_INTRIN,
    *get_wmma_load_intrin(32, 8, 16, "int8", "shared", True, False),
)

WMMA_LOAD_32x8x16_S8_B_DYN_INTRIN = "wmma_load_32x8x16_s8_b_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_32x8x16_S8_B_DYN_INTRIN,
    *get_wmma_load_intrin(32, 8, 16, "int8", "shared.dyn", True, False),
)

WMMA_LOAD_32x8x16_S8_A_TRANS_INTRIN = "wmma_load_32x8x16_s8_a_trans_shared"
TensorIntrin.register(
    WMMA_LOAD_32x8x16_S8_A_TRANS_INTRIN,
    *get_wmma_load_intrin(32, 8, 16, "int8", "shared", False, True),
)

WMMA_LOAD_32x8x16_S8_A_TRANS_DYN_INTRIN = "wmma_load_32x8x16_s8_a_trans_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_32x8x16_S8_A_TRANS_DYN_INTRIN,
    *get_wmma_load_intrin(32, 8, 16, "int8", "shared.dyn", False, True),
)

WMMA_LOAD_32x8x16_S8_B_TRANS_INTRIN = "wmma_load_32x8x16_s8_b_trans_shared"
TensorIntrin.register(
    WMMA_LOAD_32x8x16_S8_B_TRANS_INTRIN,
    *get_wmma_load_intrin(32, 8, 16, "int8", "shared", True, True),
)

WMMA_LOAD_32x8x16_S8_B_TRANS_DYN_INTRIN = "wmma_load_32x8x16_s8_b_trans_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_32x8x16_S8_B_TRANS_DYN_INTRIN,
    *get_wmma_load_intrin(32, 8, 16, "int8", "shared.dyn", True, True),
)

# shape 8*32*16
WMMA_LOAD_8x32x16_F16_A_INTRIN = "wmma_load_8x32x16_f16_a_shared"
TensorIntrin.register(
    WMMA_LOAD_8x32x16_F16_A_INTRIN,
    *get_wmma_load_intrin(8, 32, 16, "float16", "shared", False, False),
)

WMMA_LOAD_8x32x16_F16_A_DYN_INTRIN = "wmma_load_8x32x16_f16_a_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_8x32x16_F16_A_DYN_INTRIN,
    *get_wmma_load_intrin(8, 32, 16, "float16", "shared.dyn", False, False),
)

WMMA_LOAD_8x32x16_F16_B_INTRIN = "wmma_load_8x32x16_f16_b_shared"
TensorIntrin.register(
    WMMA_LOAD_8x32x16_F16_B_INTRIN,
    *get_wmma_load_intrin(8, 32, 16, "float16", "shared", True, False),
)

WMMA_LOAD_8x32x16_F16_B_DYN_INTRIN = "wmma_load_8x32x16_f16_b_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_8x32x16_F16_B_DYN_INTRIN,
    *get_wmma_load_intrin(8, 32, 16, "float16", "shared.dyn", True, False),
)

WMMA_LOAD_8x32x16_F16_A_TRANS_INTRIN = "wmma_load_8x32x16_f16_a_trans_shared"
TensorIntrin.register(
    WMMA_LOAD_8x32x16_F16_A_TRANS_INTRIN,
    *get_wmma_load_intrin(8, 32, 16, "float16", "shared", False, True),
)

WMMA_LOAD_8x32x16_F16_A_TRANS_DYN_INTRIN = "wmma_load_8x32x16_f16_a_trans_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_8x32x16_F16_A_TRANS_DYN_INTRIN,
    *get_wmma_load_intrin(8, 32, 16, "float16", "shared.dyn", False, True),
)

WMMA_LOAD_8x32x16_F16_B_TRANS_INTRIN = "wmma_load_8x32x16_f16_b_trans_shared"
TensorIntrin.register(
    WMMA_LOAD_8x32x16_F16_B_TRANS_INTRIN,
    *get_wmma_load_intrin(8, 32, 16, "float16", "shared", True, True),
)

WMMA_LOAD_8x32x16_F16_B_TRANS_DYN_INTRIN = "wmma_load_8x32x16_f16_b_trans_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_8x32x16_F16_B_TRANS_DYN_INTRIN,
    *get_wmma_load_intrin(8, 32, 16, "float16", "shared.dyn", True, True),
)

WMMA_LOAD_8x32x16_S8_A_INTRIN = "wmma_load_8x32x16_s8_a_shared"
TensorIntrin.register(
    WMMA_LOAD_8x32x16_S8_A_INTRIN,
    *get_wmma_load_intrin(8, 32, 16, "int8", "shared", False, False),
)

WMMA_LOAD_8x32x16_S8_A_DYN_INTRIN = "wmma_load_8x32x16_s8_a_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_8x32x16_S8_A_DYN_INTRIN,
    *get_wmma_load_intrin(8, 32, 16, "int8", "shared.dyn", False, False),
)

WMMA_LOAD_8x32x16_S8_B_INTRIN = "wmma_load_8x32x16_s8_b_shared"
TensorIntrin.register(
    WMMA_LOAD_8x32x16_S8_B_INTRIN,
    *get_wmma_load_intrin(8, 32, 16, "int8", "shared", True, False),
)

WMMA_LOAD_8x32x16_S8_B_DYN_INTRIN = "wmma_load_8x32x16_s8_b_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_8x32x16_S8_B_DYN_INTRIN,
    *get_wmma_load_intrin(8, 32, 16, "int8", "shared.dyn", True, False),
)

WMMA_LOAD_8x32x16_S8_A_TRANS_INTRIN = "wmma_load_8x32x16_s8_a_trans_shared"
TensorIntrin.register(
    WMMA_LOAD_8x32x16_S8_A_TRANS_INTRIN,
    *get_wmma_load_intrin(8, 32, 16, "int8", "shared", False, True),
)

WMMA_LOAD_8x32x16_S8_A_TRANS_DYN_INTRIN = "wmma_load_8x32x16_s8_a_trans_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_8x32x16_S8_A_TRANS_DYN_INTRIN,
    *get_wmma_load_intrin(8, 32, 16, "int8", "shared.dyn", False, True),
)

WMMA_LOAD_8x32x16_S8_B_TRANS_INTRIN = "wmma_load_8x32x16_s8_b_trans_shared"
TensorIntrin.register(
    WMMA_LOAD_8x32x16_S8_B_TRANS_INTRIN,
    *get_wmma_load_intrin(8, 32, 16, "int8", "shared", True, True),
)

WMMA_LOAD_8x32x16_S8_B_TRANS_DYN_INTRIN = "wmma_load_8x32x16_s8_b_trans_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_8x32x16_S8_B_TRANS_DYN_INTRIN,
    *get_wmma_load_intrin(8, 32, 16, "int8", "shared.dyn", True, True),
)

# wmma fill intrin
# shape32*8*16
WMMA_FILL_32x8x16_F32_INTRIN = "wmma_fill_32x8x16_f32"
TensorIntrin.register(
    WMMA_FILL_32x8x16_F32_INTRIN, *get_wmma_fill_intrin(32, 8, 16, "float32")
)

WMMA_FILL_32x8x16_F16_INTRIN = "wmma_fill_32x8x16_f16"
TensorIntrin.register(
    WMMA_FILL_32x8x16_F16_INTRIN, *get_wmma_fill_intrin(32, 8, 16, "float16")
)

WMMA_FILL_32x8x16_S32_INTRIN = "wmma_fill_32x8x16_s32"
TensorIntrin.register(
    WMMA_FILL_32x8x16_S32_INTRIN, *get_wmma_fill_intrin(32, 8, 16, "int32")
)
# shape 8*32*16
WMMA_FILL_8x32x16_F32_INTRIN = "wmma_fill_8x32x16_f32"
TensorIntrin.register(
    WMMA_FILL_8x32x16_F32_INTRIN, *get_wmma_fill_intrin(8, 32, 16, "float32")
)

WMMA_FILL_8x32x16_F16_INTRIN = "wmma_fill_8x32x16_f16"
TensorIntrin.register(
    WMMA_FILL_8x32x16_F16_INTRIN, *get_wmma_fill_intrin(8, 32, 16, "float16")
)

WMMA_FILL_8x32x16_S32_INTRIN = "wmma_fill_8x32x16_s32"
TensorIntrin.register(
    WMMA_FILL_8x32x16_S32_INTRIN, *get_wmma_fill_intrin(8, 32, 16, "int32")
)

# wmma stor intrin
# 32*8*16
WMMA_STORE_32x8x16_F32_SHARED_INTRIN = "wmma_store_32x8x16_f32_shared"
TensorIntrin.register(
    WMMA_STORE_32x8x16_F32_SHARED_INTRIN,
    *get_wmma_store_intrin(32, 8, 16, "float32", "shared"),
)

WMMA_STORE_32x8x16_F32_SHARED_DYN_INTRIN = "wmma_store_32x8x16_f32_shared_dyn"
TensorIntrin.register(
    WMMA_STORE_32x8x16_F32_SHARED_DYN_INTRIN,
    *get_wmma_store_intrin(32, 8, 16, "float32", "shared.dyn"),
)

WMMA_STORE_32x8x16_F16_SHARED_INTRIN = "wmma_store_32x8x16_f16_shared"
TensorIntrin.register(
    WMMA_STORE_32x8x16_F16_SHARED_INTRIN,
    *get_wmma_store_intrin(32, 8, 16, "float16", "shared"),
)

WMMA_STORE_32x8x16_F16_SHARED_DYN_INTRIN = "wmma_store_32x8x16_f16_shared_dyn"
TensorIntrin.register(
    WMMA_STORE_32x8x16_F16_SHARED_DYN_INTRIN,
    *get_wmma_store_intrin(32, 8, 16, "float16", "shared.dyn"),
)

WMMA_STORE_32x8x16_S32_SHARED_INTRIN = "wmma_store_32x8x16_s32_shared"
TensorIntrin.register(
    WMMA_STORE_32x8x16_S32_SHARED_INTRIN,
    *get_wmma_store_intrin(32, 8, 16, "int32", "shared"),
)

WMMA_STORE_32x8x16_S32_SHARED_DYN_INTRIN = "wmma_store_32x8x16_s32_shared_dyn"
TensorIntrin.register(
    WMMA_STORE_32x8x16_S32_SHARED_DYN_INTRIN,
    *get_wmma_store_intrin(32, 8, 16, "int32", "shared.dyn"),
)

# shape 8*32*16
WMMA_STORE_8x32x16_F32_SHARED_INTRIN = "wmma_store_8x32x16_f32_shared"
TensorIntrin.register(
    WMMA_STORE_8x32x16_F32_SHARED_INTRIN,
    *get_wmma_store_intrin(8, 32, 16, "float32", "shared"),
)

WMMA_STORE_8x32x16_F32_SHARED_DYN_INTRIN = "wmma_store_8x32x16_f32_shared_dyn"
TensorIntrin.register(
    WMMA_STORE_8x32x16_F32_SHARED_DYN_INTRIN,
    *get_wmma_store_intrin(8, 32, 16, "float32", "shared.dyn"),
)

WMMA_STORE_8x32x16_F16_SHARED_INTRIN = "wmma_store_8x32x16_f16_shared"
TensorIntrin.register(
    WMMA_STORE_8x32x16_F16_SHARED_INTRIN,
    *get_wmma_store_intrin(8, 32, 16, "float16", "shared"),
)

WMMA_STORE_8x32x16_F16_SHARED_DYN_INTRIN = "wmma_store_8x32x16_f16_shared_dyn"
TensorIntrin.register(
    WMMA_STORE_8x32x16_F16_SHARED_DYN_INTRIN,
    *get_wmma_store_intrin(8, 32, 16, "float16", "shared.dyn"),
)

WMMA_STORE_8x32x16_S32_SHARED_INTRIN = "wmma_store_8x32x16_s32_shared"
TensorIntrin.register(
    WMMA_STORE_8x32x16_S32_SHARED_INTRIN,
    *get_wmma_store_intrin(8, 32, 16, "int32", "shared"),
)

WMMA_STORE_8x32x16_S32_SHARED_DYN_INTRIN = "wmma_store_8x32x16_s32_shared_dyn"
TensorIntrin.register(
    WMMA_STORE_8x32x16_S32_SHARED_DYN_INTRIN,
    *get_wmma_store_intrin(8, 32, 16, "int32", "shared.dyn"),
)

# wmma_sync_intrin
WMMA_SYNC_32x8x16_f16f16f32_INTRIN = "wmma_sync_32x8x16_f16f16f32"
TensorIntrin.register(
    WMMA_SYNC_32x8x16_f16f16f32_INTRIN,
    *get_wmma_sync_intrin(32, 8, 16, "float16", "float32", False),
)

WMMA_SYNC_32x8x16_f16f16f16_INTRIN = "wmma_sync_32x8x16_f16f16f16"
TensorIntrin.register(
    WMMA_SYNC_32x8x16_f16f16f16_INTRIN,
    *get_wmma_sync_intrin(32, 8, 16, "float16", "float16", False),
)

WMMA_SYNC_8x32x16_f16f16f32_INTRIN = "wmma_sync_8x32x16_f16f16f32"
TensorIntrin.register(
    WMMA_SYNC_8x32x16_f16f16f32_INTRIN,
    *get_wmma_sync_intrin(8, 32, 16, "float16", "float32", False),
)

WMMA_SYNC_8x32x16_f16f16f16_INTRIN = "wmma_sync_8x32x16_f16f16f16"
TensorIntrin.register(
    WMMA_SYNC_8x32x16_f16f16f16_INTRIN,
    *get_wmma_sync_intrin(8, 32, 16, "float16", "float16", False),
)


def get_wmma_intrin_group_diy(
    shape: Literal["16x16x16", "32x8x16", "8x32x16", "8x8x32"],
    load_scope: Literal["shared", "shared.dyn"],
    store_scope: Literal["global", "shared", "shared.dyn"],
    in_dtype: str,
    out_dtype: str,
    trans_a: bool,
    trans_b: bool,
) -> Dict[str, str]:
    """Get a group of intrinsics for wmma tensor core with the given configurations

    Parameters
    ----------
    load_scope : Literal["shared", "shared.dyn"]
        The memory scope of the input buffer.

    store_scope : Literal["global", "shared", "shared.dyn"]
        The memory scope of the result buffer.

    in_dtype : str
        The input data type.

    out_dtype : str
        The output data dtype.

    trans_b : bool
        Whether the input matrix B is transposed.

    Returns
    -------
    ret : Dict[str, str]
        A group of tensor intrinsics.
    """
    assert shape in ["16x16x16", "32x8x16", "8x32x16", "8x8x32"]
    assert load_scope in ["shared", "shared.dyn"]
    assert store_scope in ["global", "shared", "shared.dyn"]
    assert in_dtype in ["float16", "int8"]
    assert out_dtype in ["float16", "float32", "int32"]
    dtype_map = {
        "int32": "s32",
        "float32": "f32",
        "float16": "f16",
        "int8": "s8",
        "int4": "s4",
    }

    in_dtype = dtype_map[in_dtype]
    out_dtype = dtype_map[out_dtype]
    # convert "shared.dyn" to "shared_dyn"
    load_scope = load_scope.replace(".", "_")
    store_scope = store_scope.replace(".", "_")
    trans_a = "_trans" if trans_a else ""
    trans_b = "_trans" if trans_b else ""

    # e.g. wmma_load_16x16x16_f16_a_shared
    load_a_intrin = f"wmma_load_{shape}_{in_dtype}_a{trans_a}_{load_scope}"
    # e.g. wmma_load_16x16x16_f16_b_trans_shared_dyn
    load_b_intrin = f"wmma_load_{shape}_{in_dtype}_b{trans_b}_{load_scope}"
    # e.g. wmma_sync_16x16x16_f16f16f32_trans
    compute_intrin = f"wmma_sync_{shape}_{in_dtype}{in_dtype}{out_dtype}{trans_b}"
    # e.g. wmma_fill_16x16x16_f16
    init_intrin = f"wmma_fill_{shape}_{out_dtype}"
    # e.g. wmma_store_16x16x16_f16_shared_dyn
    store_intrin = f"wmma_store_{shape}_{out_dtype}_{store_scope}"

    return {
        "init": init_intrin,
        "load_a": load_a_intrin,
        "load_b": load_b_intrin,
        "compute": compute_intrin,
        "store": store_intrin,
    }
