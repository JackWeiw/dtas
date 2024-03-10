from typing import List

import numpy as np
import tvm
from tvm import tir


def get_ref_tensor(shape: list, device: str, dtype: str) -> tvm.runtime.NDArray:
    # dtype = torch.__getattribute__(str(dtype))
    # if dtype.is_floating_point:
    if dtype == "float32" or dtype == "float16":
        return tvm.nd.array(np.random.uniform(0, 1, shape).astype(dtype), device=device)
    else:
        return tvm.nd.array(
            np.random.randint(0, 100, shape, dtype=dtype), device=device
        )


# ## 暂时没卵用
# def get_reference_output(
#     func: tir.PrimFunc, args, device="cuda:0", seed=0
# ) -> List[np.ndarray]:
#     torch.cuda.set_device(device)
#     torch.random.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     values = []
#     for tensor in args:
#         shape = list(map(int, tensor.shape))
#         arr = get_ref_tensor(shape, device, tensor.dtype)
#         arr = tvm.nd.array(arr.cpu().numpy())
#         values.append(arr)
#     schedule = tvm.te.create_schedule(args[-1].op)
#     mod = tvm.build(schedule, args, target="llvm")
#     mod(*values)
#     return values
