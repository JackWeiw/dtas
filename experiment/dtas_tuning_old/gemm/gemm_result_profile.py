import tvm
import numpy as np
import WTM
from WTM.codegen.runtime_packer import load_module_from_file
dtype = "float16"

dev = tvm.cuda(6)
rt_mod = load_module_from_file("./m_n2560_k10240_top20_256")
timer=rt_mod.time_evaluator(func_name="gemm", dev=dev, number=10,  min_repeat_ms=50 )

hidden_size = 2560
result = {}

for i in range(4096, 4097):
    x = tvm.nd.array(np.random.uniform(0, 1, (1, i, 4*hidden_size)).astype(dtype), dev)
    weight = tvm.nd.array(
        np.random.uniform(0, 1, (hidden_size, 4*hidden_size)).astype(dtype), dev
    )
    bias = tvm.nd.array(np.random.uniform(0, 1, (hidden_size,)).astype(dtype), dev)
    z = tvm.nd.array(np.random.uniform(0, 1, (1, i, hidden_size)).astype(dtype), dev)
    # rt_mod["gemm"](x, weight, bias, z)
    result[i] = timer(x, weight, bias, z).mean*1e6
    del x, weight, bias


def save_to_json(profile, filename):
    import json
    with open(filename, "w") as f:
        f.write(json.dumps(profile, indent=4))


save_to_json(
    result,
    "/home/weitao/XIAG8XX/profile/wtm_gemm/m_n2560_k10240/wtm_tuned.json",
)   