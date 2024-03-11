import tvm
import numpy as np
import dtas
from dtas.codegen.runtime_packer import load_module_from_file

dev = tvm.cuda(6)
rt_mod = load_module_from_file("/home/weitao/XIAG8XX/profile/dtas_tuned/general_reduction/softmax/top1_256")
timer=rt_mod.time_evaluator(func_name="fused_softmax_cast", dev=dev, number=10,  min_repeat_ms=50 )

hidden_size = 2560
result = {}
     
timer = rt_mod.time_evaluator("fused_softmax_cast", dev, number=1)
# n = 1000
import json
for n in range(1, 4097):
    x = tvm.nd.array(np.random.uniform(0, 1, (1, 1000, n)).astype("float32"), dev)
    z = tvm.nd.array(np.random.uniform(0, 1, (1, 1000, n)).astype("float16"), dev)                       
    # print(timer(x, z)) 
    # print(time)
    result[n] = timer(x, z).mean * 1000000
    del x, z


def save_to_json(profile, filename):
    import json
    with open(filename, "w") as f:
        f.write(json.dumps(profile, indent=4))


save_to_json(
    result,
    "/home/weitao/XIAG8XX/profile/dtas_tuned/general_reduction/softmax/top1_256/latency.json",
)   
