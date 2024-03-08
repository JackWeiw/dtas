import tvm
import numpy as np
import dtas
from dtas.codegen.runtime_packer import load_module_from_file
dtype = "float32"

dev = tvm.cuda(4)
rt_mod = load_module_from_file("/home/weitao/XIAG8XX/profile/dtas_tuned/general_reduction/layernorm/top10_256")
timer=rt_mod.time_evaluator(func_name="fused_layer_norm_cast1", dev=dev, number=5,  min_repeat_ms=50 )

hidden_size = 2560
result = {}
# bandwidth = {}
# if dtype =="int8":
#     byte = 1
# elif dtype == "float16":
#     byte = 2
# elif dtype == "float32":
#     byte = 4
     
for i in range(1, 4097):
    x = tvm.nd.array(np.random.uniform(0, 1, (1, i, hidden_size)).astype(dtype), dev)
    beta = tvm.nd.array(np.random.uniform(0, 1, (hidden_size)).astype(dtype), dev)
    gama = tvm.nd.array(np.random.uniform(0, 1, (hidden_size)).astype(dtype), dev)
    z = tvm.nd.array(np.random.uniform(0, 1, (1, i, hidden_size)).astype("float16"), dev)
    # rt_mod["gemm"](x, weight, bias, z)
    time = timer(x, beta, gama, z).mean
    # print(f"latency: {time} S, {byte}" )
    result[i] = time * 1e6
    # print(time)
    # bandwidth[i] = i * hidden_size * 3 * byte / (1024*1024*1024)/ time 
    # print(f"bandwidth: {bandwidth} GB/S" )
    del x, beta, gama, z


def save_to_json(profile, filename):
    import json
    with open(filename, "w") as f:
        f.write(json.dumps(profile, indent=4))


save_to_json(
    result,
    "/home/weitao/XIAG8XX/profile/dtas_tuned/general_reduction/layernorm/top10_256/laytency.json",
)   
