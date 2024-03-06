import numpy as np

import tvm
from tvm import relax as rx
from tvm.ir.module import IRModule
from tvm.target import Target
from tvm._ffi import get_global_func


class CompileResult:
    """
    Class to store the result of compilation
    """

    def __init__(self, name, config, sch, rt_mod):
        self.name = name
        self.config = config
        self.sch = sch
        self.rt_mod = rt_mod
        self.latency = 1e9
        self.profile_tensors = []
        self.time_evaluator = None

    def profile(self):
        """
        根据range 均匀分割取得的profile tensors
        TODO 是否有normalize的方法?
        """
        try:
            time = []
            for tensors in self.profile_tensors:
                time.append(self.time_evaluator(*tensors).mean * 1e3)
            self.profile_tensors = None # cancel reference to free memory
            return np.mean(time)
        except:
            self.profile_tensors = None
            return 1e9
        
    def get_code_and_kernel_info(self, target, range_tuple=None):
        with tvm.transform.PassContext(
            disabled_pass=["tir.AnnotateEntryFunc"],
            config={"tir.use_async_copy": True},
        ):
            target = Target(target, "c")
            # mod = tvm.IRModule({"main": self.sch.mod["main"]})
            # mod = rx.transform.AttachGlobalSymbol()(mod)
            mod = tvm.lower(self.sch.mod["main"])
            mixed_mod_passes = get_global_func("driver.mixed_mod_passes")(mod, target)
            device_mod_passes = get_global_func("driver.device_mod_passes")(mod, target)
            mod = mixed_mod_passes(mod)
            mod = device_mod_passes(mod)
            suffix = ""
            if range_tuple != None:
                for r in range_tuple:
                    suffix += r.to_suffix()

            kernel_info_dic = {
                self.name
                + suffix
                + "_kernel": {
                    "name": self.name + suffix + "_kernel",
                    "arg_types": [param.dtype for param in mod["main_kernel"].params],
                    "launch_param_tags": [
                        tag
                        for tag in mod["main_kernel"].attrs["tir.kernel_launch_params"]
                    ],
                }
            }
            lib = tvm.build(
                self.sch.mod["main"], target=Target(target, "c"), name=self.name
            )
            host_code = lib.get_source()
            device_code = lib.imported_modules[0].get_source()
        return host_code, device_code, kernel_info_dic
