from __future__ import absolute_import as _abs
import os
import subprocess
import tempfile
import shlex
import warnings

import tvm
import tvm._ffi
from tvm._ffi import get_global_func
from tvm._ffi.base import py_str
from tvm.runtime import Module
from tvm.target import Target
from tvm.contrib import utils
from tvm.contrib.nvcc import get_target_compute_version
from tvm.contrib import cc
from .code_gen import CodeGenerator
from .header import (
    host_default_header,
    cuda_default_header,
    cuda_default_tail_header,
    cuda_fp16_header,
)
from ..logging import get_log_level, debug_info

_cuda_compile = get_global_func("tvm_callback_cuda_compile")
_load_cumodule = get_global_func("runtime.module.loadfile_cubin")
_CSourceModuleCreate = get_global_func("runtime.CSourceModuleCreate")
_load_dso = get_global_func("runtime.module.loadfile_so")
# _metal_module=get_global_func("runtime.module.create_metal_module")

kernel_entry_info = """ 
struct kernel_entry_info
{
  int64_t launch_args[7];
  void *kernel_handle;
  std::string name;
};

"""


class RuntimePacker:
    
    def __init__(
        self,
        kernel_handles,
        host_forward_declares,
        host_main_bodys,
        device_forward_declares,
        device_main_bodys,
        kernel_func_infos,
        func_names,
        work_dir = None,
        use_fp16=False,
        target_kind="cuda",
    ) -> None:
        self.kernel_handles = kernel_handles
        self.host_forward_declares = host_forward_declares
        self.host_main_bodys = host_main_bodys
        self.device_forward_declares = device_forward_declares
        self.device_main_bodys = device_main_bodys
        self.func_names = func_names
        self.use_fp16 = use_fp16
        self.metadata = {"tvm_version": "0.1.0", "func_info": kernel_func_infos}
        self.target_kind = target_kind
        if work_dir == None:
           work_dir = tempfile.TemporaryDirectory().name
        os.makedirs(work_dir, exist_ok=True)
        self.work_dir = work_dir
        if get_log_level()>=1: debug_info(f"work_dir for compile: {self.work_dir}")
        
    def pack_to_cuda_runtime(
        self,
        device_code,
        target_format="cubin",
        arch=None,
        options=None,
    ) -> Module:
        assert _cuda_compile != None, "need tvm built with cuda"
        assert device_code != None, "need device code"
        """Compile cuda code with NVCC from env.

        Parameters
        ----------
        code : str
            The cuda code.

        target_format : str
            The target format of nvcc compiler.

        arch : str
            The cuda architecture.

        options : str or list of str
            The additional options.

        path_target : str, optional
            Output file.

        Return
        ------
        CudaModule : tvm.runtime.Module
            The cuda device runtime Module.
        """
        if arch is None:
            # If None, then it will use `tvm.target.Target.current().arch`.
            # Target arch could be a str like "sm_xx", or a list, such as
            # [
            #   "-gencode", "arch=compute_52,code=sm_52",
            #   "-gencode", "arch=compute_70,code=sm_70"
            # ]
            compute_version = "".join(
                get_target_compute_version(Target.current(allow_none=True)).split(".")
            )
            arch = [
                "-gencode",
                f"arch=compute_{compute_version},code=sm_{compute_version}",
            ]

        

        

        if target_format not in ["cubin", "ptx", "fatbin"]:
            raise ValueError("target_format must be in cubin, ptx, fatbin")
        
        device_code_path = os.path.join(self.work_dir, "tvm_device_kernels.cu")
        meta_data_path = os.path.join(self.work_dir, "tvm_device.tvm_meta.json")
        device_lib_path = os.path.join(self.work_dir,f"tvm_device.{target_format}")
        
        pass_context = tvm.get_global_func("transform.GetCurrentPassContext")()

        with open(device_code_path, "w") as file:
            file.write(device_code)

        import json

        with open(meta_data_path, "w") as json_file:
            json.dump(self.metadata, json_file, indent=2)

        cmd = ["nvcc"]
        cmd += [f"--{target_format}", "-O3"]
        # if kernels_output_dir is not None:
        #     cmd += ["-lineinfo"]
        if isinstance(arch, list):
            cmd += arch
        elif isinstance(arch, str):
            cmd += ["-arch", arch]

        if options:
            if isinstance(options, str):
                cmd += [options]
            elif isinstance(options, list):
                cmd += options
            else:
                raise ValueError("options must be str or list of str")

        cmd += ["-o", device_lib_path]
        cmd += [device_code_path]
        if get_log_level() >= 1:
            debug_info("nvcc command: " + " ".join(cmd))
        # NOTE: ccbin option can be used to tell nvcc where to find the c++ compiler
        # just in case it is not in the path. On Windows it is not in the path by default.
        # However, we cannot use TVM_CXX_COMPILER_PATH because the runtime env.
        # Because it is hard to do runtime compiler detection, we require nvcc is configured
        # correctly by default.
        # if cxx_compiler_path != "":
        #    cmd += ["-ccbin", cxx_compiler_path]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        (out, _) = proc.communicate()

        if proc.returncode != 0:
            msg = device_code
            msg += "\nCompilation error:\n"
            msg += py_str(out)
            raise RuntimeError(msg)

        CudaModule = _load_cumodule(device_lib_path, target_format)
        return CudaModule

    # def _pack_to_metal_runtime(self)->Module:
    #     return _metal_module(self.host_code,device_code)

    def pack_to_host_runtime(self, host_code) -> Module:
        if get_log_level() >= 2:
            debug_info(f"host_func_name: {self.func_names}")
            
        host_code_path = os.path.join(self.work_dir, "host.cc")
        host_so_path = os.path.join(self.work_dir,"host.so")
        with open(host_code_path, "w") as file:
            file.write(host_code)
            # 获取临时文件的路径
            
        TVM_HOME = os.environ.get("TVM_HOME")
        if TVM_HOME is None:
            raise RuntimeError(
                "TVM_HOME is not set, please set TVM_HOME to the root of TVM project"
            )

        include_path = (
            " -I"
            + os.path.join(TVM_HOME, "include")
            + " -I"
            + os.path.join(TVM_HOME, "3rdparty/dlpack/include")
        )
        libtvm_path = " -L" + os.path.join(TVM_HOME, "build")
        option = include_path + libtvm_path + " -ltvm -ltvm_runtime"
        options = shlex.split(option)
        cc.create_shared(host_so_path, host_code_path, options=options)
        host_rt_mod = _load_dso(host_so_path)
        return host_rt_mod

    def get_host_code(self):
        return (
            host_default_header
            + kernel_entry_info
            + "".join(self.kernel_handles)
            + "".join(self.host_forward_declares)
            + "".join(self.host_main_bodys)
        )

    def get_device_code(self):
        device_header = cuda_default_header
        if self.use_fp16:
            device_header += cuda_fp16_header
        device_header += cuda_default_tail_header
        device_code = (
            device_header
            + "".join(self.device_forward_declares)
            + "".join(self.device_main_bodys)
        )
        return device_code

    def pack_to_tvm_runtime(
        self,
    ) -> Module:
        """A function to pack the host and device code to tvm runtime module

        Returns
        -------
        Module
            tvm.runtime.Module
        """        
        host_code = self.get_host_code()
        host_rt_module = self.pack_to_host_runtime(host_code)

        if self.target_kind == "cuda":
            print("device")
            device_code = self.get_device_code()
            device_rt_module = self.pack_to_cuda_runtime(
                device_code=device_code,
                target_format="cubin",
            )
        host_rt_module.import_module(device_rt_module)
        return host_rt_module

def load_module_from_file(work_dir):
    """load .so and .cubin file from the specified directory and pack it to tvm runtime module

    Parameters
    ----------
    work_dir : str
        dtas working directory

    Returns
    -------
        tvm.runtime.Module
    """    
    if not os.path.exists(work_dir):
        raise FileNotFoundError(f"Directory {work_dir} not found.")
    host_so_path = os.path.join(work_dir, "host.so")
    device_lib_path = os.path.join(work_dir, "tvm_device.cubin")

    if not (os.path.exists(host_so_path) and os.path.exists(device_lib_path)):
        raise FileNotFoundError("host.so or tvm_device.cubin not found in the specified directory.")

    host_rt_module = _load_dso(host_so_path)
    device_rt_module = _load_cumodule(device_lib_path, "cubin")
    host_rt_module.import_module(device_rt_module)
    return host_rt_module