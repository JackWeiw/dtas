from typing import Dict, Tuple
import regex as re

from .compile_result import CompileResult
from ..common.config import Range


def simplify(expr):
    e = str(expr)
    e = re.sub(r"T\.int64\((\d+|\w+)\)", r"\1", e)
    e = re.sub(r"//", "/", e)
    return e

def save_code(file, code):
    with open(file, "w") as f:
        f.write(code)

class CodeGenerator:
    """_Codegenerator_
        
    
    """
    def __init__(self, target) -> None:
        self.target = target
        self.kernel_handles = [
            "void* __tvm_module_ctx = NULL;\n",
            "static void* __tvm_set_device_packed = NULL;\n",
        ]
        self.host_forward_declares = []
        self.host_main_bodys = []
        self.device_forward_declares = []
        self.device_main_bodys = []
        self.kernel_info_dic = {}
        self.func_names = []

    def get_kernel_handle(self,func_name, host_code: str,range_tuple, multiple_range=True) -> str:
        """
        get
        static void* fused_NT_matmul_add_kernel_packed = NULL;
        in host_code
        """
        index0 = host_code.rindex("static void*")
        index1 = host_code.index("NULL;", index0)
        kernel_handle = host_code[index0 : index1 + 5] + "\n"
        if multiple_range:
            suffix = ""
            for r in range_tuple:
                suffix += r.to_suffix()
            kernel_handle = kernel_handle.replace(func_name, func_name + suffix)
        return kernel_handle 

    def get_host_forward_declare(self, host_code: str) -> str:
        """
        get
        #ifdef __cplusplus
        extern "C"
        #endif
        TVM_DLL int32_t fused_NT_matmul_add(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle);
        in host_code
        """
        index0 = host_code.index("#ifdef __cplusplus")
        index1 = host_code.index(";", index0)
        
        return host_code[index0 : index1 + 1] + "\n"

    def get_host_main_body_partial(self, host_code: str, len_args) -> str:
        index0 = host_code.rindex("#ifdef __cplusplus")
        index1 = host_code.rindex("stack_tcode)[{}".format(len_args - 1))
        index2 = host_code.index("(", index1)
        return host_code[index0:index2] + "\n"

    def get_host_main_body(self, host_code: str) -> str:
        index0 = host_code.rindex("#ifdef __cplusplus")
        return host_code[index0:] + "\n"

    def get_host_launch_args_str(self, host_code: str, len_args):
        """
        TODO : when tuning gemeral reduction <1024 cache, >1024 shared.dyn >4096 cache need to take extra control
        find
        (((TVMValue*)stack_value)[5].v_int64) = (int64_t)1;
        ((int32_t*)stack_tcode)[5] = 0;
        (((TVMValue*)stack_value)[6].v_int64) = ((n + (int64_t)127) >> (int64_t)7);
        ((int32_t*)stack_tcode)[6] = 0;
        (((TVMValue*)stack_value)[7].v_int64) = (int64_t)20;
        ((int32_t*)stack_tcode)[7] = 0;
        (((TVMValue*)stack_value)[8].v_int64) = (int64_t)32;
        ((int32_t*)stack_tcode)[8] = 0;
        (((TVMValue*)stack_value)[9].v_int64) = (int64_t)16;
        ((int32_t*)stack_tcode)[9] = 0;
        (((TVMValue*)stack_value)[10].v_int64) = ((int64_t)73728);
        ((int32_t*)stack_tcode)[10] = 0;
        """
        index0 = host_code.rindex("stack_tcode)[{}".format(len_args - 1))
        index1 = host_code.index(";", index0)
        index4 = host_code.index("(", index1)
        
        index2 = host_code.rindex("*)stack_tcode)")
        index3 = host_code.index(";", index2)
        c = host_code[index4: index3+1].split("\n")
        stack_value_stmts = [
            c[i][: c[i].index("=") + 1] + " " for i in range(0, len(c), 2)
        ]
        launch_args = (
            "{"
            + ",".join([c[i][c[i].index("=") + 1 : -1] for i in range(0, len(c), 2)])
            + "}"
        )
        stack_tcode_stmts = [c[i + 1] + "\n" for i in range(0, len(c), 2)]
        # result=[c[i]+";"+c[i+1]+";" for i in range(0,len(c),2)]
        return stack_value_stmts, stack_tcode_stmts, launch_args

    def get_host_func_call_stmts(self, len_args) -> str:
        return """
  if (info.kernel_handle == NULL)
  {{
    if (TVMBackendGetFuncFromEnv(__tvm_module_ctx, info.name.c_str(), &info.kernel_handle) != 0)
    {{
      return -1;
    }}
  }}
  TVMValue ret_val_1;
  int ret_type_code_1;
  if (TVMFuncCall(info.kernel_handle, (TVMValue *)stack_value, (int *)stack_tcode, {}, &ret_val_1, &ret_type_code_1) != 0)
  {{
    return -1;
  }}
  return 0;
}}     
    """.format(
            len_args
        )

    def get_device_forward_declare_and_code(
        self, func_name, kernel_code: str, range_tuple, multiple_range=True
    ):

        index_forward_declare = kernel_code.index('extern "C"')
        index_forward_declare2 = kernel_code.index(";", index_forward_declare)
        forward_decalare = kernel_code[
            index_forward_declare : index_forward_declare2 + 2
        ]
        index_code = kernel_code.rindex('extern "C"')
        code = kernel_code[index_code:]
        if multiple_range:
            suffix = ""
            for r in range_tuple:
                suffix += r.to_suffix()
            forward_decalare = forward_decalare.replace(func_name, func_name + suffix)
            code = code.replace(func_name, func_name + suffix)
        return forward_decalare, code

    def gen_code_for_func(
        self, ranges_best_result, index_table, index_stmt, len_args
    ) -> str:
        # kernel_handle = ["static void* __tvm_set_device_packed = NULL;\n"]
        host_main_body = []
        if len(ranges_best_result) == 1:
            for range_tuple in ranges_best_result.keys():
                compile_result = ranges_best_result[range_tuple]
                (
                    host_code,
                    device_code,
                    kernel_info,
                ) = compile_result.get_code_and_kernel_info(self.target)
                
                self.kernel_info_dic.update(kernel_info)
                self.kernel_handles.append(self.get_kernel_handle(compile_result.name, host_code, range_tuple, False))
                self.host_forward_declares.append(
                    self.get_host_forward_declare(host_code)
                )
                self.host_main_bodys.append(self.get_host_main_body(host_code))
                (
                    _device_forward_declare,
                    _device_main_body,
                ) = self.get_device_forward_declare_and_code(
                    compile_result.name, device_code, range_tuple, False
                )
                self.device_forward_declares.append(_device_forward_declare)
                self.device_main_bodys.append(_device_main_body)
                self.func_names.append(compile_result.name)
        else:
            host_main_body_partial = ""
            call_table = ["  kernel_entry_info call_table[] = {\n"]
            i = 0
            for range_tuple in ranges_best_result.keys():
                compile_result = ranges_best_result[range_tuple]
                (
                    host_code,
                    device_code,
                    kernel_info,
                ) = compile_result.get_code_and_kernel_info(self.target, range_tuple)
                self.kernel_info_dic.update(kernel_info)
                # save_code(f"host_code_{i}.cc", host_code)
                # save_code(f"device_code_{i}.cc", device_code)
                if i == 0:
                    i += 1
                    self.host_forward_declares.append(self.get_host_forward_declare(host_code))
                    host_main_body_partial = self.get_host_main_body_partial(
                        host_code, len_args
                    )
                    (
                        stack_value_stmts,
                        stack_tcode_stmts,
                        launch_args,
                    ) = self.get_host_launch_args_str(host_code, len_args)
                    len_stack_value_stmts = len(stack_value_stmts)
                    stack_value_stmt = "".join(
                        [
                            stack_value_stmts[j]
                            + "info.launch_args[{}];\n".format(j)
                            + stack_tcode_stmts[j]
                            for j in range(len(stack_value_stmts))
                        ]
                    )
                else:
                    (
                        _,
                        _,
                        launch_args,
                    ) = self.get_host_launch_args_str(host_code, len_args)    
                    
                suffix = ""
                for r in range_tuple:
                    suffix += r.to_suffix()
                call_table.append(
                    "    {"
                    + launch_args
                    + ', {}_kernel_packed, "{}_kernel"'.format(
                        compile_result.name+suffix, compile_result.name+suffix
                    )
                    + "},\n"
                )
                self.kernel_handles.append(self.get_kernel_handle(compile_result.name, host_code, range_tuple, True))

                (
                    _device_forward_declare,
                    _device_main_body,
                ) = self.get_device_forward_declare_and_code(
                    compile_result.name, device_code, range_tuple
                )
                self.device_forward_declares.append(_device_forward_declare)
                self.device_main_bodys.append(_device_main_body)

            index_table_stmt = (
                "  int64_t index_table[] = {"
                + ",".join([str(index) for index in index_table])
                + "};\n"
            )
            kernel_entry_stmt = (
                "  kernel_entry_info info = call_table[index_table[index]];\n"
            )
            call_table.append("     };\n")
            host_main_body = (
                host_main_body_partial
                + "".join(call_table)
                +index_stmt
                + index_table_stmt
                + kernel_entry_stmt
                + stack_value_stmt
                + self.get_host_func_call_stmts(len_args + len_stack_value_stmts)
            )
            self.host_main_bodys.append(host_main_body)
            self.func_names.append(compile_result.name)

    def save_to_file(self, filename, code):
        with open(filename, "w") as f:
            f.write(code)
