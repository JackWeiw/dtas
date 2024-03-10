
// tvm target: c -keys=cpu
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#include <stdbool.h>
#include <string>
 
struct kernel_entry_info
{
  int64_t launch_args[7];
  void *kernel_handle;
  std::string name;
};

void* __tvm_module_ctx = NULL;
static void* __tvm_set_device_packed = NULL;
static void* fused_layer_norm_cast1_n_1_to_256__kernel_packed = NULL;
static void* fused_layer_norm_cast1_n_257_to_512__kernel_packed = NULL;
static void* fused_layer_norm_cast1_n_513_to_1280__kernel_packed = NULL;
static void* fused_layer_norm_cast1_n_1281_to_2816__kernel_packed = NULL;
static void* fused_layer_norm_cast1_n_2817_to_3072__kernel_packed = NULL;
static void* fused_layer_norm_cast1_n_3073_to_4096__kernel_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_layer_norm_cast1(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_layer_norm_cast1(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  TVMValue stack[5];
  void* stack_tcode = stack;
  TVMValue stack_1[9];
  void* stack_value = stack_1;
  int32_t p_lv6_code = arg_type_ids[0];
  int32_t param_1_handle_code = arg_type_ids[1];
  int32_t param_2_handle_code = arg_type_ids[2];
  int32_t p_output0_code = arg_type_ids[3];
  void* p_lv6 = (((TVMValue*)args)[0].v_handle);
  void* param_1_handle = (((TVMValue*)args)[1].v_handle);
  void* param_2_handle = (((TVMValue*)args)[2].v_handle);
  void* p_output0 = (((TVMValue*)args)[3].v_handle);
  void* lv6 = (((DLTensor*)p_lv6)[0].data);
  void* fused_layer_norm_cast1_p_lv6_shape = (((DLTensor*)p_lv6)[0].shape);
  int64_t n = ((int64_t*)fused_layer_norm_cast1_p_lv6_shape)[1];
  void* fused_layer_norm_cast1_p_lv6_strides = (((DLTensor*)p_lv6)[0].strides);
  int32_t dev_id = (((DLTensor*)p_lv6)[0].device.device_id);
  void* param_1 = (((DLTensor*)param_1_handle)[0].data);
  void* fused_layer_norm_cast1_param_1_handle_shape = (((DLTensor*)param_1_handle)[0].shape);
  void* fused_layer_norm_cast1_param_1_handle_strides = (((DLTensor*)param_1_handle)[0].strides);
  void* param_2 = (((DLTensor*)param_2_handle)[0].data);
  void* fused_layer_norm_cast1_param_2_handle_shape = (((DLTensor*)param_2_handle)[0].shape);
  void* fused_layer_norm_cast1_param_2_handle_strides = (((DLTensor*)param_2_handle)[0].strides);
  void* compute_intermediate = (((DLTensor*)p_output0)[0].data);
  void* fused_layer_norm_cast1_p_output0_shape = (((DLTensor*)p_output0)[0].shape);
  void* fused_layer_norm_cast1_p_output0_strides = (((DLTensor*)p_output0)[0].strides);
  if (!(fused_layer_norm_cast1_p_lv6_strides == NULL)) {
  }
  if (!(fused_layer_norm_cast1_param_1_handle_strides == NULL)) {
  }
  if (!(fused_layer_norm_cast1_param_2_handle_strides == NULL)) {
  }
  if (!(fused_layer_norm_cast1_p_output0_strides == NULL)) {
  }
  (((TVMValue*)stack_value)[0].v_int64) = ((int64_t)2);
  ((int32_t*)stack_tcode)[0] = 0;
  (((TVMValue*)stack_value)[1].v_int64) = ((int64_t)dev_id);
  ((int32_t*)stack_tcode)[1] = 0;
  if (__tvm_set_device_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_module_ctx, "__tvm_set_device", &__tvm_set_device_packed) != 0) {
      return -1;
    }
  }
  TVMValue ret_val;
  int ret_type_code;
  if (TVMFuncCall(__tvm_set_device_packed, (TVMValue*) stack_value, (int*) stack_tcode, 2, &ret_val, &ret_type_code) != 0) {
    return -1;
  }
  (((TVMValue*)stack_value)[0].v_handle) = compute_intermediate;
  if (compute_intermediate == NULL) {
    ((int32_t*)stack_tcode)[0] = 4;
  } else {
    ((int32_t*)stack_tcode)[0] = 3;
  }
  (((TVMValue*)stack_value)[1].v_handle) = lv6;
  if (lv6 == NULL) {
    ((int32_t*)stack_tcode)[1] = 4;
  } else {
    ((int32_t*)stack_tcode)[1] = 3;
  }
  (((TVMValue*)stack_value)[2].v_handle) = param_1;
  if (param_1 == NULL) {
    ((int32_t*)stack_tcode)[2] = 4;
  } else {
    ((int32_t*)stack_tcode)[2] = 3;
  }
  (((TVMValue*)stack_value)[3].v_handle) = param_2;
  if (param_2 == NULL) {
    ((int32_t*)stack_tcode)[3] = 4;
  } else {
    ((int32_t*)stack_tcode)[3] = 3;
  }
  (((TVMValue*)stack_value)[4].v_int64) = n;
  ((int32_t*)stack_tcode)[4] = 0;
  
  kernel_entry_info call_table[] = {
    {{ n, (int64_t)320, ((int64_t)10240)}, fused_layer_norm_cast1_n_1_to_256__kernel_packed, "fused_layer_norm_cast1_n_1_to_256__kernel"},
    {{ n, (int64_t)128, ((int64_t)10240)}, fused_layer_norm_cast1_n_257_to_512__kernel_packed, "fused_layer_norm_cast1_n_257_to_512__kernel"},
    {{ n, (int64_t)320, ((int64_t)10240)}, fused_layer_norm_cast1_n_513_to_1280__kernel_packed, "fused_layer_norm_cast1_n_513_to_1280__kernel"},
    {{ n, (int64_t)512, ((int64_t)10240)}, fused_layer_norm_cast1_n_1281_to_2816__kernel_packed, "fused_layer_norm_cast1_n_1281_to_2816__kernel"},
    {{ n, (int64_t)192, ((int64_t)10240)}, fused_layer_norm_cast1_n_2817_to_3072__kernel_packed, "fused_layer_norm_cast1_n_2817_to_3072__kernel"},
    {{ n, (int64_t)512, ((int64_t)10240)}, fused_layer_norm_cast1_n_3073_to_4096__kernel_packed, "fused_layer_norm_cast1_n_3073_to_4096__kernel"},
     };
  int64_t index = (n/256) > 15 ? 15 : n/256;
  int64_t index_table[] = {0,1,2,2,2,3,3,3,3,3,3,4,5,5,5,5};
  kernel_entry_info info = call_table[index_table[index]];
(((TVMValue*)stack_value)[5].v_int64) = info.launch_args[0];
  ((int32_t*)stack_tcode)[5] = 0;
  (((TVMValue*)stack_value)[6].v_int64) = info.launch_args[1];
  ((int32_t*)stack_tcode)[6] = 0;
  (((TVMValue*)stack_value)[7].v_int64) = info.launch_args[2];
  ((int32_t*)stack_tcode)[7] = 0;

  if (info.kernel_handle == NULL)
  {
    if (TVMBackendGetFuncFromEnv(__tvm_module_ctx, info.name.c_str(), &info.kernel_handle) != 0)
    {
      return -1;
    }
  }
  TVMValue ret_val_1;
  int ret_type_code_1;
  if (TVMFuncCall(info.kernel_handle, (TVMValue *)stack_value, (int *)stack_tcode, 8, &ret_val_1, &ret_type_code_1) != 0)
  {
    return -1;
  }
  return 0;
}     
    