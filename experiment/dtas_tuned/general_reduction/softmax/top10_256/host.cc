
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
static void* fused_softmax_cast_n_1_to_1024__kernel_packed = NULL;
static void* fused_softmax_cast_n_1025_to_1536__kernel_packed = NULL;
static void* fused_softmax_cast_n_1537_to_2048__kernel_packed = NULL;
static void* fused_softmax_cast_n_2049_to_2560__kernel_packed = NULL;
static void* fused_softmax_cast_n_2561_to_3072__kernel_packed = NULL;
static void* fused_softmax_cast_n_3073_to_3584__kernel_packed = NULL;
static void* fused_softmax_cast_n_3585_to_4096__kernel_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_softmax_cast(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_softmax_cast(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  TVMValue stack[3];
  void* stack_tcode = stack;
  TVMValue stack_1[6];
  void* stack_value = stack_1;
  int32_t p_A_code = arg_type_ids[0];
  int32_t p_output0_code = arg_type_ids[1];
  void* p_A = (((TVMValue*)args)[0].v_handle);
  void* p_output0 = (((TVMValue*)args)[1].v_handle);
  void* A = (((DLTensor*)p_A)[0].data);
  void* fused_softmax_cast_p_A_shape = (((DLTensor*)p_A)[0].shape);
  int64_t n = ((int64_t*)fused_softmax_cast_p_A_shape)[2];
  void* fused_softmax_cast_p_A_strides = (((DLTensor*)p_A)[0].strides);
  int32_t dev_id = (((DLTensor*)p_A)[0].device.device_id);
  void* compute_intermediate = (((DLTensor*)p_output0)[0].data);
  void* fused_softmax_cast_p_output0_shape = (((DLTensor*)p_output0)[0].shape);
  void* fused_softmax_cast_p_output0_strides = (((DLTensor*)p_output0)[0].strides);
  if (!(fused_softmax_cast_p_A_strides == NULL)) {
  }
  if (!(fused_softmax_cast_p_output0_strides == NULL)) {
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
  (((TVMValue*)stack_value)[0].v_handle) = A;
  if (A == NULL) {
    ((int32_t*)stack_tcode)[0] = 4;
  } else {
    ((int32_t*)stack_tcode)[0] = 3;
  }
  (((TVMValue*)stack_value)[1].v_handle) = compute_intermediate;
  if (compute_intermediate == NULL) {
    ((int32_t*)stack_tcode)[1] = 4;
  } else {
    ((int32_t*)stack_tcode)[1] = 3;
  }
  (((TVMValue*)stack_value)[2].v_int64) = n;
  ((int32_t*)stack_tcode)[2] = 0;
  
  kernel_entry_info call_table[] = {
    {{ (int64_t)1000, (int64_t)736,-1}, fused_softmax_cast_n_1_to_1024__kernel_packed, "fused_softmax_cast_n_1_to_1024__kernel"},
    {{ (int64_t)1000, (int64_t)192, (n * (int64_t)4)}, fused_softmax_cast_n_1025_to_1536__kernel_packed, "fused_softmax_cast_n_1025_to_1536__kernel"},
    {{ (int64_t)1000, (int64_t)256, (n * (int64_t)4)}, fused_softmax_cast_n_1537_to_2048__kernel_packed, "fused_softmax_cast_n_1537_to_2048__kernel"},
    {{ (int64_t)1000, (int64_t)224, (n * (int64_t)4)}, fused_softmax_cast_n_2049_to_2560__kernel_packed, "fused_softmax_cast_n_2049_to_2560__kernel"},
    {{ (int64_t)1000, (int64_t)256, (n * (int64_t)4)}, fused_softmax_cast_n_2561_to_3072__kernel_packed, "fused_softmax_cast_n_2561_to_3072__kernel"},
    {{ (int64_t)1000, (int64_t)288, (n * (int64_t)4)}, fused_softmax_cast_n_3073_to_3584__kernel_packed, "fused_softmax_cast_n_3073_to_3584__kernel"},
    {{ (int64_t)1000, (int64_t)320, (n * (int64_t)4)}, fused_softmax_cast_n_3585_to_4096__kernel_packed, "fused_softmax_cast_n_3585_to_4096__kernel"},
     };
  int64_t index = (n/256) > 15 ? 15 : n/256;
  int64_t index_table[] = {0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6};
  kernel_entry_info info = call_table[index_table[index]];
(((TVMValue*)stack_value)[3].v_int64) = info.launch_args[0];
  ((int32_t*)stack_tcode)[3] = 0;
  (((TVMValue*)stack_value)[4].v_int64) = info.launch_args[1];
  ((int32_t*)stack_tcode)[4] = 0;
if (info.launch_args[2] != -1){
  (((TVMValue*)stack_value)[5].v_int64) = info.launch_args[2];
  ((int32_t*)stack_tcode)[5] = 0;
}

  if (info.kernel_handle == NULL)
  {
    if (TVMBackendGetFuncFromEnv(__tvm_module_ctx, info.name.c_str(), &info.kernel_handle) != 0)
    {
      return -1;
    }
  }
  TVMValue ret_val_1;
  int ret_type_code_1;
  if (TVMFuncCall(info.kernel_handle, (TVMValue *)stack_value, (int *)stack_tcode, info.launch_args[2]==-1?5:6, &ret_val_1, &ret_type_code_1) != 0)
  {
    return -1;
  }
  return 0;
}     
    