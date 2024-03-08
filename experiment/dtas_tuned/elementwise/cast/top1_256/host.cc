
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
static void* cast_n_1_to_256__kernel_packed = NULL;
static void* cast_n_257_to_512__kernel_packed = NULL;
static void* cast_n_513_to_768__kernel_packed = NULL;
static void* cast_n_769_to_1024__kernel_packed = NULL;
static void* cast_n_1025_to_1280__kernel_packed = NULL;
static void* cast_n_1281_to_1536__kernel_packed = NULL;
static void* cast_n_1537_to_1792__kernel_packed = NULL;
static void* cast_n_1793_to_2048__kernel_packed = NULL;
static void* cast_n_2049_to_2304__kernel_packed = NULL;
static void* cast_n_2305_to_2560__kernel_packed = NULL;
static void* cast_n_2561_to_2816__kernel_packed = NULL;
static void* cast_n_2817_to_3072__kernel_packed = NULL;
static void* cast_n_3073_to_3328__kernel_packed = NULL;
static void* cast_n_3329_to_3584__kernel_packed = NULL;
static void* cast_n_3585_to_3840__kernel_packed = NULL;
static void* cast_n_3841_to_4096__kernel_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t cast(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t cast(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  TVMValue stack[3];
  void* stack_tcode = stack;
  TVMValue stack_1[6];
  void* stack_value = stack_1;
  int32_t var_A_code = arg_type_ids[0];
  int32_t var_compute_code = arg_type_ids[1];
  void* var_A = (((TVMValue*)args)[0].v_handle);
  void* var_compute = (((TVMValue*)args)[1].v_handle);
  void* A = (((DLTensor*)var_A)[0].data);
  void* cast_var_A_shape = (((DLTensor*)var_A)[0].shape);
  int64_t n = ((int64_t*)cast_var_A_shape)[1];
  void* cast_var_A_strides = (((DLTensor*)var_A)[0].strides);
  int32_t dev_id = (((DLTensor*)var_A)[0].device.device_id);
  void* compute = (((DLTensor*)var_compute)[0].data);
  void* cast_var_compute_shape = (((DLTensor*)var_compute)[0].shape);
  void* cast_var_compute_strides = (((DLTensor*)var_compute)[0].strides);
  if (!(cast_var_A_strides == NULL)) {
  }
  if (!(cast_var_compute_strides == NULL)) {
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
  (((TVMValue*)stack_value)[1].v_handle) = compute;
  if (compute == NULL) {
    ((int32_t*)stack_tcode)[1] = 4;
  } else {
    ((int32_t*)stack_tcode)[1] = 3;
  }
  (((TVMValue*)stack_value)[2].v_int64) = n;
  ((int32_t*)stack_tcode)[2] = 0;
  
  kernel_entry_info call_table[] = {
    {{ (int64_t)320, (int64_t)256}, cast_n_1_to_256__kernel_packed, "cast_n_1_to_256__kernel"},
    {{ (int64_t)640, (int64_t)256}, cast_n_257_to_512__kernel_packed, "cast_n_257_to_512__kernel"},
    {{ (int64_t)960, (int64_t)256}, cast_n_513_to_768__kernel_packed, "cast_n_513_to_768__kernel"},
    {{ (int64_t)1280, (int64_t)256}, cast_n_769_to_1024__kernel_packed, "cast_n_769_to_1024__kernel"},
    {{ (int64_t)1600, (int64_t)256}, cast_n_1025_to_1280__kernel_packed, "cast_n_1025_to_1280__kernel"},
    {{ (int64_t)1920, (int64_t)256}, cast_n_1281_to_1536__kernel_packed, "cast_n_1281_to_1536__kernel"},
    {{ (int64_t)2240, (int64_t)256}, cast_n_1537_to_1792__kernel_packed, "cast_n_1537_to_1792__kernel"},
    {{ (int64_t)2560, (int64_t)256}, cast_n_1793_to_2048__kernel_packed, "cast_n_1793_to_2048__kernel"},
    {{ (int64_t)2880, (int64_t)256}, cast_n_2049_to_2304__kernel_packed, "cast_n_2049_to_2304__kernel"},
    {{ (int64_t)3200, (int64_t)256}, cast_n_2305_to_2560__kernel_packed, "cast_n_2305_to_2560__kernel"},
    {{ (int64_t)3520, (int64_t)256}, cast_n_2561_to_2816__kernel_packed, "cast_n_2561_to_2816__kernel"},
    {{ (int64_t)3840, (int64_t)256}, cast_n_2817_to_3072__kernel_packed, "cast_n_2817_to_3072__kernel"},
    {{ (int64_t)4160, (int64_t)256}, cast_n_3073_to_3328__kernel_packed, "cast_n_3073_to_3328__kernel"},
    {{ (int64_t)4480, (int64_t)256}, cast_n_3329_to_3584__kernel_packed, "cast_n_3329_to_3584__kernel"},
    {{ (int64_t)4800, (int64_t)256}, cast_n_3585_to_3840__kernel_packed, "cast_n_3585_to_3840__kernel"},
    {{ (int64_t)5120, (int64_t)256}, cast_n_3841_to_4096__kernel_packed, "cast_n_3841_to_4096__kernel"},
     };
  int64_t index = (n/256) > 15 ? 15 : n/256;
  int64_t index_table[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  kernel_entry_info info = call_table[index_table[index]];
(((TVMValue*)stack_value)[3].v_int64) = info.launch_args[0];
  ((int32_t*)stack_tcode)[3] = 0;
  (((TVMValue*)stack_value)[4].v_int64) = info.launch_args[1];
  ((int32_t*)stack_tcode)[4] = 0;

  if (info.kernel_handle == NULL)
  {
    if (TVMBackendGetFuncFromEnv(__tvm_module_ctx, info.name.c_str(), &info.kernel_handle) != 0)
    {
      return -1;
    }
  }
  TVMValue ret_val_1;
  int ret_type_code_1;
  if (TVMFuncCall(info.kernel_handle, (TVMValue *)stack_value, (int *)stack_tcode, 5, &ret_val_1, &ret_type_code_1) != 0)
  {
    return -1;
  }
  return 0;
}     
    