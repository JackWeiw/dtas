
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
static void* gemm_n_1_to_256__kernel_packed = NULL;
static void* gemm_n_257_to_512__kernel_packed = NULL;
static void* gemm_n_513_to_768__kernel_packed = NULL;
static void* gemm_n_769_to_1024__kernel_packed = NULL;
static void* gemm_n_1025_to_1280__kernel_packed = NULL;
static void* gemm_n_1281_to_1536__kernel_packed = NULL;
static void* gemm_n_1537_to_1792__kernel_packed = NULL;
static void* gemm_n_1793_to_2048__kernel_packed = NULL;
static void* gemm_n_2049_to_2304__kernel_packed = NULL;
static void* gemm_n_2305_to_2560__kernel_packed = NULL;
static void* gemm_n_2561_to_2816__kernel_packed = NULL;
static void* gemm_n_2817_to_3072__kernel_packed = NULL;
static void* gemm_n_3073_to_3328__kernel_packed = NULL;
static void* gemm_n_3329_to_3584__kernel_packed = NULL;
static void* gemm_n_3585_to_3840__kernel_packed = NULL;
static void* gemm_n_3841_to_4096__kernel_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t gemm(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t gemm(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  TVMValue stack[6];
  void* stack_tcode = stack;
  TVMValue stack_1[12];
  void* stack_value = stack_1;
  int32_t p_lv9_code = arg_type_ids[0];
  int32_t lv3_handle_code = arg_type_ids[1];
  int32_t lv4_handle_code = arg_type_ids[2];
  int32_t p_output0_code = arg_type_ids[3];
  void* p_lv9 = (((TVMValue*)args)[0].v_handle);
  void* lv3_handle = (((TVMValue*)args)[1].v_handle);
  void* lv4_handle = (((TVMValue*)args)[2].v_handle);
  void* p_output0 = (((TVMValue*)args)[3].v_handle);
  void* lv9 = (((DLTensor*)p_lv9)[0].data);
  void* gemm_p_lv9_shape = (((DLTensor*)p_lv9)[0].shape);
  int64_t n = ((int64_t*)gemm_p_lv9_shape)[1];
  void* gemm_p_lv9_strides = (((DLTensor*)p_lv9)[0].strides);
  int32_t dev_id = (((DLTensor*)p_lv9)[0].device.device_id);
  void* lv3 = (((DLTensor*)lv3_handle)[0].data);
  void* gemm_lv3_handle_shape = (((DLTensor*)lv3_handle)[0].shape);
  void* gemm_lv3_handle_strides = (((DLTensor*)lv3_handle)[0].strides);
  void* lv4 = (((DLTensor*)lv4_handle)[0].data);
  void* gemm_lv4_handle_shape = (((DLTensor*)lv4_handle)[0].shape);
  void* gemm_lv4_handle_strides = (((DLTensor*)lv4_handle)[0].strides);
  void* var_T_add_intermediate = (((DLTensor*)p_output0)[0].data);
  void* gemm_p_output0_shape = (((DLTensor*)p_output0)[0].shape);
  void* gemm_p_output0_strides = (((DLTensor*)p_output0)[0].strides);
  if (!(gemm_p_lv9_strides == NULL)) {
  }
  if (!(gemm_lv3_handle_strides == NULL)) {
  }
  if (!(gemm_lv4_handle_strides == NULL)) {
  }
  if (!(gemm_p_output0_strides == NULL)) {
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
  (((TVMValue*)stack_value)[0].v_handle) = lv3;
  if (lv3 == NULL) {
    ((int32_t*)stack_tcode)[0] = 4;
  } else {
    ((int32_t*)stack_tcode)[0] = 3;
  }
  (((TVMValue*)stack_value)[1].v_handle) = lv4;
  if (lv4 == NULL) {
    ((int32_t*)stack_tcode)[1] = 4;
  } else {
    ((int32_t*)stack_tcode)[1] = 3;
  }
  (((TVMValue*)stack_value)[2].v_handle) = lv9;
  if (lv9 == NULL) {
    ((int32_t*)stack_tcode)[2] = 4;
  } else {
    ((int32_t*)stack_tcode)[2] = 3;
  }
  (((TVMValue*)stack_value)[3].v_handle) = var_T_add_intermediate;
  if (var_T_add_intermediate == NULL) {
    ((int32_t*)stack_tcode)[3] = 4;
  } else {
    ((int32_t*)stack_tcode)[3] = 3;
  }
  (((TVMValue*)stack_value)[4].v_int64) = n;
  ((int32_t*)stack_tcode)[4] = 0;
  
  kernel_entry_info call_table[] = {
    {{ (int64_t)1, (((n + (int64_t)191) / (int64_t)192) + (((n + (int64_t)191) % (int64_t)192) >> (int64_t)63)), (int64_t)20, (int64_t)32, (int64_t)12, ((int64_t)92160)}, gemm_n_1_to_256__kernel_packed, "gemm_n_1_to_256__kernel"},
    {{ (int64_t)1, (((n + (int64_t)191) / (int64_t)192) + (((n + (int64_t)191) % (int64_t)192) >> (int64_t)63)), (int64_t)20, (int64_t)32, (int64_t)12, ((int64_t)92160)}, gemm_n_257_to_512__kernel_packed, "gemm_n_257_to_512__kernel"},
    {{ (int64_t)1, (((n + (int64_t)191) / (int64_t)192) + (((n + (int64_t)191) % (int64_t)192) >> (int64_t)63)), (int64_t)20, (int64_t)32, (int64_t)12, ((int64_t)92160)}, gemm_n_513_to_768__kernel_packed, "gemm_n_513_to_768__kernel"},
    {{ (int64_t)1, ((n + (int64_t)127) >> (int64_t)7), (int64_t)14, (int64_t)32, (int64_t)12, ((int64_t)92160)}, gemm_n_769_to_1024__kernel_packed, "gemm_n_769_to_1024__kernel"},
    {{ (int64_t)1, (((n + (int64_t)191) / (int64_t)192) + (((n + (int64_t)191) % (int64_t)192) >> (int64_t)63)), (int64_t)20, (int64_t)32, (int64_t)12, ((int64_t)92160)}, gemm_n_1025_to_1280__kernel_packed, "gemm_n_1025_to_1280__kernel"},
    {{ (int64_t)1, (((n + (int64_t)191) / (int64_t)192) + (((n + (int64_t)191) % (int64_t)192) >> (int64_t)63)), (int64_t)20, (int64_t)32, (int64_t)12, ((int64_t)92160)}, gemm_n_1281_to_1536__kernel_packed, "gemm_n_1281_to_1536__kernel"},
    {{ (int64_t)1, ((((n + (int64_t)895) / (int64_t)896) + (((n + (int64_t)895) % (int64_t)896) >> (int64_t)63)) * (int64_t)4), (int64_t)20, (int64_t)32, (int64_t)7, ((int64_t)101376)}, gemm_n_1537_to_1792__kernel_packed, "gemm_n_1537_to_1792__kernel"},
    {{ (int64_t)1, (((n + (int64_t)191) / (int64_t)192) + (((n + (int64_t)191) % (int64_t)192) >> (int64_t)63)), (int64_t)20, (int64_t)32, (int64_t)12, ((int64_t)92160)}, gemm_n_1793_to_2048__kernel_packed, "gemm_n_1793_to_2048__kernel"},
    {{ (int64_t)1, (((n + (int64_t)191) / (int64_t)192) + (((n + (int64_t)191) % (int64_t)192) >> (int64_t)63)), (int64_t)20, (int64_t)32, (int64_t)12, ((int64_t)92160)}, gemm_n_2049_to_2304__kernel_packed, "gemm_n_2049_to_2304__kernel"},
    {{ (int64_t)1, ((((n + (int64_t)895) / (int64_t)896) + (((n + (int64_t)895) % (int64_t)896) >> (int64_t)63)) * (int64_t)4), (int64_t)20, (int64_t)32, (int64_t)7, ((int64_t)101376)}, gemm_n_2305_to_2560__kernel_packed, "gemm_n_2305_to_2560__kernel"},
    {{ (int64_t)1, (((n + (int64_t)191) / (int64_t)192) + (((n + (int64_t)191) % (int64_t)192) >> (int64_t)63)), (int64_t)20, (int64_t)32, (int64_t)12, ((int64_t)92160)}, gemm_n_2561_to_2816__kernel_packed, "gemm_n_2561_to_2816__kernel"},
    {{ (int64_t)1, (((n + (int64_t)191) / (int64_t)192) + (((n + (int64_t)191) % (int64_t)192) >> (int64_t)63)), (int64_t)20, (int64_t)32, (int64_t)12, ((int64_t)92160)}, gemm_n_2817_to_3072__kernel_packed, "gemm_n_2817_to_3072__kernel"},
    {{ (int64_t)1, (((n + (int64_t)191) / (int64_t)192) + (((n + (int64_t)191) % (int64_t)192) >> (int64_t)63)), (int64_t)20, (int64_t)32, (int64_t)12, ((int64_t)92160)}, gemm_n_3073_to_3328__kernel_packed, "gemm_n_3073_to_3328__kernel"},
    {{ (int64_t)1, (((n + (int64_t)191) / (int64_t)192) + (((n + (int64_t)191) % (int64_t)192) >> (int64_t)63)), (int64_t)20, (int64_t)32, (int64_t)12, ((int64_t)92160)}, gemm_n_3329_to_3584__kernel_packed, "gemm_n_3329_to_3584__kernel"},
    {{ (int64_t)1, (((n + (int64_t)191) / (int64_t)192) + (((n + (int64_t)191) % (int64_t)192) >> (int64_t)63)), (int64_t)20, (int64_t)32, (int64_t)12, ((int64_t)92160)}, gemm_n_3585_to_3840__kernel_packed, "gemm_n_3585_to_3840__kernel"},
    {{ (int64_t)1, (((n + (int64_t)191) / (int64_t)192) + (((n + (int64_t)191) % (int64_t)192) >> (int64_t)63)), (int64_t)20, (int64_t)32, (int64_t)16, ((int64_t)92160)}, gemm_n_3841_to_4096__kernel_packed, "gemm_n_3841_to_4096__kernel"},
     };
  int64_t index = (n/256) > 15 ? 15 : n/256;
  int64_t index_table[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  kernel_entry_info info = call_table[index_table[index]];
(((TVMValue*)stack_value)[5].v_int64) = info.launch_args[0];
  ((int32_t*)stack_tcode)[5] = 0;
  (((TVMValue*)stack_value)[6].v_int64) = info.launch_args[1];
  ((int32_t*)stack_tcode)[6] = 0;
  (((TVMValue*)stack_value)[7].v_int64) = info.launch_args[2];
  ((int32_t*)stack_tcode)[7] = 0;
  (((TVMValue*)stack_value)[8].v_int64) = info.launch_args[3];
  ((int32_t*)stack_tcode)[8] = 0;
  (((TVMValue*)stack_value)[9].v_int64) = info.launch_args[4];
  ((int32_t*)stack_tcode)[9] = 0;
  (((TVMValue*)stack_value)[10].v_int64) = info.launch_args[5];
  ((int32_t*)stack_tcode)[10] = 0;

  if (info.kernel_handle == NULL)
  {
    if (TVMBackendGetFuncFromEnv(__tvm_module_ctx, info.name.c_str(), &info.kernel_handle) != 0)
    {
      return -1;
    }
  }
  TVMValue ret_val_1;
  int ret_type_code_1;
  if (TVMFuncCall(info.kernel_handle, (TVMValue *)stack_value, (int *)stack_tcode, 11, &ret_val_1, &ret_type_code_1) != 0)
  {
    return -1;
  }
  return 0;
}     
    