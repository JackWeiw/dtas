
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
static void* softmax_n_1_to_1024__m_1_to_1024__kernel_packed = NULL;
static void* softmax_n_1_to_256__m_769_to_1024__kernel_packed = NULL;
static void* softmax_n_1_to_256__m_1025_to_1536__kernel_packed = NULL;
static void* softmax_n_1_to_1280__m_1025_to_2048__kernel_packed = NULL;
static void* softmax_n_769_to_2048__m_1_to_1024__kernel_packed = NULL;
static void* softmax_n_769_to_1024__m_513_to_768__kernel_packed = NULL;
static void* softmax_n_1025_to_1280__m_1025_to_1792__kernel_packed = NULL;
static void* softmax_n_1025_to_1280__m_1281_to_2048__kernel_packed = NULL;
static void* softmax_n_1281_to_1536__m_1025_to_2048__kernel_packed = NULL;
static void* softmax_n_1281_to_1536__m_1281_to_2048__kernel_packed = NULL;
static void* softmax_n_1537_to_1792__m_1025_to_2048__kernel_packed = NULL;
static void* softmax_n_1793_to_2048__m_1281_to_2048__kernel_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t softmax(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t softmax(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  TVMValue stack[4];
  void* stack_tcode = stack;
  TVMValue stack_1[7];
  void* stack_value = stack_1;
  int32_t p_lv38_code = arg_type_ids[0];
  int32_t p_output0_code = arg_type_ids[1];
  void* p_lv38 = (((TVMValue*)args)[0].v_handle);
  void* p_output0 = (((TVMValue*)args)[1].v_handle);
  void* lv38 = (((DLTensor*)p_lv38)[0].data);
  void* softmax_p_lv38_shape = (((DLTensor*)p_lv38)[0].shape);
  int64_t n = ((int64_t*)softmax_p_lv38_shape)[2];
  int64_t m = ((int64_t*)softmax_p_lv38_shape)[3];
  void* softmax_p_lv38_strides = (((DLTensor*)p_lv38)[0].strides);
  int32_t dev_id = (((DLTensor*)p_lv38)[0].device.device_id);
  void* var_compute_intermediate = (((DLTensor*)p_output0)[0].data);
  void* softmax_p_output0_shape = (((DLTensor*)p_output0)[0].shape);
  void* softmax_p_output0_strides = (((DLTensor*)p_output0)[0].strides);
  if (!(softmax_p_lv38_strides == NULL)) {
  }
  if (!(softmax_p_output0_strides == NULL)) {
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
  (((TVMValue*)stack_value)[0].v_handle) = lv38;
  if (lv38 == NULL) {
    ((int32_t*)stack_tcode)[0] = 4;
  } else {
    ((int32_t*)stack_tcode)[0] = 3;
  }
  (((TVMValue*)stack_value)[1].v_handle) = var_compute_intermediate;
  if (var_compute_intermediate == NULL) {
    ((int32_t*)stack_tcode)[1] = 4;
  } else {
    ((int32_t*)stack_tcode)[1] = 3;
  }
  (((TVMValue*)stack_value)[2].v_int64) = m;
  ((int32_t*)stack_tcode)[2] = 0;
  (((TVMValue*)stack_value)[3].v_int64) = n;
  ((int32_t*)stack_tcode)[3] = 0;
  
  kernel_entry_info call_table[] = {
    {{ (n * (int64_t)32), (int64_t)128}, softmax_n_1_to_1024__m_1_to_1024__kernel_packed, "softmax_n_1_to_1024__m_1_to_1024__kernel"},
    {{ (n * (int64_t)32), (int64_t)256}, softmax_n_1_to_256__m_769_to_1024__kernel_packed, "softmax_n_1_to_256__m_769_to_1024__kernel"},
    {{ (n * (int64_t)32), (int64_t)160, (int64_t)0, (int64_t)0, (((v_) < (v__1) ? (v_) : (v__1)) + ((v__2) > (v__3) ? (v__2) : (v__3))) + m, ((((m + (int64_t)159) / (int64_t)160) + (((m + (int64_t)159) % (int64_t)160) >> (int64_t)63)) * (int64_t)160) - (int64_t)1, (int64_t)1 - ((((m + (int64_t)159) / (int64_t)160) + (((m + (int64_t)159) % (int64_t)160) >> (int64_t)63)) * (int64_t)160), (int64_t)0, (((m) < (v__11) ? (m) : (v__11)) * (int64_t)4)}, softmax_n_1_to_256__m_1025_to_1536__kernel_packed, "softmax_n_1_to_256__m_1025_to_1536__kernel"},
    {{ (n * (int64_t)32), (int64_t)128, (int64_t)0, (int64_t)0, (((v_) < (v__1) ? (v_) : (v__1)) + ((v__2) > (v__3) ? (v__2) : (v__3))) + m, (((m + (int64_t)127) >> (int64_t)7) * (int64_t)128) - (int64_t)1, (int64_t)1 - (((m + (int64_t)127) >> (int64_t)7) * (int64_t)128), (int64_t)0, (((m) < (v__11) ? (m) : (v__11)) * (int64_t)4)}, softmax_n_1_to_1280__m_1025_to_2048__kernel_packed, "softmax_n_1_to_1280__m_1025_to_2048__kernel"},
    {{ (n * (int64_t)32), (int64_t)128}, softmax_n_769_to_2048__m_1_to_1024__kernel_packed, "softmax_n_769_to_2048__m_1_to_1024__kernel"},
    {{ (n * (int64_t)32), (int64_t)160}, softmax_n_769_to_1024__m_513_to_768__kernel_packed, "softmax_n_769_to_1024__m_513_to_768__kernel"},
    {{ (n * (int64_t)32), (int64_t)160, (int64_t)0, (int64_t)0, (((v_) < (v__1) ? (v_) : (v__1)) + ((v__2) > (v__3) ? (v__2) : (v__3))) + m, ((((m + (int64_t)159) / (int64_t)160) + (((m + (int64_t)159) % (int64_t)160) >> (int64_t)63)) * (int64_t)160) - (int64_t)1, (int64_t)1 - ((((m + (int64_t)159) / (int64_t)160) + (((m + (int64_t)159) % (int64_t)160) >> (int64_t)63)) * (int64_t)160), (int64_t)0, (((m) < (v__11) ? (m) : (v__11)) * (int64_t)4)}, softmax_n_1025_to_1280__m_1025_to_1792__kernel_packed, "softmax_n_1025_to_1280__m_1025_to_1792__kernel"},
    {{ (n * (int64_t)32), (int64_t)128, (int64_t)0, (int64_t)0, (((v_) < (v__1) ? (v_) : (v__1)) + ((v__2) > (v__3) ? (v__2) : (v__3))) + m, (((m + (int64_t)127) >> (int64_t)7) * (int64_t)128) - (int64_t)1, (int64_t)1 - (((m + (int64_t)127) >> (int64_t)7) * (int64_t)128), (int64_t)0, (((m) < (v__11) ? (m) : (v__11)) * (int64_t)4)}, softmax_n_1025_to_1280__m_1281_to_2048__kernel_packed, "softmax_n_1025_to_1280__m_1281_to_2048__kernel"},
    {{ (n * (int64_t)32), (int64_t)128, (int64_t)0, (int64_t)0, (((v_) < (v__1) ? (v_) : (v__1)) + ((v__2) > (v__3) ? (v__2) : (v__3))) + m, (((m + (int64_t)127) >> (int64_t)7) * (int64_t)128) - (int64_t)1, (int64_t)1 - (((m + (int64_t)127) >> (int64_t)7) * (int64_t)128), (int64_t)0, (((m) < (v__11) ? (m) : (v__11)) * (int64_t)4)}, softmax_n_1281_to_1536__m_1025_to_2048__kernel_packed, "softmax_n_1281_to_1536__m_1025_to_2048__kernel"},
    {{ (n * (int64_t)32), (int64_t)128, (int64_t)0, (int64_t)0, (((v_) < (v__1) ? (v_) : (v__1)) + ((v__2) > (v__3) ? (v__2) : (v__3))) + m, (((m + (int64_t)127) >> (int64_t)7) * (int64_t)128) - (int64_t)1, (int64_t)1 - (((m + (int64_t)127) >> (int64_t)7) * (int64_t)128), (int64_t)0, (((m) < (v__11) ? (m) : (v__11)) * (int64_t)4)}, softmax_n_1281_to_1536__m_1281_to_2048__kernel_packed, "softmax_n_1281_to_1536__m_1281_to_2048__kernel"},
    {{ (n * (int64_t)32), (int64_t)128, (int64_t)0, (int64_t)0, (((v_) < (v__1) ? (v_) : (v__1)) + ((v__2) > (v__3) ? (v__2) : (v__3))) + m, (((m + (int64_t)127) >> (int64_t)7) * (int64_t)128) - (int64_t)1, (int64_t)1 - (((m + (int64_t)127) >> (int64_t)7) * (int64_t)128), (int64_t)0, (((m) < (v__11) ? (m) : (v__11)) * (int64_t)4)}, softmax_n_1537_to_1792__m_1025_to_2048__kernel_packed, "softmax_n_1537_to_1792__m_1025_to_2048__kernel"},
    {{ (n * (int64_t)32), (int64_t)128, (int64_t)0, (int64_t)0, (((v_) < (v__1) ? (v_) : (v__1)) + ((v__2) > (v__3) ? (v__2) : (v__3))) + m, (((m + (int64_t)127) >> (int64_t)7) * (int64_t)128) - (int64_t)1, (int64_t)1 - (((m + (int64_t)127) >> (int64_t)7) * (int64_t)128), (int64_t)0, (((m) < (v__11) ? (m) : (v__11)) * (int64_t)4)}, softmax_n_1793_to_2048__m_1281_to_2048__kernel_packed, "softmax_n_1793_to_2048__m_1281_to_2048__kernel"},
     };
  int64_t index = (n/256 * 8 + m/256) > 63 ? 63 : (n/256 * 8 + m/256);
  int64_t index_table[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,6,6,6,7,7,7,8,8,8,8,9,9,9,10,10,10,10,11,11,11};
  kernel_entry_info info = call_table[index_table[index]];
(((TVMValue*)stack_value)[4].v_int64) = info.launch_args[0];
  ((int32_t*)stack_tcode)[4] = 0;
  (((TVMValue*)stack_value)[5].v_int64) = info.launch_args[1];
  ((int32_t*)stack_tcode)[5] = 0;

  if (info.kernel_handle == NULL)
  {
    if (TVMBackendGetFuncFromEnv(__tvm_module_ctx, info.name.c_str(), &info.kernel_handle) != 0)
    {
      return -1;
    }
  }
  TVMValue ret_val_1;
  int ret_type_code_1;
  if (TVMFuncCall(info.kernel_handle, (TVMValue *)stack_value, (int *)stack_tcode, 6, &ret_val_1, &ret_type_code_1) != 0)
  {
    return -1;
  }
  return 0;
}     
    