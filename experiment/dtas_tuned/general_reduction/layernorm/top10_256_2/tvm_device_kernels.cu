
#include <cuda_runtime.h>
#include <math.h>
#include <mma.h>
//for int8
#include <sm_61_intrinsics.h>

#include <cuda_fp16.h>
__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}

#define __int8_t_defined

#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) static inline __device__ __host__ half HALF_MATH_NAME(half x, half y) {     float tmp_x = __half2float(x);                                            float tmp_y = __half2float(y);                                            float result = FP32_MATH_NAME(tmp_x, tmp_y);                              return __float2half(result);                                            }

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) static inline __device__ __host__ half HALF_MATH_NAME(half x) {            float tmp_x = __half2float(x);                                           float result = FP32_MATH_NAME(tmp_x);                                    return __float2half(result);                                           }

CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY

// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// There is no make_int8 in cuda, but TVM codegen seem to use it
static inline __device__ longlong4 make_int8(int x0, int x1, int x2, int x3, int x4, int x5, int x6, int x7) {
  int2 i0 = make_int2(x0, x1);
  int2 i1 = make_int2(x2, x3);
  int2 i2 = make_int2(x4, x5);
  int2 i3 = make_int2(x6, x7);
  long long l0 = *(long long*)&i0;
  long long l1 = *(long long*)&i1;
  long long l2 = *(long long*)&i2;
  long long l3 = *(long long*)&i3;
  return make_longlong4(l0, l1, l2, l3);
}


#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) ||      (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(320) fused_layer_norm_cast1_n_1_to_256__kernel(half* __restrict__ compute_intermediate, float* __restrict__ lv6, float* __restrict__ param_1, float* __restrict__ param_2, int64_t n);
extern "C" __global__ void __launch_bounds__(128) fused_layer_norm_cast1_n_257_to_512__kernel(half* __restrict__ compute_intermediate, float* __restrict__ lv6, float* __restrict__ param_1, float* __restrict__ param_2, int64_t n);
extern "C" __global__ void __launch_bounds__(320) fused_layer_norm_cast1_n_513_to_1280__kernel(half* __restrict__ compute_intermediate, float* __restrict__ lv6, float* __restrict__ param_1, float* __restrict__ param_2, int64_t n);
extern "C" __global__ void __launch_bounds__(512) fused_layer_norm_cast1_n_1281_to_2816__kernel(half* __restrict__ compute_intermediate, float* __restrict__ lv6, float* __restrict__ param_1, float* __restrict__ param_2, int64_t n);
extern "C" __global__ void __launch_bounds__(192) fused_layer_norm_cast1_n_2817_to_3072__kernel(half* __restrict__ compute_intermediate, float* __restrict__ lv6, float* __restrict__ param_1, float* __restrict__ param_2, int64_t n);
extern "C" __global__ void __launch_bounds__(512) fused_layer_norm_cast1_n_3073_to_4096__kernel(half* __restrict__ compute_intermediate, float* __restrict__ lv6, float* __restrict__ param_1, float* __restrict__ param_2, int64_t n);
extern "C" __global__ void __launch_bounds__(320) fused_layer_norm_cast1_n_1_to_256__kernel(half* __restrict__ compute_intermediate, float* __restrict__ lv6, float* __restrict__ param_1, float* __restrict__ param_2, int64_t n) {
  float4 lv6_shared_dyn_local[2];
  extern __shared__ float lv6_shared_dyn[];
  float in_thread_A_red_temp_v0_shared[1];
  float in_thread_A_red_temp_v1_shared[1];
  __shared__ float red_result[1];
  __shared__ float red_result_1[1];
  __shared__ float A_red_temp_v0_shared[1];
  __shared__ float A_red_temp_v1_shared[1];
  for (int64_t ax2_0 = 0; ax2_0 < (int64_t)2; ++ax2_0) {
    lv6_shared_dyn_local[ax2_0] = *(float4*)(lv6 + (((((int64_t)blockIdx.x) * (int64_t)2560) + (ax2_0 * (int64_t)1280)) + (((int64_t)threadIdx.x) * (int64_t)4)));
  }
  for (int ax2_0_1 = 0; ax2_0_1 < 2; ++ax2_0_1) {
    *(float4*)(lv6_shared_dyn + ((((int64_t)ax2_0_1) * (int64_t)1280) + (((int64_t)threadIdx.x) * (int64_t)4))) = lv6_shared_dyn_local[ax2_0_1];
  }
  in_thread_A_red_temp_v0_shared[0] = 0.000000e+00f;
  in_thread_A_red_temp_v1_shared[0] = 0.000000e+00f;
  __syncthreads();
  float v_A_red_temp_v0 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[((int64_t)threadIdx.x)]);
  float v_A_red_temp_v1 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[((int64_t)threadIdx.x)] * lv6_shared_dyn[((int64_t)threadIdx.x)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1;
  float v_A_red_temp_v0_1 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)320)]);
  float v_A_red_temp_v1_1 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)320)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)320)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_1;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_1;
  float v_A_red_temp_v0_2 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)640)]);
  float v_A_red_temp_v1_2 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)640)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)640)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_2;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_2;
  float v_A_red_temp_v0_3 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)960)]);
  float v_A_red_temp_v1_3 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)960)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)960)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_3;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_3;
  float v_A_red_temp_v0_4 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1280)]);
  float v_A_red_temp_v1_4 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1280)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1280)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_4;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_4;
  float v_A_red_temp_v0_5 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1600)]);
  float v_A_red_temp_v1_5 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1600)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1600)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_5;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_5;
  float v_A_red_temp_v0_6 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1920)]);
  float v_A_red_temp_v1_6 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1920)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1920)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_6;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_6;
  float v_A_red_temp_v0_7 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2240)]);
  float v_A_red_temp_v1_7 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2240)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2240)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_7;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_7;
  float red_buf1[1];
  float red_buf0[1];
  uint mask[1];
  float t1[1];
  float t0[1];
  float red_buf1_1[1];
  float red_buf0_1[1];
  uint mask_1[1];
  float t1_1[1];
  float t0_1[1];
  __shared__ float red_buf_staging[10];
  __shared__ float red_buf_staging_1[10];
  red_buf0_1[0] = in_thread_A_red_temp_v0_shared[0];
  red_buf1_1[0] = in_thread_A_red_temp_v1_shared[0];
  mask_1[0] = __activemask();
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 16, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 8, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 4, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 2, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 1, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging_1[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_1[0];
    red_buf_staging[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf1_1[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)10) {
    red_buf0[0] = red_buf_staging_1[((int64_t)threadIdx.x)];
    red_buf1[0] = red_buf_staging[((int64_t)threadIdx.x)];
  }
  mask[0] = (__activemask() & (uint)1023);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  t1[0] = __shfl_down_sync(mask[0], red_buf1[0], 8, 32);
  if (((int64_t)threadIdx.x) < (int64_t)2) {
    red_buf0[0] = (red_buf0[0] + t0[0]);
    red_buf1[0] = (red_buf1[0] + t1[0]);
  }
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  t1[0] = __shfl_down_sync(mask[0], red_buf1[0], 4, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf1[0] = (red_buf1[0] + t1[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  t1[0] = __shfl_down_sync(mask[0], red_buf1[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf1[0] = (red_buf1[0] + t1[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  t1[0] = __shfl_down_sync(mask[0], red_buf1[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf1[0] = (red_buf1[0] + t1[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result)[0] = red_buf0[0];
    ((volatile float*)red_result_1)[0] = red_buf1[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    A_red_temp_v0_shared[0] = ((volatile float*)red_result)[0];
    A_red_temp_v1_shared[0] = ((volatile float*)red_result_1)[0];
  }
  __syncthreads();
  compute_intermediate[((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x))] = ((half)((((lv6_shared_dyn[((int64_t)threadIdx.x)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[((int64_t)threadIdx.x)]) + param_2[((int64_t)threadIdx.x)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)320)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)320)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)320)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)320)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)640)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)640)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)640)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)640)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)960)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)960)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)960)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)960)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)1280)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1280)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)1280)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)1280)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)1600)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1600)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)1600)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)1600)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)1920)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1920)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)1920)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)1920)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)2240)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2240)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)2240)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)2240)]));
}

extern "C" __global__ void __launch_bounds__(128) fused_layer_norm_cast1_n_257_to_512__kernel(half* __restrict__ compute_intermediate, float* __restrict__ lv6, float* __restrict__ param_1, float* __restrict__ param_2, int64_t n) {
  float4 lv6_shared_dyn_local[5];
  extern __shared__ float lv6_shared_dyn[];
  float in_thread_A_red_temp_v0_shared[1];
  float in_thread_A_red_temp_v1_shared[1];
  __shared__ float red_result[1];
  __shared__ float red_result_1[1];
  __shared__ float A_red_temp_v0_shared[1];
  __shared__ float A_red_temp_v1_shared[1];
  for (int64_t ax2_0 = 0; ax2_0 < (int64_t)5; ++ax2_0) {
    lv6_shared_dyn_local[ax2_0] = *(float4*)(lv6 + (((((int64_t)blockIdx.x) * (int64_t)2560) + (ax2_0 * (int64_t)512)) + (((int64_t)threadIdx.x) * (int64_t)4)));
  }
  for (int ax2_0_1 = 0; ax2_0_1 < 5; ++ax2_0_1) {
    *(float4*)(lv6_shared_dyn + ((((int64_t)ax2_0_1) * (int64_t)512) + (((int64_t)threadIdx.x) * (int64_t)4))) = lv6_shared_dyn_local[ax2_0_1];
  }
  in_thread_A_red_temp_v0_shared[0] = 0.000000e+00f;
  in_thread_A_red_temp_v1_shared[0] = 0.000000e+00f;
  __syncthreads();
  float v_A_red_temp_v0 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[((int64_t)threadIdx.x)]);
  float v_A_red_temp_v1 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[((int64_t)threadIdx.x)] * lv6_shared_dyn[((int64_t)threadIdx.x)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1;
  float v_A_red_temp_v0_1 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)128)]);
  float v_A_red_temp_v1_1 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)128)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)128)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_1;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_1;
  float v_A_red_temp_v0_2 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)256)]);
  float v_A_red_temp_v1_2 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)256)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)256)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_2;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_2;
  float v_A_red_temp_v0_3 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)384)]);
  float v_A_red_temp_v1_3 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)384)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)384)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_3;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_3;
  float v_A_red_temp_v0_4 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)512)]);
  float v_A_red_temp_v1_4 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)512)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)512)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_4;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_4;
  float v_A_red_temp_v0_5 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)640)]);
  float v_A_red_temp_v1_5 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)640)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)640)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_5;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_5;
  float v_A_red_temp_v0_6 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)768)]);
  float v_A_red_temp_v1_6 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)768)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)768)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_6;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_6;
  float v_A_red_temp_v0_7 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)896)]);
  float v_A_red_temp_v1_7 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)896)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)896)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_7;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_7;
  float v_A_red_temp_v0_8 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1024)]);
  float v_A_red_temp_v1_8 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1024)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1024)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_8;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_8;
  float v_A_red_temp_v0_9 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1152)]);
  float v_A_red_temp_v1_9 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1152)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1152)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_9;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_9;
  float v_A_red_temp_v0_10 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1280)]);
  float v_A_red_temp_v1_10 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1280)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1280)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_10;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_10;
  float v_A_red_temp_v0_11 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1408)]);
  float v_A_red_temp_v1_11 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1408)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1408)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_11;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_11;
  float v_A_red_temp_v0_12 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1536)]);
  float v_A_red_temp_v1_12 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1536)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1536)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_12;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_12;
  float v_A_red_temp_v0_13 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1664)]);
  float v_A_red_temp_v1_13 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1664)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1664)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_13;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_13;
  float v_A_red_temp_v0_14 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1792)]);
  float v_A_red_temp_v1_14 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1792)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1792)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_14;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_14;
  float v_A_red_temp_v0_15 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1920)]);
  float v_A_red_temp_v1_15 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1920)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1920)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_15;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_15;
  float v_A_red_temp_v0_16 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2048)]);
  float v_A_red_temp_v1_16 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2048)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2048)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_16;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_16;
  float v_A_red_temp_v0_17 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2176)]);
  float v_A_red_temp_v1_17 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2176)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2176)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_17;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_17;
  float v_A_red_temp_v0_18 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2304)]);
  float v_A_red_temp_v1_18 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2304)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2304)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_18;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_18;
  float v_A_red_temp_v0_19 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2432)]);
  float v_A_red_temp_v1_19 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2432)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2432)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_19;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_19;
  float red_buf1[1];
  float red_buf0[1];
  uint mask[1];
  float t1[1];
  float t0[1];
  float red_buf1_1[1];
  float red_buf0_1[1];
  uint mask_1[1];
  float t1_1[1];
  float t0_1[1];
  __shared__ float red_buf_staging[4];
  __shared__ float red_buf_staging_1[4];
  red_buf0_1[0] = in_thread_A_red_temp_v0_shared[0];
  red_buf1_1[0] = in_thread_A_red_temp_v1_shared[0];
  mask_1[0] = __activemask();
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 16, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 8, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 4, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 2, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 1, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging_1[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_1[0];
    red_buf_staging[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf1_1[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)4) {
    red_buf0[0] = red_buf_staging_1[((int64_t)threadIdx.x)];
    red_buf1[0] = red_buf_staging[((int64_t)threadIdx.x)];
  }
  mask[0] = (__activemask() & (uint)15);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  t1[0] = __shfl_down_sync(mask[0], red_buf1[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf1[0] = (red_buf1[0] + t1[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  t1[0] = __shfl_down_sync(mask[0], red_buf1[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf1[0] = (red_buf1[0] + t1[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result)[0] = red_buf0[0];
    ((volatile float*)red_result_1)[0] = red_buf1[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    A_red_temp_v0_shared[0] = ((volatile float*)red_result)[0];
    A_red_temp_v1_shared[0] = ((volatile float*)red_result_1)[0];
  }
  __syncthreads();
  compute_intermediate[((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x))] = ((half)((((lv6_shared_dyn[((int64_t)threadIdx.x)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[((int64_t)threadIdx.x)]) + param_2[((int64_t)threadIdx.x)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)128)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)128)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)128)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)128)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)256)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)256)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)256)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)256)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)384)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)384)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)384)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)384)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)512)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)512)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)512)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)512)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)640)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)640)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)640)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)640)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)768)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)768)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)768)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)768)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)896)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)896)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)896)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)896)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)1024)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1024)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)1024)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)1024)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)1152)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1152)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)1152)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)1152)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)1280)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1280)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)1280)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)1280)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)1408)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1408)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)1408)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)1408)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)1536)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1536)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)1536)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)1536)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)1664)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1664)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)1664)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)1664)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)1792)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1792)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)1792)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)1792)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)1920)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1920)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)1920)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)1920)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)2048)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2048)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)2048)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)2048)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)2176)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2176)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)2176)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)2176)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)2304)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2304)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)2304)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)2304)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)2432)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2432)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)2432)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)2432)]));
}

extern "C" __global__ void __launch_bounds__(320) fused_layer_norm_cast1_n_513_to_1280__kernel(half* __restrict__ compute_intermediate, float* __restrict__ lv6, float* __restrict__ param_1, float* __restrict__ param_2, int64_t n) {
  float4 lv6_shared_dyn_local[2];
  extern __shared__ float lv6_shared_dyn[];
  float in_thread_A_red_temp_v0_shared[1];
  float in_thread_A_red_temp_v1_shared[1];
  __shared__ float red_result[1];
  __shared__ float red_result_1[1];
  __shared__ float A_red_temp_v0_shared[1];
  __shared__ float A_red_temp_v1_shared[1];
  for (int64_t ax2_0 = 0; ax2_0 < (int64_t)2; ++ax2_0) {
    lv6_shared_dyn_local[ax2_0] = *(float4*)(lv6 + (((((int64_t)blockIdx.x) * (int64_t)2560) + (ax2_0 * (int64_t)1280)) + (((int64_t)threadIdx.x) * (int64_t)4)));
  }
  for (int ax2_0_1 = 0; ax2_0_1 < 2; ++ax2_0_1) {
    *(float4*)(lv6_shared_dyn + ((((int64_t)ax2_0_1) * (int64_t)1280) + (((int64_t)threadIdx.x) * (int64_t)4))) = lv6_shared_dyn_local[ax2_0_1];
  }
  in_thread_A_red_temp_v0_shared[0] = 0.000000e+00f;
  in_thread_A_red_temp_v1_shared[0] = 0.000000e+00f;
  __syncthreads();
  float v_A_red_temp_v0 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[((int64_t)threadIdx.x)]);
  float v_A_red_temp_v1 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[((int64_t)threadIdx.x)] * lv6_shared_dyn[((int64_t)threadIdx.x)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1;
  float v_A_red_temp_v0_1 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)320)]);
  float v_A_red_temp_v1_1 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)320)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)320)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_1;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_1;
  float v_A_red_temp_v0_2 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)640)]);
  float v_A_red_temp_v1_2 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)640)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)640)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_2;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_2;
  float v_A_red_temp_v0_3 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)960)]);
  float v_A_red_temp_v1_3 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)960)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)960)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_3;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_3;
  float v_A_red_temp_v0_4 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1280)]);
  float v_A_red_temp_v1_4 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1280)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1280)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_4;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_4;
  float v_A_red_temp_v0_5 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1600)]);
  float v_A_red_temp_v1_5 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1600)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1600)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_5;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_5;
  float v_A_red_temp_v0_6 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1920)]);
  float v_A_red_temp_v1_6 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1920)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1920)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_6;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_6;
  float v_A_red_temp_v0_7 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2240)]);
  float v_A_red_temp_v1_7 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2240)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2240)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_7;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_7;
  float red_buf1[1];
  float red_buf0[1];
  uint mask[1];
  float t1[1];
  float t0[1];
  float red_buf1_1[1];
  float red_buf0_1[1];
  uint mask_1[1];
  float t1_1[1];
  float t0_1[1];
  __shared__ float red_buf_staging[10];
  __shared__ float red_buf_staging_1[10];
  red_buf0_1[0] = in_thread_A_red_temp_v0_shared[0];
  red_buf1_1[0] = in_thread_A_red_temp_v1_shared[0];
  mask_1[0] = __activemask();
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 16, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 8, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 4, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 2, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 1, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging_1[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_1[0];
    red_buf_staging[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf1_1[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)10) {
    red_buf0[0] = red_buf_staging_1[((int64_t)threadIdx.x)];
    red_buf1[0] = red_buf_staging[((int64_t)threadIdx.x)];
  }
  mask[0] = (__activemask() & (uint)1023);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  t1[0] = __shfl_down_sync(mask[0], red_buf1[0], 8, 32);
  if (((int64_t)threadIdx.x) < (int64_t)2) {
    red_buf0[0] = (red_buf0[0] + t0[0]);
    red_buf1[0] = (red_buf1[0] + t1[0]);
  }
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  t1[0] = __shfl_down_sync(mask[0], red_buf1[0], 4, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf1[0] = (red_buf1[0] + t1[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  t1[0] = __shfl_down_sync(mask[0], red_buf1[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf1[0] = (red_buf1[0] + t1[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  t1[0] = __shfl_down_sync(mask[0], red_buf1[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf1[0] = (red_buf1[0] + t1[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result)[0] = red_buf0[0];
    ((volatile float*)red_result_1)[0] = red_buf1[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    A_red_temp_v0_shared[0] = ((volatile float*)red_result)[0];
    A_red_temp_v1_shared[0] = ((volatile float*)red_result_1)[0];
  }
  __syncthreads();
  compute_intermediate[((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x))] = ((half)((((lv6_shared_dyn[((int64_t)threadIdx.x)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[((int64_t)threadIdx.x)]) + param_2[((int64_t)threadIdx.x)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)320)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)320)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)320)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)320)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)640)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)640)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)640)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)640)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)960)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)960)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)960)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)960)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)1280)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1280)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)1280)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)1280)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)1600)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1600)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)1600)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)1600)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)1920)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1920)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)1920)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)1920)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)2240)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2240)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)2240)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)2240)]));
}

extern "C" __global__ void __launch_bounds__(512) fused_layer_norm_cast1_n_1281_to_2816__kernel(half* __restrict__ compute_intermediate, float* __restrict__ lv6, float* __restrict__ param_1, float* __restrict__ param_2, int64_t n) {
  float4 lv6_shared_dyn_local[2];
  extern __shared__ float lv6_shared_dyn[];
  float in_thread_A_red_temp_v0_shared[1];
  float in_thread_A_red_temp_v1_shared[1];
  __shared__ float red_result[1];
  __shared__ float red_result_1[1];
  __shared__ float A_red_temp_v0_shared[1];
  __shared__ float A_red_temp_v1_shared[1];
  for (int64_t ax2_0 = 0; ax2_0 < (int64_t)2; ++ax2_0) {
    if (((ax2_0 * (int64_t)4) + (((int64_t)threadIdx.x) >> (int64_t)7)) < (int64_t)5) {
      lv6_shared_dyn_local[ax2_0] = *(float4*)(lv6 + (((((int64_t)blockIdx.x) * (int64_t)2560) + (ax2_0 * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)4)));
    }
  }
  for (int ax2_0_1 = 0; ax2_0_1 < 2; ++ax2_0_1) {
    if (((((int64_t)ax2_0_1) * (int64_t)4) + (((int64_t)threadIdx.x) >> (int64_t)7)) < (int64_t)5) {
      *(float4*)(lv6_shared_dyn + ((((int64_t)ax2_0_1) * (int64_t)2048) + (((int64_t)threadIdx.x) * (int64_t)4))) = lv6_shared_dyn_local[ax2_0_1];
    }
  }
  in_thread_A_red_temp_v0_shared[0] = 0.000000e+00f;
  in_thread_A_red_temp_v1_shared[0] = 0.000000e+00f;
  __syncthreads();
  float v_A_red_temp_v0 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[((int64_t)threadIdx.x)]);
  float v_A_red_temp_v1 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[((int64_t)threadIdx.x)] * lv6_shared_dyn[((int64_t)threadIdx.x)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1;
  float v_A_red_temp_v0_1 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)512)]);
  float v_A_red_temp_v1_1 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)512)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)512)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_1;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_1;
  float v_A_red_temp_v0_2 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1024)]);
  float v_A_red_temp_v1_2 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1024)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1024)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_2;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_2;
  float v_A_red_temp_v0_3 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1536)]);
  float v_A_red_temp_v1_3 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1536)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1536)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_3;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_3;
  float v_A_red_temp_v0_4 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2048)]);
  float v_A_red_temp_v1_4 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2048)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2048)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_4;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_4;
  float red_buf1[1];
  float red_buf0[1];
  uint mask[1];
  float t1[1];
  float t0[1];
  float red_buf1_1[1];
  float red_buf0_1[1];
  uint mask_1[1];
  float t1_1[1];
  float t0_1[1];
  __shared__ float red_buf_staging[16];
  __shared__ float red_buf_staging_1[16];
  red_buf0_1[0] = in_thread_A_red_temp_v0_shared[0];
  red_buf1_1[0] = in_thread_A_red_temp_v1_shared[0];
  mask_1[0] = __activemask();
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 16, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 8, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 4, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 2, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 1, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging_1[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_1[0];
    red_buf_staging[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf1_1[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)16) {
    red_buf0[0] = red_buf_staging_1[((int64_t)threadIdx.x)];
    red_buf1[0] = red_buf_staging[((int64_t)threadIdx.x)];
  }
  mask[0] = (__activemask() & (uint)65535);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  t1[0] = __shfl_down_sync(mask[0], red_buf1[0], 8, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf1[0] = (red_buf1[0] + t1[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  t1[0] = __shfl_down_sync(mask[0], red_buf1[0], 4, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf1[0] = (red_buf1[0] + t1[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  t1[0] = __shfl_down_sync(mask[0], red_buf1[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf1[0] = (red_buf1[0] + t1[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  t1[0] = __shfl_down_sync(mask[0], red_buf1[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf1[0] = (red_buf1[0] + t1[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result)[0] = red_buf0[0];
    ((volatile float*)red_result_1)[0] = red_buf1[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    A_red_temp_v0_shared[0] = ((volatile float*)red_result)[0];
    A_red_temp_v1_shared[0] = ((volatile float*)red_result_1)[0];
  }
  __syncthreads();
  compute_intermediate[((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x))] = ((half)((((lv6_shared_dyn[((int64_t)threadIdx.x)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[((int64_t)threadIdx.x)]) + param_2[((int64_t)threadIdx.x)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)512)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)512)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)512)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)512)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)1024)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1024)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)1024)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)1024)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)1536)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1536)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)1536)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)1536)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)2048)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2048)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)2048)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)2048)]));
}

extern "C" __global__ void __launch_bounds__(192) fused_layer_norm_cast1_n_2817_to_3072__kernel(half* __restrict__ compute_intermediate, float* __restrict__ lv6, float* __restrict__ param_1, float* __restrict__ param_2, int64_t n) {
  float4 lv6_shared_dyn_local[4];
  extern __shared__ float lv6_shared_dyn[];
  float in_thread_A_red_temp_v0_shared[1];
  float in_thread_A_red_temp_v1_shared[1];
  __shared__ float red_result[1];
  __shared__ float red_result_1[1];
  __shared__ float A_red_temp_v0_shared[1];
  __shared__ float A_red_temp_v1_shared[1];
  for (int64_t ax2_0 = 0; ax2_0 < (int64_t)4; ++ax2_0) {
    if (((ax2_0 * (int64_t)3) + (((int64_t)threadIdx.x) >> (int64_t)6)) < (int64_t)10) {
      lv6_shared_dyn_local[ax2_0] = *(float4*)(lv6 + (((((int64_t)blockIdx.x) * (int64_t)2560) + (ax2_0 * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)4)));
    }
  }
  for (int ax2_0_1 = 0; ax2_0_1 < 4; ++ax2_0_1) {
    if (((((int64_t)ax2_0_1) * (int64_t)3) + (((int64_t)threadIdx.x) >> (int64_t)6)) < (int64_t)10) {
      *(float4*)(lv6_shared_dyn + ((((int64_t)ax2_0_1) * (int64_t)768) + (((int64_t)threadIdx.x) * (int64_t)4))) = lv6_shared_dyn_local[ax2_0_1];
    }
  }
  in_thread_A_red_temp_v0_shared[0] = 0.000000e+00f;
  in_thread_A_red_temp_v1_shared[0] = 0.000000e+00f;
  __syncthreads();
  float v_A_red_temp_v0 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[((int64_t)threadIdx.x)]);
  float v_A_red_temp_v1 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[((int64_t)threadIdx.x)] * lv6_shared_dyn[((int64_t)threadIdx.x)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1;
  float v_A_red_temp_v0_1 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)192)]);
  float v_A_red_temp_v1_1 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)192)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)192)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_1;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_1;
  float v_A_red_temp_v0_2 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)384)]);
  float v_A_red_temp_v1_2 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)384)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)384)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_2;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_2;
  float v_A_red_temp_v0_3 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)576)]);
  float v_A_red_temp_v1_3 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)576)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)576)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_3;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_3;
  float v_A_red_temp_v0_4 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)768)]);
  float v_A_red_temp_v1_4 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)768)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)768)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_4;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_4;
  float v_A_red_temp_v0_5 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)960)]);
  float v_A_red_temp_v1_5 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)960)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)960)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_5;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_5;
  float v_A_red_temp_v0_6 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1152)]);
  float v_A_red_temp_v1_6 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1152)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1152)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_6;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_6;
  float v_A_red_temp_v0_7 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1344)]);
  float v_A_red_temp_v1_7 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1344)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1344)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_7;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_7;
  float v_A_red_temp_v0_8 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1536)]);
  float v_A_red_temp_v1_8 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1536)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1536)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_8;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_8;
  float v_A_red_temp_v0_9 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1728)]);
  float v_A_red_temp_v1_9 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1728)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1728)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_9;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_9;
  float v_A_red_temp_v0_10 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1920)]);
  float v_A_red_temp_v1_10 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1920)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1920)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_10;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_10;
  float v_A_red_temp_v0_11 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2112)]);
  float v_A_red_temp_v1_11 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2112)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2112)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_11;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_11;
  float v_A_red_temp_v0_12 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2304)]);
  float v_A_red_temp_v1_12 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2304)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2304)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_12;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_12;
  if (((int64_t)threadIdx.x) < (int64_t)64) {
    float v_A_red_temp_v0_13 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2496)]);
    float v_A_red_temp_v1_13 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2496)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2496)]));
    in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_13;
    in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_13;
  }
  float red_buf1[1];
  float red_buf0[1];
  uint mask[1];
  float t1[1];
  float t0[1];
  float red_buf1_1[1];
  float red_buf0_1[1];
  uint mask_1[1];
  float t1_1[1];
  float t0_1[1];
  __shared__ float red_buf_staging[6];
  __shared__ float red_buf_staging_1[6];
  red_buf0_1[0] = in_thread_A_red_temp_v0_shared[0];
  red_buf1_1[0] = in_thread_A_red_temp_v1_shared[0];
  mask_1[0] = __activemask();
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 16, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 8, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 4, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 2, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 1, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging_1[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_1[0];
    red_buf_staging[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf1_1[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)6) {
    red_buf0[0] = red_buf_staging_1[((int64_t)threadIdx.x)];
    red_buf1[0] = red_buf_staging[((int64_t)threadIdx.x)];
  }
  mask[0] = (__activemask() & (uint)63);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  t1[0] = __shfl_down_sync(mask[0], red_buf1[0], 4, 32);
  if (((int64_t)threadIdx.x) < (int64_t)2) {
    red_buf0[0] = (red_buf0[0] + t0[0]);
    red_buf1[0] = (red_buf1[0] + t1[0]);
  }
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  t1[0] = __shfl_down_sync(mask[0], red_buf1[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf1[0] = (red_buf1[0] + t1[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  t1[0] = __shfl_down_sync(mask[0], red_buf1[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf1[0] = (red_buf1[0] + t1[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result)[0] = red_buf0[0];
    ((volatile float*)red_result_1)[0] = red_buf1[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    A_red_temp_v0_shared[0] = ((volatile float*)red_result)[0];
    A_red_temp_v1_shared[0] = ((volatile float*)red_result_1)[0];
  }
  __syncthreads();
  compute_intermediate[((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x))] = ((half)((((lv6_shared_dyn[((int64_t)threadIdx.x)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[((int64_t)threadIdx.x)]) + param_2[((int64_t)threadIdx.x)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)192)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)192)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)192)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)192)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)384)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)384)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)384)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)384)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)576)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)576)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)576)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)576)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)768)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)768)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)768)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)768)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)960)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)960)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)960)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)960)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)1152)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1152)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)1152)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)1152)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)1344)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1344)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)1344)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)1344)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)1536)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1536)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)1536)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)1536)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)1728)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1728)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)1728)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)1728)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)1920)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1920)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)1920)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)1920)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)2112)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2112)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)2112)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)2112)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)2304)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2304)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)2304)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)2304)]));
  if (((int64_t)threadIdx.x) < (int64_t)64) {
    compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)2496)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2496)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)2496)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)2496)]));
  }
}

extern "C" __global__ void __launch_bounds__(512) fused_layer_norm_cast1_n_3073_to_4096__kernel(half* __restrict__ compute_intermediate, float* __restrict__ lv6, float* __restrict__ param_1, float* __restrict__ param_2, int64_t n) {
  float4 lv6_shared_dyn_local[2];
  extern __shared__ float lv6_shared_dyn[];
  float in_thread_A_red_temp_v0_shared[1];
  float in_thread_A_red_temp_v1_shared[1];
  __shared__ float red_result[1];
  __shared__ float red_result_1[1];
  __shared__ float A_red_temp_v0_shared[1];
  __shared__ float A_red_temp_v1_shared[1];
  for (int64_t ax2_0 = 0; ax2_0 < (int64_t)2; ++ax2_0) {
    if (((ax2_0 * (int64_t)4) + (((int64_t)threadIdx.x) >> (int64_t)7)) < (int64_t)5) {
      lv6_shared_dyn_local[ax2_0] = *(float4*)(lv6 + (((((int64_t)blockIdx.x) * (int64_t)2560) + (ax2_0 * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)4)));
    }
  }
  for (int ax2_0_1 = 0; ax2_0_1 < 2; ++ax2_0_1) {
    if (((((int64_t)ax2_0_1) * (int64_t)4) + (((int64_t)threadIdx.x) >> (int64_t)7)) < (int64_t)5) {
      *(float4*)(lv6_shared_dyn + ((((int64_t)ax2_0_1) * (int64_t)2048) + (((int64_t)threadIdx.x) * (int64_t)4))) = lv6_shared_dyn_local[ax2_0_1];
    }
  }
  in_thread_A_red_temp_v0_shared[0] = 0.000000e+00f;
  in_thread_A_red_temp_v1_shared[0] = 0.000000e+00f;
  __syncthreads();
  float v_A_red_temp_v0 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[((int64_t)threadIdx.x)]);
  float v_A_red_temp_v1 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[((int64_t)threadIdx.x)] * lv6_shared_dyn[((int64_t)threadIdx.x)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1;
  float v_A_red_temp_v0_1 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)512)]);
  float v_A_red_temp_v1_1 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)512)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)512)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_1;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_1;
  float v_A_red_temp_v0_2 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1024)]);
  float v_A_red_temp_v1_2 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1024)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1024)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_2;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_2;
  float v_A_red_temp_v0_3 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1536)]);
  float v_A_red_temp_v1_3 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1536)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1536)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_3;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_3;
  float v_A_red_temp_v0_4 = (in_thread_A_red_temp_v0_shared[0] + lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2048)]);
  float v_A_red_temp_v1_4 = (in_thread_A_red_temp_v1_shared[0] + (lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2048)] * lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2048)]));
  in_thread_A_red_temp_v0_shared[0] = v_A_red_temp_v0_4;
  in_thread_A_red_temp_v1_shared[0] = v_A_red_temp_v1_4;
  float red_buf1[1];
  float red_buf0[1];
  uint mask[1];
  float t1[1];
  float t0[1];
  float red_buf1_1[1];
  float red_buf0_1[1];
  uint mask_1[1];
  float t1_1[1];
  float t0_1[1];
  __shared__ float red_buf_staging[16];
  __shared__ float red_buf_staging_1[16];
  red_buf0_1[0] = in_thread_A_red_temp_v0_shared[0];
  red_buf1_1[0] = in_thread_A_red_temp_v1_shared[0];
  mask_1[0] = __activemask();
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 16, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 8, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 4, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 2, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
  t1_1[0] = __shfl_down_sync(mask_1[0], red_buf1_1[0], 1, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf1_1[0] = (red_buf1_1[0] + t1_1[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging_1[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_1[0];
    red_buf_staging[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf1_1[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)16) {
    red_buf0[0] = red_buf_staging_1[((int64_t)threadIdx.x)];
    red_buf1[0] = red_buf_staging[((int64_t)threadIdx.x)];
  }
  mask[0] = (__activemask() & (uint)65535);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  t1[0] = __shfl_down_sync(mask[0], red_buf1[0], 8, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf1[0] = (red_buf1[0] + t1[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  t1[0] = __shfl_down_sync(mask[0], red_buf1[0], 4, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf1[0] = (red_buf1[0] + t1[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  t1[0] = __shfl_down_sync(mask[0], red_buf1[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf1[0] = (red_buf1[0] + t1[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  t1[0] = __shfl_down_sync(mask[0], red_buf1[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf1[0] = (red_buf1[0] + t1[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result)[0] = red_buf0[0];
    ((volatile float*)red_result_1)[0] = red_buf1[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    A_red_temp_v0_shared[0] = ((volatile float*)red_result)[0];
    A_red_temp_v1_shared[0] = ((volatile float*)red_result_1)[0];
  }
  __syncthreads();
  compute_intermediate[((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x))] = ((half)((((lv6_shared_dyn[((int64_t)threadIdx.x)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[((int64_t)threadIdx.x)]) + param_2[((int64_t)threadIdx.x)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)512)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)512)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)512)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)512)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)1024)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1024)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)1024)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)1024)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)1536)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)1536)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)1536)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)1536)]));
  compute_intermediate[(((((int64_t)blockIdx.x) * (int64_t)2560) + ((int64_t)threadIdx.x)) + (int64_t)2048)] = ((half)((((lv6_shared_dyn[(((int64_t)threadIdx.x) + (int64_t)2048)] - (A_red_temp_v0_shared[0] * 3.906250e-04f)) * (1.000000e+00f / sqrtf((((A_red_temp_v1_shared[0] * 3.906250e-04f) - ((A_red_temp_v0_shared[0] * 3.906250e-04f) * (A_red_temp_v0_shared[0] * 3.906250e-04f))) + 1.000000e-05f)))) * param_1[(((int64_t)threadIdx.x) + (int64_t)2048)]) + param_2[(((int64_t)threadIdx.x) + (int64_t)2048)]));
}

