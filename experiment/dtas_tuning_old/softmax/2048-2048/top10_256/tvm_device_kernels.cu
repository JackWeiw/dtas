
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
extern "C" __global__ void __launch_bounds__(128) softmax_n_1_to_2048__m_1_to_1024__kernel(float* __restrict__ lv38, half* __restrict__ var_compute_intermediate, int64_t m, int64_t n);
extern "C" __global__ void __launch_bounds__(256) softmax_n_1_to_256__m_769_to_1024__kernel(float* __restrict__ lv38, half* __restrict__ var_compute_intermediate, int64_t m, int64_t n);
extern "C" __global__ void __launch_bounds__(128) softmax_n_1_to_768__m_1025_to_1792__kernel(float* __restrict__ lv38, half* __restrict__ var_compute_intermediate, int64_t m, int64_t n);
extern "C" __global__ void __launch_bounds__(160) softmax_n_1_to_1792__m_1537_to_2048__kernel(float* __restrict__ lv38, half* __restrict__ var_compute_intermediate, int64_t m, int64_t n);
extern "C" __global__ void __launch_bounds__(128) softmax_n_513_to_768__m_1281_to_2048__kernel(float* __restrict__ lv38, half* __restrict__ var_compute_intermediate, int64_t m, int64_t n);
extern "C" __global__ void __launch_bounds__(128) softmax_n_769_to_1280__m_1025_to_1792__kernel(float* __restrict__ lv38, half* __restrict__ var_compute_intermediate, int64_t m, int64_t n);
extern "C" __global__ void __launch_bounds__(128) softmax_n_1025_to_1280__m_1281_to_2048__kernel(float* __restrict__ lv38, half* __restrict__ var_compute_intermediate, int64_t m, int64_t n);
extern "C" __global__ void __launch_bounds__(128) softmax_n_1281_to_1792__m_1025_to_1792__kernel(float* __restrict__ lv38, half* __restrict__ var_compute_intermediate, int64_t m, int64_t n);
extern "C" __global__ void __launch_bounds__(128) softmax_n_1793_to_2048__m_1025_to_1536__kernel(float* __restrict__ lv38, half* __restrict__ var_compute_intermediate, int64_t m, int64_t n);
extern "C" __global__ void __launch_bounds__(160) softmax_n_1793_to_2048__m_1793_to_2048__kernel(float* __restrict__ lv38, half* __restrict__ var_compute_intermediate, int64_t m, int64_t n);
extern "C" __global__ void __launch_bounds__(128) softmax_n_1_to_2048__m_1_to_1024__kernel(float* __restrict__ lv38, half* __restrict__ var_compute_intermediate, int64_t m, int64_t n) {
  float in_thread_T_softmax_n_1_to_2048__m_1_to_1024__maxelem_shared[1];
  __shared__ float red_result[1];
  __shared__ float T_softmax_n_1_to_2048__m_1_to_1024__maxelem_shared[1];
  float in_thread_T_softmax_n_1_to_2048__m_1_to_1024__expsum_shared[1];
  __shared__ float red_result_1[1];
  __shared__ float T_softmax_n_1_to_2048__m_1_to_1024__expsum_shared[1];
  in_thread_T_softmax_n_1_to_2048__m_1_to_1024__maxelem_shared[0] = -3.402823e+38f;
  for (int64_t ax2_fused_0 = 0; ax2_fused_0 < ((m + (int64_t)127) >> (int64_t)7); ++ax2_fused_0) {
    if (((ax2_fused_0 * (int64_t)128) + ((int64_t)threadIdx.x)) < m) {
      in_thread_T_softmax_n_1_to_2048__m_1_to_1024__maxelem_shared[0] = max(in_thread_T_softmax_n_1_to_2048__m_1_to_1024__maxelem_shared[0], lv38[(((ax2_fused_0 * (int64_t)128) + (((int64_t)blockIdx.x) * m)) + ((int64_t)threadIdx.x))]);
    }
  }
  float red_buf0[1];
  uint mask[1];
  float t0[1];
  float red_buf0_1[1];
  uint mask_1[1];
  float t0_1[1];
  __shared__ float red_buf_staging[4];
  red_buf0_1[0] = in_thread_T_softmax_n_1_to_2048__m_1_to_1024__maxelem_shared[0];
  mask_1[0] = __activemask();
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_1[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)4) {
    red_buf0[0] = red_buf_staging[((int64_t)threadIdx.x)];
  }
  mask[0] = (__activemask() & (uint)15);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result)[0] = red_buf0[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    T_softmax_n_1_to_2048__m_1_to_1024__maxelem_shared[0] = ((volatile float*)red_result)[0];
  }
  in_thread_T_softmax_n_1_to_2048__m_1_to_1024__expsum_shared[0] = 0.000000e+00f;
  __syncthreads();
  for (int64_t ax2_fused_0_1 = 0; ax2_fused_0_1 < ((m + (int64_t)127) >> (int64_t)7); ++ax2_fused_0_1) {
    if (((ax2_fused_0_1 * (int64_t)128) + ((int64_t)threadIdx.x)) < m) {
      in_thread_T_softmax_n_1_to_2048__m_1_to_1024__expsum_shared[0] = (in_thread_T_softmax_n_1_to_2048__m_1_to_1024__expsum_shared[0] + __expf((lv38[(((ax2_fused_0_1 * (int64_t)128) + (((int64_t)blockIdx.x) * m)) + ((int64_t)threadIdx.x))] - T_softmax_n_1_to_2048__m_1_to_1024__maxelem_shared[0])));
    }
  }
  float red_buf0_2[1];
  uint mask_2[1];
  float t0_2[1];
  float red_buf0_3[1];
  uint mask_3[1];
  float t0_3[1];
  __shared__ float red_buf_staging_1[4];
  red_buf0_3[0] = in_thread_T_softmax_n_1_to_2048__m_1_to_1024__expsum_shared[0];
  mask_3[0] = __activemask();
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 16, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 8, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 4, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 2, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 1, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging_1[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_3[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)4) {
    red_buf0_2[0] = red_buf_staging_1[((int64_t)threadIdx.x)];
  }
  mask_2[0] = (__activemask() & (uint)15);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 2, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 1, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result_1)[0] = red_buf0_2[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    T_softmax_n_1_to_2048__m_1_to_1024__expsum_shared[0] = ((volatile float*)red_result_1)[0];
  }
  __syncthreads();
  for (int64_t ax2_0 = 0; ax2_0 < ((m + (int64_t)127) >> (int64_t)7); ++ax2_0) {
    if (((ax2_0 * (int64_t)128) + ((int64_t)threadIdx.x)) < m) {
      var_compute_intermediate[(((ax2_0 * (int64_t)128) + (((int64_t)blockIdx.x) * m)) + ((int64_t)threadIdx.x))] = ((half)(__expf((lv38[(((ax2_0 * (int64_t)128) + (((int64_t)blockIdx.x) * m)) + ((int64_t)threadIdx.x))] - T_softmax_n_1_to_2048__m_1_to_1024__maxelem_shared[0])) / T_softmax_n_1_to_2048__m_1_to_1024__expsum_shared[0]));
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) softmax_n_1_to_256__m_769_to_1024__kernel(float* __restrict__ lv38, half* __restrict__ var_compute_intermediate, int64_t m, int64_t n) {
  float in_thread_T_softmax_n_1_to_256__m_769_to_1024__maxelem_shared[1];
  __shared__ float red_result[1];
  __shared__ float T_softmax_n_1_to_256__m_769_to_1024__maxelem_shared[1];
  float in_thread_T_softmax_n_1_to_256__m_769_to_1024__expsum_shared[1];
  __shared__ float red_result_1[1];
  __shared__ float T_softmax_n_1_to_256__m_769_to_1024__expsum_shared[1];
  in_thread_T_softmax_n_1_to_256__m_769_to_1024__maxelem_shared[0] = -3.402823e+38f;
  for (int64_t ax2_fused_0 = 0; ax2_fused_0 < ((m + (int64_t)255) >> (int64_t)8); ++ax2_fused_0) {
    if (((ax2_fused_0 * (int64_t)256) + ((int64_t)threadIdx.x)) < m) {
      in_thread_T_softmax_n_1_to_256__m_769_to_1024__maxelem_shared[0] = max(in_thread_T_softmax_n_1_to_256__m_769_to_1024__maxelem_shared[0], lv38[(((ax2_fused_0 * (int64_t)256) + (((int64_t)blockIdx.x) * m)) + ((int64_t)threadIdx.x))]);
    }
  }
  float red_buf0[1];
  uint mask[1];
  float t0[1];
  float red_buf0_1[1];
  uint mask_1[1];
  float t0_1[1];
  __shared__ float red_buf_staging[8];
  red_buf0_1[0] = in_thread_T_softmax_n_1_to_256__m_769_to_1024__maxelem_shared[0];
  mask_1[0] = __activemask();
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_1[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)8) {
    red_buf0[0] = red_buf_staging[((int64_t)threadIdx.x)];
  }
  mask[0] = (__activemask() & (uint)255);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result)[0] = red_buf0[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    T_softmax_n_1_to_256__m_769_to_1024__maxelem_shared[0] = ((volatile float*)red_result)[0];
  }
  in_thread_T_softmax_n_1_to_256__m_769_to_1024__expsum_shared[0] = 0.000000e+00f;
  __syncthreads();
  for (int64_t ax2_fused_0_1 = 0; ax2_fused_0_1 < ((m + (int64_t)255) >> (int64_t)8); ++ax2_fused_0_1) {
    if (((ax2_fused_0_1 * (int64_t)256) + ((int64_t)threadIdx.x)) < m) {
      in_thread_T_softmax_n_1_to_256__m_769_to_1024__expsum_shared[0] = (in_thread_T_softmax_n_1_to_256__m_769_to_1024__expsum_shared[0] + __expf((lv38[(((ax2_fused_0_1 * (int64_t)256) + (((int64_t)blockIdx.x) * m)) + ((int64_t)threadIdx.x))] - T_softmax_n_1_to_256__m_769_to_1024__maxelem_shared[0])));
    }
  }
  float red_buf0_2[1];
  uint mask_2[1];
  float t0_2[1];
  float red_buf0_3[1];
  uint mask_3[1];
  float t0_3[1];
  __shared__ float red_buf_staging_1[8];
  red_buf0_3[0] = in_thread_T_softmax_n_1_to_256__m_769_to_1024__expsum_shared[0];
  mask_3[0] = __activemask();
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 16, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 8, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 4, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 2, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 1, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging_1[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_3[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)8) {
    red_buf0_2[0] = red_buf_staging_1[((int64_t)threadIdx.x)];
  }
  mask_2[0] = (__activemask() & (uint)255);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 4, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 2, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 1, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result_1)[0] = red_buf0_2[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    T_softmax_n_1_to_256__m_769_to_1024__expsum_shared[0] = ((volatile float*)red_result_1)[0];
  }
  __syncthreads();
  for (int64_t ax2_0 = 0; ax2_0 < ((m + (int64_t)255) >> (int64_t)8); ++ax2_0) {
    if (((ax2_0 * (int64_t)256) + ((int64_t)threadIdx.x)) < m) {
      var_compute_intermediate[(((ax2_0 * (int64_t)256) + (((int64_t)blockIdx.x) * m)) + ((int64_t)threadIdx.x))] = ((half)(__expf((lv38[(((ax2_0 * (int64_t)256) + (((int64_t)blockIdx.x) * m)) + ((int64_t)threadIdx.x))] - T_softmax_n_1_to_256__m_769_to_1024__maxelem_shared[0])) / T_softmax_n_1_to_256__m_769_to_1024__expsum_shared[0]));
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) softmax_n_1_to_768__m_1025_to_1792__kernel(float* __restrict__ lv38, half* __restrict__ var_compute_intermediate, int64_t m, int64_t n) {
  extern __shared__ float lv38_shared_dyn[];
  float in_thread_T_softmax_n_1_to_768__m_1025_to_1792__maxelem_shared[1];
  __shared__ float red_result[1];
  __shared__ float T_softmax_n_1_to_768__m_1025_to_1792__maxelem_shared[1];
  float in_thread_T_softmax_n_1_to_768__m_1025_to_1792__expsum_shared[1];
  __shared__ float red_result_1[1];
  __shared__ float T_softmax_n_1_to_768__m_1025_to_1792__expsum_shared[1];
  for (int64_t ax3_0 = 0; ax3_0 < ((m + (int64_t)511) >> (int64_t)9); ++ax3_0) {
    for (int64_t ax3_2_s = 0; ax3_2_s < (int64_t)4; ++ax3_2_s) {
      if (((((ax3_0 * (int64_t)512) + (((int64_t)threadIdx.x) * (int64_t)4)) + ax3_2_s) < m) && ((((ax3_0 * (int64_t)128) + ((int64_t)threadIdx.x)) - (((m + (int64_t)127) >> (int64_t)7) * (int64_t)32)) < (int64_t)0)) {
        lv38_shared_dyn[(((ax3_0 * (int64_t)512) + (((int64_t)threadIdx.x) * (int64_t)4)) + ax3_2_s)] = lv38[((((ax3_0 * (int64_t)512) + (((int64_t)threadIdx.x) * (int64_t)4)) + (((int64_t)blockIdx.x) * m)) + ax3_2_s)];
      }
    }
  }
  in_thread_T_softmax_n_1_to_768__m_1025_to_1792__maxelem_shared[0] = -3.402823e+38f;
  __syncthreads();
  for (int64_t ax2_fused_0 = 0; ax2_fused_0 < ((m + (int64_t)127) >> (int64_t)7); ++ax2_fused_0) {
    if (((ax2_fused_0 * (int64_t)128) + ((int64_t)threadIdx.x)) < m) {
      in_thread_T_softmax_n_1_to_768__m_1025_to_1792__maxelem_shared[0] = max(in_thread_T_softmax_n_1_to_768__m_1025_to_1792__maxelem_shared[0], lv38_shared_dyn[((ax2_fused_0 * (int64_t)128) + ((int64_t)threadIdx.x))]);
    }
  }
  float red_buf0[1];
  uint mask[1];
  float t0[1];
  float red_buf0_1[1];
  uint mask_1[1];
  float t0_1[1];
  __shared__ float red_buf_staging[4];
  red_buf0_1[0] = in_thread_T_softmax_n_1_to_768__m_1025_to_1792__maxelem_shared[0];
  mask_1[0] = __activemask();
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_1[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)4) {
    red_buf0[0] = red_buf_staging[((int64_t)threadIdx.x)];
  }
  mask[0] = (__activemask() & (uint)15);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result)[0] = red_buf0[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    T_softmax_n_1_to_768__m_1025_to_1792__maxelem_shared[0] = ((volatile float*)red_result)[0];
  }
  in_thread_T_softmax_n_1_to_768__m_1025_to_1792__expsum_shared[0] = 0.000000e+00f;
  __syncthreads();
  for (int64_t ax2_fused_0_1 = 0; ax2_fused_0_1 < ((m + (int64_t)127) >> (int64_t)7); ++ax2_fused_0_1) {
    if (((ax2_fused_0_1 * (int64_t)128) + ((int64_t)threadIdx.x)) < m) {
      in_thread_T_softmax_n_1_to_768__m_1025_to_1792__expsum_shared[0] = (in_thread_T_softmax_n_1_to_768__m_1025_to_1792__expsum_shared[0] + __expf((lv38_shared_dyn[((ax2_fused_0_1 * (int64_t)128) + ((int64_t)threadIdx.x))] - T_softmax_n_1_to_768__m_1025_to_1792__maxelem_shared[0])));
    }
  }
  float red_buf0_2[1];
  uint mask_2[1];
  float t0_2[1];
  float red_buf0_3[1];
  uint mask_3[1];
  float t0_3[1];
  __shared__ float red_buf_staging_1[4];
  red_buf0_3[0] = in_thread_T_softmax_n_1_to_768__m_1025_to_1792__expsum_shared[0];
  mask_3[0] = __activemask();
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 16, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 8, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 4, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 2, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 1, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging_1[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_3[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)4) {
    red_buf0_2[0] = red_buf_staging_1[((int64_t)threadIdx.x)];
  }
  mask_2[0] = (__activemask() & (uint)15);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 2, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 1, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result_1)[0] = red_buf0_2[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    T_softmax_n_1_to_768__m_1025_to_1792__expsum_shared[0] = ((volatile float*)red_result_1)[0];
  }
  __syncthreads();
  for (int64_t ax2_0 = 0; ax2_0 < ((m + (int64_t)127) >> (int64_t)7); ++ax2_0) {
    if (((ax2_0 * (int64_t)128) + ((int64_t)threadIdx.x)) < m) {
      var_compute_intermediate[(((ax2_0 * (int64_t)128) + (((int64_t)blockIdx.x) * m)) + ((int64_t)threadIdx.x))] = ((half)(__expf((lv38_shared_dyn[((ax2_0 * (int64_t)128) + ((int64_t)threadIdx.x))] - T_softmax_n_1_to_768__m_1025_to_1792__maxelem_shared[0])) / T_softmax_n_1_to_768__m_1025_to_1792__expsum_shared[0]));
    }
  }
}

extern "C" __global__ void __launch_bounds__(160) softmax_n_1_to_1792__m_1537_to_2048__kernel(float* __restrict__ lv38, half* __restrict__ var_compute_intermediate, int64_t m, int64_t n) {
  extern __shared__ float lv38_shared_dyn[];
  float in_thread_T_softmax_n_1_to_1792__m_1537_to_2048__maxelem_shared[1];
  __shared__ float red_result[1];
  __shared__ float T_softmax_n_1_to_1792__m_1537_to_2048__maxelem_shared[1];
  float in_thread_T_softmax_n_1_to_1792__m_1537_to_2048__expsum_shared[1];
  __shared__ float red_result_1[1];
  __shared__ float T_softmax_n_1_to_1792__m_1537_to_2048__expsum_shared[1];
  for (int64_t ax3_0 = 0; ax3_0 < (((m + (int64_t)639) / (int64_t)640) + (((m + (int64_t)639) % (int64_t)640) >> (int64_t)63)); ++ax3_0) {
    for (int64_t ax3_2_s = 0; ax3_2_s < (int64_t)4; ++ax3_2_s) {
      if (((((ax3_0 * (int64_t)640) + (((int64_t)threadIdx.x) * (int64_t)4)) + ax3_2_s) < m) && ((((ax3_0 * (int64_t)160) + ((int64_t)threadIdx.x)) - ((((m + (int64_t)159) / (int64_t)160) + (((m + (int64_t)159) % (int64_t)160) >> (int64_t)63)) * (int64_t)40)) < (int64_t)0)) {
        lv38_shared_dyn[(((ax3_0 * (int64_t)640) + (((int64_t)threadIdx.x) * (int64_t)4)) + ax3_2_s)] = lv38[((((ax3_0 * (int64_t)640) + (((int64_t)threadIdx.x) * (int64_t)4)) + (((int64_t)blockIdx.x) * m)) + ax3_2_s)];
      }
    }
  }
  in_thread_T_softmax_n_1_to_1792__m_1537_to_2048__maxelem_shared[0] = -3.402823e+38f;
  __syncthreads();
  for (int64_t ax2_fused_0 = 0; ax2_fused_0 < (((m + (int64_t)159) / (int64_t)160) + (((m + (int64_t)159) % (int64_t)160) >> (int64_t)63)); ++ax2_fused_0) {
    if (((ax2_fused_0 * (int64_t)160) + ((int64_t)threadIdx.x)) < m) {
      in_thread_T_softmax_n_1_to_1792__m_1537_to_2048__maxelem_shared[0] = max(in_thread_T_softmax_n_1_to_1792__m_1537_to_2048__maxelem_shared[0], lv38_shared_dyn[((ax2_fused_0 * (int64_t)160) + ((int64_t)threadIdx.x))]);
    }
  }
  float red_buf0[1];
  uint mask[1];
  float t0[1];
  float red_buf0_1[1];
  uint mask_1[1];
  float t0_1[1];
  __shared__ float red_buf_staging[5];
  red_buf0_1[0] = in_thread_T_softmax_n_1_to_1792__m_1537_to_2048__maxelem_shared[0];
  mask_1[0] = __activemask();
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_1[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)5) {
    red_buf0[0] = red_buf_staging[((int64_t)threadIdx.x)];
  }
  mask[0] = (__activemask() & (uint)31);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  if (((int64_t)threadIdx.x) < (int64_t)1) {
    red_buf0[0] = max(red_buf0[0], t0[0]);
  }
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result)[0] = red_buf0[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    T_softmax_n_1_to_1792__m_1537_to_2048__maxelem_shared[0] = ((volatile float*)red_result)[0];
  }
  in_thread_T_softmax_n_1_to_1792__m_1537_to_2048__expsum_shared[0] = 0.000000e+00f;
  __syncthreads();
  for (int64_t ax2_fused_0_1 = 0; ax2_fused_0_1 < (((m + (int64_t)159) / (int64_t)160) + (((m + (int64_t)159) % (int64_t)160) >> (int64_t)63)); ++ax2_fused_0_1) {
    if (((ax2_fused_0_1 * (int64_t)160) + ((int64_t)threadIdx.x)) < m) {
      in_thread_T_softmax_n_1_to_1792__m_1537_to_2048__expsum_shared[0] = (in_thread_T_softmax_n_1_to_1792__m_1537_to_2048__expsum_shared[0] + __expf((lv38_shared_dyn[((ax2_fused_0_1 * (int64_t)160) + ((int64_t)threadIdx.x))] - T_softmax_n_1_to_1792__m_1537_to_2048__maxelem_shared[0])));
    }
  }
  float red_buf0_2[1];
  uint mask_2[1];
  float t0_2[1];
  float red_buf0_3[1];
  uint mask_3[1];
  float t0_3[1];
  __shared__ float red_buf_staging_1[5];
  red_buf0_3[0] = in_thread_T_softmax_n_1_to_1792__m_1537_to_2048__expsum_shared[0];
  mask_3[0] = __activemask();
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 16, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 8, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 4, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 2, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 1, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging_1[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_3[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)5) {
    red_buf0_2[0] = red_buf_staging_1[((int64_t)threadIdx.x)];
  }
  mask_2[0] = (__activemask() & (uint)31);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 4, 32);
  if (((int64_t)threadIdx.x) < (int64_t)1) {
    red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  }
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 2, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 1, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result_1)[0] = red_buf0_2[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    T_softmax_n_1_to_1792__m_1537_to_2048__expsum_shared[0] = ((volatile float*)red_result_1)[0];
  }
  __syncthreads();
  for (int64_t ax2_0 = 0; ax2_0 < (((m + (int64_t)159) / (int64_t)160) + (((m + (int64_t)159) % (int64_t)160) >> (int64_t)63)); ++ax2_0) {
    if (((ax2_0 * (int64_t)160) + ((int64_t)threadIdx.x)) < m) {
      var_compute_intermediate[(((ax2_0 * (int64_t)160) + (((int64_t)blockIdx.x) * m)) + ((int64_t)threadIdx.x))] = ((half)(__expf((lv38_shared_dyn[((ax2_0 * (int64_t)160) + ((int64_t)threadIdx.x))] - T_softmax_n_1_to_1792__m_1537_to_2048__maxelem_shared[0])) / T_softmax_n_1_to_1792__m_1537_to_2048__expsum_shared[0]));
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) softmax_n_513_to_768__m_1281_to_2048__kernel(float* __restrict__ lv38, half* __restrict__ var_compute_intermediate, int64_t m, int64_t n) {
  extern __shared__ float lv38_shared_dyn[];
  float in_thread_T_softmax_n_513_to_768__m_1281_to_2048__maxelem_shared[1];
  __shared__ float red_result[1];
  __shared__ float T_softmax_n_513_to_768__m_1281_to_2048__maxelem_shared[1];
  float in_thread_T_softmax_n_513_to_768__m_1281_to_2048__expsum_shared[1];
  __shared__ float red_result_1[1];
  __shared__ float T_softmax_n_513_to_768__m_1281_to_2048__expsum_shared[1];
  for (int64_t ax3_0 = 0; ax3_0 < ((m + (int64_t)511) >> (int64_t)9); ++ax3_0) {
    for (int64_t ax3_2_s = 0; ax3_2_s < (int64_t)4; ++ax3_2_s) {
      if (((((ax3_0 * (int64_t)512) + (((int64_t)threadIdx.x) * (int64_t)4)) + ax3_2_s) < m) && ((((ax3_0 * (int64_t)128) + ((int64_t)threadIdx.x)) - (((m + (int64_t)127) >> (int64_t)7) * (int64_t)32)) < (int64_t)0)) {
        lv38_shared_dyn[(((ax3_0 * (int64_t)512) + (((int64_t)threadIdx.x) * (int64_t)4)) + ax3_2_s)] = lv38[((((ax3_0 * (int64_t)512) + (((int64_t)threadIdx.x) * (int64_t)4)) + (((int64_t)blockIdx.x) * m)) + ax3_2_s)];
      }
    }
  }
  in_thread_T_softmax_n_513_to_768__m_1281_to_2048__maxelem_shared[0] = -3.402823e+38f;
  __syncthreads();
  for (int64_t ax2_fused_0 = 0; ax2_fused_0 < ((m + (int64_t)127) >> (int64_t)7); ++ax2_fused_0) {
    if (((ax2_fused_0 * (int64_t)128) + ((int64_t)threadIdx.x)) < m) {
      in_thread_T_softmax_n_513_to_768__m_1281_to_2048__maxelem_shared[0] = max(in_thread_T_softmax_n_513_to_768__m_1281_to_2048__maxelem_shared[0], lv38_shared_dyn[((ax2_fused_0 * (int64_t)128) + ((int64_t)threadIdx.x))]);
    }
  }
  float red_buf0[1];
  uint mask[1];
  float t0[1];
  float red_buf0_1[1];
  uint mask_1[1];
  float t0_1[1];
  __shared__ float red_buf_staging[4];
  red_buf0_1[0] = in_thread_T_softmax_n_513_to_768__m_1281_to_2048__maxelem_shared[0];
  mask_1[0] = __activemask();
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_1[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)4) {
    red_buf0[0] = red_buf_staging[((int64_t)threadIdx.x)];
  }
  mask[0] = (__activemask() & (uint)15);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result)[0] = red_buf0[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    T_softmax_n_513_to_768__m_1281_to_2048__maxelem_shared[0] = ((volatile float*)red_result)[0];
  }
  in_thread_T_softmax_n_513_to_768__m_1281_to_2048__expsum_shared[0] = 0.000000e+00f;
  __syncthreads();
  for (int64_t ax2_fused_0_1 = 0; ax2_fused_0_1 < ((m + (int64_t)127) >> (int64_t)7); ++ax2_fused_0_1) {
    if (((ax2_fused_0_1 * (int64_t)128) + ((int64_t)threadIdx.x)) < m) {
      in_thread_T_softmax_n_513_to_768__m_1281_to_2048__expsum_shared[0] = (in_thread_T_softmax_n_513_to_768__m_1281_to_2048__expsum_shared[0] + __expf((lv38_shared_dyn[((ax2_fused_0_1 * (int64_t)128) + ((int64_t)threadIdx.x))] - T_softmax_n_513_to_768__m_1281_to_2048__maxelem_shared[0])));
    }
  }
  float red_buf0_2[1];
  uint mask_2[1];
  float t0_2[1];
  float red_buf0_3[1];
  uint mask_3[1];
  float t0_3[1];
  __shared__ float red_buf_staging_1[4];
  red_buf0_3[0] = in_thread_T_softmax_n_513_to_768__m_1281_to_2048__expsum_shared[0];
  mask_3[0] = __activemask();
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 16, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 8, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 4, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 2, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 1, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging_1[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_3[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)4) {
    red_buf0_2[0] = red_buf_staging_1[((int64_t)threadIdx.x)];
  }
  mask_2[0] = (__activemask() & (uint)15);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 2, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 1, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result_1)[0] = red_buf0_2[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    T_softmax_n_513_to_768__m_1281_to_2048__expsum_shared[0] = ((volatile float*)red_result_1)[0];
  }
  __syncthreads();
  for (int64_t ax2_0 = 0; ax2_0 < ((m + (int64_t)127) >> (int64_t)7); ++ax2_0) {
    if (((ax2_0 * (int64_t)128) + ((int64_t)threadIdx.x)) < m) {
      var_compute_intermediate[(((ax2_0 * (int64_t)128) + (((int64_t)blockIdx.x) * m)) + ((int64_t)threadIdx.x))] = ((half)(__expf((lv38_shared_dyn[((ax2_0 * (int64_t)128) + ((int64_t)threadIdx.x))] - T_softmax_n_513_to_768__m_1281_to_2048__maxelem_shared[0])) / T_softmax_n_513_to_768__m_1281_to_2048__expsum_shared[0]));
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) softmax_n_769_to_1280__m_1025_to_1792__kernel(float* __restrict__ lv38, half* __restrict__ var_compute_intermediate, int64_t m, int64_t n) {
  extern __shared__ float lv38_shared_dyn[];
  float in_thread_T_softmax_n_769_to_1280__m_1025_to_1792__maxelem_shared[1];
  __shared__ float red_result[1];
  __shared__ float T_softmax_n_769_to_1280__m_1025_to_1792__maxelem_shared[1];
  float in_thread_T_softmax_n_769_to_1280__m_1025_to_1792__expsum_shared[1];
  __shared__ float red_result_1[1];
  __shared__ float T_softmax_n_769_to_1280__m_1025_to_1792__expsum_shared[1];
  for (int64_t ax3_0 = 0; ax3_0 < ((m + (int64_t)511) >> (int64_t)9); ++ax3_0) {
    for (int64_t ax3_2_s = 0; ax3_2_s < (int64_t)4; ++ax3_2_s) {
      if (((((ax3_0 * (int64_t)512) + (((int64_t)threadIdx.x) * (int64_t)4)) + ax3_2_s) < m) && ((((ax3_0 * (int64_t)128) + ((int64_t)threadIdx.x)) - (((m + (int64_t)127) >> (int64_t)7) * (int64_t)32)) < (int64_t)0)) {
        lv38_shared_dyn[(((ax3_0 * (int64_t)512) + (((int64_t)threadIdx.x) * (int64_t)4)) + ax3_2_s)] = lv38[((((ax3_0 * (int64_t)512) + (((int64_t)threadIdx.x) * (int64_t)4)) + (((int64_t)blockIdx.x) * m)) + ax3_2_s)];
      }
    }
  }
  in_thread_T_softmax_n_769_to_1280__m_1025_to_1792__maxelem_shared[0] = -3.402823e+38f;
  __syncthreads();
  for (int64_t ax2_fused_0 = 0; ax2_fused_0 < ((m + (int64_t)127) >> (int64_t)7); ++ax2_fused_0) {
    if (((ax2_fused_0 * (int64_t)128) + ((int64_t)threadIdx.x)) < m) {
      in_thread_T_softmax_n_769_to_1280__m_1025_to_1792__maxelem_shared[0] = max(in_thread_T_softmax_n_769_to_1280__m_1025_to_1792__maxelem_shared[0], lv38_shared_dyn[((ax2_fused_0 * (int64_t)128) + ((int64_t)threadIdx.x))]);
    }
  }
  float red_buf0[1];
  uint mask[1];
  float t0[1];
  float red_buf0_1[1];
  uint mask_1[1];
  float t0_1[1];
  __shared__ float red_buf_staging[4];
  red_buf0_1[0] = in_thread_T_softmax_n_769_to_1280__m_1025_to_1792__maxelem_shared[0];
  mask_1[0] = __activemask();
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_1[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)4) {
    red_buf0[0] = red_buf_staging[((int64_t)threadIdx.x)];
  }
  mask[0] = (__activemask() & (uint)15);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result)[0] = red_buf0[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    T_softmax_n_769_to_1280__m_1025_to_1792__maxelem_shared[0] = ((volatile float*)red_result)[0];
  }
  in_thread_T_softmax_n_769_to_1280__m_1025_to_1792__expsum_shared[0] = 0.000000e+00f;
  __syncthreads();
  for (int64_t ax2_fused_0_1 = 0; ax2_fused_0_1 < ((m + (int64_t)127) >> (int64_t)7); ++ax2_fused_0_1) {
    if (((ax2_fused_0_1 * (int64_t)128) + ((int64_t)threadIdx.x)) < m) {
      in_thread_T_softmax_n_769_to_1280__m_1025_to_1792__expsum_shared[0] = (in_thread_T_softmax_n_769_to_1280__m_1025_to_1792__expsum_shared[0] + __expf((lv38_shared_dyn[((ax2_fused_0_1 * (int64_t)128) + ((int64_t)threadIdx.x))] - T_softmax_n_769_to_1280__m_1025_to_1792__maxelem_shared[0])));
    }
  }
  float red_buf0_2[1];
  uint mask_2[1];
  float t0_2[1];
  float red_buf0_3[1];
  uint mask_3[1];
  float t0_3[1];
  __shared__ float red_buf_staging_1[4];
  red_buf0_3[0] = in_thread_T_softmax_n_769_to_1280__m_1025_to_1792__expsum_shared[0];
  mask_3[0] = __activemask();
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 16, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 8, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 4, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 2, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 1, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging_1[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_3[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)4) {
    red_buf0_2[0] = red_buf_staging_1[((int64_t)threadIdx.x)];
  }
  mask_2[0] = (__activemask() & (uint)15);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 2, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 1, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result_1)[0] = red_buf0_2[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    T_softmax_n_769_to_1280__m_1025_to_1792__expsum_shared[0] = ((volatile float*)red_result_1)[0];
  }
  __syncthreads();
  for (int64_t ax2_0 = 0; ax2_0 < ((m + (int64_t)127) >> (int64_t)7); ++ax2_0) {
    if (((ax2_0 * (int64_t)128) + ((int64_t)threadIdx.x)) < m) {
      var_compute_intermediate[(((ax2_0 * (int64_t)128) + (((int64_t)blockIdx.x) * m)) + ((int64_t)threadIdx.x))] = ((half)(__expf((lv38_shared_dyn[((ax2_0 * (int64_t)128) + ((int64_t)threadIdx.x))] - T_softmax_n_769_to_1280__m_1025_to_1792__maxelem_shared[0])) / T_softmax_n_769_to_1280__m_1025_to_1792__expsum_shared[0]));
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) softmax_n_1025_to_1280__m_1281_to_2048__kernel(float* __restrict__ lv38, half* __restrict__ var_compute_intermediate, int64_t m, int64_t n) {
  extern __shared__ float lv38_shared_dyn[];
  float in_thread_T_softmax_n_1025_to_1280__m_1281_to_2048__maxelem_shared[1];
  __shared__ float red_result[1];
  __shared__ float T_softmax_n_1025_to_1280__m_1281_to_2048__maxelem_shared[1];
  float in_thread_T_softmax_n_1025_to_1280__m_1281_to_2048__expsum_shared[1];
  __shared__ float red_result_1[1];
  __shared__ float T_softmax_n_1025_to_1280__m_1281_to_2048__expsum_shared[1];
  for (int64_t ax3_0 = 0; ax3_0 < ((m + (int64_t)511) >> (int64_t)9); ++ax3_0) {
    for (int64_t ax3_2_s = 0; ax3_2_s < (int64_t)4; ++ax3_2_s) {
      if (((((ax3_0 * (int64_t)512) + (((int64_t)threadIdx.x) * (int64_t)4)) + ax3_2_s) < m) && ((((ax3_0 * (int64_t)128) + ((int64_t)threadIdx.x)) - (((m + (int64_t)127) >> (int64_t)7) * (int64_t)32)) < (int64_t)0)) {
        lv38_shared_dyn[(((ax3_0 * (int64_t)512) + (((int64_t)threadIdx.x) * (int64_t)4)) + ax3_2_s)] = lv38[((((ax3_0 * (int64_t)512) + (((int64_t)threadIdx.x) * (int64_t)4)) + (((int64_t)blockIdx.x) * m)) + ax3_2_s)];
      }
    }
  }
  in_thread_T_softmax_n_1025_to_1280__m_1281_to_2048__maxelem_shared[0] = -3.402823e+38f;
  __syncthreads();
  for (int64_t ax2_fused_0 = 0; ax2_fused_0 < ((m + (int64_t)127) >> (int64_t)7); ++ax2_fused_0) {
    if (((ax2_fused_0 * (int64_t)128) + ((int64_t)threadIdx.x)) < m) {
      in_thread_T_softmax_n_1025_to_1280__m_1281_to_2048__maxelem_shared[0] = max(in_thread_T_softmax_n_1025_to_1280__m_1281_to_2048__maxelem_shared[0], lv38_shared_dyn[((ax2_fused_0 * (int64_t)128) + ((int64_t)threadIdx.x))]);
    }
  }
  float red_buf0[1];
  uint mask[1];
  float t0[1];
  float red_buf0_1[1];
  uint mask_1[1];
  float t0_1[1];
  __shared__ float red_buf_staging[4];
  red_buf0_1[0] = in_thread_T_softmax_n_1025_to_1280__m_1281_to_2048__maxelem_shared[0];
  mask_1[0] = __activemask();
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_1[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)4) {
    red_buf0[0] = red_buf_staging[((int64_t)threadIdx.x)];
  }
  mask[0] = (__activemask() & (uint)15);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result)[0] = red_buf0[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    T_softmax_n_1025_to_1280__m_1281_to_2048__maxelem_shared[0] = ((volatile float*)red_result)[0];
  }
  in_thread_T_softmax_n_1025_to_1280__m_1281_to_2048__expsum_shared[0] = 0.000000e+00f;
  __syncthreads();
  for (int64_t ax2_fused_0_1 = 0; ax2_fused_0_1 < ((m + (int64_t)127) >> (int64_t)7); ++ax2_fused_0_1) {
    if (((ax2_fused_0_1 * (int64_t)128) + ((int64_t)threadIdx.x)) < m) {
      in_thread_T_softmax_n_1025_to_1280__m_1281_to_2048__expsum_shared[0] = (in_thread_T_softmax_n_1025_to_1280__m_1281_to_2048__expsum_shared[0] + __expf((lv38_shared_dyn[((ax2_fused_0_1 * (int64_t)128) + ((int64_t)threadIdx.x))] - T_softmax_n_1025_to_1280__m_1281_to_2048__maxelem_shared[0])));
    }
  }
  float red_buf0_2[1];
  uint mask_2[1];
  float t0_2[1];
  float red_buf0_3[1];
  uint mask_3[1];
  float t0_3[1];
  __shared__ float red_buf_staging_1[4];
  red_buf0_3[0] = in_thread_T_softmax_n_1025_to_1280__m_1281_to_2048__expsum_shared[0];
  mask_3[0] = __activemask();
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 16, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 8, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 4, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 2, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 1, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging_1[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_3[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)4) {
    red_buf0_2[0] = red_buf_staging_1[((int64_t)threadIdx.x)];
  }
  mask_2[0] = (__activemask() & (uint)15);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 2, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 1, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result_1)[0] = red_buf0_2[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    T_softmax_n_1025_to_1280__m_1281_to_2048__expsum_shared[0] = ((volatile float*)red_result_1)[0];
  }
  __syncthreads();
  for (int64_t ax2_0 = 0; ax2_0 < ((m + (int64_t)127) >> (int64_t)7); ++ax2_0) {
    if (((ax2_0 * (int64_t)128) + ((int64_t)threadIdx.x)) < m) {
      var_compute_intermediate[(((ax2_0 * (int64_t)128) + (((int64_t)blockIdx.x) * m)) + ((int64_t)threadIdx.x))] = ((half)(__expf((lv38_shared_dyn[((ax2_0 * (int64_t)128) + ((int64_t)threadIdx.x))] - T_softmax_n_1025_to_1280__m_1281_to_2048__maxelem_shared[0])) / T_softmax_n_1025_to_1280__m_1281_to_2048__expsum_shared[0]));
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) softmax_n_1281_to_1792__m_1025_to_1792__kernel(float* __restrict__ lv38, half* __restrict__ var_compute_intermediate, int64_t m, int64_t n) {
  extern __shared__ float lv38_shared_dyn[];
  float in_thread_T_softmax_n_1281_to_1792__m_1025_to_1792__maxelem_shared[1];
  __shared__ float red_result[1];
  __shared__ float T_softmax_n_1281_to_1792__m_1025_to_1792__maxelem_shared[1];
  float in_thread_T_softmax_n_1281_to_1792__m_1025_to_1792__expsum_shared[1];
  __shared__ float red_result_1[1];
  __shared__ float T_softmax_n_1281_to_1792__m_1025_to_1792__expsum_shared[1];
  for (int64_t ax3_0 = 0; ax3_0 < ((m + (int64_t)511) >> (int64_t)9); ++ax3_0) {
    for (int64_t ax3_2_s = 0; ax3_2_s < (int64_t)4; ++ax3_2_s) {
      if (((((ax3_0 * (int64_t)512) + (((int64_t)threadIdx.x) * (int64_t)4)) + ax3_2_s) < m) && ((((ax3_0 * (int64_t)128) + ((int64_t)threadIdx.x)) - (((m + (int64_t)127) >> (int64_t)7) * (int64_t)32)) < (int64_t)0)) {
        lv38_shared_dyn[(((ax3_0 * (int64_t)512) + (((int64_t)threadIdx.x) * (int64_t)4)) + ax3_2_s)] = lv38[((((ax3_0 * (int64_t)512) + (((int64_t)threadIdx.x) * (int64_t)4)) + (((int64_t)blockIdx.x) * m)) + ax3_2_s)];
      }
    }
  }
  in_thread_T_softmax_n_1281_to_1792__m_1025_to_1792__maxelem_shared[0] = -3.402823e+38f;
  __syncthreads();
  for (int64_t ax2_fused_0 = 0; ax2_fused_0 < ((m + (int64_t)127) >> (int64_t)7); ++ax2_fused_0) {
    if (((ax2_fused_0 * (int64_t)128) + ((int64_t)threadIdx.x)) < m) {
      in_thread_T_softmax_n_1281_to_1792__m_1025_to_1792__maxelem_shared[0] = max(in_thread_T_softmax_n_1281_to_1792__m_1025_to_1792__maxelem_shared[0], lv38_shared_dyn[((ax2_fused_0 * (int64_t)128) + ((int64_t)threadIdx.x))]);
    }
  }
  float red_buf0[1];
  uint mask[1];
  float t0[1];
  float red_buf0_1[1];
  uint mask_1[1];
  float t0_1[1];
  __shared__ float red_buf_staging[4];
  red_buf0_1[0] = in_thread_T_softmax_n_1281_to_1792__m_1025_to_1792__maxelem_shared[0];
  mask_1[0] = __activemask();
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_1[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)4) {
    red_buf0[0] = red_buf_staging[((int64_t)threadIdx.x)];
  }
  mask[0] = (__activemask() & (uint)15);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result)[0] = red_buf0[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    T_softmax_n_1281_to_1792__m_1025_to_1792__maxelem_shared[0] = ((volatile float*)red_result)[0];
  }
  in_thread_T_softmax_n_1281_to_1792__m_1025_to_1792__expsum_shared[0] = 0.000000e+00f;
  __syncthreads();
  for (int64_t ax2_fused_0_1 = 0; ax2_fused_0_1 < ((m + (int64_t)127) >> (int64_t)7); ++ax2_fused_0_1) {
    if (((ax2_fused_0_1 * (int64_t)128) + ((int64_t)threadIdx.x)) < m) {
      in_thread_T_softmax_n_1281_to_1792__m_1025_to_1792__expsum_shared[0] = (in_thread_T_softmax_n_1281_to_1792__m_1025_to_1792__expsum_shared[0] + __expf((lv38_shared_dyn[((ax2_fused_0_1 * (int64_t)128) + ((int64_t)threadIdx.x))] - T_softmax_n_1281_to_1792__m_1025_to_1792__maxelem_shared[0])));
    }
  }
  float red_buf0_2[1];
  uint mask_2[1];
  float t0_2[1];
  float red_buf0_3[1];
  uint mask_3[1];
  float t0_3[1];
  __shared__ float red_buf_staging_1[4];
  red_buf0_3[0] = in_thread_T_softmax_n_1281_to_1792__m_1025_to_1792__expsum_shared[0];
  mask_3[0] = __activemask();
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 16, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 8, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 4, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 2, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 1, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging_1[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_3[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)4) {
    red_buf0_2[0] = red_buf_staging_1[((int64_t)threadIdx.x)];
  }
  mask_2[0] = (__activemask() & (uint)15);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 2, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 1, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result_1)[0] = red_buf0_2[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    T_softmax_n_1281_to_1792__m_1025_to_1792__expsum_shared[0] = ((volatile float*)red_result_1)[0];
  }
  __syncthreads();
  for (int64_t ax2_0 = 0; ax2_0 < ((m + (int64_t)127) >> (int64_t)7); ++ax2_0) {
    if (((ax2_0 * (int64_t)128) + ((int64_t)threadIdx.x)) < m) {
      var_compute_intermediate[(((ax2_0 * (int64_t)128) + (((int64_t)blockIdx.x) * m)) + ((int64_t)threadIdx.x))] = ((half)(__expf((lv38_shared_dyn[((ax2_0 * (int64_t)128) + ((int64_t)threadIdx.x))] - T_softmax_n_1281_to_1792__m_1025_to_1792__maxelem_shared[0])) / T_softmax_n_1281_to_1792__m_1025_to_1792__expsum_shared[0]));
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) softmax_n_1793_to_2048__m_1025_to_1536__kernel(float* __restrict__ lv38, half* __restrict__ var_compute_intermediate, int64_t m, int64_t n) {
  extern __shared__ float lv38_shared_dyn[];
  float in_thread_T_softmax_n_1793_to_2048__m_1025_to_1536__maxelem_shared[1];
  __shared__ float red_result[1];
  __shared__ float T_softmax_n_1793_to_2048__m_1025_to_1536__maxelem_shared[1];
  float in_thread_T_softmax_n_1793_to_2048__m_1025_to_1536__expsum_shared[1];
  __shared__ float red_result_1[1];
  __shared__ float T_softmax_n_1793_to_2048__m_1025_to_1536__expsum_shared[1];
  for (int64_t ax3_0 = 0; ax3_0 < ((m + (int64_t)511) >> (int64_t)9); ++ax3_0) {
    for (int64_t ax3_2_s = 0; ax3_2_s < (int64_t)4; ++ax3_2_s) {
      if (((((ax3_0 * (int64_t)512) + (((int64_t)threadIdx.x) * (int64_t)4)) + ax3_2_s) < m) && ((((ax3_0 * (int64_t)128) + ((int64_t)threadIdx.x)) - (((m + (int64_t)127) >> (int64_t)7) * (int64_t)32)) < (int64_t)0)) {
        lv38_shared_dyn[(((ax3_0 * (int64_t)512) + (((int64_t)threadIdx.x) * (int64_t)4)) + ax3_2_s)] = lv38[((((ax3_0 * (int64_t)512) + (((int64_t)threadIdx.x) * (int64_t)4)) + (((int64_t)blockIdx.x) * m)) + ax3_2_s)];
      }
    }
  }
  in_thread_T_softmax_n_1793_to_2048__m_1025_to_1536__maxelem_shared[0] = -3.402823e+38f;
  __syncthreads();
  for (int64_t ax2_fused_0 = 0; ax2_fused_0 < ((m + (int64_t)127) >> (int64_t)7); ++ax2_fused_0) {
    if (((ax2_fused_0 * (int64_t)128) + ((int64_t)threadIdx.x)) < m) {
      in_thread_T_softmax_n_1793_to_2048__m_1025_to_1536__maxelem_shared[0] = max(in_thread_T_softmax_n_1793_to_2048__m_1025_to_1536__maxelem_shared[0], lv38_shared_dyn[((ax2_fused_0 * (int64_t)128) + ((int64_t)threadIdx.x))]);
    }
  }
  float red_buf0[1];
  uint mask[1];
  float t0[1];
  float red_buf0_1[1];
  uint mask_1[1];
  float t0_1[1];
  __shared__ float red_buf_staging[4];
  red_buf0_1[0] = in_thread_T_softmax_n_1793_to_2048__m_1025_to_1536__maxelem_shared[0];
  mask_1[0] = __activemask();
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_1[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)4) {
    red_buf0[0] = red_buf_staging[((int64_t)threadIdx.x)];
  }
  mask[0] = (__activemask() & (uint)15);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result)[0] = red_buf0[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    T_softmax_n_1793_to_2048__m_1025_to_1536__maxelem_shared[0] = ((volatile float*)red_result)[0];
  }
  in_thread_T_softmax_n_1793_to_2048__m_1025_to_1536__expsum_shared[0] = 0.000000e+00f;
  __syncthreads();
  for (int64_t ax2_fused_0_1 = 0; ax2_fused_0_1 < ((m + (int64_t)127) >> (int64_t)7); ++ax2_fused_0_1) {
    if (((ax2_fused_0_1 * (int64_t)128) + ((int64_t)threadIdx.x)) < m) {
      in_thread_T_softmax_n_1793_to_2048__m_1025_to_1536__expsum_shared[0] = (in_thread_T_softmax_n_1793_to_2048__m_1025_to_1536__expsum_shared[0] + __expf((lv38_shared_dyn[((ax2_fused_0_1 * (int64_t)128) + ((int64_t)threadIdx.x))] - T_softmax_n_1793_to_2048__m_1025_to_1536__maxelem_shared[0])));
    }
  }
  float red_buf0_2[1];
  uint mask_2[1];
  float t0_2[1];
  float red_buf0_3[1];
  uint mask_3[1];
  float t0_3[1];
  __shared__ float red_buf_staging_1[4];
  red_buf0_3[0] = in_thread_T_softmax_n_1793_to_2048__m_1025_to_1536__expsum_shared[0];
  mask_3[0] = __activemask();
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 16, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 8, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 4, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 2, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 1, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging_1[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_3[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)4) {
    red_buf0_2[0] = red_buf_staging_1[((int64_t)threadIdx.x)];
  }
  mask_2[0] = (__activemask() & (uint)15);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 2, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 1, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result_1)[0] = red_buf0_2[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    T_softmax_n_1793_to_2048__m_1025_to_1536__expsum_shared[0] = ((volatile float*)red_result_1)[0];
  }
  __syncthreads();
  for (int64_t ax2_0 = 0; ax2_0 < ((m + (int64_t)127) >> (int64_t)7); ++ax2_0) {
    if (((ax2_0 * (int64_t)128) + ((int64_t)threadIdx.x)) < m) {
      var_compute_intermediate[(((ax2_0 * (int64_t)128) + (((int64_t)blockIdx.x) * m)) + ((int64_t)threadIdx.x))] = ((half)(__expf((lv38_shared_dyn[((ax2_0 * (int64_t)128) + ((int64_t)threadIdx.x))] - T_softmax_n_1793_to_2048__m_1025_to_1536__maxelem_shared[0])) / T_softmax_n_1793_to_2048__m_1025_to_1536__expsum_shared[0]));
    }
  }
}

extern "C" __global__ void __launch_bounds__(160) softmax_n_1793_to_2048__m_1793_to_2048__kernel(float* __restrict__ lv38, half* __restrict__ var_compute_intermediate, int64_t m, int64_t n) {
  extern __shared__ float lv38_shared_dyn[];
  float in_thread_T_softmax_n_1793_to_2048__m_1793_to_2048__maxelem_shared[1];
  __shared__ float red_result[1];
  __shared__ float T_softmax_n_1793_to_2048__m_1793_to_2048__maxelem_shared[1];
  float in_thread_T_softmax_n_1793_to_2048__m_1793_to_2048__expsum_shared[1];
  __shared__ float red_result_1[1];
  __shared__ float T_softmax_n_1793_to_2048__m_1793_to_2048__expsum_shared[1];
  for (int64_t ax3_0 = 0; ax3_0 < (((m + (int64_t)639) / (int64_t)640) + (((m + (int64_t)639) % (int64_t)640) >> (int64_t)63)); ++ax3_0) {
    for (int64_t ax3_2_s = 0; ax3_2_s < (int64_t)4; ++ax3_2_s) {
      if (((((ax3_0 * (int64_t)640) + (((int64_t)threadIdx.x) * (int64_t)4)) + ax3_2_s) < m) && ((((ax3_0 * (int64_t)160) + ((int64_t)threadIdx.x)) - ((((m + (int64_t)159) / (int64_t)160) + (((m + (int64_t)159) % (int64_t)160) >> (int64_t)63)) * (int64_t)40)) < (int64_t)0)) {
        lv38_shared_dyn[(((ax3_0 * (int64_t)640) + (((int64_t)threadIdx.x) * (int64_t)4)) + ax3_2_s)] = lv38[((((ax3_0 * (int64_t)640) + (((int64_t)threadIdx.x) * (int64_t)4)) + (((int64_t)blockIdx.x) * m)) + ax3_2_s)];
      }
    }
  }
  in_thread_T_softmax_n_1793_to_2048__m_1793_to_2048__maxelem_shared[0] = -3.402823e+38f;
  __syncthreads();
  for (int64_t ax2_fused_0 = 0; ax2_fused_0 < (((m + (int64_t)159) / (int64_t)160) + (((m + (int64_t)159) % (int64_t)160) >> (int64_t)63)); ++ax2_fused_0) {
    if (((ax2_fused_0 * (int64_t)160) + ((int64_t)threadIdx.x)) < m) {
      in_thread_T_softmax_n_1793_to_2048__m_1793_to_2048__maxelem_shared[0] = max(in_thread_T_softmax_n_1793_to_2048__m_1793_to_2048__maxelem_shared[0], lv38_shared_dyn[((ax2_fused_0 * (int64_t)160) + ((int64_t)threadIdx.x))]);
    }
  }
  float red_buf0[1];
  uint mask[1];
  float t0[1];
  float red_buf0_1[1];
  uint mask_1[1];
  float t0_1[1];
  __shared__ float red_buf_staging[5];
  red_buf0_1[0] = in_thread_T_softmax_n_1793_to_2048__m_1793_to_2048__maxelem_shared[0];
  mask_1[0] = __activemask();
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_1[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)5) {
    red_buf0[0] = red_buf_staging[((int64_t)threadIdx.x)];
  }
  mask[0] = (__activemask() & (uint)31);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  if (((int64_t)threadIdx.x) < (int64_t)1) {
    red_buf0[0] = max(red_buf0[0], t0[0]);
  }
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result)[0] = red_buf0[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    T_softmax_n_1793_to_2048__m_1793_to_2048__maxelem_shared[0] = ((volatile float*)red_result)[0];
  }
  in_thread_T_softmax_n_1793_to_2048__m_1793_to_2048__expsum_shared[0] = 0.000000e+00f;
  __syncthreads();
  for (int64_t ax2_fused_0_1 = 0; ax2_fused_0_1 < (((m + (int64_t)159) / (int64_t)160) + (((m + (int64_t)159) % (int64_t)160) >> (int64_t)63)); ++ax2_fused_0_1) {
    if (((ax2_fused_0_1 * (int64_t)160) + ((int64_t)threadIdx.x)) < m) {
      in_thread_T_softmax_n_1793_to_2048__m_1793_to_2048__expsum_shared[0] = (in_thread_T_softmax_n_1793_to_2048__m_1793_to_2048__expsum_shared[0] + __expf((lv38_shared_dyn[((ax2_fused_0_1 * (int64_t)160) + ((int64_t)threadIdx.x))] - T_softmax_n_1793_to_2048__m_1793_to_2048__maxelem_shared[0])));
    }
  }
  float red_buf0_2[1];
  uint mask_2[1];
  float t0_2[1];
  float red_buf0_3[1];
  uint mask_3[1];
  float t0_3[1];
  __shared__ float red_buf_staging_1[5];
  red_buf0_3[0] = in_thread_T_softmax_n_1793_to_2048__m_1793_to_2048__expsum_shared[0];
  mask_3[0] = __activemask();
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 16, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 8, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 4, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 2, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 1, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  if ((((int64_t)threadIdx.x) % (int64_t)32) == (int64_t)0) {
    red_buf_staging_1[(((int64_t)threadIdx.x) >> (int64_t)5)] = red_buf0_3[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)5) {
    red_buf0_2[0] = red_buf_staging_1[((int64_t)threadIdx.x)];
  }
  mask_2[0] = (__activemask() & (uint)31);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 4, 32);
  if (((int64_t)threadIdx.x) < (int64_t)1) {
    red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  }
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 2, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 1, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    ((volatile float*)red_result_1)[0] = red_buf0_2[0];
  }
  __syncthreads();
  if (((int64_t)threadIdx.x) == (int64_t)0) {
    T_softmax_n_1793_to_2048__m_1793_to_2048__expsum_shared[0] = ((volatile float*)red_result_1)[0];
  }
  __syncthreads();
  for (int64_t ax2_0 = 0; ax2_0 < (((m + (int64_t)159) / (int64_t)160) + (((m + (int64_t)159) % (int64_t)160) >> (int64_t)63)); ++ax2_0) {
    if (((ax2_0 * (int64_t)160) + ((int64_t)threadIdx.x)) < m) {
      var_compute_intermediate[(((ax2_0 * (int64_t)160) + (((int64_t)blockIdx.x) * m)) + ((int64_t)threadIdx.x))] = ((half)(__expf((lv38_shared_dyn[((ax2_0 * (int64_t)160) + ((int64_t)threadIdx.x))] - T_softmax_n_1793_to_2048__m_1793_to_2048__maxelem_shared[0])) / T_softmax_n_1793_to_2048__m_1793_to_2048__expsum_shared[0]));
    }
  }
}

