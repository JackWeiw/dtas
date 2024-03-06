
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
extern "C" __global__ void __launch_bounds__(256) cast_n_1_to_128__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_129_to_256__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_257_to_384__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_385_to_512__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_513_to_640__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_641_to_768__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_769_to_896__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_897_to_1024__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_1025_to_1152__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_1153_to_1280__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_1281_to_1408__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_1409_to_1536__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_1537_to_1664__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_1665_to_1792__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_1793_to_1920__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_1921_to_2048__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_2049_to_2176__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_2177_to_2304__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_2305_to_2432__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_2433_to_2560__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_2561_to_2688__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_2689_to_2816__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_2817_to_2944__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_2945_to_3072__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_3073_to_3200__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_3201_to_3328__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_3329_to_3456__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_3457_to_3584__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_3585_to_3712__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_3713_to_3840__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_3841_to_3968__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_3969_to_4096__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n);
extern "C" __global__ void __launch_bounds__(256) cast_n_1_to_128__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < ((n + (int64_t)127) >> (int64_t)7); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * ((n + (int64_t)127) >> (int64_t)7)) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)127) >> (int64_t)7)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)127) >> (int64_t)7)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_129_to_256__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < ((n + (int64_t)255) >> (int64_t)8); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * ((n + (int64_t)255) >> (int64_t)8)) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)255) >> (int64_t)8)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)255) >> (int64_t)8)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_257_to_384__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)383) / (int64_t)384) + (((n + (int64_t)383) % (int64_t)384) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)383) / (int64_t)384) + (((n + (int64_t)383) % (int64_t)384) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)383) / (int64_t)384) + (((n + (int64_t)383) % (int64_t)384) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)383) / (int64_t)384) + (((n + (int64_t)383) % (int64_t)384) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_385_to_512__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < ((n + (int64_t)511) >> (int64_t)9); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * ((n + (int64_t)511) >> (int64_t)9)) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)511) >> (int64_t)9)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)511) >> (int64_t)9)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_513_to_640__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)639) / (int64_t)640) + (((n + (int64_t)639) % (int64_t)640) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)639) / (int64_t)640) + (((n + (int64_t)639) % (int64_t)640) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)639) / (int64_t)640) + (((n + (int64_t)639) % (int64_t)640) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)639) / (int64_t)640) + (((n + (int64_t)639) % (int64_t)640) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_641_to_768__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)767) / (int64_t)768) + (((n + (int64_t)767) % (int64_t)768) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)767) / (int64_t)768) + (((n + (int64_t)767) % (int64_t)768) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)767) / (int64_t)768) + (((n + (int64_t)767) % (int64_t)768) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)767) / (int64_t)768) + (((n + (int64_t)767) % (int64_t)768) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_769_to_896__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)895) / (int64_t)896) + (((n + (int64_t)895) % (int64_t)896) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)895) / (int64_t)896) + (((n + (int64_t)895) % (int64_t)896) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)895) / (int64_t)896) + (((n + (int64_t)895) % (int64_t)896) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)895) / (int64_t)896) + (((n + (int64_t)895) % (int64_t)896) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_897_to_1024__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < ((n + (int64_t)1023) >> (int64_t)10); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * ((n + (int64_t)1023) >> (int64_t)10)) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)1023) >> (int64_t)10)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)1023) >> (int64_t)10)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_1025_to_1152__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)1151) / (int64_t)1152) + (((n + (int64_t)1151) % (int64_t)1152) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)1151) / (int64_t)1152) + (((n + (int64_t)1151) % (int64_t)1152) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1151) / (int64_t)1152) + (((n + (int64_t)1151) % (int64_t)1152) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1151) / (int64_t)1152) + (((n + (int64_t)1151) % (int64_t)1152) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_1153_to_1280__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)1279) / (int64_t)1280) + (((n + (int64_t)1279) % (int64_t)1280) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)1279) / (int64_t)1280) + (((n + (int64_t)1279) % (int64_t)1280) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1279) / (int64_t)1280) + (((n + (int64_t)1279) % (int64_t)1280) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1279) / (int64_t)1280) + (((n + (int64_t)1279) % (int64_t)1280) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_1281_to_1408__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)1407) / (int64_t)1408) + (((n + (int64_t)1407) % (int64_t)1408) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)1407) / (int64_t)1408) + (((n + (int64_t)1407) % (int64_t)1408) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1407) / (int64_t)1408) + (((n + (int64_t)1407) % (int64_t)1408) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1407) / (int64_t)1408) + (((n + (int64_t)1407) % (int64_t)1408) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_1409_to_1536__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_1537_to_1664__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)1663) / (int64_t)1664) + (((n + (int64_t)1663) % (int64_t)1664) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)1663) / (int64_t)1664) + (((n + (int64_t)1663) % (int64_t)1664) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1663) / (int64_t)1664) + (((n + (int64_t)1663) % (int64_t)1664) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1663) / (int64_t)1664) + (((n + (int64_t)1663) % (int64_t)1664) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_1665_to_1792__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)1791) / (int64_t)1792) + (((n + (int64_t)1791) % (int64_t)1792) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)1791) / (int64_t)1792) + (((n + (int64_t)1791) % (int64_t)1792) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1791) / (int64_t)1792) + (((n + (int64_t)1791) % (int64_t)1792) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1791) / (int64_t)1792) + (((n + (int64_t)1791) % (int64_t)1792) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_1793_to_1920__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)1919) / (int64_t)1920) + (((n + (int64_t)1919) % (int64_t)1920) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)1919) / (int64_t)1920) + (((n + (int64_t)1919) % (int64_t)1920) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1919) / (int64_t)1920) + (((n + (int64_t)1919) % (int64_t)1920) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1919) / (int64_t)1920) + (((n + (int64_t)1919) % (int64_t)1920) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_1921_to_2048__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < ((n + (int64_t)2047) >> (int64_t)11); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * ((n + (int64_t)2047) >> (int64_t)11)) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)2047) >> (int64_t)11)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)2047) >> (int64_t)11)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_2049_to_2176__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)2175) / (int64_t)2176) + (((n + (int64_t)2175) % (int64_t)2176) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)2175) / (int64_t)2176) + (((n + (int64_t)2175) % (int64_t)2176) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2175) / (int64_t)2176) + (((n + (int64_t)2175) % (int64_t)2176) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2175) / (int64_t)2176) + (((n + (int64_t)2175) % (int64_t)2176) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_2177_to_2304__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)2303) / (int64_t)2304) + (((n + (int64_t)2303) % (int64_t)2304) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)2303) / (int64_t)2304) + (((n + (int64_t)2303) % (int64_t)2304) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2303) / (int64_t)2304) + (((n + (int64_t)2303) % (int64_t)2304) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2303) / (int64_t)2304) + (((n + (int64_t)2303) % (int64_t)2304) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_2305_to_2432__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)2431) / (int64_t)2432) + (((n + (int64_t)2431) % (int64_t)2432) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)2431) / (int64_t)2432) + (((n + (int64_t)2431) % (int64_t)2432) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2431) / (int64_t)2432) + (((n + (int64_t)2431) % (int64_t)2432) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2431) / (int64_t)2432) + (((n + (int64_t)2431) % (int64_t)2432) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_2433_to_2560__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)2559) / (int64_t)2560) + (((n + (int64_t)2559) % (int64_t)2560) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)2559) / (int64_t)2560) + (((n + (int64_t)2559) % (int64_t)2560) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2559) / (int64_t)2560) + (((n + (int64_t)2559) % (int64_t)2560) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2559) / (int64_t)2560) + (((n + (int64_t)2559) % (int64_t)2560) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_2561_to_2688__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)2687) / (int64_t)2688) + (((n + (int64_t)2687) % (int64_t)2688) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)2687) / (int64_t)2688) + (((n + (int64_t)2687) % (int64_t)2688) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2687) / (int64_t)2688) + (((n + (int64_t)2687) % (int64_t)2688) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2687) / (int64_t)2688) + (((n + (int64_t)2687) % (int64_t)2688) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_2689_to_2816__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)2815) / (int64_t)2816) + (((n + (int64_t)2815) % (int64_t)2816) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)2815) / (int64_t)2816) + (((n + (int64_t)2815) % (int64_t)2816) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2815) / (int64_t)2816) + (((n + (int64_t)2815) % (int64_t)2816) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2815) / (int64_t)2816) + (((n + (int64_t)2815) % (int64_t)2816) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_2817_to_2944__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)2943) / (int64_t)2944) + (((n + (int64_t)2943) % (int64_t)2944) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)2943) / (int64_t)2944) + (((n + (int64_t)2943) % (int64_t)2944) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2943) / (int64_t)2944) + (((n + (int64_t)2943) % (int64_t)2944) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2943) / (int64_t)2944) + (((n + (int64_t)2943) % (int64_t)2944) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_2945_to_3072__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_3073_to_3200__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)3199) / (int64_t)3200) + (((n + (int64_t)3199) % (int64_t)3200) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)3199) / (int64_t)3200) + (((n + (int64_t)3199) % (int64_t)3200) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3199) / (int64_t)3200) + (((n + (int64_t)3199) % (int64_t)3200) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3199) / (int64_t)3200) + (((n + (int64_t)3199) % (int64_t)3200) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_3201_to_3328__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)3327) / (int64_t)3328) + (((n + (int64_t)3327) % (int64_t)3328) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)3327) / (int64_t)3328) + (((n + (int64_t)3327) % (int64_t)3328) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3327) / (int64_t)3328) + (((n + (int64_t)3327) % (int64_t)3328) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3327) / (int64_t)3328) + (((n + (int64_t)3327) % (int64_t)3328) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_3329_to_3456__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)3455) / (int64_t)3456) + (((n + (int64_t)3455) % (int64_t)3456) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)3455) / (int64_t)3456) + (((n + (int64_t)3455) % (int64_t)3456) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3455) / (int64_t)3456) + (((n + (int64_t)3455) % (int64_t)3456) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3455) / (int64_t)3456) + (((n + (int64_t)3455) % (int64_t)3456) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_3457_to_3584__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)3583) / (int64_t)3584) + (((n + (int64_t)3583) % (int64_t)3584) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)3583) / (int64_t)3584) + (((n + (int64_t)3583) % (int64_t)3584) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3583) / (int64_t)3584) + (((n + (int64_t)3583) % (int64_t)3584) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3583) / (int64_t)3584) + (((n + (int64_t)3583) % (int64_t)3584) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_3585_to_3712__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)3711) / (int64_t)3712) + (((n + (int64_t)3711) % (int64_t)3712) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)3711) / (int64_t)3712) + (((n + (int64_t)3711) % (int64_t)3712) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3711) / (int64_t)3712) + (((n + (int64_t)3711) % (int64_t)3712) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3711) / (int64_t)3712) + (((n + (int64_t)3711) % (int64_t)3712) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_3713_to_3840__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)3839) / (int64_t)3840) + (((n + (int64_t)3839) % (int64_t)3840) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)3839) / (int64_t)3840) + (((n + (int64_t)3839) % (int64_t)3840) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3839) / (int64_t)3840) + (((n + (int64_t)3839) % (int64_t)3840) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3839) / (int64_t)3840) + (((n + (int64_t)3839) % (int64_t)3840) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_3841_to_3968__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)3967) / (int64_t)3968) + (((n + (int64_t)3967) % (int64_t)3968) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)3967) / (int64_t)3968) + (((n + (int64_t)3967) % (int64_t)3968) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3967) / (int64_t)3968) + (((n + (int64_t)3967) % (int64_t)3968) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3967) / (int64_t)3968) + (((n + (int64_t)3967) % (int64_t)3968) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) cast_n_3969_to_4096__kernel(half* __restrict__ A, float* __restrict__ compute, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < ((n + (int64_t)4095) >> (int64_t)12); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * ((n + (int64_t)4095) >> (int64_t)12)) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      ulonglong4 __1;
      uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)4095) >> (int64_t)12)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
      ((float2*)(&(__1.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__1.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__1.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__1.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__1.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__1.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__1.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__1.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      *(ulonglong4*)(compute + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)4095) >> (int64_t)12)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

