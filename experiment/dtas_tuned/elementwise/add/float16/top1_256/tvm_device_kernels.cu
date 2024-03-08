
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
extern "C" __global__ void __launch_bounds__(256) add_n_1_to_256__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_1_to_256_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_257_to_512__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_257_to_512_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_513_to_768__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_513_to_768_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_769_to_1024__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_769_to_1024_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_1025_to_1280__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_1025_to_1280_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_1281_to_1536__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_1281_to_1536_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_1537_to_1792__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_1537_to_1792_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_1793_to_2048__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_1793_to_2048_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_2049_to_2304__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_2049_to_2304_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_2305_to_2560__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_2305_to_2560_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_2561_to_2816__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_2561_to_2816_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_2817_to_3072__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_2817_to_3072_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_3073_to_3328__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_3073_to_3328_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_3329_to_3584__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_3329_to_3584_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_3585_to_3840__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_3585_to_3840_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_3841_to_4096__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_3841_to_4096_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_1_to_256__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_1_to_256_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < ((n + (int64_t)255) >> (int64_t)8); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * ((n + (int64_t)255) >> (int64_t)8)) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      uint4 __1;
        uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)255) >> (int64_t)8)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        uint4 v__1 = *(uint4*)(B + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)255) >> (int64_t)8)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(T_add_n_1_to_256_ + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)255) >> (int64_t)8)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_257_to_512__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_257_to_512_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < ((n + (int64_t)511) >> (int64_t)9); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * ((n + (int64_t)511) >> (int64_t)9)) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      uint4 __1;
        uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)511) >> (int64_t)9)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        uint4 v__1 = *(uint4*)(B + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)511) >> (int64_t)9)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(T_add_n_257_to_512_ + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)511) >> (int64_t)9)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_513_to_768__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_513_to_768_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)767) / (int64_t)768) + (((n + (int64_t)767) % (int64_t)768) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)767) / (int64_t)768) + (((n + (int64_t)767) % (int64_t)768) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      uint4 __1;
        uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)767) / (int64_t)768) + (((n + (int64_t)767) % (int64_t)768) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        uint4 v__1 = *(uint4*)(B + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)767) / (int64_t)768) + (((n + (int64_t)767) % (int64_t)768) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(T_add_n_513_to_768_ + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)767) / (int64_t)768) + (((n + (int64_t)767) % (int64_t)768) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_769_to_1024__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_769_to_1024_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < ((n + (int64_t)1023) >> (int64_t)10); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * ((n + (int64_t)1023) >> (int64_t)10)) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      uint4 __1;
        uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)1023) >> (int64_t)10)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        uint4 v__1 = *(uint4*)(B + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)1023) >> (int64_t)10)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(T_add_n_769_to_1024_ + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)1023) >> (int64_t)10)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_1025_to_1280__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_1025_to_1280_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)1279) / (int64_t)1280) + (((n + (int64_t)1279) % (int64_t)1280) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)1279) / (int64_t)1280) + (((n + (int64_t)1279) % (int64_t)1280) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      uint4 __1;
        uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1279) / (int64_t)1280) + (((n + (int64_t)1279) % (int64_t)1280) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        uint4 v__1 = *(uint4*)(B + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1279) / (int64_t)1280) + (((n + (int64_t)1279) % (int64_t)1280) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(T_add_n_1025_to_1280_ + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1279) / (int64_t)1280) + (((n + (int64_t)1279) % (int64_t)1280) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_1281_to_1536__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_1281_to_1536_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      uint4 __1;
        uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        uint4 v__1 = *(uint4*)(B + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(T_add_n_1281_to_1536_ + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_1537_to_1792__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_1537_to_1792_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)1791) / (int64_t)1792) + (((n + (int64_t)1791) % (int64_t)1792) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)1791) / (int64_t)1792) + (((n + (int64_t)1791) % (int64_t)1792) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      uint4 __1;
        uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1791) / (int64_t)1792) + (((n + (int64_t)1791) % (int64_t)1792) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        uint4 v__1 = *(uint4*)(B + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1791) / (int64_t)1792) + (((n + (int64_t)1791) % (int64_t)1792) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(T_add_n_1537_to_1792_ + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1791) / (int64_t)1792) + (((n + (int64_t)1791) % (int64_t)1792) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_1793_to_2048__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_1793_to_2048_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < ((n + (int64_t)2047) >> (int64_t)11); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * ((n + (int64_t)2047) >> (int64_t)11)) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      uint4 __1;
        uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)2047) >> (int64_t)11)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        uint4 v__1 = *(uint4*)(B + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)2047) >> (int64_t)11)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(T_add_n_1793_to_2048_ + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)2047) >> (int64_t)11)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_2049_to_2304__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_2049_to_2304_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)2303) / (int64_t)2304) + (((n + (int64_t)2303) % (int64_t)2304) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)2303) / (int64_t)2304) + (((n + (int64_t)2303) % (int64_t)2304) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      uint4 __1;
        uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2303) / (int64_t)2304) + (((n + (int64_t)2303) % (int64_t)2304) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        uint4 v__1 = *(uint4*)(B + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2303) / (int64_t)2304) + (((n + (int64_t)2303) % (int64_t)2304) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(T_add_n_2049_to_2304_ + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2303) / (int64_t)2304) + (((n + (int64_t)2303) % (int64_t)2304) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_2305_to_2560__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_2305_to_2560_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)2559) / (int64_t)2560) + (((n + (int64_t)2559) % (int64_t)2560) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)2559) / (int64_t)2560) + (((n + (int64_t)2559) % (int64_t)2560) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      uint4 __1;
        uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2559) / (int64_t)2560) + (((n + (int64_t)2559) % (int64_t)2560) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        uint4 v__1 = *(uint4*)(B + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2559) / (int64_t)2560) + (((n + (int64_t)2559) % (int64_t)2560) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(T_add_n_2305_to_2560_ + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2559) / (int64_t)2560) + (((n + (int64_t)2559) % (int64_t)2560) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_2561_to_2816__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_2561_to_2816_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)2815) / (int64_t)2816) + (((n + (int64_t)2815) % (int64_t)2816) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)2815) / (int64_t)2816) + (((n + (int64_t)2815) % (int64_t)2816) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      uint4 __1;
        uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2815) / (int64_t)2816) + (((n + (int64_t)2815) % (int64_t)2816) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        uint4 v__1 = *(uint4*)(B + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2815) / (int64_t)2816) + (((n + (int64_t)2815) % (int64_t)2816) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(T_add_n_2561_to_2816_ + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2815) / (int64_t)2816) + (((n + (int64_t)2815) % (int64_t)2816) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_2817_to_3072__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_2817_to_3072_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      uint4 __1;
        uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        uint4 v__1 = *(uint4*)(B + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(T_add_n_2817_to_3072_ + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_3073_to_3328__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_3073_to_3328_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)3327) / (int64_t)3328) + (((n + (int64_t)3327) % (int64_t)3328) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)3327) / (int64_t)3328) + (((n + (int64_t)3327) % (int64_t)3328) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      uint4 __1;
        uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3327) / (int64_t)3328) + (((n + (int64_t)3327) % (int64_t)3328) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        uint4 v__1 = *(uint4*)(B + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3327) / (int64_t)3328) + (((n + (int64_t)3327) % (int64_t)3328) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(T_add_n_3073_to_3328_ + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3327) / (int64_t)3328) + (((n + (int64_t)3327) % (int64_t)3328) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_3329_to_3584__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_3329_to_3584_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)3583) / (int64_t)3584) + (((n + (int64_t)3583) % (int64_t)3584) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)3583) / (int64_t)3584) + (((n + (int64_t)3583) % (int64_t)3584) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      uint4 __1;
        uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3583) / (int64_t)3584) + (((n + (int64_t)3583) % (int64_t)3584) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        uint4 v__1 = *(uint4*)(B + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3583) / (int64_t)3584) + (((n + (int64_t)3583) % (int64_t)3584) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(T_add_n_3329_to_3584_ + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3583) / (int64_t)3584) + (((n + (int64_t)3583) % (int64_t)3584) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_3585_to_3840__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_3585_to_3840_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)3839) / (int64_t)3840) + (((n + (int64_t)3839) % (int64_t)3840) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)3839) / (int64_t)3840) + (((n + (int64_t)3839) % (int64_t)3840) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      uint4 __1;
        uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3839) / (int64_t)3840) + (((n + (int64_t)3839) % (int64_t)3840) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        uint4 v__1 = *(uint4*)(B + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3839) / (int64_t)3840) + (((n + (int64_t)3839) % (int64_t)3840) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(T_add_n_3585_to_3840_ + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3839) / (int64_t)3840) + (((n + (int64_t)3839) % (int64_t)3840) >> (int64_t)63))) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_3841_to_4096__kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ T_add_n_3841_to_4096_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < ((n + (int64_t)4095) >> (int64_t)12); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * ((n + (int64_t)4095) >> (int64_t)12)) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)320)) < (int64_t)0) {
      uint4 __1;
        uint4 v_ = *(uint4*)(A + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)4095) >> (int64_t)12)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        uint4 v__1 = *(uint4*)(B + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)4095) >> (int64_t)12)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(T_add_n_3841_to_4096_ + (((ax0_ax1_fused_1 * (int64_t)2048) + ((((int64_t)blockIdx.x) * ((n + (int64_t)4095) >> (int64_t)12)) * (int64_t)2048)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __1;
    }
  }
}

