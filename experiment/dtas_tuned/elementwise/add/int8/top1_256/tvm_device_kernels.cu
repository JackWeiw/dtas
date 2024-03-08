
#include <cuda_runtime.h>
#include <math.h>
#include <mma.h>
//for int8
#include <sm_61_intrinsics.h>

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
extern "C" __global__ void __launch_bounds__(256) add_n_1_to_256__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_1_to_256_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_257_to_512__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_257_to_512_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_513_to_768__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_513_to_768_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_769_to_1024__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_769_to_1024_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_1025_to_1280__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_1025_to_1280_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_1281_to_1536__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_1281_to_1536_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_1537_to_1792__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_1537_to_1792_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_1793_to_2048__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_1793_to_2048_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_2049_to_2304__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_2049_to_2304_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_2305_to_2560__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_2305_to_2560_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_2561_to_2816__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_2561_to_2816_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_2817_to_3072__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_2817_to_3072_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_3073_to_3328__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_3073_to_3328_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_3329_to_3584__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_3329_to_3584_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_3585_to_3840__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_3585_to_3840_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_3841_to_4096__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_3841_to_4096_, int64_t n);
extern "C" __global__ void __launch_bounds__(256) add_n_1_to_256__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_1_to_256_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < ((n + (int64_t)255) >> (int64_t)8); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * ((n + (int64_t)255) >> (int64_t)8)) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)160)) < (int64_t)0) {
      int4 __1;
        int4 v_ = *(int4*)(A + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * ((n + (int64_t)255) >> (int64_t)8)) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        int4 v__1 = *(int4*)(B + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * ((n + (int64_t)255) >> (int64_t)8)) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        __1.x=((((char)(v_.x >> 0))+((char)(v__1.x >> 0))) << 0);
        __1.x=__1.x & ~(0x000000ff << 8) |((((char)(v_.x >> 8))+((char)(v__1.x >> 8))) << 8);
        __1.x=__1.x & ~(0x000000ff << 16) |((((char)(v_.x >> 16))+((char)(v__1.x >> 16))) << 16);
        __1.x=__1.x & ~(0x000000ff << 24) |((((char)(v_.x >> 24))+((char)(v__1.x >> 24))) << 24);
        __1.y=__1.y & ~(0x000000ff << 0) |((((char)(v_.y >> 0))+((char)(v__1.y >> 0))) << 0);
        __1.y=__1.y & ~(0x000000ff << 8) |((((char)(v_.y >> 8))+((char)(v__1.y >> 8))) << 8);
        __1.y=__1.y & ~(0x000000ff << 16) |((((char)(v_.y >> 16))+((char)(v__1.y >> 16))) << 16);
        __1.y=__1.y & ~(0x000000ff << 24) |((((char)(v_.y >> 24))+((char)(v__1.y >> 24))) << 24);
        __1.z=__1.z & ~(0x000000ff << 0) |((((char)(v_.z >> 0))+((char)(v__1.z >> 0))) << 0);
        __1.z=__1.z & ~(0x000000ff << 8) |((((char)(v_.z >> 8))+((char)(v__1.z >> 8))) << 8);
        __1.z=__1.z & ~(0x000000ff << 16) |((((char)(v_.z >> 16))+((char)(v__1.z >> 16))) << 16);
        __1.z=__1.z & ~(0x000000ff << 24) |((((char)(v_.z >> 24))+((char)(v__1.z >> 24))) << 24);
        __1.w=__1.w & ~(0x000000ff << 0) |((((char)(v_.w >> 0))+((char)(v__1.w >> 0))) << 0);
        __1.w=__1.w & ~(0x000000ff << 8) |((((char)(v_.w >> 8))+((char)(v__1.w >> 8))) << 8);
        __1.w=__1.w & ~(0x000000ff << 16) |((((char)(v_.w >> 16))+((char)(v__1.w >> 16))) << 16);
        __1.w=__1.w & ~(0x000000ff << 24) |((((char)(v_.w >> 24))+((char)(v__1.w >> 24))) << 24);
      *(int4*)(T_add_n_1_to_256_ + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * ((n + (int64_t)255) >> (int64_t)8)) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_257_to_512__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_257_to_512_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < ((n + (int64_t)511) >> (int64_t)9); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * ((n + (int64_t)511) >> (int64_t)9)) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)160)) < (int64_t)0) {
      int4 __1;
        int4 v_ = *(int4*)(A + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * ((n + (int64_t)511) >> (int64_t)9)) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        int4 v__1 = *(int4*)(B + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * ((n + (int64_t)511) >> (int64_t)9)) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        __1.x=((((char)(v_.x >> 0))+((char)(v__1.x >> 0))) << 0);
        __1.x=__1.x & ~(0x000000ff << 8) |((((char)(v_.x >> 8))+((char)(v__1.x >> 8))) << 8);
        __1.x=__1.x & ~(0x000000ff << 16) |((((char)(v_.x >> 16))+((char)(v__1.x >> 16))) << 16);
        __1.x=__1.x & ~(0x000000ff << 24) |((((char)(v_.x >> 24))+((char)(v__1.x >> 24))) << 24);
        __1.y=__1.y & ~(0x000000ff << 0) |((((char)(v_.y >> 0))+((char)(v__1.y >> 0))) << 0);
        __1.y=__1.y & ~(0x000000ff << 8) |((((char)(v_.y >> 8))+((char)(v__1.y >> 8))) << 8);
        __1.y=__1.y & ~(0x000000ff << 16) |((((char)(v_.y >> 16))+((char)(v__1.y >> 16))) << 16);
        __1.y=__1.y & ~(0x000000ff << 24) |((((char)(v_.y >> 24))+((char)(v__1.y >> 24))) << 24);
        __1.z=__1.z & ~(0x000000ff << 0) |((((char)(v_.z >> 0))+((char)(v__1.z >> 0))) << 0);
        __1.z=__1.z & ~(0x000000ff << 8) |((((char)(v_.z >> 8))+((char)(v__1.z >> 8))) << 8);
        __1.z=__1.z & ~(0x000000ff << 16) |((((char)(v_.z >> 16))+((char)(v__1.z >> 16))) << 16);
        __1.z=__1.z & ~(0x000000ff << 24) |((((char)(v_.z >> 24))+((char)(v__1.z >> 24))) << 24);
        __1.w=__1.w & ~(0x000000ff << 0) |((((char)(v_.w >> 0))+((char)(v__1.w >> 0))) << 0);
        __1.w=__1.w & ~(0x000000ff << 8) |((((char)(v_.w >> 8))+((char)(v__1.w >> 8))) << 8);
        __1.w=__1.w & ~(0x000000ff << 16) |((((char)(v_.w >> 16))+((char)(v__1.w >> 16))) << 16);
        __1.w=__1.w & ~(0x000000ff << 24) |((((char)(v_.w >> 24))+((char)(v__1.w >> 24))) << 24);
      *(int4*)(T_add_n_257_to_512_ + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * ((n + (int64_t)511) >> (int64_t)9)) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_513_to_768__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_513_to_768_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)767) / (int64_t)768) + (((n + (int64_t)767) % (int64_t)768) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)767) / (int64_t)768) + (((n + (int64_t)767) % (int64_t)768) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)160)) < (int64_t)0) {
      int4 __1;
        int4 v_ = *(int4*)(A + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)767) / (int64_t)768) + (((n + (int64_t)767) % (int64_t)768) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        int4 v__1 = *(int4*)(B + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)767) / (int64_t)768) + (((n + (int64_t)767) % (int64_t)768) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        __1.x=((((char)(v_.x >> 0))+((char)(v__1.x >> 0))) << 0);
        __1.x=__1.x & ~(0x000000ff << 8) |((((char)(v_.x >> 8))+((char)(v__1.x >> 8))) << 8);
        __1.x=__1.x & ~(0x000000ff << 16) |((((char)(v_.x >> 16))+((char)(v__1.x >> 16))) << 16);
        __1.x=__1.x & ~(0x000000ff << 24) |((((char)(v_.x >> 24))+((char)(v__1.x >> 24))) << 24);
        __1.y=__1.y & ~(0x000000ff << 0) |((((char)(v_.y >> 0))+((char)(v__1.y >> 0))) << 0);
        __1.y=__1.y & ~(0x000000ff << 8) |((((char)(v_.y >> 8))+((char)(v__1.y >> 8))) << 8);
        __1.y=__1.y & ~(0x000000ff << 16) |((((char)(v_.y >> 16))+((char)(v__1.y >> 16))) << 16);
        __1.y=__1.y & ~(0x000000ff << 24) |((((char)(v_.y >> 24))+((char)(v__1.y >> 24))) << 24);
        __1.z=__1.z & ~(0x000000ff << 0) |((((char)(v_.z >> 0))+((char)(v__1.z >> 0))) << 0);
        __1.z=__1.z & ~(0x000000ff << 8) |((((char)(v_.z >> 8))+((char)(v__1.z >> 8))) << 8);
        __1.z=__1.z & ~(0x000000ff << 16) |((((char)(v_.z >> 16))+((char)(v__1.z >> 16))) << 16);
        __1.z=__1.z & ~(0x000000ff << 24) |((((char)(v_.z >> 24))+((char)(v__1.z >> 24))) << 24);
        __1.w=__1.w & ~(0x000000ff << 0) |((((char)(v_.w >> 0))+((char)(v__1.w >> 0))) << 0);
        __1.w=__1.w & ~(0x000000ff << 8) |((((char)(v_.w >> 8))+((char)(v__1.w >> 8))) << 8);
        __1.w=__1.w & ~(0x000000ff << 16) |((((char)(v_.w >> 16))+((char)(v__1.w >> 16))) << 16);
        __1.w=__1.w & ~(0x000000ff << 24) |((((char)(v_.w >> 24))+((char)(v__1.w >> 24))) << 24);
      *(int4*)(T_add_n_513_to_768_ + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)767) / (int64_t)768) + (((n + (int64_t)767) % (int64_t)768) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_769_to_1024__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_769_to_1024_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < ((n + (int64_t)1023) >> (int64_t)10); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * ((n + (int64_t)1023) >> (int64_t)10)) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)160)) < (int64_t)0) {
      int4 __1;
        int4 v_ = *(int4*)(A + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * ((n + (int64_t)1023) >> (int64_t)10)) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        int4 v__1 = *(int4*)(B + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * ((n + (int64_t)1023) >> (int64_t)10)) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        __1.x=((((char)(v_.x >> 0))+((char)(v__1.x >> 0))) << 0);
        __1.x=__1.x & ~(0x000000ff << 8) |((((char)(v_.x >> 8))+((char)(v__1.x >> 8))) << 8);
        __1.x=__1.x & ~(0x000000ff << 16) |((((char)(v_.x >> 16))+((char)(v__1.x >> 16))) << 16);
        __1.x=__1.x & ~(0x000000ff << 24) |((((char)(v_.x >> 24))+((char)(v__1.x >> 24))) << 24);
        __1.y=__1.y & ~(0x000000ff << 0) |((((char)(v_.y >> 0))+((char)(v__1.y >> 0))) << 0);
        __1.y=__1.y & ~(0x000000ff << 8) |((((char)(v_.y >> 8))+((char)(v__1.y >> 8))) << 8);
        __1.y=__1.y & ~(0x000000ff << 16) |((((char)(v_.y >> 16))+((char)(v__1.y >> 16))) << 16);
        __1.y=__1.y & ~(0x000000ff << 24) |((((char)(v_.y >> 24))+((char)(v__1.y >> 24))) << 24);
        __1.z=__1.z & ~(0x000000ff << 0) |((((char)(v_.z >> 0))+((char)(v__1.z >> 0))) << 0);
        __1.z=__1.z & ~(0x000000ff << 8) |((((char)(v_.z >> 8))+((char)(v__1.z >> 8))) << 8);
        __1.z=__1.z & ~(0x000000ff << 16) |((((char)(v_.z >> 16))+((char)(v__1.z >> 16))) << 16);
        __1.z=__1.z & ~(0x000000ff << 24) |((((char)(v_.z >> 24))+((char)(v__1.z >> 24))) << 24);
        __1.w=__1.w & ~(0x000000ff << 0) |((((char)(v_.w >> 0))+((char)(v__1.w >> 0))) << 0);
        __1.w=__1.w & ~(0x000000ff << 8) |((((char)(v_.w >> 8))+((char)(v__1.w >> 8))) << 8);
        __1.w=__1.w & ~(0x000000ff << 16) |((((char)(v_.w >> 16))+((char)(v__1.w >> 16))) << 16);
        __1.w=__1.w & ~(0x000000ff << 24) |((((char)(v_.w >> 24))+((char)(v__1.w >> 24))) << 24);
      *(int4*)(T_add_n_769_to_1024_ + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * ((n + (int64_t)1023) >> (int64_t)10)) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_1025_to_1280__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_1025_to_1280_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)1279) / (int64_t)1280) + (((n + (int64_t)1279) % (int64_t)1280) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)1279) / (int64_t)1280) + (((n + (int64_t)1279) % (int64_t)1280) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)160)) < (int64_t)0) {
      int4 __1;
        int4 v_ = *(int4*)(A + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1279) / (int64_t)1280) + (((n + (int64_t)1279) % (int64_t)1280) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        int4 v__1 = *(int4*)(B + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1279) / (int64_t)1280) + (((n + (int64_t)1279) % (int64_t)1280) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        __1.x=((((char)(v_.x >> 0))+((char)(v__1.x >> 0))) << 0);
        __1.x=__1.x & ~(0x000000ff << 8) |((((char)(v_.x >> 8))+((char)(v__1.x >> 8))) << 8);
        __1.x=__1.x & ~(0x000000ff << 16) |((((char)(v_.x >> 16))+((char)(v__1.x >> 16))) << 16);
        __1.x=__1.x & ~(0x000000ff << 24) |((((char)(v_.x >> 24))+((char)(v__1.x >> 24))) << 24);
        __1.y=__1.y & ~(0x000000ff << 0) |((((char)(v_.y >> 0))+((char)(v__1.y >> 0))) << 0);
        __1.y=__1.y & ~(0x000000ff << 8) |((((char)(v_.y >> 8))+((char)(v__1.y >> 8))) << 8);
        __1.y=__1.y & ~(0x000000ff << 16) |((((char)(v_.y >> 16))+((char)(v__1.y >> 16))) << 16);
        __1.y=__1.y & ~(0x000000ff << 24) |((((char)(v_.y >> 24))+((char)(v__1.y >> 24))) << 24);
        __1.z=__1.z & ~(0x000000ff << 0) |((((char)(v_.z >> 0))+((char)(v__1.z >> 0))) << 0);
        __1.z=__1.z & ~(0x000000ff << 8) |((((char)(v_.z >> 8))+((char)(v__1.z >> 8))) << 8);
        __1.z=__1.z & ~(0x000000ff << 16) |((((char)(v_.z >> 16))+((char)(v__1.z >> 16))) << 16);
        __1.z=__1.z & ~(0x000000ff << 24) |((((char)(v_.z >> 24))+((char)(v__1.z >> 24))) << 24);
        __1.w=__1.w & ~(0x000000ff << 0) |((((char)(v_.w >> 0))+((char)(v__1.w >> 0))) << 0);
        __1.w=__1.w & ~(0x000000ff << 8) |((((char)(v_.w >> 8))+((char)(v__1.w >> 8))) << 8);
        __1.w=__1.w & ~(0x000000ff << 16) |((((char)(v_.w >> 16))+((char)(v__1.w >> 16))) << 16);
        __1.w=__1.w & ~(0x000000ff << 24) |((((char)(v_.w >> 24))+((char)(v__1.w >> 24))) << 24);
      *(int4*)(T_add_n_1025_to_1280_ + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1279) / (int64_t)1280) + (((n + (int64_t)1279) % (int64_t)1280) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_1281_to_1536__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_1281_to_1536_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)160)) < (int64_t)0) {
      int4 __1;
        int4 v_ = *(int4*)(A + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        int4 v__1 = *(int4*)(B + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        __1.x=((((char)(v_.x >> 0))+((char)(v__1.x >> 0))) << 0);
        __1.x=__1.x & ~(0x000000ff << 8) |((((char)(v_.x >> 8))+((char)(v__1.x >> 8))) << 8);
        __1.x=__1.x & ~(0x000000ff << 16) |((((char)(v_.x >> 16))+((char)(v__1.x >> 16))) << 16);
        __1.x=__1.x & ~(0x000000ff << 24) |((((char)(v_.x >> 24))+((char)(v__1.x >> 24))) << 24);
        __1.y=__1.y & ~(0x000000ff << 0) |((((char)(v_.y >> 0))+((char)(v__1.y >> 0))) << 0);
        __1.y=__1.y & ~(0x000000ff << 8) |((((char)(v_.y >> 8))+((char)(v__1.y >> 8))) << 8);
        __1.y=__1.y & ~(0x000000ff << 16) |((((char)(v_.y >> 16))+((char)(v__1.y >> 16))) << 16);
        __1.y=__1.y & ~(0x000000ff << 24) |((((char)(v_.y >> 24))+((char)(v__1.y >> 24))) << 24);
        __1.z=__1.z & ~(0x000000ff << 0) |((((char)(v_.z >> 0))+((char)(v__1.z >> 0))) << 0);
        __1.z=__1.z & ~(0x000000ff << 8) |((((char)(v_.z >> 8))+((char)(v__1.z >> 8))) << 8);
        __1.z=__1.z & ~(0x000000ff << 16) |((((char)(v_.z >> 16))+((char)(v__1.z >> 16))) << 16);
        __1.z=__1.z & ~(0x000000ff << 24) |((((char)(v_.z >> 24))+((char)(v__1.z >> 24))) << 24);
        __1.w=__1.w & ~(0x000000ff << 0) |((((char)(v_.w >> 0))+((char)(v__1.w >> 0))) << 0);
        __1.w=__1.w & ~(0x000000ff << 8) |((((char)(v_.w >> 8))+((char)(v__1.w >> 8))) << 8);
        __1.w=__1.w & ~(0x000000ff << 16) |((((char)(v_.w >> 16))+((char)(v__1.w >> 16))) << 16);
        __1.w=__1.w & ~(0x000000ff << 24) |((((char)(v_.w >> 24))+((char)(v__1.w >> 24))) << 24);
      *(int4*)(T_add_n_1281_to_1536_ + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_1537_to_1792__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_1537_to_1792_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)1791) / (int64_t)1792) + (((n + (int64_t)1791) % (int64_t)1792) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)1791) / (int64_t)1792) + (((n + (int64_t)1791) % (int64_t)1792) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)160)) < (int64_t)0) {
      int4 __1;
        int4 v_ = *(int4*)(A + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1791) / (int64_t)1792) + (((n + (int64_t)1791) % (int64_t)1792) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        int4 v__1 = *(int4*)(B + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1791) / (int64_t)1792) + (((n + (int64_t)1791) % (int64_t)1792) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        __1.x=((((char)(v_.x >> 0))+((char)(v__1.x >> 0))) << 0);
        __1.x=__1.x & ~(0x000000ff << 8) |((((char)(v_.x >> 8))+((char)(v__1.x >> 8))) << 8);
        __1.x=__1.x & ~(0x000000ff << 16) |((((char)(v_.x >> 16))+((char)(v__1.x >> 16))) << 16);
        __1.x=__1.x & ~(0x000000ff << 24) |((((char)(v_.x >> 24))+((char)(v__1.x >> 24))) << 24);
        __1.y=__1.y & ~(0x000000ff << 0) |((((char)(v_.y >> 0))+((char)(v__1.y >> 0))) << 0);
        __1.y=__1.y & ~(0x000000ff << 8) |((((char)(v_.y >> 8))+((char)(v__1.y >> 8))) << 8);
        __1.y=__1.y & ~(0x000000ff << 16) |((((char)(v_.y >> 16))+((char)(v__1.y >> 16))) << 16);
        __1.y=__1.y & ~(0x000000ff << 24) |((((char)(v_.y >> 24))+((char)(v__1.y >> 24))) << 24);
        __1.z=__1.z & ~(0x000000ff << 0) |((((char)(v_.z >> 0))+((char)(v__1.z >> 0))) << 0);
        __1.z=__1.z & ~(0x000000ff << 8) |((((char)(v_.z >> 8))+((char)(v__1.z >> 8))) << 8);
        __1.z=__1.z & ~(0x000000ff << 16) |((((char)(v_.z >> 16))+((char)(v__1.z >> 16))) << 16);
        __1.z=__1.z & ~(0x000000ff << 24) |((((char)(v_.z >> 24))+((char)(v__1.z >> 24))) << 24);
        __1.w=__1.w & ~(0x000000ff << 0) |((((char)(v_.w >> 0))+((char)(v__1.w >> 0))) << 0);
        __1.w=__1.w & ~(0x000000ff << 8) |((((char)(v_.w >> 8))+((char)(v__1.w >> 8))) << 8);
        __1.w=__1.w & ~(0x000000ff << 16) |((((char)(v_.w >> 16))+((char)(v__1.w >> 16))) << 16);
        __1.w=__1.w & ~(0x000000ff << 24) |((((char)(v_.w >> 24))+((char)(v__1.w >> 24))) << 24);
      *(int4*)(T_add_n_1537_to_1792_ + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1791) / (int64_t)1792) + (((n + (int64_t)1791) % (int64_t)1792) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_1793_to_2048__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_1793_to_2048_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < ((n + (int64_t)2047) >> (int64_t)11); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * ((n + (int64_t)2047) >> (int64_t)11)) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)160)) < (int64_t)0) {
      int4 __1;
        int4 v_ = *(int4*)(A + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * ((n + (int64_t)2047) >> (int64_t)11)) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        int4 v__1 = *(int4*)(B + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * ((n + (int64_t)2047) >> (int64_t)11)) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        __1.x=((((char)(v_.x >> 0))+((char)(v__1.x >> 0))) << 0);
        __1.x=__1.x & ~(0x000000ff << 8) |((((char)(v_.x >> 8))+((char)(v__1.x >> 8))) << 8);
        __1.x=__1.x & ~(0x000000ff << 16) |((((char)(v_.x >> 16))+((char)(v__1.x >> 16))) << 16);
        __1.x=__1.x & ~(0x000000ff << 24) |((((char)(v_.x >> 24))+((char)(v__1.x >> 24))) << 24);
        __1.y=__1.y & ~(0x000000ff << 0) |((((char)(v_.y >> 0))+((char)(v__1.y >> 0))) << 0);
        __1.y=__1.y & ~(0x000000ff << 8) |((((char)(v_.y >> 8))+((char)(v__1.y >> 8))) << 8);
        __1.y=__1.y & ~(0x000000ff << 16) |((((char)(v_.y >> 16))+((char)(v__1.y >> 16))) << 16);
        __1.y=__1.y & ~(0x000000ff << 24) |((((char)(v_.y >> 24))+((char)(v__1.y >> 24))) << 24);
        __1.z=__1.z & ~(0x000000ff << 0) |((((char)(v_.z >> 0))+((char)(v__1.z >> 0))) << 0);
        __1.z=__1.z & ~(0x000000ff << 8) |((((char)(v_.z >> 8))+((char)(v__1.z >> 8))) << 8);
        __1.z=__1.z & ~(0x000000ff << 16) |((((char)(v_.z >> 16))+((char)(v__1.z >> 16))) << 16);
        __1.z=__1.z & ~(0x000000ff << 24) |((((char)(v_.z >> 24))+((char)(v__1.z >> 24))) << 24);
        __1.w=__1.w & ~(0x000000ff << 0) |((((char)(v_.w >> 0))+((char)(v__1.w >> 0))) << 0);
        __1.w=__1.w & ~(0x000000ff << 8) |((((char)(v_.w >> 8))+((char)(v__1.w >> 8))) << 8);
        __1.w=__1.w & ~(0x000000ff << 16) |((((char)(v_.w >> 16))+((char)(v__1.w >> 16))) << 16);
        __1.w=__1.w & ~(0x000000ff << 24) |((((char)(v_.w >> 24))+((char)(v__1.w >> 24))) << 24);
      *(int4*)(T_add_n_1793_to_2048_ + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * ((n + (int64_t)2047) >> (int64_t)11)) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_2049_to_2304__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_2049_to_2304_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)2303) / (int64_t)2304) + (((n + (int64_t)2303) % (int64_t)2304) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)2303) / (int64_t)2304) + (((n + (int64_t)2303) % (int64_t)2304) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)160)) < (int64_t)0) {
      int4 __1;
        int4 v_ = *(int4*)(A + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2303) / (int64_t)2304) + (((n + (int64_t)2303) % (int64_t)2304) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        int4 v__1 = *(int4*)(B + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2303) / (int64_t)2304) + (((n + (int64_t)2303) % (int64_t)2304) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        __1.x=((((char)(v_.x >> 0))+((char)(v__1.x >> 0))) << 0);
        __1.x=__1.x & ~(0x000000ff << 8) |((((char)(v_.x >> 8))+((char)(v__1.x >> 8))) << 8);
        __1.x=__1.x & ~(0x000000ff << 16) |((((char)(v_.x >> 16))+((char)(v__1.x >> 16))) << 16);
        __1.x=__1.x & ~(0x000000ff << 24) |((((char)(v_.x >> 24))+((char)(v__1.x >> 24))) << 24);
        __1.y=__1.y & ~(0x000000ff << 0) |((((char)(v_.y >> 0))+((char)(v__1.y >> 0))) << 0);
        __1.y=__1.y & ~(0x000000ff << 8) |((((char)(v_.y >> 8))+((char)(v__1.y >> 8))) << 8);
        __1.y=__1.y & ~(0x000000ff << 16) |((((char)(v_.y >> 16))+((char)(v__1.y >> 16))) << 16);
        __1.y=__1.y & ~(0x000000ff << 24) |((((char)(v_.y >> 24))+((char)(v__1.y >> 24))) << 24);
        __1.z=__1.z & ~(0x000000ff << 0) |((((char)(v_.z >> 0))+((char)(v__1.z >> 0))) << 0);
        __1.z=__1.z & ~(0x000000ff << 8) |((((char)(v_.z >> 8))+((char)(v__1.z >> 8))) << 8);
        __1.z=__1.z & ~(0x000000ff << 16) |((((char)(v_.z >> 16))+((char)(v__1.z >> 16))) << 16);
        __1.z=__1.z & ~(0x000000ff << 24) |((((char)(v_.z >> 24))+((char)(v__1.z >> 24))) << 24);
        __1.w=__1.w & ~(0x000000ff << 0) |((((char)(v_.w >> 0))+((char)(v__1.w >> 0))) << 0);
        __1.w=__1.w & ~(0x000000ff << 8) |((((char)(v_.w >> 8))+((char)(v__1.w >> 8))) << 8);
        __1.w=__1.w & ~(0x000000ff << 16) |((((char)(v_.w >> 16))+((char)(v__1.w >> 16))) << 16);
        __1.w=__1.w & ~(0x000000ff << 24) |((((char)(v_.w >> 24))+((char)(v__1.w >> 24))) << 24);
      *(int4*)(T_add_n_2049_to_2304_ + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2303) / (int64_t)2304) + (((n + (int64_t)2303) % (int64_t)2304) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_2305_to_2560__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_2305_to_2560_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)2559) / (int64_t)2560) + (((n + (int64_t)2559) % (int64_t)2560) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)2559) / (int64_t)2560) + (((n + (int64_t)2559) % (int64_t)2560) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)160)) < (int64_t)0) {
      int4 __1;
        int4 v_ = *(int4*)(A + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2559) / (int64_t)2560) + (((n + (int64_t)2559) % (int64_t)2560) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        int4 v__1 = *(int4*)(B + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2559) / (int64_t)2560) + (((n + (int64_t)2559) % (int64_t)2560) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        __1.x=((((char)(v_.x >> 0))+((char)(v__1.x >> 0))) << 0);
        __1.x=__1.x & ~(0x000000ff << 8) |((((char)(v_.x >> 8))+((char)(v__1.x >> 8))) << 8);
        __1.x=__1.x & ~(0x000000ff << 16) |((((char)(v_.x >> 16))+((char)(v__1.x >> 16))) << 16);
        __1.x=__1.x & ~(0x000000ff << 24) |((((char)(v_.x >> 24))+((char)(v__1.x >> 24))) << 24);
        __1.y=__1.y & ~(0x000000ff << 0) |((((char)(v_.y >> 0))+((char)(v__1.y >> 0))) << 0);
        __1.y=__1.y & ~(0x000000ff << 8) |((((char)(v_.y >> 8))+((char)(v__1.y >> 8))) << 8);
        __1.y=__1.y & ~(0x000000ff << 16) |((((char)(v_.y >> 16))+((char)(v__1.y >> 16))) << 16);
        __1.y=__1.y & ~(0x000000ff << 24) |((((char)(v_.y >> 24))+((char)(v__1.y >> 24))) << 24);
        __1.z=__1.z & ~(0x000000ff << 0) |((((char)(v_.z >> 0))+((char)(v__1.z >> 0))) << 0);
        __1.z=__1.z & ~(0x000000ff << 8) |((((char)(v_.z >> 8))+((char)(v__1.z >> 8))) << 8);
        __1.z=__1.z & ~(0x000000ff << 16) |((((char)(v_.z >> 16))+((char)(v__1.z >> 16))) << 16);
        __1.z=__1.z & ~(0x000000ff << 24) |((((char)(v_.z >> 24))+((char)(v__1.z >> 24))) << 24);
        __1.w=__1.w & ~(0x000000ff << 0) |((((char)(v_.w >> 0))+((char)(v__1.w >> 0))) << 0);
        __1.w=__1.w & ~(0x000000ff << 8) |((((char)(v_.w >> 8))+((char)(v__1.w >> 8))) << 8);
        __1.w=__1.w & ~(0x000000ff << 16) |((((char)(v_.w >> 16))+((char)(v__1.w >> 16))) << 16);
        __1.w=__1.w & ~(0x000000ff << 24) |((((char)(v_.w >> 24))+((char)(v__1.w >> 24))) << 24);
      *(int4*)(T_add_n_2305_to_2560_ + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2559) / (int64_t)2560) + (((n + (int64_t)2559) % (int64_t)2560) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_2561_to_2816__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_2561_to_2816_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)2815) / (int64_t)2816) + (((n + (int64_t)2815) % (int64_t)2816) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)2815) / (int64_t)2816) + (((n + (int64_t)2815) % (int64_t)2816) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)160)) < (int64_t)0) {
      int4 __1;
        int4 v_ = *(int4*)(A + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2815) / (int64_t)2816) + (((n + (int64_t)2815) % (int64_t)2816) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        int4 v__1 = *(int4*)(B + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2815) / (int64_t)2816) + (((n + (int64_t)2815) % (int64_t)2816) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        __1.x=((((char)(v_.x >> 0))+((char)(v__1.x >> 0))) << 0);
        __1.x=__1.x & ~(0x000000ff << 8) |((((char)(v_.x >> 8))+((char)(v__1.x >> 8))) << 8);
        __1.x=__1.x & ~(0x000000ff << 16) |((((char)(v_.x >> 16))+((char)(v__1.x >> 16))) << 16);
        __1.x=__1.x & ~(0x000000ff << 24) |((((char)(v_.x >> 24))+((char)(v__1.x >> 24))) << 24);
        __1.y=__1.y & ~(0x000000ff << 0) |((((char)(v_.y >> 0))+((char)(v__1.y >> 0))) << 0);
        __1.y=__1.y & ~(0x000000ff << 8) |((((char)(v_.y >> 8))+((char)(v__1.y >> 8))) << 8);
        __1.y=__1.y & ~(0x000000ff << 16) |((((char)(v_.y >> 16))+((char)(v__1.y >> 16))) << 16);
        __1.y=__1.y & ~(0x000000ff << 24) |((((char)(v_.y >> 24))+((char)(v__1.y >> 24))) << 24);
        __1.z=__1.z & ~(0x000000ff << 0) |((((char)(v_.z >> 0))+((char)(v__1.z >> 0))) << 0);
        __1.z=__1.z & ~(0x000000ff << 8) |((((char)(v_.z >> 8))+((char)(v__1.z >> 8))) << 8);
        __1.z=__1.z & ~(0x000000ff << 16) |((((char)(v_.z >> 16))+((char)(v__1.z >> 16))) << 16);
        __1.z=__1.z & ~(0x000000ff << 24) |((((char)(v_.z >> 24))+((char)(v__1.z >> 24))) << 24);
        __1.w=__1.w & ~(0x000000ff << 0) |((((char)(v_.w >> 0))+((char)(v__1.w >> 0))) << 0);
        __1.w=__1.w & ~(0x000000ff << 8) |((((char)(v_.w >> 8))+((char)(v__1.w >> 8))) << 8);
        __1.w=__1.w & ~(0x000000ff << 16) |((((char)(v_.w >> 16))+((char)(v__1.w >> 16))) << 16);
        __1.w=__1.w & ~(0x000000ff << 24) |((((char)(v_.w >> 24))+((char)(v__1.w >> 24))) << 24);
      *(int4*)(T_add_n_2561_to_2816_ + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2815) / (int64_t)2816) + (((n + (int64_t)2815) % (int64_t)2816) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_2817_to_3072__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_2817_to_3072_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)160)) < (int64_t)0) {
      int4 __1;
        int4 v_ = *(int4*)(A + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        int4 v__1 = *(int4*)(B + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        __1.x=((((char)(v_.x >> 0))+((char)(v__1.x >> 0))) << 0);
        __1.x=__1.x & ~(0x000000ff << 8) |((((char)(v_.x >> 8))+((char)(v__1.x >> 8))) << 8);
        __1.x=__1.x & ~(0x000000ff << 16) |((((char)(v_.x >> 16))+((char)(v__1.x >> 16))) << 16);
        __1.x=__1.x & ~(0x000000ff << 24) |((((char)(v_.x >> 24))+((char)(v__1.x >> 24))) << 24);
        __1.y=__1.y & ~(0x000000ff << 0) |((((char)(v_.y >> 0))+((char)(v__1.y >> 0))) << 0);
        __1.y=__1.y & ~(0x000000ff << 8) |((((char)(v_.y >> 8))+((char)(v__1.y >> 8))) << 8);
        __1.y=__1.y & ~(0x000000ff << 16) |((((char)(v_.y >> 16))+((char)(v__1.y >> 16))) << 16);
        __1.y=__1.y & ~(0x000000ff << 24) |((((char)(v_.y >> 24))+((char)(v__1.y >> 24))) << 24);
        __1.z=__1.z & ~(0x000000ff << 0) |((((char)(v_.z >> 0))+((char)(v__1.z >> 0))) << 0);
        __1.z=__1.z & ~(0x000000ff << 8) |((((char)(v_.z >> 8))+((char)(v__1.z >> 8))) << 8);
        __1.z=__1.z & ~(0x000000ff << 16) |((((char)(v_.z >> 16))+((char)(v__1.z >> 16))) << 16);
        __1.z=__1.z & ~(0x000000ff << 24) |((((char)(v_.z >> 24))+((char)(v__1.z >> 24))) << 24);
        __1.w=__1.w & ~(0x000000ff << 0) |((((char)(v_.w >> 0))+((char)(v__1.w >> 0))) << 0);
        __1.w=__1.w & ~(0x000000ff << 8) |((((char)(v_.w >> 8))+((char)(v__1.w >> 8))) << 8);
        __1.w=__1.w & ~(0x000000ff << 16) |((((char)(v_.w >> 16))+((char)(v__1.w >> 16))) << 16);
        __1.w=__1.w & ~(0x000000ff << 24) |((((char)(v_.w >> 24))+((char)(v__1.w >> 24))) << 24);
      *(int4*)(T_add_n_2817_to_3072_ + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_3073_to_3328__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_3073_to_3328_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)3327) / (int64_t)3328) + (((n + (int64_t)3327) % (int64_t)3328) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)3327) / (int64_t)3328) + (((n + (int64_t)3327) % (int64_t)3328) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)160)) < (int64_t)0) {
      int4 __1;
        int4 v_ = *(int4*)(A + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3327) / (int64_t)3328) + (((n + (int64_t)3327) % (int64_t)3328) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        int4 v__1 = *(int4*)(B + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3327) / (int64_t)3328) + (((n + (int64_t)3327) % (int64_t)3328) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        __1.x=((((char)(v_.x >> 0))+((char)(v__1.x >> 0))) << 0);
        __1.x=__1.x & ~(0x000000ff << 8) |((((char)(v_.x >> 8))+((char)(v__1.x >> 8))) << 8);
        __1.x=__1.x & ~(0x000000ff << 16) |((((char)(v_.x >> 16))+((char)(v__1.x >> 16))) << 16);
        __1.x=__1.x & ~(0x000000ff << 24) |((((char)(v_.x >> 24))+((char)(v__1.x >> 24))) << 24);
        __1.y=__1.y & ~(0x000000ff << 0) |((((char)(v_.y >> 0))+((char)(v__1.y >> 0))) << 0);
        __1.y=__1.y & ~(0x000000ff << 8) |((((char)(v_.y >> 8))+((char)(v__1.y >> 8))) << 8);
        __1.y=__1.y & ~(0x000000ff << 16) |((((char)(v_.y >> 16))+((char)(v__1.y >> 16))) << 16);
        __1.y=__1.y & ~(0x000000ff << 24) |((((char)(v_.y >> 24))+((char)(v__1.y >> 24))) << 24);
        __1.z=__1.z & ~(0x000000ff << 0) |((((char)(v_.z >> 0))+((char)(v__1.z >> 0))) << 0);
        __1.z=__1.z & ~(0x000000ff << 8) |((((char)(v_.z >> 8))+((char)(v__1.z >> 8))) << 8);
        __1.z=__1.z & ~(0x000000ff << 16) |((((char)(v_.z >> 16))+((char)(v__1.z >> 16))) << 16);
        __1.z=__1.z & ~(0x000000ff << 24) |((((char)(v_.z >> 24))+((char)(v__1.z >> 24))) << 24);
        __1.w=__1.w & ~(0x000000ff << 0) |((((char)(v_.w >> 0))+((char)(v__1.w >> 0))) << 0);
        __1.w=__1.w & ~(0x000000ff << 8) |((((char)(v_.w >> 8))+((char)(v__1.w >> 8))) << 8);
        __1.w=__1.w & ~(0x000000ff << 16) |((((char)(v_.w >> 16))+((char)(v__1.w >> 16))) << 16);
        __1.w=__1.w & ~(0x000000ff << 24) |((((char)(v_.w >> 24))+((char)(v__1.w >> 24))) << 24);
      *(int4*)(T_add_n_3073_to_3328_ + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3327) / (int64_t)3328) + (((n + (int64_t)3327) % (int64_t)3328) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_3329_to_3584__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_3329_to_3584_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)3583) / (int64_t)3584) + (((n + (int64_t)3583) % (int64_t)3584) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)3583) / (int64_t)3584) + (((n + (int64_t)3583) % (int64_t)3584) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)160)) < (int64_t)0) {
      int4 __1;
        int4 v_ = *(int4*)(A + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3583) / (int64_t)3584) + (((n + (int64_t)3583) % (int64_t)3584) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        int4 v__1 = *(int4*)(B + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3583) / (int64_t)3584) + (((n + (int64_t)3583) % (int64_t)3584) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        __1.x=((((char)(v_.x >> 0))+((char)(v__1.x >> 0))) << 0);
        __1.x=__1.x & ~(0x000000ff << 8) |((((char)(v_.x >> 8))+((char)(v__1.x >> 8))) << 8);
        __1.x=__1.x & ~(0x000000ff << 16) |((((char)(v_.x >> 16))+((char)(v__1.x >> 16))) << 16);
        __1.x=__1.x & ~(0x000000ff << 24) |((((char)(v_.x >> 24))+((char)(v__1.x >> 24))) << 24);
        __1.y=__1.y & ~(0x000000ff << 0) |((((char)(v_.y >> 0))+((char)(v__1.y >> 0))) << 0);
        __1.y=__1.y & ~(0x000000ff << 8) |((((char)(v_.y >> 8))+((char)(v__1.y >> 8))) << 8);
        __1.y=__1.y & ~(0x000000ff << 16) |((((char)(v_.y >> 16))+((char)(v__1.y >> 16))) << 16);
        __1.y=__1.y & ~(0x000000ff << 24) |((((char)(v_.y >> 24))+((char)(v__1.y >> 24))) << 24);
        __1.z=__1.z & ~(0x000000ff << 0) |((((char)(v_.z >> 0))+((char)(v__1.z >> 0))) << 0);
        __1.z=__1.z & ~(0x000000ff << 8) |((((char)(v_.z >> 8))+((char)(v__1.z >> 8))) << 8);
        __1.z=__1.z & ~(0x000000ff << 16) |((((char)(v_.z >> 16))+((char)(v__1.z >> 16))) << 16);
        __1.z=__1.z & ~(0x000000ff << 24) |((((char)(v_.z >> 24))+((char)(v__1.z >> 24))) << 24);
        __1.w=__1.w & ~(0x000000ff << 0) |((((char)(v_.w >> 0))+((char)(v__1.w >> 0))) << 0);
        __1.w=__1.w & ~(0x000000ff << 8) |((((char)(v_.w >> 8))+((char)(v__1.w >> 8))) << 8);
        __1.w=__1.w & ~(0x000000ff << 16) |((((char)(v_.w >> 16))+((char)(v__1.w >> 16))) << 16);
        __1.w=__1.w & ~(0x000000ff << 24) |((((char)(v_.w >> 24))+((char)(v__1.w >> 24))) << 24);
      *(int4*)(T_add_n_3329_to_3584_ + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3583) / (int64_t)3584) + (((n + (int64_t)3583) % (int64_t)3584) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_3585_to_3840__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_3585_to_3840_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)3839) / (int64_t)3840) + (((n + (int64_t)3839) % (int64_t)3840) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)3839) / (int64_t)3840) + (((n + (int64_t)3839) % (int64_t)3840) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)160)) < (int64_t)0) {
      int4 __1;
        int4 v_ = *(int4*)(A + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3839) / (int64_t)3840) + (((n + (int64_t)3839) % (int64_t)3840) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        int4 v__1 = *(int4*)(B + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3839) / (int64_t)3840) + (((n + (int64_t)3839) % (int64_t)3840) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        __1.x=((((char)(v_.x >> 0))+((char)(v__1.x >> 0))) << 0);
        __1.x=__1.x & ~(0x000000ff << 8) |((((char)(v_.x >> 8))+((char)(v__1.x >> 8))) << 8);
        __1.x=__1.x & ~(0x000000ff << 16) |((((char)(v_.x >> 16))+((char)(v__1.x >> 16))) << 16);
        __1.x=__1.x & ~(0x000000ff << 24) |((((char)(v_.x >> 24))+((char)(v__1.x >> 24))) << 24);
        __1.y=__1.y & ~(0x000000ff << 0) |((((char)(v_.y >> 0))+((char)(v__1.y >> 0))) << 0);
        __1.y=__1.y & ~(0x000000ff << 8) |((((char)(v_.y >> 8))+((char)(v__1.y >> 8))) << 8);
        __1.y=__1.y & ~(0x000000ff << 16) |((((char)(v_.y >> 16))+((char)(v__1.y >> 16))) << 16);
        __1.y=__1.y & ~(0x000000ff << 24) |((((char)(v_.y >> 24))+((char)(v__1.y >> 24))) << 24);
        __1.z=__1.z & ~(0x000000ff << 0) |((((char)(v_.z >> 0))+((char)(v__1.z >> 0))) << 0);
        __1.z=__1.z & ~(0x000000ff << 8) |((((char)(v_.z >> 8))+((char)(v__1.z >> 8))) << 8);
        __1.z=__1.z & ~(0x000000ff << 16) |((((char)(v_.z >> 16))+((char)(v__1.z >> 16))) << 16);
        __1.z=__1.z & ~(0x000000ff << 24) |((((char)(v_.z >> 24))+((char)(v__1.z >> 24))) << 24);
        __1.w=__1.w & ~(0x000000ff << 0) |((((char)(v_.w >> 0))+((char)(v__1.w >> 0))) << 0);
        __1.w=__1.w & ~(0x000000ff << 8) |((((char)(v_.w >> 8))+((char)(v__1.w >> 8))) << 8);
        __1.w=__1.w & ~(0x000000ff << 16) |((((char)(v_.w >> 16))+((char)(v__1.w >> 16))) << 16);
        __1.w=__1.w & ~(0x000000ff << 24) |((((char)(v_.w >> 24))+((char)(v__1.w >> 24))) << 24);
      *(int4*)(T_add_n_3585_to_3840_ + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3839) / (int64_t)3840) + (((n + (int64_t)3839) % (int64_t)3840) >> (int64_t)63))) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) add_n_3841_to_4096__kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ T_add_n_3841_to_4096_, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < ((n + (int64_t)4095) >> (int64_t)12); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * ((n + (int64_t)4095) >> (int64_t)12)) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)160)) < (int64_t)0) {
      int4 __1;
        int4 v_ = *(int4*)(A + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * ((n + (int64_t)4095) >> (int64_t)12)) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        int4 v__1 = *(int4*)(B + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * ((n + (int64_t)4095) >> (int64_t)12)) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16)));
        __1.x=((((char)(v_.x >> 0))+((char)(v__1.x >> 0))) << 0);
        __1.x=__1.x & ~(0x000000ff << 8) |((((char)(v_.x >> 8))+((char)(v__1.x >> 8))) << 8);
        __1.x=__1.x & ~(0x000000ff << 16) |((((char)(v_.x >> 16))+((char)(v__1.x >> 16))) << 16);
        __1.x=__1.x & ~(0x000000ff << 24) |((((char)(v_.x >> 24))+((char)(v__1.x >> 24))) << 24);
        __1.y=__1.y & ~(0x000000ff << 0) |((((char)(v_.y >> 0))+((char)(v__1.y >> 0))) << 0);
        __1.y=__1.y & ~(0x000000ff << 8) |((((char)(v_.y >> 8))+((char)(v__1.y >> 8))) << 8);
        __1.y=__1.y & ~(0x000000ff << 16) |((((char)(v_.y >> 16))+((char)(v__1.y >> 16))) << 16);
        __1.y=__1.y & ~(0x000000ff << 24) |((((char)(v_.y >> 24))+((char)(v__1.y >> 24))) << 24);
        __1.z=__1.z & ~(0x000000ff << 0) |((((char)(v_.z >> 0))+((char)(v__1.z >> 0))) << 0);
        __1.z=__1.z & ~(0x000000ff << 8) |((((char)(v_.z >> 8))+((char)(v__1.z >> 8))) << 8);
        __1.z=__1.z & ~(0x000000ff << 16) |((((char)(v_.z >> 16))+((char)(v__1.z >> 16))) << 16);
        __1.z=__1.z & ~(0x000000ff << 24) |((((char)(v_.z >> 24))+((char)(v__1.z >> 24))) << 24);
        __1.w=__1.w & ~(0x000000ff << 0) |((((char)(v_.w >> 0))+((char)(v__1.w >> 0))) << 0);
        __1.w=__1.w & ~(0x000000ff << 8) |((((char)(v_.w >> 8))+((char)(v__1.w >> 8))) << 8);
        __1.w=__1.w & ~(0x000000ff << 16) |((((char)(v_.w >> 16))+((char)(v__1.w >> 16))) << 16);
        __1.w=__1.w & ~(0x000000ff << 24) |((((char)(v_.w >> 24))+((char)(v__1.w >> 24))) << 24);
      *(int4*)(T_add_n_3841_to_4096_ + (((ax0_ax1_fused_1 * (int64_t)4096) + ((((int64_t)blockIdx.x) * ((n + (int64_t)4095) >> (int64_t)12)) * (int64_t)4096)) + (((int64_t)threadIdx.x) * (int64_t)16))) = __1;
    }
  }
}

