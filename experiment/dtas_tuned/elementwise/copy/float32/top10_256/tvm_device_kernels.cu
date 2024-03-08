
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
extern "C" __global__ void __launch_bounds__(256) copy_n_1_to_256__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n);
extern "C" __global__ void __launch_bounds__(256) copy_n_257_to_512__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n);
extern "C" __global__ void __launch_bounds__(256) copy_n_513_to_768__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n);
extern "C" __global__ void __launch_bounds__(256) copy_n_769_to_1024__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n);
extern "C" __global__ void __launch_bounds__(256) copy_n_1025_to_1280__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n);
extern "C" __global__ void __launch_bounds__(256) copy_n_1281_to_1536__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n);
extern "C" __global__ void __launch_bounds__(256) copy_n_1537_to_1792__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n);
extern "C" __global__ void __launch_bounds__(256) copy_n_1793_to_2048__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n);
extern "C" __global__ void __launch_bounds__(256) copy_n_2049_to_2304__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n);
extern "C" __global__ void __launch_bounds__(256) copy_n_2305_to_2560__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n);
extern "C" __global__ void __launch_bounds__(256) copy_n_2561_to_2816__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n);
extern "C" __global__ void __launch_bounds__(256) copy_n_2817_to_3072__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n);
extern "C" __global__ void __launch_bounds__(256) copy_n_3073_to_3328__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n);
extern "C" __global__ void __launch_bounds__(256) copy_n_3329_to_3584__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n);
extern "C" __global__ void __launch_bounds__(256) copy_n_3585_to_3840__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n);
extern "C" __global__ void __launch_bounds__(256) copy_n_3841_to_4096__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n);
extern "C" __global__ void __launch_bounds__(256) copy_n_1_to_256__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < ((n + (int64_t)255) >> (int64_t)8); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * ((n + (int64_t)255) >> (int64_t)8)) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)640)) < (int64_t)0) {
      *(float4*)(B + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * ((n + (int64_t)255) >> (int64_t)8)) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4))) = *(float4*)(A + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * ((n + (int64_t)255) >> (int64_t)8)) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4)));
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) copy_n_257_to_512__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < ((n + (int64_t)511) >> (int64_t)9); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * ((n + (int64_t)511) >> (int64_t)9)) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)640)) < (int64_t)0) {
      *(float4*)(B + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * ((n + (int64_t)511) >> (int64_t)9)) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4))) = *(float4*)(A + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * ((n + (int64_t)511) >> (int64_t)9)) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4)));
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) copy_n_513_to_768__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)767) / (int64_t)768) + (((n + (int64_t)767) % (int64_t)768) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)767) / (int64_t)768) + (((n + (int64_t)767) % (int64_t)768) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)640)) < (int64_t)0) {
      *(float4*)(B + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * (((n + (int64_t)767) / (int64_t)768) + (((n + (int64_t)767) % (int64_t)768) >> (int64_t)63))) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4))) = *(float4*)(A + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * (((n + (int64_t)767) / (int64_t)768) + (((n + (int64_t)767) % (int64_t)768) >> (int64_t)63))) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4)));
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) copy_n_769_to_1024__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < ((n + (int64_t)1023) >> (int64_t)10); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * ((n + (int64_t)1023) >> (int64_t)10)) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)640)) < (int64_t)0) {
      *(float4*)(B + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * ((n + (int64_t)1023) >> (int64_t)10)) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4))) = *(float4*)(A + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * ((n + (int64_t)1023) >> (int64_t)10)) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4)));
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) copy_n_1025_to_1280__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)1279) / (int64_t)1280) + (((n + (int64_t)1279) % (int64_t)1280) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)1279) / (int64_t)1280) + (((n + (int64_t)1279) % (int64_t)1280) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)640)) < (int64_t)0) {
      *(float4*)(B + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1279) / (int64_t)1280) + (((n + (int64_t)1279) % (int64_t)1280) >> (int64_t)63))) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4))) = *(float4*)(A + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1279) / (int64_t)1280) + (((n + (int64_t)1279) % (int64_t)1280) >> (int64_t)63))) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4)));
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) copy_n_1281_to_1536__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if (((((((int64_t)blockIdx.x) * (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63))) * (int64_t)768) + (ax0_ax1_fused_1 * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3)) < (n * (int64_t)2560)) {
      B[(((ax0_ax1_fused_1 * (int64_t)768) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63))) * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3))] = A[(((ax0_ax1_fused_1 * (int64_t)768) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63))) * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3))];
    }
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63))) * (int64_t)768) + (ax0_ax1_fused_1 * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3)) + (int64_t)1) < (n * (int64_t)2560)) {
      B[((((ax0_ax1_fused_1 * (int64_t)768) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63))) * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3)) + (int64_t)1)] = A[((((ax0_ax1_fused_1 * (int64_t)768) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63))) * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3)) + (int64_t)1)];
    }
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63))) * (int64_t)768) + (ax0_ax1_fused_1 * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3)) + (int64_t)2) < (n * (int64_t)2560)) {
      B[((((ax0_ax1_fused_1 * (int64_t)768) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63))) * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3)) + (int64_t)2)] = A[((((ax0_ax1_fused_1 * (int64_t)768) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1535) / (int64_t)1536) + (((n + (int64_t)1535) % (int64_t)1536) >> (int64_t)63))) * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3)) + (int64_t)2)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) copy_n_1537_to_1792__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)1791) / (int64_t)1792) + (((n + (int64_t)1791) % (int64_t)1792) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)1791) / (int64_t)1792) + (((n + (int64_t)1791) % (int64_t)1792) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)640)) < (int64_t)0) {
      *(float4*)(B + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1791) / (int64_t)1792) + (((n + (int64_t)1791) % (int64_t)1792) >> (int64_t)63))) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4))) = *(float4*)(A + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * (((n + (int64_t)1791) / (int64_t)1792) + (((n + (int64_t)1791) % (int64_t)1792) >> (int64_t)63))) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4)));
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) copy_n_1793_to_2048__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < ((n + (int64_t)2047) >> (int64_t)11); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * ((n + (int64_t)2047) >> (int64_t)11)) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)640)) < (int64_t)0) {
      *(float4*)(B + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * ((n + (int64_t)2047) >> (int64_t)11)) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4))) = *(float4*)(A + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * ((n + (int64_t)2047) >> (int64_t)11)) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4)));
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) copy_n_2049_to_2304__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)2303) / (int64_t)2304) + (((n + (int64_t)2303) % (int64_t)2304) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)2303) / (int64_t)2304) + (((n + (int64_t)2303) % (int64_t)2304) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)640)) < (int64_t)0) {
      *(float4*)(B + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2303) / (int64_t)2304) + (((n + (int64_t)2303) % (int64_t)2304) >> (int64_t)63))) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4))) = *(float4*)(A + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2303) / (int64_t)2304) + (((n + (int64_t)2303) % (int64_t)2304) >> (int64_t)63))) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4)));
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) copy_n_2305_to_2560__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)2559) / (int64_t)2560) + (((n + (int64_t)2559) % (int64_t)2560) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)2559) / (int64_t)2560) + (((n + (int64_t)2559) % (int64_t)2560) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)640)) < (int64_t)0) {
      *(float4*)(B + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2559) / (int64_t)2560) + (((n + (int64_t)2559) % (int64_t)2560) >> (int64_t)63))) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4))) = *(float4*)(A + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2559) / (int64_t)2560) + (((n + (int64_t)2559) % (int64_t)2560) >> (int64_t)63))) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4)));
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) copy_n_2561_to_2816__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)2815) / (int64_t)2816) + (((n + (int64_t)2815) % (int64_t)2816) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)2815) / (int64_t)2816) + (((n + (int64_t)2815) % (int64_t)2816) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)640)) < (int64_t)0) {
      *(float4*)(B + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2815) / (int64_t)2816) + (((n + (int64_t)2815) % (int64_t)2816) >> (int64_t)63))) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4))) = *(float4*)(A + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * (((n + (int64_t)2815) / (int64_t)2816) + (((n + (int64_t)2815) % (int64_t)2816) >> (int64_t)63))) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4)));
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) copy_n_2817_to_3072__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if (((((((int64_t)blockIdx.x) * (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63))) * (int64_t)768) + (ax0_ax1_fused_1 * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3)) < (n * (int64_t)2560)) {
      B[(((ax0_ax1_fused_1 * (int64_t)768) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63))) * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3))] = A[(((ax0_ax1_fused_1 * (int64_t)768) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63))) * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3))];
    }
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63))) * (int64_t)768) + (ax0_ax1_fused_1 * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3)) + (int64_t)1) < (n * (int64_t)2560)) {
      B[((((ax0_ax1_fused_1 * (int64_t)768) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63))) * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3)) + (int64_t)1)] = A[((((ax0_ax1_fused_1 * (int64_t)768) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63))) * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3)) + (int64_t)1)];
    }
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63))) * (int64_t)768) + (ax0_ax1_fused_1 * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3)) + (int64_t)2) < (n * (int64_t)2560)) {
      B[((((ax0_ax1_fused_1 * (int64_t)768) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63))) * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3)) + (int64_t)2)] = A[((((ax0_ax1_fused_1 * (int64_t)768) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3071) / (int64_t)3072) + (((n + (int64_t)3071) % (int64_t)3072) >> (int64_t)63))) * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3)) + (int64_t)2)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) copy_n_3073_to_3328__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < ((((n * (int64_t)5) + (int64_t)16640) / (int64_t)16641) + ((((n * (int64_t)5) + (int64_t)16640) % (int64_t)16641) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if (((((((int64_t)blockIdx.x) * ((((n * (int64_t)5) + (int64_t)16640) / (int64_t)16641) + ((((n * (int64_t)5) + (int64_t)16640) % (int64_t)16641) >> (int64_t)63))) * (int64_t)768) + (ax0_ax1_fused_1 * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3)) < (n * (int64_t)2560)) {
      B[(((ax0_ax1_fused_1 * (int64_t)768) + ((((int64_t)blockIdx.x) * ((((n * (int64_t)5) + (int64_t)16640) / (int64_t)16641) + ((((n * (int64_t)5) + (int64_t)16640) % (int64_t)16641) >> (int64_t)63))) * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3))] = A[(((ax0_ax1_fused_1 * (int64_t)768) + ((((int64_t)blockIdx.x) * ((((n * (int64_t)5) + (int64_t)16640) / (int64_t)16641) + ((((n * (int64_t)5) + (int64_t)16640) % (int64_t)16641) >> (int64_t)63))) * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3))];
    }
    if ((((((((int64_t)blockIdx.x) * ((((n * (int64_t)5) + (int64_t)16640) / (int64_t)16641) + ((((n * (int64_t)5) + (int64_t)16640) % (int64_t)16641) >> (int64_t)63))) * (int64_t)768) + (ax0_ax1_fused_1 * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3)) + (int64_t)1) < (n * (int64_t)2560)) {
      B[((((ax0_ax1_fused_1 * (int64_t)768) + ((((int64_t)blockIdx.x) * ((((n * (int64_t)5) + (int64_t)16640) / (int64_t)16641) + ((((n * (int64_t)5) + (int64_t)16640) % (int64_t)16641) >> (int64_t)63))) * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3)) + (int64_t)1)] = A[((((ax0_ax1_fused_1 * (int64_t)768) + ((((int64_t)blockIdx.x) * ((((n * (int64_t)5) + (int64_t)16640) / (int64_t)16641) + ((((n * (int64_t)5) + (int64_t)16640) % (int64_t)16641) >> (int64_t)63))) * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3)) + (int64_t)1)];
    }
    if ((((((((int64_t)blockIdx.x) * ((((n * (int64_t)5) + (int64_t)16640) / (int64_t)16641) + ((((n * (int64_t)5) + (int64_t)16640) % (int64_t)16641) >> (int64_t)63))) * (int64_t)768) + (ax0_ax1_fused_1 * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3)) + (int64_t)2) < (n * (int64_t)2560)) {
      B[((((ax0_ax1_fused_1 * (int64_t)768) + ((((int64_t)blockIdx.x) * ((((n * (int64_t)5) + (int64_t)16640) / (int64_t)16641) + ((((n * (int64_t)5) + (int64_t)16640) % (int64_t)16641) >> (int64_t)63))) * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3)) + (int64_t)2)] = A[((((ax0_ax1_fused_1 * (int64_t)768) + ((((int64_t)blockIdx.x) * ((((n * (int64_t)5) + (int64_t)16640) / (int64_t)16641) + ((((n * (int64_t)5) + (int64_t)16640) % (int64_t)16641) >> (int64_t)63))) * (int64_t)768)) + (((int64_t)threadIdx.x) * (int64_t)3)) + (int64_t)2)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) copy_n_3329_to_3584__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)3583) / (int64_t)3584) + (((n + (int64_t)3583) % (int64_t)3584) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)3583) / (int64_t)3584) + (((n + (int64_t)3583) % (int64_t)3584) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)640)) < (int64_t)0) {
      *(float4*)(B + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3583) / (int64_t)3584) + (((n + (int64_t)3583) % (int64_t)3584) >> (int64_t)63))) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4))) = *(float4*)(A + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3583) / (int64_t)3584) + (((n + (int64_t)3583) % (int64_t)3584) >> (int64_t)63))) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4)));
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) copy_n_3585_to_3840__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < (((n + (int64_t)3839) / (int64_t)3840) + (((n + (int64_t)3839) % (int64_t)3840) >> (int64_t)63)); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * (((n + (int64_t)3839) / (int64_t)3840) + (((n + (int64_t)3839) % (int64_t)3840) >> (int64_t)63))) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)640)) < (int64_t)0) {
      *(float4*)(B + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3839) / (int64_t)3840) + (((n + (int64_t)3839) % (int64_t)3840) >> (int64_t)63))) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4))) = *(float4*)(A + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * (((n + (int64_t)3839) / (int64_t)3840) + (((n + (int64_t)3839) % (int64_t)3840) >> (int64_t)63))) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4)));
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) copy_n_3841_to_4096__kernel(float* __restrict__ A, float* __restrict__ B, int64_t n) {
  for (int64_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < ((n + (int64_t)4095) >> (int64_t)12); ++ax0_ax1_fused_1) {
    if ((((((((int64_t)blockIdx.x) * ((n + (int64_t)4095) >> (int64_t)12)) * (int64_t)256) + (ax0_ax1_fused_1 * (int64_t)256)) + ((int64_t)threadIdx.x)) - (n * (int64_t)640)) < (int64_t)0) {
      *(float4*)(B + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * ((n + (int64_t)4095) >> (int64_t)12)) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4))) = *(float4*)(A + (((ax0_ax1_fused_1 * (int64_t)1024) + ((((int64_t)blockIdx.x) * ((n + (int64_t)4095) >> (int64_t)12)) * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)4)));
    }
  }
}

