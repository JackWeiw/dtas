
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
extern "C" __global__ void __launch_bounds__(640) gemm_n_1_to_256__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n);
extern "C" __global__ void __launch_bounds__(384) gemm_n_257_to_512__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n);
extern "C" __global__ void __launch_bounds__(384) gemm_n_513_to_768__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n);
extern "C" __global__ void __launch_bounds__(320) gemm_n_769_to_1024__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n);
extern "C" __global__ void __launch_bounds__(384) gemm_n_1025_to_1280__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n);
extern "C" __global__ void __launch_bounds__(384) gemm_n_1281_to_1536__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n);
extern "C" __global__ void __launch_bounds__(224) gemm_n_1537_to_1792__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n);
extern "C" __global__ void __launch_bounds__(384) gemm_n_1793_to_2048__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n);
extern "C" __global__ void __launch_bounds__(384) gemm_n_2049_to_2304__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n);
extern "C" __global__ void __launch_bounds__(224) gemm_n_2305_to_2560__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n);
extern "C" __global__ void __launch_bounds__(384) gemm_n_2561_to_2816__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n);
extern "C" __global__ void __launch_bounds__(512) gemm_n_2817_to_3072__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n);
extern "C" __global__ void __launch_bounds__(384) gemm_n_3073_to_3328__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n);
extern "C" __global__ void __launch_bounds__(384) gemm_n_3329_to_3584__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n);
extern "C" __global__ void __launch_bounds__(384) gemm_n_3585_to_3840__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n);
extern "C" __global__ void __launch_bounds__(512) gemm_n_3841_to_4096__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n);
extern "C" __global__ void __launch_bounds__(640) gemm_n_1_to_256__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[4];
  uint4 lv9_reindex_pad_shared_dyn_local[2];
  uint4 lv3_reindex_shared_dyn_local[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> lv9_reindex_pad_shared_dyn_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> lv3_reindex_shared_dyn_wmma_matrix_b[4];
  for (int ax1_0_3_init = 0; ax1_0_3_init < 2; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 2; ++ax2_0_3_init) {
      nvcuda::wmma::fill_fragment(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_init * 2) + ax2_0_3_init)], 0.000000e+00f);
    }
  }
  for (int64_t ax1_ax2_fused_0_0_0 = 0; ax1_ax2_fused_0_0_0 < (int64_t)2; ++ax1_ax2_fused_0_0_0) {
    uint4 condval;
    if ((((((((int64_t)blockIdx.x) * (int64_t)160) + (ax1_ax2_fused_0_0_0 * (int64_t)80)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
      condval = *(uint4*)(lv9 + (((((((int64_t)blockIdx.x) * (int64_t)1638400) + (ax1_ax2_fused_0_0_0 * (int64_t)819200)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
    }
    lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0] = condval;
  }
  for (int ax1_ax2_fused_0_0_0_1 = 0; ax1_ax2_fused_0_0_0_1 < 2; ++ax1_ax2_fused_0_0_0_1) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)5) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
      lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_1] = *(uint4*)(lv3 + (((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)819200)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    }
  }
  for (int ax1_ax2_fused_0_0_0_2 = 0; ax1_ax2_fused_0_0_0_2 < 2; ++ax1_ax2_fused_0_0_0_2) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((int64_t)ax1_ax2_fused_0_0_0_2) * (int64_t)5760) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_2];
  }
  for (int ax1_ax2_fused_0_0_0_3 = 0; ax1_ax2_fused_0_0_0_3 < 2; ++ax1_ax2_fused_0_0_0_3) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)5) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)5760) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_3];
    }
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)2304) + (int64_t)18432)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)2304) + (int64_t)19584)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)2304)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)2304) + (int64_t)1152)])), 72);
  for (int64_t ax3_0_0 = 0; ax3_0_0 < (int64_t)159; ++ax3_0_0) {
    for (int64_t ax1_ax2_fused_0_0_0_4 = 0; ax1_ax2_fused_0_0_0_4 < (int64_t)2; ++ax1_ax2_fused_0_0_0_4) {
      uint4 condval_1;
      if ((((((((int64_t)blockIdx.x) * (int64_t)160) + (ax1_ax2_fused_0_0_0_4 * (int64_t)80)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
        condval_1 = *(uint4*)(lv9 + (((((((((int64_t)blockIdx.x) * (int64_t)1638400) + (ax1_ax2_fused_0_0_0_4 * (int64_t)819200)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      } else {
        condval_1 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
      lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_4] = condval_1;
    }
    for (int ax1_ax2_fused_0_0_0_5 = 0; ax1_ax2_fused_0_0_0_5 < 2; ++ax1_ax2_fused_0_0_0_5) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)5) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
        lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_5] = *(uint4*)(lv3 + (((((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)819200)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      }
    }
    for (int ax3_0_1 = 0; ax3_0_1 < 3; ++ax3_0_1) {
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)11520) + ((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)18448)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)11520) + ((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)19600)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)16)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)1168)])), 72);
      for (int ax1_0_3 = 0; ax1_0_3 < 2; ++ax1_0_3) {
        for (int ax2_0_3 = 0; ax2_0_3 < 2; ++ax2_0_3) {
          nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 2) + ax2_0_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 & 1) * 2) + ax1_0_3)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 & 1) * 2) + ax2_0_3)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 2) + ax2_0_3)]);
        }
      }
    }
    __syncthreads();
    for (int ax1_ax2_fused_0_0_0_6 = 0; ax1_ax2_fused_0_0_0_6 < 2; ++ax1_ax2_fused_0_0_0_6) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)11520) + (((int64_t)ax1_ax2_fused_0_0_0_6) * (int64_t)5760)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_6];
    }
    for (int ax1_ax2_fused_0_0_0_7 = 0; ax1_ax2_fused_0_0_0_7 < 2; ++ax1_ax2_fused_0_0_0_7) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)5) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
        *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + (((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)5760)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_7];
      }
    }
    __syncthreads();
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)11520) + ((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)2304)) + (int64_t)18432)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)11520) + ((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)2304)) + (int64_t)19584)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)2304))])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)2304)) + (int64_t)1152)])), 72);
    for (int ax1_0_3_1 = 0; ax1_0_3_1 < 2; ++ax1_0_3_1) {
      for (int ax2_0_3_1 = 0; ax2_0_3_1 < 2; ++ax2_0_3_1) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 2) + ax2_0_3_1)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_1 + 2)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_1 + 2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 2) + ax2_0_3_1)]);
      }
    }
  }
  for (int ax3_0_1_1 = 0; ax3_0_1_1 < 3; ++ax3_0_1_1) {
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)29968)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)31120)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)9232)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)10384)])), 72);
    for (int ax1_0_3_2 = 0; ax1_0_3_2 < 2; ++ax1_0_3_2) {
      for (int ax2_0_3_2 = 0; ax2_0_3_2 < 2; ++ax2_0_3_2) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 2) + ax2_0_3_2)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 & 1) * 2) + ax1_0_3_2)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 & 1) * 2) + ax2_0_3_2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 2) + ax2_0_3_2)]);
      }
    }
  }
  for (int ax1_0_3_3 = 0; ax1_0_3_3 < 2; ++ax1_0_3_3) {
    for (int ax2_0_3_3 = 0; ax2_0_3_3 < 2; ++ax2_0_3_3) {
      nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 2) + ax2_0_3_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_3 + 2)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_3 + 2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 2) + ax2_0_3_3)]);
    }
  }
  __syncthreads();
  for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {
    for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0) {
      nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)4096) + (((int64_t)ax0_0) * (int64_t)2048)) + ((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)32)) + (((int64_t)ax1_0) * (int64_t)16)) + (int64_t)18432)])), var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax0_0 * 2) + ax1_0)], 128, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int64_t ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < (int64_t)4; ++ax0_ax1_fused_0) {
    if (((((((int64_t)blockIdx.x) * (int64_t)160) + ((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)32)) + (ax0_ax1_fused_0 * (int64_t)8)) + (((int64_t)threadIdx.x) >> (int64_t)2)) < n) {
      uint4 __1;
        uint4 v_ = *(uint4*)(((half*)buf_dyn_shmem) + (((((((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)4096) + (ax0_ax1_fused_0 * (int64_t)1024)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)) + (int64_t)18432));
        uint4 v__1 = *(uint4*)(lv4 + (((((int64_t)blockIdx.y) * (int64_t)128) + ((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(var_T_add_intermediate + (((((((((int64_t)blockIdx.x) * (int64_t)409600) + ((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)81920)) + (ax0_ax1_fused_0 * (int64_t)20480)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)2560)) + (((int64_t)blockIdx.y) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(384) gemm_n_257_to_512__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[8];
  uint4 lv9_reindex_pad_shared_dyn_local[4];
  uint4 lv3_reindex_shared_dyn_local[3];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> lv9_reindex_pad_shared_dyn_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> lv3_reindex_shared_dyn_wmma_matrix_b[8];
  for (int ax1_0_3_init = 0; ax1_0_3_init < 2; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 4; ++ax2_0_3_init) {
      nvcuda::wmma::fill_fragment(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_init * 4) + ax2_0_3_init)], 0.000000e+00f);
    }
  }
  for (int64_t ax1_ax2_fused_0_0_0 = 0; ax1_ax2_fused_0_0_0 < (int64_t)4; ++ax1_ax2_fused_0_0_0) {
    uint4 condval;
    if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0 * (int64_t)48)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
      condval = *(uint4*)(lv9 + (((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0 * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
    }
    lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0] = condval;
  }
  for (int ax1_ax2_fused_0_0_0_1 = 0; ax1_ax2_fused_0_0_0_1 < 3; ++ax1_ax2_fused_0_0_0_1) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
      lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_1] = *(uint4*)(lv3 + (((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    }
  }
  for (int ax1_ax2_fused_0_0_0_2 = 0; ax1_ax2_fused_0_0_0_2 < 4; ++ax1_ax2_fused_0_0_0_2) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((int64_t)ax1_ax2_fused_0_0_0_2) * (int64_t)3456) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_2];
  }
  for (int ax1_ax2_fused_0_0_0_3 = 0; ax1_ax2_fused_0_0_0_3 < 3; ++ax1_ax2_fused_0_0_0_3) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)3456) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_3];
    }
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (int64_t)18432)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (int64_t)19584)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (int64_t)1152)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (int64_t)2304)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (int64_t)3456)])), 72);
  for (int64_t ax3_0_0 = 0; ax3_0_0 < (int64_t)159; ++ax3_0_0) {
    for (int64_t ax1_ax2_fused_0_0_0_4 = 0; ax1_ax2_fused_0_0_0_4 < (int64_t)4; ++ax1_ax2_fused_0_0_0_4) {
      uint4 condval_1;
      if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0_4 * (int64_t)48)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
        condval_1 = *(uint4*)(lv9 + (((((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0_4 * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      } else {
        condval_1 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
      lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_4] = condval_1;
    }
    for (int ax1_ax2_fused_0_0_0_5 = 0; ax1_ax2_fused_0_0_0_5 < 3; ++ax1_ax2_fused_0_0_0_5) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
        lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_5] = *(uint4*)(lv3 + (((((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      }
    }
    for (int ax3_0_1 = 0; ax3_0_1 < 3; ++ax3_0_1) {
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)18448)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)19600)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)16)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)1168)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)2320)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)3472)])), 72);
      for (int ax1_0_3 = 0; ax1_0_3 < 2; ++ax1_0_3) {
        for (int ax2_0_3 = 0; ax2_0_3 < 4; ++ax2_0_3) {
          nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 4) + ax2_0_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 & 1) * 2) + ax1_0_3)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 & 1) * 4) + ax2_0_3)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 4) + ax2_0_3)]);
        }
      }
    }
    __syncthreads();
    for (int ax1_ax2_fused_0_0_0_6 = 0; ax1_ax2_fused_0_0_0_6 < 4; ++ax1_ax2_fused_0_0_0_6) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + (((int64_t)ax1_ax2_fused_0_0_0_6) * (int64_t)3456)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_6];
    }
    for (int ax1_ax2_fused_0_0_0_7 = 0; ax1_ax2_fused_0_0_0_7 < 3; ++ax1_ax2_fused_0_0_0_7) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
        *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + (((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)3456)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_7];
      }
    }
    __syncthreads();
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (int64_t)18432)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (int64_t)19584)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608))])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (int64_t)1152)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (int64_t)2304)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (int64_t)3456)])), 72);
    for (int ax1_0_3_1 = 0; ax1_0_3_1 < 2; ++ax1_0_3_1) {
      for (int ax2_0_3_1 = 0; ax2_0_3_1 < 4; ++ax2_0_3_1) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 4) + ax2_0_3_1)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_1 + 2)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_1 + 4)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 4) + ax2_0_3_1)]);
      }
    }
  }
  for (int ax3_0_1_1 = 0; ax3_0_1_1 < 3; ++ax3_0_1_1) {
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)32272)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)33424)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)9232)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)10384)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)11536)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)12688)])), 72);
    for (int ax1_0_3_2 = 0; ax1_0_3_2 < 2; ++ax1_0_3_2) {
      for (int ax2_0_3_2 = 0; ax2_0_3_2 < 4; ++ax2_0_3_2) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 4) + ax2_0_3_2)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 & 1) * 2) + ax1_0_3_2)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 & 1) * 4) + ax2_0_3_2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 4) + ax2_0_3_2)]);
      }
    }
  }
  for (int ax1_0_3_3 = 0; ax1_0_3_3 < 2; ++ax1_0_3_3) {
    for (int ax2_0_3_3 = 0; ax2_0_3_3 < 4; ++ax2_0_3_3) {
      nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 4) + ax2_0_3_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_3 + 2)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_3 + 4)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 4) + ax2_0_3_3)]);
    }
  }
  __syncthreads();
  for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {
    for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
      nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)4096) + (((int64_t)ax0_0) * (int64_t)2048)) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + (((int64_t)ax1_0) * (int64_t)16)) + (int64_t)18432)])), var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax0_0 * 4) + ax1_0)], 128, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int64_t ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < (int64_t)8; ++ax0_ax1_fused_0) {
    if (((((((int64_t)blockIdx.x) * (int64_t)192) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)32)) + (ax0_ax1_fused_0 * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n) {
      uint4 __1;
        uint4 v_ = *(uint4*)(((half*)buf_dyn_shmem) + (((((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)4096) + (ax0_ax1_fused_0 * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432));
        uint4 v__1 = *(uint4*)(lv4 + (((((int64_t)blockIdx.y) * (int64_t)128) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(var_T_add_intermediate + (((((((((int64_t)blockIdx.x) * (int64_t)491520) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)81920)) + (ax0_ax1_fused_0 * (int64_t)10240)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)2560)) + (((int64_t)blockIdx.y) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(384) gemm_n_513_to_768__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[8];
  uint4 lv9_reindex_pad_shared_dyn_local[4];
  uint4 lv3_reindex_shared_dyn_local[3];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> lv9_reindex_pad_shared_dyn_wmma_matrix_a[8];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> lv3_reindex_shared_dyn_wmma_matrix_b[4];
  for (int ax1_0_3_init = 0; ax1_0_3_init < 4; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 2; ++ax2_0_3_init) {
      nvcuda::wmma::fill_fragment(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_init * 2) + ax2_0_3_init)], 0.000000e+00f);
    }
  }
  for (int64_t ax1_ax2_fused_0_0_0 = 0; ax1_ax2_fused_0_0_0 < (int64_t)4; ++ax1_ax2_fused_0_0_0) {
    uint4 condval;
    if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0 * (int64_t)48)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
      condval = *(uint4*)(lv9 + (((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0 * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
    }
    lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0] = condval;
  }
  for (int ax1_ax2_fused_0_0_0_1 = 0; ax1_ax2_fused_0_0_0_1 < 3; ++ax1_ax2_fused_0_0_0_1) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
      lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_1] = *(uint4*)(lv3 + (((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    }
  }
  for (int ax1_ax2_fused_0_0_0_2 = 0; ax1_ax2_fused_0_0_0_2 < 4; ++ax1_ax2_fused_0_0_0_2) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((int64_t)ax1_ax2_fused_0_0_0_2) * (int64_t)3456) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_2];
  }
  for (int ax1_ax2_fused_0_0_0_3 = 0; ax1_ax2_fused_0_0_0_3 < 3; ++ax1_ax2_fused_0_0_0_3) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)3456) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_3];
    }
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (int64_t)18432)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (int64_t)19584)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (int64_t)20736)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (int64_t)21888)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304) + (int64_t)1152)])), 72);
  for (int64_t ax3_0_0 = 0; ax3_0_0 < (int64_t)159; ++ax3_0_0) {
    for (int64_t ax1_ax2_fused_0_0_0_4 = 0; ax1_ax2_fused_0_0_0_4 < (int64_t)4; ++ax1_ax2_fused_0_0_0_4) {
      uint4 condval_1;
      if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0_4 * (int64_t)48)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
        condval_1 = *(uint4*)(lv9 + (((((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0_4 * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      } else {
        condval_1 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
      lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_4] = condval_1;
    }
    for (int ax1_ax2_fused_0_0_0_5 = 0; ax1_ax2_fused_0_0_0_5 < 3; ++ax1_ax2_fused_0_0_0_5) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
        lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_5] = *(uint4*)(lv3 + (((((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      }
    }
    for (int ax3_0_1 = 0; ax3_0_1 < 3; ++ax3_0_1) {
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)18448)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)19600)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)20752)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)21904)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)16)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)1168)])), 72);
      for (int ax1_0_3 = 0; ax1_0_3 < 4; ++ax1_0_3) {
        for (int ax2_0_3 = 0; ax2_0_3 < 2; ++ax2_0_3) {
          nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 2) + ax2_0_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 & 1) * 4) + ax1_0_3)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 & 1) * 2) + ax2_0_3)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 2) + ax2_0_3)]);
        }
      }
    }
    __syncthreads();
    for (int ax1_ax2_fused_0_0_0_6 = 0; ax1_ax2_fused_0_0_0_6 < 4; ++ax1_ax2_fused_0_0_0_6) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + (((int64_t)ax1_ax2_fused_0_0_0_6) * (int64_t)3456)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_6];
    }
    for (int ax1_ax2_fused_0_0_0_7 = 0; ax1_ax2_fused_0_0_0_7 < 3; ++ax1_ax2_fused_0_0_0_7) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
        *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + (((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)3456)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_7];
      }
    }
    __syncthreads();
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (int64_t)18432)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (int64_t)19584)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (int64_t)20736)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (int64_t)21888)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304))])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304)) + (int64_t)1152)])), 72);
    for (int ax1_0_3_1 = 0; ax1_0_3_1 < 4; ++ax1_0_3_1) {
      for (int ax2_0_3_1 = 0; ax2_0_3_1 < 2; ++ax2_0_3_1) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 2) + ax2_0_3_1)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_1 + 4)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_1 + 2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 2) + ax2_0_3_1)]);
      }
    }
  }
  for (int ax3_0_1_1 = 0; ax3_0_1_1 < 3; ++ax3_0_1_1) {
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)32272)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)33424)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)34576)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)35728)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)9232)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)10384)])), 72);
    for (int ax1_0_3_2 = 0; ax1_0_3_2 < 4; ++ax1_0_3_2) {
      for (int ax2_0_3_2 = 0; ax2_0_3_2 < 2; ++ax2_0_3_2) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 2) + ax2_0_3_2)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 & 1) * 4) + ax1_0_3_2)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 & 1) * 2) + ax2_0_3_2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 2) + ax2_0_3_2)]);
      }
    }
  }
  for (int ax1_0_3_3 = 0; ax1_0_3_3 < 4; ++ax1_0_3_3) {
    for (int ax2_0_3_3 = 0; ax2_0_3_3 < 2; ++ax2_0_3_3) {
      nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 2) + ax2_0_3_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_3 + 4)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_3 + 2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 2) + ax2_0_3_3)]);
    }
  }
  __syncthreads();
  for (int ax0_0 = 0; ax0_0 < 4; ++ax0_0) {
    for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0) {
      nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)8192) + (((int64_t)ax0_0) * (int64_t)2048)) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)32)) + (((int64_t)ax1_0) * (int64_t)16)) + (int64_t)18432)])), var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax0_0 * 2) + ax1_0)], 128, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int64_t ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < (int64_t)8; ++ax0_ax1_fused_0) {
    if (((((((int64_t)blockIdx.x) * (int64_t)192) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)64)) + (ax0_ax1_fused_0 * (int64_t)8)) + (((int64_t)threadIdx.x) >> (int64_t)2)) < n) {
      uint4 __1;
        uint4 v_ = *(uint4*)(((half*)buf_dyn_shmem) + (((((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)8192) + (ax0_ax1_fused_0 * (int64_t)1024)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)) + (int64_t)18432));
        uint4 v__1 = *(uint4*)(lv4 + (((((int64_t)blockIdx.y) * (int64_t)128) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(var_T_add_intermediate + (((((((((int64_t)blockIdx.x) * (int64_t)491520) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)163840)) + (ax0_ax1_fused_0 * (int64_t)20480)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)2560)) + (((int64_t)blockIdx.y) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(320) gemm_n_769_to_1024__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[8];
  uint4 lv9_reindex_pad_shared_dyn_local[4];
  uint4 lv3_reindex_shared_dyn_local[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> lv9_reindex_pad_shared_dyn_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> lv3_reindex_shared_dyn_wmma_matrix_b[8];
  for (int ax1_0_3_init = 0; ax1_0_3_init < 2; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 4; ++ax2_0_3_init) {
      nvcuda::wmma::fill_fragment(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_init * 4) + ax2_0_3_init)], 0.000000e+00f);
    }
  }
  for (int64_t ax1_ax2_fused_0_0_0 = 0; ax1_ax2_fused_0_0_0 < (int64_t)4; ++ax1_ax2_fused_0_0_0) {
    uint4 condval;
    if ((((((((int64_t)blockIdx.x) * (int64_t)160) + (ax1_ax2_fused_0_0_0 * (int64_t)40)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
      condval = *(uint4*)(lv9 + (((((((int64_t)blockIdx.x) * (int64_t)1638400) + (ax1_ax2_fused_0_0_0 * (int64_t)409600)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
    }
    lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0] = condval;
  }
  for (int ax1_ax2_fused_0_0_0_1 = 0; ax1_ax2_fused_0_0_0_1 < 4; ++ax1_ax2_fused_0_0_0_1) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)5) + (((int64_t)threadIdx.y) >> (int64_t)1)) < (int64_t)16) {
      lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_1] = *(uint4*)(lv3 + (((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)409600)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    }
  }
  for (int ax1_ax2_fused_0_0_0_2 = 0; ax1_ax2_fused_0_0_0_2 < 4; ++ax1_ax2_fused_0_0_0_2) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((int64_t)ax1_ax2_fused_0_0_0_2) * (int64_t)2880) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_2];
  }
  for (int ax1_ax2_fused_0_0_0_3 = 0; ax1_ax2_fused_0_0_0_3 < 4; ++ax1_ax2_fused_0_0_0_3) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)5) + (((int64_t)threadIdx.y) >> (int64_t)1)) < (int64_t)16) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)2880) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_3];
    }
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)2304) + (int64_t)18432)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)2304) + (int64_t)19584)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)4608)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)4608) + (int64_t)1152)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)4608) + (int64_t)2304)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)4608) + (int64_t)3456)])), 72);
  for (int64_t ax3_0_0 = 0; ax3_0_0 < (int64_t)159; ++ax3_0_0) {
    for (int64_t ax1_ax2_fused_0_0_0_4 = 0; ax1_ax2_fused_0_0_0_4 < (int64_t)4; ++ax1_ax2_fused_0_0_0_4) {
      uint4 condval_1;
      if ((((((((int64_t)blockIdx.x) * (int64_t)160) + (ax1_ax2_fused_0_0_0_4 * (int64_t)40)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
        condval_1 = *(uint4*)(lv9 + (((((((((int64_t)blockIdx.x) * (int64_t)1638400) + (ax1_ax2_fused_0_0_0_4 * (int64_t)409600)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      } else {
        condval_1 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
      lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_4] = condval_1;
    }
    for (int ax1_ax2_fused_0_0_0_5 = 0; ax1_ax2_fused_0_0_0_5 < 4; ++ax1_ax2_fused_0_0_0_5) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)5) + (((int64_t)threadIdx.y) >> (int64_t)1)) < (int64_t)16) {
        lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_5] = *(uint4*)(lv3 + (((((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)409600)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      }
    }
    for (int ax3_0_1 = 0; ax3_0_1 < 3; ++ax3_0_1) {
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)11520) + ((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)18448)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)11520) + ((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)19600)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)16)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)1168)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)2320)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)3472)])), 72);
      for (int ax1_0_3 = 0; ax1_0_3 < 2; ++ax1_0_3) {
        for (int ax2_0_3 = 0; ax2_0_3 < 4; ++ax2_0_3) {
          nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 4) + ax2_0_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 & 1) * 2) + ax1_0_3)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 & 1) * 4) + ax2_0_3)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 4) + ax2_0_3)]);
        }
      }
    }
    __syncthreads();
    for (int ax1_ax2_fused_0_0_0_6 = 0; ax1_ax2_fused_0_0_0_6 < 4; ++ax1_ax2_fused_0_0_0_6) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)11520) + (((int64_t)ax1_ax2_fused_0_0_0_6) * (int64_t)2880)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_6];
    }
    for (int ax1_ax2_fused_0_0_0_7 = 0; ax1_ax2_fused_0_0_0_7 < 4; ++ax1_ax2_fused_0_0_0_7) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)5) + (((int64_t)threadIdx.y) >> (int64_t)1)) < (int64_t)16) {
        *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + (((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)2880)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_7];
      }
    }
    __syncthreads();
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)11520) + ((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)2304)) + (int64_t)18432)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)11520) + ((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)2304)) + (int64_t)19584)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)4608))])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)4608)) + (int64_t)1152)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)4608)) + (int64_t)2304)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)4608)) + (int64_t)3456)])), 72);
    for (int ax1_0_3_1 = 0; ax1_0_3_1 < 2; ++ax1_0_3_1) {
      for (int ax2_0_3_1 = 0; ax2_0_3_1 < 4; ++ax2_0_3_1) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 4) + ax2_0_3_1)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_1 + 2)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_1 + 4)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 4) + ax2_0_3_1)]);
      }
    }
  }
  for (int ax3_0_1_1 = 0; ax3_0_1_1 < 3; ++ax3_0_1_1) {
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)29968)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)31120)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)9232)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)10384)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)11536)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)12688)])), 72);
    for (int ax1_0_3_2 = 0; ax1_0_3_2 < 2; ++ax1_0_3_2) {
      for (int ax2_0_3_2 = 0; ax2_0_3_2 < 4; ++ax2_0_3_2) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 4) + ax2_0_3_2)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 & 1) * 2) + ax1_0_3_2)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 & 1) * 4) + ax2_0_3_2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 4) + ax2_0_3_2)]);
      }
    }
  }
  for (int ax1_0_3_3 = 0; ax1_0_3_3 < 2; ++ax1_0_3_3) {
    for (int ax2_0_3_3 = 0; ax2_0_3_3 < 4; ++ax2_0_3_3) {
      nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 4) + ax2_0_3_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_3 + 2)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_3 + 4)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 4) + ax2_0_3_3)]);
    }
  }
  __syncthreads();
  for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {
    for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
      nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)4096) + (((int64_t)ax0_0) * (int64_t)2048)) + ((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)64)) + (((int64_t)ax1_0) * (int64_t)16)) + (int64_t)18432)])), var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax0_0 * 4) + ax1_0)], 128, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int64_t ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < (int64_t)8; ++ax0_ax1_fused_0) {
    if (((((((int64_t)blockIdx.x) * (int64_t)160) + ((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)32)) + (ax0_ax1_fused_0 * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n) {
      uint4 __1;
        uint4 v_ = *(uint4*)(((half*)buf_dyn_shmem) + (((((((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)4096) + (ax0_ax1_fused_0 * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432));
        uint4 v__1 = *(uint4*)(lv4 + (((((int64_t)blockIdx.y) * (int64_t)128) + ((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(var_T_add_intermediate + (((((((((int64_t)blockIdx.x) * (int64_t)409600) + ((((int64_t)threadIdx.y) % (int64_t)5) * (int64_t)81920)) + (ax0_ax1_fused_0 * (int64_t)10240)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)2560)) + (((int64_t)blockIdx.y) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)5) * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(384) gemm_n_1025_to_1280__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[8];
  uint4 lv9_reindex_pad_shared_dyn_local[4];
  uint4 lv3_reindex_shared_dyn_local[3];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> lv9_reindex_pad_shared_dyn_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> lv3_reindex_shared_dyn_wmma_matrix_b[8];
  for (int ax1_0_3_init = 0; ax1_0_3_init < 2; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 4; ++ax2_0_3_init) {
      nvcuda::wmma::fill_fragment(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_init * 4) + ax2_0_3_init)], 0.000000e+00f);
    }
  }
  for (int64_t ax1_ax2_fused_0_0_0 = 0; ax1_ax2_fused_0_0_0 < (int64_t)4; ++ax1_ax2_fused_0_0_0) {
    uint4 condval;
    if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0 * (int64_t)48)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
      condval = *(uint4*)(lv9 + (((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0 * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
    }
    lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0] = condval;
  }
  for (int ax1_ax2_fused_0_0_0_1 = 0; ax1_ax2_fused_0_0_0_1 < 3; ++ax1_ax2_fused_0_0_0_1) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
      lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_1] = *(uint4*)(lv3 + (((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    }
  }
  for (int ax1_ax2_fused_0_0_0_2 = 0; ax1_ax2_fused_0_0_0_2 < 4; ++ax1_ax2_fused_0_0_0_2) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((int64_t)ax1_ax2_fused_0_0_0_2) * (int64_t)3456) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_2];
  }
  for (int ax1_ax2_fused_0_0_0_3 = 0; ax1_ax2_fused_0_0_0_3 < 3; ++ax1_ax2_fused_0_0_0_3) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)3456) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_3];
    }
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (int64_t)18432)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (int64_t)19584)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (int64_t)1152)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (int64_t)2304)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (int64_t)3456)])), 72);
  for (int64_t ax3_0_0 = 0; ax3_0_0 < (int64_t)159; ++ax3_0_0) {
    for (int64_t ax1_ax2_fused_0_0_0_4 = 0; ax1_ax2_fused_0_0_0_4 < (int64_t)4; ++ax1_ax2_fused_0_0_0_4) {
      uint4 condval_1;
      if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0_4 * (int64_t)48)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
        condval_1 = *(uint4*)(lv9 + (((((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0_4 * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      } else {
        condval_1 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
      lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_4] = condval_1;
    }
    for (int ax1_ax2_fused_0_0_0_5 = 0; ax1_ax2_fused_0_0_0_5 < 3; ++ax1_ax2_fused_0_0_0_5) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
        lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_5] = *(uint4*)(lv3 + (((((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      }
    }
    for (int ax3_0_1 = 0; ax3_0_1 < 3; ++ax3_0_1) {
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)18448)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)19600)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)16)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)1168)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)2320)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)3472)])), 72);
      for (int ax1_0_3 = 0; ax1_0_3 < 2; ++ax1_0_3) {
        for (int ax2_0_3 = 0; ax2_0_3 < 4; ++ax2_0_3) {
          nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 4) + ax2_0_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 & 1) * 2) + ax1_0_3)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 & 1) * 4) + ax2_0_3)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 4) + ax2_0_3)]);
        }
      }
    }
    __syncthreads();
    for (int ax1_ax2_fused_0_0_0_6 = 0; ax1_ax2_fused_0_0_0_6 < 4; ++ax1_ax2_fused_0_0_0_6) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + (((int64_t)ax1_ax2_fused_0_0_0_6) * (int64_t)3456)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_6];
    }
    for (int ax1_ax2_fused_0_0_0_7 = 0; ax1_ax2_fused_0_0_0_7 < 3; ++ax1_ax2_fused_0_0_0_7) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
        *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + (((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)3456)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_7];
      }
    }
    __syncthreads();
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (int64_t)18432)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (int64_t)19584)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608))])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (int64_t)1152)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (int64_t)2304)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (int64_t)3456)])), 72);
    for (int ax1_0_3_1 = 0; ax1_0_3_1 < 2; ++ax1_0_3_1) {
      for (int ax2_0_3_1 = 0; ax2_0_3_1 < 4; ++ax2_0_3_1) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 4) + ax2_0_3_1)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_1 + 2)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_1 + 4)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 4) + ax2_0_3_1)]);
      }
    }
  }
  for (int ax3_0_1_1 = 0; ax3_0_1_1 < 3; ++ax3_0_1_1) {
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)32272)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)33424)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)9232)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)10384)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)11536)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)12688)])), 72);
    for (int ax1_0_3_2 = 0; ax1_0_3_2 < 2; ++ax1_0_3_2) {
      for (int ax2_0_3_2 = 0; ax2_0_3_2 < 4; ++ax2_0_3_2) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 4) + ax2_0_3_2)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 & 1) * 2) + ax1_0_3_2)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 & 1) * 4) + ax2_0_3_2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 4) + ax2_0_3_2)]);
      }
    }
  }
  for (int ax1_0_3_3 = 0; ax1_0_3_3 < 2; ++ax1_0_3_3) {
    for (int ax2_0_3_3 = 0; ax2_0_3_3 < 4; ++ax2_0_3_3) {
      nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 4) + ax2_0_3_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_3 + 2)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_3 + 4)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 4) + ax2_0_3_3)]);
    }
  }
  __syncthreads();
  for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {
    for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
      nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)4096) + (((int64_t)ax0_0) * (int64_t)2048)) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + (((int64_t)ax1_0) * (int64_t)16)) + (int64_t)18432)])), var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax0_0 * 4) + ax1_0)], 128, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int64_t ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < (int64_t)8; ++ax0_ax1_fused_0) {
    if (((((((int64_t)blockIdx.x) * (int64_t)192) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)32)) + (ax0_ax1_fused_0 * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n) {
      uint4 __1;
        uint4 v_ = *(uint4*)(((half*)buf_dyn_shmem) + (((((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)4096) + (ax0_ax1_fused_0 * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432));
        uint4 v__1 = *(uint4*)(lv4 + (((((int64_t)blockIdx.y) * (int64_t)128) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(var_T_add_intermediate + (((((((((int64_t)blockIdx.x) * (int64_t)491520) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)81920)) + (ax0_ax1_fused_0 * (int64_t)10240)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)2560)) + (((int64_t)blockIdx.y) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(384) gemm_n_1281_to_1536__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[8];
  uint4 lv9_reindex_pad_shared_dyn_local[4];
  uint4 lv3_reindex_shared_dyn_local[3];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> lv9_reindex_pad_shared_dyn_wmma_matrix_a[8];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> lv3_reindex_shared_dyn_wmma_matrix_b[4];
  for (int ax1_0_3_init = 0; ax1_0_3_init < 4; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 2; ++ax2_0_3_init) {
      nvcuda::wmma::fill_fragment(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_init * 2) + ax2_0_3_init)], 0.000000e+00f);
    }
  }
  for (int64_t ax1_ax2_fused_0_0_0 = 0; ax1_ax2_fused_0_0_0 < (int64_t)4; ++ax1_ax2_fused_0_0_0) {
    uint4 condval;
    if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0 * (int64_t)48)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
      condval = *(uint4*)(lv9 + (((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0 * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
    }
    lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0] = condval;
  }
  for (int ax1_ax2_fused_0_0_0_1 = 0; ax1_ax2_fused_0_0_0_1 < 3; ++ax1_ax2_fused_0_0_0_1) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
      lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_1] = *(uint4*)(lv3 + (((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    }
  }
  for (int ax1_ax2_fused_0_0_0_2 = 0; ax1_ax2_fused_0_0_0_2 < 4; ++ax1_ax2_fused_0_0_0_2) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((int64_t)ax1_ax2_fused_0_0_0_2) * (int64_t)3456) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_2];
  }
  for (int ax1_ax2_fused_0_0_0_3 = 0; ax1_ax2_fused_0_0_0_3 < 3; ++ax1_ax2_fused_0_0_0_3) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)3456) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_3];
    }
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (int64_t)18432)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (int64_t)19584)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (int64_t)20736)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (int64_t)21888)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304) + (int64_t)1152)])), 72);
  for (int64_t ax3_0_0 = 0; ax3_0_0 < (int64_t)159; ++ax3_0_0) {
    for (int64_t ax1_ax2_fused_0_0_0_4 = 0; ax1_ax2_fused_0_0_0_4 < (int64_t)4; ++ax1_ax2_fused_0_0_0_4) {
      uint4 condval_1;
      if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0_4 * (int64_t)48)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
        condval_1 = *(uint4*)(lv9 + (((((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0_4 * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      } else {
        condval_1 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
      lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_4] = condval_1;
    }
    for (int ax1_ax2_fused_0_0_0_5 = 0; ax1_ax2_fused_0_0_0_5 < 3; ++ax1_ax2_fused_0_0_0_5) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
        lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_5] = *(uint4*)(lv3 + (((((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      }
    }
    for (int ax3_0_1 = 0; ax3_0_1 < 3; ++ax3_0_1) {
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)18448)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)19600)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)20752)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)21904)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)16)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)1168)])), 72);
      for (int ax1_0_3 = 0; ax1_0_3 < 4; ++ax1_0_3) {
        for (int ax2_0_3 = 0; ax2_0_3 < 2; ++ax2_0_3) {
          nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 2) + ax2_0_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 & 1) * 4) + ax1_0_3)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 & 1) * 2) + ax2_0_3)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 2) + ax2_0_3)]);
        }
      }
    }
    __syncthreads();
    for (int ax1_ax2_fused_0_0_0_6 = 0; ax1_ax2_fused_0_0_0_6 < 4; ++ax1_ax2_fused_0_0_0_6) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + (((int64_t)ax1_ax2_fused_0_0_0_6) * (int64_t)3456)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_6];
    }
    for (int ax1_ax2_fused_0_0_0_7 = 0; ax1_ax2_fused_0_0_0_7 < 3; ++ax1_ax2_fused_0_0_0_7) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
        *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + (((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)3456)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_7];
      }
    }
    __syncthreads();
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (int64_t)18432)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (int64_t)19584)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (int64_t)20736)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (int64_t)21888)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304))])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304)) + (int64_t)1152)])), 72);
    for (int ax1_0_3_1 = 0; ax1_0_3_1 < 4; ++ax1_0_3_1) {
      for (int ax2_0_3_1 = 0; ax2_0_3_1 < 2; ++ax2_0_3_1) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 2) + ax2_0_3_1)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_1 + 4)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_1 + 2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 2) + ax2_0_3_1)]);
      }
    }
  }
  for (int ax3_0_1_1 = 0; ax3_0_1_1 < 3; ++ax3_0_1_1) {
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)32272)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)33424)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)34576)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)35728)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)9232)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)10384)])), 72);
    for (int ax1_0_3_2 = 0; ax1_0_3_2 < 4; ++ax1_0_3_2) {
      for (int ax2_0_3_2 = 0; ax2_0_3_2 < 2; ++ax2_0_3_2) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 2) + ax2_0_3_2)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 & 1) * 4) + ax1_0_3_2)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 & 1) * 2) + ax2_0_3_2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 2) + ax2_0_3_2)]);
      }
    }
  }
  for (int ax1_0_3_3 = 0; ax1_0_3_3 < 4; ++ax1_0_3_3) {
    for (int ax2_0_3_3 = 0; ax2_0_3_3 < 2; ++ax2_0_3_3) {
      nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 2) + ax2_0_3_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_3 + 4)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_3 + 2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 2) + ax2_0_3_3)]);
    }
  }
  __syncthreads();
  for (int ax0_0 = 0; ax0_0 < 4; ++ax0_0) {
    for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0) {
      nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)8192) + (((int64_t)ax0_0) * (int64_t)2048)) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)32)) + (((int64_t)ax1_0) * (int64_t)16)) + (int64_t)18432)])), var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax0_0 * 2) + ax1_0)], 128, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int64_t ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < (int64_t)8; ++ax0_ax1_fused_0) {
    if (((((((int64_t)blockIdx.x) * (int64_t)192) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)64)) + (ax0_ax1_fused_0 * (int64_t)8)) + (((int64_t)threadIdx.x) >> (int64_t)2)) < n) {
      uint4 __1;
        uint4 v_ = *(uint4*)(((half*)buf_dyn_shmem) + (((((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)8192) + (ax0_ax1_fused_0 * (int64_t)1024)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)) + (int64_t)18432));
        uint4 v__1 = *(uint4*)(lv4 + (((((int64_t)blockIdx.y) * (int64_t)128) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(var_T_add_intermediate + (((((((((int64_t)blockIdx.x) * (int64_t)491520) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)163840)) + (ax0_ax1_fused_0 * (int64_t)20480)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)2560)) + (((int64_t)blockIdx.y) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(224) gemm_n_1537_to_1792__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, half> var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[16];
  uint4 lv9_reindex_pad_shared_dyn_local[8];
  uint4 lv3_reindex_shared_dyn_local[5];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, half, nvcuda::wmma::row_major> lv9_reindex_pad_shared_dyn_wmma_matrix_a[8];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, half, nvcuda::wmma::col_major> lv3_reindex_shared_dyn_wmma_matrix_b[8];
  for (int ax1_0_3_init = 0; ax1_0_3_init < 4; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 4; ++ax2_0_3_init) {
      nvcuda::wmma::fill_fragment(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_init * 4) + ax2_0_3_init)], 0.000000e+00f);
    }
  }
  for (int64_t ax1_ax2_fused_0_0_0 = 0; ax1_ax2_fused_0_0_0 < (int64_t)8; ++ax1_ax2_fused_0_0_0) {
    uint4 condval;
    if ((((((((int64_t)blockIdx.x) * (int64_t)224) + (ax1_ax2_fused_0_0_0 * (int64_t)28)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
      condval = *(uint4*)(lv9 + (((((((int64_t)blockIdx.x) * (int64_t)2293760) + (ax1_ax2_fused_0_0_0 * (int64_t)286720)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
    }
    lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0] = condval;
  }
  for (int ax1_ax2_fused_0_0_0_1 = 0; ax1_ax2_fused_0_0_0_1 < 5; ++ax1_ax2_fused_0_0_0_1) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)7) + ((int64_t)threadIdx.y)) < (int64_t)32) {
      lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_1] = *(uint4*)(lv3 + (((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)286720)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    }
  }
  for (int ax1_ax2_fused_0_0_0_2 = 0; ax1_ax2_fused_0_0_0_2 < 8; ++ax1_ax2_fused_0_0_0_2) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((int64_t)ax1_ax2_fused_0_0_0_2) * (int64_t)2016) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_2];
  }
  for (int ax1_ax2_fused_0_0_0_3 = 0; ax1_ax2_fused_0_0_0_3 < 5; ++ax1_ax2_fused_0_0_0_3) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)7) + ((int64_t)threadIdx.y)) < (int64_t)32) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)2016) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_3];
    }
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int64_t)threadIdx.y) * (int64_t)2304) + (int64_t)18432)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int64_t)threadIdx.y) * (int64_t)2304) + (int64_t)19008)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[((((int64_t)threadIdx.y) * (int64_t)2304) + (int64_t)19584)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[((((int64_t)threadIdx.y) * (int64_t)2304) + (int64_t)20160)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[2304])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[4608])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[6912])), 72);
  for (int64_t ax3_0_0 = 0; ax3_0_0 < (int64_t)159; ++ax3_0_0) {
    for (int64_t ax1_ax2_fused_0_0_0_4 = 0; ax1_ax2_fused_0_0_0_4 < (int64_t)8; ++ax1_ax2_fused_0_0_0_4) {
      uint4 condval_1;
      if ((((((((int64_t)blockIdx.x) * (int64_t)224) + (ax1_ax2_fused_0_0_0_4 * (int64_t)28)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
        condval_1 = *(uint4*)(lv9 + (((((((((int64_t)blockIdx.x) * (int64_t)2293760) + (ax1_ax2_fused_0_0_0_4 * (int64_t)286720)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      } else {
        condval_1 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
      lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_4] = condval_1;
    }
    for (int ax1_ax2_fused_0_0_0_5 = 0; ax1_ax2_fused_0_0_0_5 < 5; ++ax1_ax2_fused_0_0_0_5) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)7) + ((int64_t)threadIdx.y)) < (int64_t)32) {
        lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_5] = *(uint4*)(lv3 + (((((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)286720)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      }
    }
    for (int ax3_0_1 = 0; ax3_0_1 < 3; ++ax3_0_1) {
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)16128) + (((int64_t)threadIdx.y) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)18448)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)16128) + (((int64_t)threadIdx.y) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)19024)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)16128) + (((int64_t)threadIdx.y) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)19600)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)16128) + (((int64_t)threadIdx.y) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)20176)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)16)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)2320)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)4624)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)6928)])), 72);
      for (int ax1_0_3 = 0; ax1_0_3 < 4; ++ax1_0_3) {
        for (int ax2_0_3 = 0; ax2_0_3 < 4; ++ax2_0_3) {
          nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 4) + ax2_0_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 & 1) * 4) + ax1_0_3)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 & 1) * 4) + ax2_0_3)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 4) + ax2_0_3)]);
        }
      }
    }
    __syncthreads();
    for (int ax1_ax2_fused_0_0_0_6 = 0; ax1_ax2_fused_0_0_0_6 < 8; ++ax1_ax2_fused_0_0_0_6) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)16128) + (((int64_t)ax1_ax2_fused_0_0_0_6) * (int64_t)2016)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_6];
    }
    for (int ax1_ax2_fused_0_0_0_7 = 0; ax1_ax2_fused_0_0_0_7 < 5; ++ax1_ax2_fused_0_0_0_7) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)7) + ((int64_t)threadIdx.y)) < (int64_t)32) {
        *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + (((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)2016)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_7];
      }
    }
    __syncthreads();
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)16128) + (((int64_t)threadIdx.y) * (int64_t)2304)) + (int64_t)18432)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)16128) + (((int64_t)threadIdx.y) * (int64_t)2304)) + (int64_t)19008)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)16128) + (((int64_t)threadIdx.y) * (int64_t)2304)) + (int64_t)19584)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)16128) + (((int64_t)threadIdx.y) * (int64_t)2304)) + (int64_t)20160)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + (int64_t)2304)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + (int64_t)4608)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + (int64_t)6912)])), 72);
    for (int ax1_0_3_1 = 0; ax1_0_3_1 < 4; ++ax1_0_3_1) {
      for (int ax2_0_3_1 = 0; ax2_0_3_1 < 4; ++ax2_0_3_1) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 4) + ax2_0_3_1)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_1 + 4)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_1 + 4)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 4) + ax2_0_3_1)]);
      }
    }
  }
  for (int ax3_0_1_1 = 0; ax3_0_1_1 < 3; ++ax3_0_1_1) {
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)34576)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)35152)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)35728)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)36304)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[((ax3_0_1_1 * 16) + 9232)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[((ax3_0_1_1 * 16) + 11536)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[((ax3_0_1_1 * 16) + 13840)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[((ax3_0_1_1 * 16) + 16144)])), 72);
    for (int ax1_0_3_2 = 0; ax1_0_3_2 < 4; ++ax1_0_3_2) {
      for (int ax2_0_3_2 = 0; ax2_0_3_2 < 4; ++ax2_0_3_2) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 4) + ax2_0_3_2)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 & 1) * 4) + ax1_0_3_2)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 & 1) * 4) + ax2_0_3_2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 4) + ax2_0_3_2)]);
      }
    }
  }
  for (int ax1_0_3_3 = 0; ax1_0_3_3 < 4; ++ax1_0_3_3) {
    for (int ax2_0_3_3 = 0; ax2_0_3_3 < 4; ++ax2_0_3_3) {
      nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 4) + ax2_0_3_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_3 + 4)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_3 + 4)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 4) + ax2_0_3_3)]);
    }
  }
  __syncthreads();
  for (int ax0_0 = 0; ax0_0 < 4; ++ax0_0) {
    for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
      nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) * (int64_t)4096) + (((int64_t)ax0_0) * (int64_t)1024)) + (((int64_t)ax1_0) * (int64_t)32)) + (int64_t)18432)])), var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax0_0 * 4) + ax1_0)], 128, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int64_t ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < (int64_t)16; ++ax0_ax1_fused_0) {
    if (((((((int64_t)blockIdx.x) * (int64_t)224) + (((int64_t)threadIdx.y) * (int64_t)32)) + (ax0_ax1_fused_0 * (int64_t)2)) + (((int64_t)threadIdx.x) >> (int64_t)4)) < n) {
      uint4 __1;
        uint4 v_ = *(uint4*)(((half*)buf_dyn_shmem) + ((((((int64_t)threadIdx.y) * (int64_t)4096) + (ax0_ax1_fused_0 * (int64_t)256)) + (((int64_t)threadIdx.x) * (int64_t)8)) + (int64_t)18432));
        uint4 v__1 = *(uint4*)(lv4 + ((((int64_t)blockIdx.y) * (int64_t)128) + ((((int64_t)threadIdx.x) & (int64_t)15) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(var_T_add_intermediate + ((((((((int64_t)blockIdx.x) * (int64_t)573440) + (((int64_t)threadIdx.y) * (int64_t)81920)) + (ax0_ax1_fused_0 * (int64_t)5120)) + ((((int64_t)threadIdx.x) >> (int64_t)4) * (int64_t)2560)) + (((int64_t)blockIdx.y) * (int64_t)128)) + ((((int64_t)threadIdx.x) & (int64_t)15) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(384) gemm_n_1793_to_2048__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[8];
  uint4 lv9_reindex_pad_shared_dyn_local[4];
  uint4 lv3_reindex_shared_dyn_local[3];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> lv9_reindex_pad_shared_dyn_wmma_matrix_a[8];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> lv3_reindex_shared_dyn_wmma_matrix_b[4];
  for (int ax1_0_3_init = 0; ax1_0_3_init < 4; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 2; ++ax2_0_3_init) {
      nvcuda::wmma::fill_fragment(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_init * 2) + ax2_0_3_init)], 0.000000e+00f);
    }
  }
  for (int64_t ax1_ax2_fused_0_0_0 = 0; ax1_ax2_fused_0_0_0 < (int64_t)4; ++ax1_ax2_fused_0_0_0) {
    uint4 condval;
    if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0 * (int64_t)48)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
      condval = *(uint4*)(lv9 + (((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0 * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
    }
    lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0] = condval;
  }
  for (int ax1_ax2_fused_0_0_0_1 = 0; ax1_ax2_fused_0_0_0_1 < 3; ++ax1_ax2_fused_0_0_0_1) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
      lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_1] = *(uint4*)(lv3 + (((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    }
  }
  for (int ax1_ax2_fused_0_0_0_2 = 0; ax1_ax2_fused_0_0_0_2 < 4; ++ax1_ax2_fused_0_0_0_2) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((int64_t)ax1_ax2_fused_0_0_0_2) * (int64_t)3456) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_2];
  }
  for (int ax1_ax2_fused_0_0_0_3 = 0; ax1_ax2_fused_0_0_0_3 < 3; ++ax1_ax2_fused_0_0_0_3) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)3456) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_3];
    }
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (int64_t)18432)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (int64_t)19584)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (int64_t)20736)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (int64_t)21888)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304) + (int64_t)1152)])), 72);
  for (int64_t ax3_0_0 = 0; ax3_0_0 < (int64_t)159; ++ax3_0_0) {
    for (int64_t ax1_ax2_fused_0_0_0_4 = 0; ax1_ax2_fused_0_0_0_4 < (int64_t)4; ++ax1_ax2_fused_0_0_0_4) {
      uint4 condval_1;
      if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0_4 * (int64_t)48)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
        condval_1 = *(uint4*)(lv9 + (((((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0_4 * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      } else {
        condval_1 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
      lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_4] = condval_1;
    }
    for (int ax1_ax2_fused_0_0_0_5 = 0; ax1_ax2_fused_0_0_0_5 < 3; ++ax1_ax2_fused_0_0_0_5) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
        lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_5] = *(uint4*)(lv3 + (((((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      }
    }
    for (int ax3_0_1 = 0; ax3_0_1 < 3; ++ax3_0_1) {
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)18448)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)19600)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)20752)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)21904)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)16)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)1168)])), 72);
      for (int ax1_0_3 = 0; ax1_0_3 < 4; ++ax1_0_3) {
        for (int ax2_0_3 = 0; ax2_0_3 < 2; ++ax2_0_3) {
          nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 2) + ax2_0_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 & 1) * 4) + ax1_0_3)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 & 1) * 2) + ax2_0_3)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 2) + ax2_0_3)]);
        }
      }
    }
    __syncthreads();
    for (int ax1_ax2_fused_0_0_0_6 = 0; ax1_ax2_fused_0_0_0_6 < 4; ++ax1_ax2_fused_0_0_0_6) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + (((int64_t)ax1_ax2_fused_0_0_0_6) * (int64_t)3456)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_6];
    }
    for (int ax1_ax2_fused_0_0_0_7 = 0; ax1_ax2_fused_0_0_0_7 < 3; ++ax1_ax2_fused_0_0_0_7) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
        *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + (((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)3456)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_7];
      }
    }
    __syncthreads();
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (int64_t)18432)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (int64_t)19584)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (int64_t)20736)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (int64_t)21888)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304))])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304)) + (int64_t)1152)])), 72);
    for (int ax1_0_3_1 = 0; ax1_0_3_1 < 4; ++ax1_0_3_1) {
      for (int ax2_0_3_1 = 0; ax2_0_3_1 < 2; ++ax2_0_3_1) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 2) + ax2_0_3_1)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_1 + 4)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_1 + 2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 2) + ax2_0_3_1)]);
      }
    }
  }
  for (int ax3_0_1_1 = 0; ax3_0_1_1 < 3; ++ax3_0_1_1) {
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)32272)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)33424)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)34576)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)35728)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)9232)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)10384)])), 72);
    for (int ax1_0_3_2 = 0; ax1_0_3_2 < 4; ++ax1_0_3_2) {
      for (int ax2_0_3_2 = 0; ax2_0_3_2 < 2; ++ax2_0_3_2) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 2) + ax2_0_3_2)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 & 1) * 4) + ax1_0_3_2)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 & 1) * 2) + ax2_0_3_2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 2) + ax2_0_3_2)]);
      }
    }
  }
  for (int ax1_0_3_3 = 0; ax1_0_3_3 < 4; ++ax1_0_3_3) {
    for (int ax2_0_3_3 = 0; ax2_0_3_3 < 2; ++ax2_0_3_3) {
      nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 2) + ax2_0_3_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_3 + 4)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_3 + 2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 2) + ax2_0_3_3)]);
    }
  }
  __syncthreads();
  for (int ax0_0 = 0; ax0_0 < 4; ++ax0_0) {
    for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0) {
      nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)8192) + (((int64_t)ax0_0) * (int64_t)2048)) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)32)) + (((int64_t)ax1_0) * (int64_t)16)) + (int64_t)18432)])), var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax0_0 * 2) + ax1_0)], 128, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int64_t ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < (int64_t)8; ++ax0_ax1_fused_0) {
    if (((((((int64_t)blockIdx.x) * (int64_t)192) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)64)) + (ax0_ax1_fused_0 * (int64_t)8)) + (((int64_t)threadIdx.x) >> (int64_t)2)) < n) {
      uint4 __1;
        uint4 v_ = *(uint4*)(((half*)buf_dyn_shmem) + (((((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)8192) + (ax0_ax1_fused_0 * (int64_t)1024)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)) + (int64_t)18432));
        uint4 v__1 = *(uint4*)(lv4 + (((((int64_t)blockIdx.y) * (int64_t)128) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(var_T_add_intermediate + (((((((((int64_t)blockIdx.x) * (int64_t)491520) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)163840)) + (ax0_ax1_fused_0 * (int64_t)20480)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)2560)) + (((int64_t)blockIdx.y) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(384) gemm_n_2049_to_2304__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[8];
  uint4 lv9_reindex_pad_shared_dyn_local[4];
  uint4 lv3_reindex_shared_dyn_local[3];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> lv9_reindex_pad_shared_dyn_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> lv3_reindex_shared_dyn_wmma_matrix_b[8];
  for (int ax1_0_3_init = 0; ax1_0_3_init < 2; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 4; ++ax2_0_3_init) {
      nvcuda::wmma::fill_fragment(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_init * 4) + ax2_0_3_init)], 0.000000e+00f);
    }
  }
  for (int64_t ax1_ax2_fused_0_0_0 = 0; ax1_ax2_fused_0_0_0 < (int64_t)4; ++ax1_ax2_fused_0_0_0) {
    uint4 condval;
    if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0 * (int64_t)48)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
      condval = *(uint4*)(lv9 + (((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0 * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
    }
    lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0] = condval;
  }
  for (int ax1_ax2_fused_0_0_0_1 = 0; ax1_ax2_fused_0_0_0_1 < 3; ++ax1_ax2_fused_0_0_0_1) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
      lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_1] = *(uint4*)(lv3 + (((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    }
  }
  for (int ax1_ax2_fused_0_0_0_2 = 0; ax1_ax2_fused_0_0_0_2 < 4; ++ax1_ax2_fused_0_0_0_2) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((int64_t)ax1_ax2_fused_0_0_0_2) * (int64_t)3456) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_2];
  }
  for (int ax1_ax2_fused_0_0_0_3 = 0; ax1_ax2_fused_0_0_0_3 < 3; ++ax1_ax2_fused_0_0_0_3) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)3456) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_3];
    }
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (int64_t)18432)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (int64_t)19584)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (int64_t)1152)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (int64_t)2304)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (int64_t)3456)])), 72);
  for (int64_t ax3_0_0 = 0; ax3_0_0 < (int64_t)159; ++ax3_0_0) {
    for (int64_t ax1_ax2_fused_0_0_0_4 = 0; ax1_ax2_fused_0_0_0_4 < (int64_t)4; ++ax1_ax2_fused_0_0_0_4) {
      uint4 condval_1;
      if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0_4 * (int64_t)48)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
        condval_1 = *(uint4*)(lv9 + (((((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0_4 * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      } else {
        condval_1 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
      lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_4] = condval_1;
    }
    for (int ax1_ax2_fused_0_0_0_5 = 0; ax1_ax2_fused_0_0_0_5 < 3; ++ax1_ax2_fused_0_0_0_5) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
        lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_5] = *(uint4*)(lv3 + (((((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      }
    }
    for (int ax3_0_1 = 0; ax3_0_1 < 3; ++ax3_0_1) {
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)18448)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)19600)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)16)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)1168)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)2320)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)3472)])), 72);
      for (int ax1_0_3 = 0; ax1_0_3 < 2; ++ax1_0_3) {
        for (int ax2_0_3 = 0; ax2_0_3 < 4; ++ax2_0_3) {
          nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 4) + ax2_0_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 & 1) * 2) + ax1_0_3)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 & 1) * 4) + ax2_0_3)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 4) + ax2_0_3)]);
        }
      }
    }
    __syncthreads();
    for (int ax1_ax2_fused_0_0_0_6 = 0; ax1_ax2_fused_0_0_0_6 < 4; ++ax1_ax2_fused_0_0_0_6) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + (((int64_t)ax1_ax2_fused_0_0_0_6) * (int64_t)3456)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_6];
    }
    for (int ax1_ax2_fused_0_0_0_7 = 0; ax1_ax2_fused_0_0_0_7 < 3; ++ax1_ax2_fused_0_0_0_7) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
        *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + (((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)3456)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_7];
      }
    }
    __syncthreads();
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (int64_t)18432)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (int64_t)19584)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608))])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (int64_t)1152)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (int64_t)2304)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (int64_t)3456)])), 72);
    for (int ax1_0_3_1 = 0; ax1_0_3_1 < 2; ++ax1_0_3_1) {
      for (int ax2_0_3_1 = 0; ax2_0_3_1 < 4; ++ax2_0_3_1) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 4) + ax2_0_3_1)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_1 + 2)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_1 + 4)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 4) + ax2_0_3_1)]);
      }
    }
  }
  for (int ax3_0_1_1 = 0; ax3_0_1_1 < 3; ++ax3_0_1_1) {
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)32272)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)33424)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)9232)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)10384)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)11536)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)12688)])), 72);
    for (int ax1_0_3_2 = 0; ax1_0_3_2 < 2; ++ax1_0_3_2) {
      for (int ax2_0_3_2 = 0; ax2_0_3_2 < 4; ++ax2_0_3_2) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 4) + ax2_0_3_2)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 & 1) * 2) + ax1_0_3_2)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 & 1) * 4) + ax2_0_3_2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 4) + ax2_0_3_2)]);
      }
    }
  }
  for (int ax1_0_3_3 = 0; ax1_0_3_3 < 2; ++ax1_0_3_3) {
    for (int ax2_0_3_3 = 0; ax2_0_3_3 < 4; ++ax2_0_3_3) {
      nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 4) + ax2_0_3_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_3 + 2)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_3 + 4)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 4) + ax2_0_3_3)]);
    }
  }
  __syncthreads();
  for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {
    for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
      nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)4096) + (((int64_t)ax0_0) * (int64_t)2048)) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + (((int64_t)ax1_0) * (int64_t)16)) + (int64_t)18432)])), var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax0_0 * 4) + ax1_0)], 128, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int64_t ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < (int64_t)8; ++ax0_ax1_fused_0) {
    if (((((((int64_t)blockIdx.x) * (int64_t)192) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)32)) + (ax0_ax1_fused_0 * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n) {
      uint4 __1;
        uint4 v_ = *(uint4*)(((half*)buf_dyn_shmem) + (((((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)4096) + (ax0_ax1_fused_0 * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432));
        uint4 v__1 = *(uint4*)(lv4 + (((((int64_t)blockIdx.y) * (int64_t)128) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(var_T_add_intermediate + (((((((((int64_t)blockIdx.x) * (int64_t)491520) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)81920)) + (ax0_ax1_fused_0 * (int64_t)10240)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)2560)) + (((int64_t)blockIdx.y) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(224) gemm_n_2305_to_2560__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, half> var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[16];
  uint4 lv9_reindex_pad_shared_dyn_local[8];
  uint4 lv3_reindex_shared_dyn_local[5];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, half, nvcuda::wmma::row_major> lv9_reindex_pad_shared_dyn_wmma_matrix_a[8];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, half, nvcuda::wmma::col_major> lv3_reindex_shared_dyn_wmma_matrix_b[8];
  for (int ax1_0_3_init = 0; ax1_0_3_init < 4; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 4; ++ax2_0_3_init) {
      nvcuda::wmma::fill_fragment(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_init * 4) + ax2_0_3_init)], 0.000000e+00f);
    }
  }
  for (int64_t ax1_ax2_fused_0_0_0 = 0; ax1_ax2_fused_0_0_0 < (int64_t)8; ++ax1_ax2_fused_0_0_0) {
    uint4 condval;
    if ((((((((int64_t)blockIdx.x) * (int64_t)224) + (ax1_ax2_fused_0_0_0 * (int64_t)28)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
      condval = *(uint4*)(lv9 + (((((((int64_t)blockIdx.x) * (int64_t)2293760) + (ax1_ax2_fused_0_0_0 * (int64_t)286720)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
    }
    lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0] = condval;
  }
  for (int ax1_ax2_fused_0_0_0_1 = 0; ax1_ax2_fused_0_0_0_1 < 5; ++ax1_ax2_fused_0_0_0_1) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)7) + ((int64_t)threadIdx.y)) < (int64_t)32) {
      lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_1] = *(uint4*)(lv3 + (((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)286720)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    }
  }
  for (int ax1_ax2_fused_0_0_0_2 = 0; ax1_ax2_fused_0_0_0_2 < 8; ++ax1_ax2_fused_0_0_0_2) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((int64_t)ax1_ax2_fused_0_0_0_2) * (int64_t)2016) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_2];
  }
  for (int ax1_ax2_fused_0_0_0_3 = 0; ax1_ax2_fused_0_0_0_3 < 5; ++ax1_ax2_fused_0_0_0_3) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)7) + ((int64_t)threadIdx.y)) < (int64_t)32) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)2016) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_3];
    }
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int64_t)threadIdx.y) * (int64_t)2304) + (int64_t)18432)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int64_t)threadIdx.y) * (int64_t)2304) + (int64_t)19008)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[((((int64_t)threadIdx.y) * (int64_t)2304) + (int64_t)19584)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[((((int64_t)threadIdx.y) * (int64_t)2304) + (int64_t)20160)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[2304])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[4608])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[6912])), 72);
  for (int64_t ax3_0_0 = 0; ax3_0_0 < (int64_t)159; ++ax3_0_0) {
    for (int64_t ax1_ax2_fused_0_0_0_4 = 0; ax1_ax2_fused_0_0_0_4 < (int64_t)8; ++ax1_ax2_fused_0_0_0_4) {
      uint4 condval_1;
      if ((((((((int64_t)blockIdx.x) * (int64_t)224) + (ax1_ax2_fused_0_0_0_4 * (int64_t)28)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
        condval_1 = *(uint4*)(lv9 + (((((((((int64_t)blockIdx.x) * (int64_t)2293760) + (ax1_ax2_fused_0_0_0_4 * (int64_t)286720)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      } else {
        condval_1 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
      lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_4] = condval_1;
    }
    for (int ax1_ax2_fused_0_0_0_5 = 0; ax1_ax2_fused_0_0_0_5 < 5; ++ax1_ax2_fused_0_0_0_5) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)7) + ((int64_t)threadIdx.y)) < (int64_t)32) {
        lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_5] = *(uint4*)(lv3 + (((((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)286720)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      }
    }
    for (int ax3_0_1 = 0; ax3_0_1 < 3; ++ax3_0_1) {
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)16128) + (((int64_t)threadIdx.y) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)18448)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)16128) + (((int64_t)threadIdx.y) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)19024)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)16128) + (((int64_t)threadIdx.y) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)19600)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)16128) + (((int64_t)threadIdx.y) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)20176)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)16)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)2320)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)4624)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)6928)])), 72);
      for (int ax1_0_3 = 0; ax1_0_3 < 4; ++ax1_0_3) {
        for (int ax2_0_3 = 0; ax2_0_3 < 4; ++ax2_0_3) {
          nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 4) + ax2_0_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 & 1) * 4) + ax1_0_3)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 & 1) * 4) + ax2_0_3)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 4) + ax2_0_3)]);
        }
      }
    }
    __syncthreads();
    for (int ax1_ax2_fused_0_0_0_6 = 0; ax1_ax2_fused_0_0_0_6 < 8; ++ax1_ax2_fused_0_0_0_6) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)16128) + (((int64_t)ax1_ax2_fused_0_0_0_6) * (int64_t)2016)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_6];
    }
    for (int ax1_ax2_fused_0_0_0_7 = 0; ax1_ax2_fused_0_0_0_7 < 5; ++ax1_ax2_fused_0_0_0_7) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)7) + ((int64_t)threadIdx.y)) < (int64_t)32) {
        *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + (((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)2016)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_7];
      }
    }
    __syncthreads();
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)16128) + (((int64_t)threadIdx.y) * (int64_t)2304)) + (int64_t)18432)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)16128) + (((int64_t)threadIdx.y) * (int64_t)2304)) + (int64_t)19008)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)16128) + (((int64_t)threadIdx.y) * (int64_t)2304)) + (int64_t)19584)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)16128) + (((int64_t)threadIdx.y) * (int64_t)2304)) + (int64_t)20160)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + (int64_t)2304)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + (int64_t)4608)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + (int64_t)6912)])), 72);
    for (int ax1_0_3_1 = 0; ax1_0_3_1 < 4; ++ax1_0_3_1) {
      for (int ax2_0_3_1 = 0; ax2_0_3_1 < 4; ++ax2_0_3_1) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 4) + ax2_0_3_1)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_1 + 4)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_1 + 4)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 4) + ax2_0_3_1)]);
      }
    }
  }
  for (int ax3_0_1_1 = 0; ax3_0_1_1 < 3; ++ax3_0_1_1) {
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)34576)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)35152)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)35728)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)36304)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[((ax3_0_1_1 * 16) + 9232)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[((ax3_0_1_1 * 16) + 11536)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[((ax3_0_1_1 * 16) + 13840)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[((ax3_0_1_1 * 16) + 16144)])), 72);
    for (int ax1_0_3_2 = 0; ax1_0_3_2 < 4; ++ax1_0_3_2) {
      for (int ax2_0_3_2 = 0; ax2_0_3_2 < 4; ++ax2_0_3_2) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 4) + ax2_0_3_2)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 & 1) * 4) + ax1_0_3_2)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 & 1) * 4) + ax2_0_3_2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 4) + ax2_0_3_2)]);
      }
    }
  }
  for (int ax1_0_3_3 = 0; ax1_0_3_3 < 4; ++ax1_0_3_3) {
    for (int ax2_0_3_3 = 0; ax2_0_3_3 < 4; ++ax2_0_3_3) {
      nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 4) + ax2_0_3_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_3 + 4)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_3 + 4)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 4) + ax2_0_3_3)]);
    }
  }
  __syncthreads();
  for (int ax0_0 = 0; ax0_0 < 4; ++ax0_0) {
    for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
      nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) * (int64_t)4096) + (((int64_t)ax0_0) * (int64_t)1024)) + (((int64_t)ax1_0) * (int64_t)32)) + (int64_t)18432)])), var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax0_0 * 4) + ax1_0)], 128, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int64_t ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < (int64_t)16; ++ax0_ax1_fused_0) {
    if (((((((int64_t)blockIdx.x) * (int64_t)224) + (((int64_t)threadIdx.y) * (int64_t)32)) + (ax0_ax1_fused_0 * (int64_t)2)) + (((int64_t)threadIdx.x) >> (int64_t)4)) < n) {
      uint4 __1;
        uint4 v_ = *(uint4*)(((half*)buf_dyn_shmem) + ((((((int64_t)threadIdx.y) * (int64_t)4096) + (ax0_ax1_fused_0 * (int64_t)256)) + (((int64_t)threadIdx.x) * (int64_t)8)) + (int64_t)18432));
        uint4 v__1 = *(uint4*)(lv4 + ((((int64_t)blockIdx.y) * (int64_t)128) + ((((int64_t)threadIdx.x) & (int64_t)15) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(var_T_add_intermediate + ((((((((int64_t)blockIdx.x) * (int64_t)573440) + (((int64_t)threadIdx.y) * (int64_t)81920)) + (ax0_ax1_fused_0 * (int64_t)5120)) + ((((int64_t)threadIdx.x) >> (int64_t)4) * (int64_t)2560)) + (((int64_t)blockIdx.y) * (int64_t)128)) + ((((int64_t)threadIdx.x) & (int64_t)15) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(384) gemm_n_2561_to_2816__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[8];
  uint4 lv9_reindex_pad_shared_dyn_local[4];
  uint4 lv3_reindex_shared_dyn_local[3];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> lv9_reindex_pad_shared_dyn_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> lv3_reindex_shared_dyn_wmma_matrix_b[8];
  for (int ax1_0_3_init = 0; ax1_0_3_init < 2; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 4; ++ax2_0_3_init) {
      nvcuda::wmma::fill_fragment(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_init * 4) + ax2_0_3_init)], 0.000000e+00f);
    }
  }
  for (int64_t ax1_ax2_fused_0_0_0 = 0; ax1_ax2_fused_0_0_0 < (int64_t)4; ++ax1_ax2_fused_0_0_0) {
    uint4 condval;
    if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0 * (int64_t)48)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
      condval = *(uint4*)(lv9 + (((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0 * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
    }
    lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0] = condval;
  }
  for (int ax1_ax2_fused_0_0_0_1 = 0; ax1_ax2_fused_0_0_0_1 < 3; ++ax1_ax2_fused_0_0_0_1) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
      lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_1] = *(uint4*)(lv3 + (((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    }
  }
  for (int ax1_ax2_fused_0_0_0_2 = 0; ax1_ax2_fused_0_0_0_2 < 4; ++ax1_ax2_fused_0_0_0_2) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((int64_t)ax1_ax2_fused_0_0_0_2) * (int64_t)3456) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_2];
  }
  for (int ax1_ax2_fused_0_0_0_3 = 0; ax1_ax2_fused_0_0_0_3 < 3; ++ax1_ax2_fused_0_0_0_3) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)3456) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_3];
    }
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (int64_t)18432)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (int64_t)19584)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (int64_t)1152)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (int64_t)2304)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (int64_t)3456)])), 72);
  for (int64_t ax3_0_0 = 0; ax3_0_0 < (int64_t)159; ++ax3_0_0) {
    for (int64_t ax1_ax2_fused_0_0_0_4 = 0; ax1_ax2_fused_0_0_0_4 < (int64_t)4; ++ax1_ax2_fused_0_0_0_4) {
      uint4 condval_1;
      if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0_4 * (int64_t)48)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
        condval_1 = *(uint4*)(lv9 + (((((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0_4 * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      } else {
        condval_1 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
      lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_4] = condval_1;
    }
    for (int ax1_ax2_fused_0_0_0_5 = 0; ax1_ax2_fused_0_0_0_5 < 3; ++ax1_ax2_fused_0_0_0_5) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
        lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_5] = *(uint4*)(lv3 + (((((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      }
    }
    for (int ax3_0_1 = 0; ax3_0_1 < 3; ++ax3_0_1) {
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)18448)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)19600)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)16)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)1168)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)2320)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)3472)])), 72);
      for (int ax1_0_3 = 0; ax1_0_3 < 2; ++ax1_0_3) {
        for (int ax2_0_3 = 0; ax2_0_3 < 4; ++ax2_0_3) {
          nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 4) + ax2_0_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 & 1) * 2) + ax1_0_3)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 & 1) * 4) + ax2_0_3)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 4) + ax2_0_3)]);
        }
      }
    }
    __syncthreads();
    for (int ax1_ax2_fused_0_0_0_6 = 0; ax1_ax2_fused_0_0_0_6 < 4; ++ax1_ax2_fused_0_0_0_6) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + (((int64_t)ax1_ax2_fused_0_0_0_6) * (int64_t)3456)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_6];
    }
    for (int ax1_ax2_fused_0_0_0_7 = 0; ax1_ax2_fused_0_0_0_7 < 3; ++ax1_ax2_fused_0_0_0_7) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
        *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + (((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)3456)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_7];
      }
    }
    __syncthreads();
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (int64_t)18432)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (int64_t)19584)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608))])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (int64_t)1152)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (int64_t)2304)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (int64_t)3456)])), 72);
    for (int ax1_0_3_1 = 0; ax1_0_3_1 < 2; ++ax1_0_3_1) {
      for (int ax2_0_3_1 = 0; ax2_0_3_1 < 4; ++ax2_0_3_1) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 4) + ax2_0_3_1)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_1 + 2)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_1 + 4)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 4) + ax2_0_3_1)]);
      }
    }
  }
  for (int ax3_0_1_1 = 0; ax3_0_1_1 < 3; ++ax3_0_1_1) {
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)32272)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)33424)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)9232)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)10384)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)11536)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)12688)])), 72);
    for (int ax1_0_3_2 = 0; ax1_0_3_2 < 2; ++ax1_0_3_2) {
      for (int ax2_0_3_2 = 0; ax2_0_3_2 < 4; ++ax2_0_3_2) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 4) + ax2_0_3_2)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 & 1) * 2) + ax1_0_3_2)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 & 1) * 4) + ax2_0_3_2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 4) + ax2_0_3_2)]);
      }
    }
  }
  for (int ax1_0_3_3 = 0; ax1_0_3_3 < 2; ++ax1_0_3_3) {
    for (int ax2_0_3_3 = 0; ax2_0_3_3 < 4; ++ax2_0_3_3) {
      nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 4) + ax2_0_3_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_3 + 2)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_3 + 4)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 4) + ax2_0_3_3)]);
    }
  }
  __syncthreads();
  for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {
    for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
      nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)4096) + (((int64_t)ax0_0) * (int64_t)2048)) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + (((int64_t)ax1_0) * (int64_t)16)) + (int64_t)18432)])), var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax0_0 * 4) + ax1_0)], 128, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int64_t ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < (int64_t)8; ++ax0_ax1_fused_0) {
    if (((((((int64_t)blockIdx.x) * (int64_t)192) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)32)) + (ax0_ax1_fused_0 * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n) {
      uint4 __1;
        uint4 v_ = *(uint4*)(((half*)buf_dyn_shmem) + (((((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)4096) + (ax0_ax1_fused_0 * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432));
        uint4 v__1 = *(uint4*)(lv4 + (((((int64_t)blockIdx.y) * (int64_t)128) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(var_T_add_intermediate + (((((((((int64_t)blockIdx.x) * (int64_t)491520) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)81920)) + (ax0_ax1_fused_0 * (int64_t)10240)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)2560)) + (((int64_t)blockIdx.y) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(512) gemm_n_2817_to_3072__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[6];
  uint4 lv9_reindex_pad_shared_dyn_local[3];
  uint4 lv3_reindex_shared_dyn_local[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> lv9_reindex_pad_shared_dyn_wmma_matrix_a[6];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> lv3_reindex_shared_dyn_wmma_matrix_b[4];
  for (int ax1_0_3_init = 0; ax1_0_3_init < 3; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 2; ++ax2_0_3_init) {
      nvcuda::wmma::fill_fragment(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_init * 2) + ax2_0_3_init)], 0.000000e+00f);
    }
  }
  for (int64_t ax1_ax2_fused_0_0_0 = 0; ax1_ax2_fused_0_0_0 < (int64_t)3; ++ax1_ax2_fused_0_0_0) {
    uint4 condval;
    if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0 * (int64_t)64)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
      condval = *(uint4*)(lv9 + (((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0 * (int64_t)655360)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
    }
    lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0] = condval;
  }
  for (int ax1_ax2_fused_0_0_0_1 = 0; ax1_ax2_fused_0_0_0_1 < 2; ++ax1_ax2_fused_0_0_0_1) {
    lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_1] = *(uint4*)(lv3 + (((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)655360)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
  }
  for (int ax1_ax2_fused_0_0_0_2 = 0; ax1_ax2_fused_0_0_0_2 < 3; ++ax1_ax2_fused_0_0_0_2) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((int64_t)ax1_ax2_fused_0_0_0_2) * (int64_t)4608) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_2];
  }
  for (int ax1_ax2_fused_0_0_0_3 = 0; ax1_ax2_fused_0_0_0_3 < 2; ++ax1_ax2_fused_0_0_0_3) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)4608) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_3];
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456) + (int64_t)18432)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456) + (int64_t)19584)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456) + (int64_t)20736)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)2304)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)2304) + (int64_t)1152)])), 72);
  for (int64_t ax3_0_0 = 0; ax3_0_0 < (int64_t)159; ++ax3_0_0) {
    for (int64_t ax1_ax2_fused_0_0_0_4 = 0; ax1_ax2_fused_0_0_0_4 < (int64_t)3; ++ax1_ax2_fused_0_0_0_4) {
      uint4 condval_1;
      if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0_4 * (int64_t)64)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
        condval_1 = *(uint4*)(lv9 + (((((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0_4 * (int64_t)655360)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      } else {
        condval_1 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
      lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_4] = condval_1;
    }
    for (int ax1_ax2_fused_0_0_0_5 = 0; ax1_ax2_fused_0_0_0_5 < 2; ++ax1_ax2_fused_0_0_0_5) {
      lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_5] = *(uint4*)(lv3 + (((((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)655360)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
    }
    for (int ax3_0_1 = 0; ax3_0_1 < 3; ++ax3_0_1) {
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 + 1) & 1) * 3)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)18448)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 3) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)19600)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 3) + 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)20752)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)16)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)1168)])), 72);
      for (int ax1_0_3 = 0; ax1_0_3 < 3; ++ax1_0_3) {
        for (int ax2_0_3 = 0; ax2_0_3 < 2; ++ax2_0_3) {
          nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 2) + ax2_0_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 & 1) * 3) + ax1_0_3)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 & 1) * 2) + ax2_0_3)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 2) + ax2_0_3)]);
        }
      }
    }
    __syncthreads();
    for (int ax1_ax2_fused_0_0_0_6 = 0; ax1_ax2_fused_0_0_0_6 < 3; ++ax1_ax2_fused_0_0_0_6) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + (((int64_t)ax1_ax2_fused_0_0_0_6) * (int64_t)4608)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_6];
    }
    for (int ax1_ax2_fused_0_0_0_7 = 0; ax1_ax2_fused_0_0_0_7 < 2; ++ax1_ax2_fused_0_0_0_7) {
      *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + (((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)4608)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_7];
    }
    __syncthreads();
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456)) + (int64_t)18432)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456)) + (int64_t)19584)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456)) + (int64_t)20736)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)2304))])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)2304)) + (int64_t)1152)])), 72);
    for (int ax1_0_3_1 = 0; ax1_0_3_1 < 3; ++ax1_0_3_1) {
      for (int ax2_0_3_1 = 0; ax2_0_3_1 < 2; ++ax2_0_3_1) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 2) + ax2_0_3_1)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_1 + 3)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_1 + 2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 2) + ax2_0_3_1)]);
      }
    }
  }
  for (int ax3_0_1_1 = 0; ax3_0_1_1 < 3; ++ax3_0_1_1) {
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 + 1) & 1) * 3)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)32272)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 3) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)33424)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 3) + 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)34576)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)9232)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)10384)])), 72);
    for (int ax1_0_3_2 = 0; ax1_0_3_2 < 3; ++ax1_0_3_2) {
      for (int ax2_0_3_2 = 0; ax2_0_3_2 < 2; ++ax2_0_3_2) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 2) + ax2_0_3_2)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 & 1) * 3) + ax1_0_3_2)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 & 1) * 2) + ax2_0_3_2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 2) + ax2_0_3_2)]);
      }
    }
  }
  for (int ax1_0_3_3 = 0; ax1_0_3_3 < 3; ++ax1_0_3_3) {
    for (int ax2_0_3_3 = 0; ax2_0_3_3 < 2; ++ax2_0_3_3) {
      nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 2) + ax2_0_3_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_3 + 3)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_3 + 2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 2) + ax2_0_3_3)]);
    }
  }
  __syncthreads();
  for (int ax0_0 = 0; ax0_0 < 3; ++ax0_0) {
    for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0) {
      nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)6144) + (((int64_t)ax0_0) * (int64_t)2048)) + ((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)32)) + (((int64_t)ax1_0) * (int64_t)16)) + (int64_t)18432)])), var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax0_0 * 2) + ax1_0)], 128, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int64_t ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < (int64_t)6; ++ax0_ax1_fused_0) {
    if (((((((int64_t)blockIdx.x) * (int64_t)192) + ((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)48)) + (ax0_ax1_fused_0 * (int64_t)8)) + (((int64_t)threadIdx.x) >> (int64_t)2)) < n) {
      uint4 __1;
        uint4 v_ = *(uint4*)(((half*)buf_dyn_shmem) + (((((((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)6144) + (ax0_ax1_fused_0 * (int64_t)1024)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)) + (int64_t)18432));
        uint4 v__1 = *(uint4*)(lv4 + (((((int64_t)blockIdx.y) * (int64_t)128) + ((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(var_T_add_intermediate + (((((((((int64_t)blockIdx.x) * (int64_t)491520) + ((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)122880)) + (ax0_ax1_fused_0 * (int64_t)20480)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)2560)) + (((int64_t)blockIdx.y) * (int64_t)128)) + ((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(384) gemm_n_3073_to_3328__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[8];
  uint4 lv9_reindex_pad_shared_dyn_local[4];
  uint4 lv3_reindex_shared_dyn_local[3];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> lv9_reindex_pad_shared_dyn_wmma_matrix_a[8];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> lv3_reindex_shared_dyn_wmma_matrix_b[4];
  for (int ax1_0_3_init = 0; ax1_0_3_init < 4; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 2; ++ax2_0_3_init) {
      nvcuda::wmma::fill_fragment(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_init * 2) + ax2_0_3_init)], 0.000000e+00f);
    }
  }
  for (int64_t ax1_ax2_fused_0_0_0 = 0; ax1_ax2_fused_0_0_0 < (int64_t)4; ++ax1_ax2_fused_0_0_0) {
    uint4 condval;
    if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0 * (int64_t)48)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
      condval = *(uint4*)(lv9 + (((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0 * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
    }
    lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0] = condval;
  }
  for (int ax1_ax2_fused_0_0_0_1 = 0; ax1_ax2_fused_0_0_0_1 < 3; ++ax1_ax2_fused_0_0_0_1) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
      lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_1] = *(uint4*)(lv3 + (((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    }
  }
  for (int ax1_ax2_fused_0_0_0_2 = 0; ax1_ax2_fused_0_0_0_2 < 4; ++ax1_ax2_fused_0_0_0_2) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((int64_t)ax1_ax2_fused_0_0_0_2) * (int64_t)3456) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_2];
  }
  for (int ax1_ax2_fused_0_0_0_3 = 0; ax1_ax2_fused_0_0_0_3 < 3; ++ax1_ax2_fused_0_0_0_3) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)3456) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_3];
    }
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (int64_t)18432)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (int64_t)19584)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (int64_t)20736)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (int64_t)21888)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304) + (int64_t)1152)])), 72);
  for (int64_t ax3_0_0 = 0; ax3_0_0 < (int64_t)159; ++ax3_0_0) {
    for (int64_t ax1_ax2_fused_0_0_0_4 = 0; ax1_ax2_fused_0_0_0_4 < (int64_t)4; ++ax1_ax2_fused_0_0_0_4) {
      uint4 condval_1;
      if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0_4 * (int64_t)48)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
        condval_1 = *(uint4*)(lv9 + (((((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0_4 * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      } else {
        condval_1 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
      lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_4] = condval_1;
    }
    for (int ax1_ax2_fused_0_0_0_5 = 0; ax1_ax2_fused_0_0_0_5 < 3; ++ax1_ax2_fused_0_0_0_5) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
        lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_5] = *(uint4*)(lv3 + (((((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      }
    }
    for (int ax3_0_1 = 0; ax3_0_1 < 3; ++ax3_0_1) {
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)18448)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)19600)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)20752)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)21904)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)16)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)1168)])), 72);
      for (int ax1_0_3 = 0; ax1_0_3 < 4; ++ax1_0_3) {
        for (int ax2_0_3 = 0; ax2_0_3 < 2; ++ax2_0_3) {
          nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 2) + ax2_0_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 & 1) * 4) + ax1_0_3)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 & 1) * 2) + ax2_0_3)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 2) + ax2_0_3)]);
        }
      }
    }
    __syncthreads();
    for (int ax1_ax2_fused_0_0_0_6 = 0; ax1_ax2_fused_0_0_0_6 < 4; ++ax1_ax2_fused_0_0_0_6) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + (((int64_t)ax1_ax2_fused_0_0_0_6) * (int64_t)3456)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_6];
    }
    for (int ax1_ax2_fused_0_0_0_7 = 0; ax1_ax2_fused_0_0_0_7 < 3; ++ax1_ax2_fused_0_0_0_7) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
        *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + (((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)3456)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_7];
      }
    }
    __syncthreads();
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (int64_t)18432)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (int64_t)19584)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (int64_t)20736)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608)) + (int64_t)21888)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304))])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304)) + (int64_t)1152)])), 72);
    for (int ax1_0_3_1 = 0; ax1_0_3_1 < 4; ++ax1_0_3_1) {
      for (int ax2_0_3_1 = 0; ax2_0_3_1 < 2; ++ax2_0_3_1) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 2) + ax2_0_3_1)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_1 + 4)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_1 + 2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 2) + ax2_0_3_1)]);
      }
    }
  }
  for (int ax3_0_1_1 = 0; ax3_0_1_1 < 3; ++ax3_0_1_1) {
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)32272)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)33424)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)34576)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)35728)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)9232)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)10384)])), 72);
    for (int ax1_0_3_2 = 0; ax1_0_3_2 < 4; ++ax1_0_3_2) {
      for (int ax2_0_3_2 = 0; ax2_0_3_2 < 2; ++ax2_0_3_2) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 2) + ax2_0_3_2)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 & 1) * 4) + ax1_0_3_2)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 & 1) * 2) + ax2_0_3_2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 2) + ax2_0_3_2)]);
      }
    }
  }
  for (int ax1_0_3_3 = 0; ax1_0_3_3 < 4; ++ax1_0_3_3) {
    for (int ax2_0_3_3 = 0; ax2_0_3_3 < 2; ++ax2_0_3_3) {
      nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 2) + ax2_0_3_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_3 + 4)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_3 + 2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 2) + ax2_0_3_3)]);
    }
  }
  __syncthreads();
  for (int ax0_0 = 0; ax0_0 < 4; ++ax0_0) {
    for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0) {
      nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)8192) + (((int64_t)ax0_0) * (int64_t)2048)) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)32)) + (((int64_t)ax1_0) * (int64_t)16)) + (int64_t)18432)])), var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax0_0 * 2) + ax1_0)], 128, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int64_t ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < (int64_t)8; ++ax0_ax1_fused_0) {
    if (((((((int64_t)blockIdx.x) * (int64_t)192) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)64)) + (ax0_ax1_fused_0 * (int64_t)8)) + (((int64_t)threadIdx.x) >> (int64_t)2)) < n) {
      uint4 __1;
        uint4 v_ = *(uint4*)(((half*)buf_dyn_shmem) + (((((((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)8192) + (ax0_ax1_fused_0 * (int64_t)1024)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)) + (int64_t)18432));
        uint4 v__1 = *(uint4*)(lv4 + (((((int64_t)blockIdx.y) * (int64_t)128) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(var_T_add_intermediate + (((((((((int64_t)blockIdx.x) * (int64_t)491520) + ((((int64_t)threadIdx.y) % (int64_t)3) * (int64_t)163840)) + (ax0_ax1_fused_0 * (int64_t)20480)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)2560)) + (((int64_t)blockIdx.y) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)3) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(384) gemm_n_3329_to_3584__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[8];
  uint4 lv9_reindex_pad_shared_dyn_local[4];
  uint4 lv3_reindex_shared_dyn_local[3];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> lv9_reindex_pad_shared_dyn_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> lv3_reindex_shared_dyn_wmma_matrix_b[8];
  for (int ax1_0_3_init = 0; ax1_0_3_init < 2; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 4; ++ax2_0_3_init) {
      nvcuda::wmma::fill_fragment(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_init * 4) + ax2_0_3_init)], 0.000000e+00f);
    }
  }
  for (int64_t ax1_ax2_fused_0_0_0 = 0; ax1_ax2_fused_0_0_0 < (int64_t)4; ++ax1_ax2_fused_0_0_0) {
    uint4 condval;
    if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0 * (int64_t)48)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
      condval = *(uint4*)(lv9 + (((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0 * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
    }
    lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0] = condval;
  }
  for (int ax1_ax2_fused_0_0_0_1 = 0; ax1_ax2_fused_0_0_0_1 < 3; ++ax1_ax2_fused_0_0_0_1) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
      lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_1] = *(uint4*)(lv3 + (((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    }
  }
  for (int ax1_ax2_fused_0_0_0_2 = 0; ax1_ax2_fused_0_0_0_2 < 4; ++ax1_ax2_fused_0_0_0_2) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((int64_t)ax1_ax2_fused_0_0_0_2) * (int64_t)3456) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_2];
  }
  for (int ax1_ax2_fused_0_0_0_3 = 0; ax1_ax2_fused_0_0_0_3 < 3; ++ax1_ax2_fused_0_0_0_3) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)3456) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_3];
    }
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (int64_t)18432)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (int64_t)19584)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (int64_t)1152)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (int64_t)2304)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (int64_t)3456)])), 72);
  for (int64_t ax3_0_0 = 0; ax3_0_0 < (int64_t)159; ++ax3_0_0) {
    for (int64_t ax1_ax2_fused_0_0_0_4 = 0; ax1_ax2_fused_0_0_0_4 < (int64_t)4; ++ax1_ax2_fused_0_0_0_4) {
      uint4 condval_1;
      if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0_4 * (int64_t)48)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
        condval_1 = *(uint4*)(lv9 + (((((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0_4 * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      } else {
        condval_1 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
      lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_4] = condval_1;
    }
    for (int ax1_ax2_fused_0_0_0_5 = 0; ax1_ax2_fused_0_0_0_5 < 3; ++ax1_ax2_fused_0_0_0_5) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
        lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_5] = *(uint4*)(lv3 + (((((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      }
    }
    for (int ax3_0_1 = 0; ax3_0_1 < 3; ++ax3_0_1) {
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)18448)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)19600)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)16)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)1168)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)2320)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)3472)])), 72);
      for (int ax1_0_3 = 0; ax1_0_3 < 2; ++ax1_0_3) {
        for (int ax2_0_3 = 0; ax2_0_3 < 4; ++ax2_0_3) {
          nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 4) + ax2_0_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 & 1) * 2) + ax1_0_3)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 & 1) * 4) + ax2_0_3)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 4) + ax2_0_3)]);
        }
      }
    }
    __syncthreads();
    for (int ax1_ax2_fused_0_0_0_6 = 0; ax1_ax2_fused_0_0_0_6 < 4; ++ax1_ax2_fused_0_0_0_6) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + (((int64_t)ax1_ax2_fused_0_0_0_6) * (int64_t)3456)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_6];
    }
    for (int ax1_ax2_fused_0_0_0_7 = 0; ax1_ax2_fused_0_0_0_7 < 3; ++ax1_ax2_fused_0_0_0_7) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
        *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + (((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)3456)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_7];
      }
    }
    __syncthreads();
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (int64_t)18432)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (int64_t)19584)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608))])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (int64_t)1152)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (int64_t)2304)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (int64_t)3456)])), 72);
    for (int ax1_0_3_1 = 0; ax1_0_3_1 < 2; ++ax1_0_3_1) {
      for (int ax2_0_3_1 = 0; ax2_0_3_1 < 4; ++ax2_0_3_1) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 4) + ax2_0_3_1)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_1 + 2)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_1 + 4)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 4) + ax2_0_3_1)]);
      }
    }
  }
  for (int ax3_0_1_1 = 0; ax3_0_1_1 < 3; ++ax3_0_1_1) {
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)32272)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)33424)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)9232)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)10384)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)11536)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)12688)])), 72);
    for (int ax1_0_3_2 = 0; ax1_0_3_2 < 2; ++ax1_0_3_2) {
      for (int ax2_0_3_2 = 0; ax2_0_3_2 < 4; ++ax2_0_3_2) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 4) + ax2_0_3_2)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 & 1) * 2) + ax1_0_3_2)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 & 1) * 4) + ax2_0_3_2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 4) + ax2_0_3_2)]);
      }
    }
  }
  for (int ax1_0_3_3 = 0; ax1_0_3_3 < 2; ++ax1_0_3_3) {
    for (int ax2_0_3_3 = 0; ax2_0_3_3 < 4; ++ax2_0_3_3) {
      nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 4) + ax2_0_3_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_3 + 2)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_3 + 4)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 4) + ax2_0_3_3)]);
    }
  }
  __syncthreads();
  for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {
    for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
      nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)4096) + (((int64_t)ax0_0) * (int64_t)2048)) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + (((int64_t)ax1_0) * (int64_t)16)) + (int64_t)18432)])), var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax0_0 * 4) + ax1_0)], 128, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int64_t ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < (int64_t)8; ++ax0_ax1_fused_0) {
    if (((((((int64_t)blockIdx.x) * (int64_t)192) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)32)) + (ax0_ax1_fused_0 * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n) {
      uint4 __1;
        uint4 v_ = *(uint4*)(((half*)buf_dyn_shmem) + (((((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)4096) + (ax0_ax1_fused_0 * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432));
        uint4 v__1 = *(uint4*)(lv4 + (((((int64_t)blockIdx.y) * (int64_t)128) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(var_T_add_intermediate + (((((((((int64_t)blockIdx.x) * (int64_t)491520) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)81920)) + (ax0_ax1_fused_0 * (int64_t)10240)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)2560)) + (((int64_t)blockIdx.y) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(384) gemm_n_3585_to_3840__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[8];
  uint4 lv9_reindex_pad_shared_dyn_local[4];
  uint4 lv3_reindex_shared_dyn_local[3];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> lv9_reindex_pad_shared_dyn_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> lv3_reindex_shared_dyn_wmma_matrix_b[8];
  for (int ax1_0_3_init = 0; ax1_0_3_init < 2; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 4; ++ax2_0_3_init) {
      nvcuda::wmma::fill_fragment(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_init * 4) + ax2_0_3_init)], 0.000000e+00f);
    }
  }
  for (int64_t ax1_ax2_fused_0_0_0 = 0; ax1_ax2_fused_0_0_0 < (int64_t)4; ++ax1_ax2_fused_0_0_0) {
    uint4 condval;
    if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0 * (int64_t)48)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
      condval = *(uint4*)(lv9 + (((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0 * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
    }
    lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0] = condval;
  }
  for (int ax1_ax2_fused_0_0_0_1 = 0; ax1_ax2_fused_0_0_0_1 < 3; ++ax1_ax2_fused_0_0_0_1) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
      lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_1] = *(uint4*)(lv3 + (((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    }
  }
  for (int ax1_ax2_fused_0_0_0_2 = 0; ax1_ax2_fused_0_0_0_2 < 4; ++ax1_ax2_fused_0_0_0_2) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((int64_t)ax1_ax2_fused_0_0_0_2) * (int64_t)3456) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_2];
  }
  for (int ax1_ax2_fused_0_0_0_3 = 0; ax1_ax2_fused_0_0_0_3 < 3; ++ax1_ax2_fused_0_0_0_3) {
    if (((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)3456) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_3];
    }
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (int64_t)18432)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (int64_t)19584)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (int64_t)1152)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (int64_t)2304)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (int64_t)3456)])), 72);
  for (int64_t ax3_0_0 = 0; ax3_0_0 < (int64_t)159; ++ax3_0_0) {
    for (int64_t ax1_ax2_fused_0_0_0_4 = 0; ax1_ax2_fused_0_0_0_4 < (int64_t)4; ++ax1_ax2_fused_0_0_0_4) {
      uint4 condval_1;
      if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0_4 * (int64_t)48)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
        condval_1 = *(uint4*)(lv9 + (((((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0_4 * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      } else {
        condval_1 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
      lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_4] = condval_1;
    }
    for (int ax1_ax2_fused_0_0_0_5 = 0; ax1_ax2_fused_0_0_0_5 < 3; ++ax1_ax2_fused_0_0_0_5) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
        lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_5] = *(uint4*)(lv3 + (((((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)491520)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      }
    }
    for (int ax3_0_1 = 0; ax3_0_1 < 3; ++ax3_0_1) {
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)18448)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)19600)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)16)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)1168)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)2320)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)3472)])), 72);
      for (int ax1_0_3 = 0; ax1_0_3 < 2; ++ax1_0_3) {
        for (int ax2_0_3 = 0; ax2_0_3 < 4; ++ax2_0_3) {
          nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 4) + ax2_0_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 & 1) * 2) + ax1_0_3)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 & 1) * 4) + ax2_0_3)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 4) + ax2_0_3)]);
        }
      }
    }
    __syncthreads();
    for (int ax1_ax2_fused_0_0_0_6 = 0; ax1_ax2_fused_0_0_0_6 < 4; ++ax1_ax2_fused_0_0_0_6) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + (((int64_t)ax1_ax2_fused_0_0_0_6) * (int64_t)3456)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_6];
    }
    for (int ax1_ax2_fused_0_0_0_7 = 0; ax1_ax2_fused_0_0_0_7 < 3; ++ax1_ax2_fused_0_0_0_7) {
      if (((((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)3) + (((int64_t)threadIdx.y) >> (int64_t)2)) < (int64_t)8) {
        *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + (((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)3456)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_7];
      }
    }
    __syncthreads();
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (int64_t)18432)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304)) + (int64_t)19584)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608))])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (int64_t)1152)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (int64_t)2304)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608)) + (int64_t)3456)])), 72);
    for (int ax1_0_3_1 = 0; ax1_0_3_1 < 2; ++ax1_0_3_1) {
      for (int ax2_0_3_1 = 0; ax2_0_3_1 < 4; ++ax2_0_3_1) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 4) + ax2_0_3_1)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_1 + 2)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_1 + 4)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 4) + ax2_0_3_1)]);
      }
    }
  }
  for (int ax3_0_1_1 = 0; ax3_0_1_1 < 3; ++ax3_0_1_1) {
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)32272)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)33424)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 + 1) & 1) * 4)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)9232)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)10384)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)11536)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 4) + 3)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)4608) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)12688)])), 72);
    for (int ax1_0_3_2 = 0; ax1_0_3_2 < 2; ++ax1_0_3_2) {
      for (int ax2_0_3_2 = 0; ax2_0_3_2 < 4; ++ax2_0_3_2) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 4) + ax2_0_3_2)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 & 1) * 2) + ax1_0_3_2)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 & 1) * 4) + ax2_0_3_2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 4) + ax2_0_3_2)]);
      }
    }
  }
  for (int ax1_0_3_3 = 0; ax1_0_3_3 < 2; ++ax1_0_3_3) {
    for (int ax2_0_3_3 = 0; ax2_0_3_3 < 4; ++ax2_0_3_3) {
      nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 4) + ax2_0_3_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_3 + 2)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_3 + 4)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 4) + ax2_0_3_3)]);
    }
  }
  __syncthreads();
  for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {
    for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
      nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)4096) + (((int64_t)ax0_0) * (int64_t)2048)) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + (((int64_t)ax1_0) * (int64_t)16)) + (int64_t)18432)])), var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax0_0 * 4) + ax1_0)], 128, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int64_t ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < (int64_t)8; ++ax0_ax1_fused_0) {
    if (((((((int64_t)blockIdx.x) * (int64_t)192) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)32)) + (ax0_ax1_fused_0 * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n) {
      uint4 __1;
        uint4 v_ = *(uint4*)(((half*)buf_dyn_shmem) + (((((((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)4096) + (ax0_ax1_fused_0 * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432));
        uint4 v__1 = *(uint4*)(lv4 + (((((int64_t)blockIdx.y) * (int64_t)128) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(var_T_add_intermediate + (((((((((int64_t)blockIdx.x) * (int64_t)491520) + ((((int64_t)threadIdx.y) % (int64_t)6) * (int64_t)81920)) + (ax0_ax1_fused_0 * (int64_t)10240)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)2560)) + (((int64_t)blockIdx.y) * (int64_t)128)) + ((((int64_t)threadIdx.y) / (int64_t)6) * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = __1;
    }
  }
}

extern "C" __global__ void __launch_bounds__(512) gemm_n_3841_to_4096__kernel(half* __restrict__ lv3, half* __restrict__ lv4, half* __restrict__ lv9, half* __restrict__ var_T_add_intermediate, int64_t n) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[6];
  uint4 lv9_reindex_pad_shared_dyn_local[3];
  uint4 lv3_reindex_shared_dyn_local[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> lv9_reindex_pad_shared_dyn_wmma_matrix_a[6];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> lv3_reindex_shared_dyn_wmma_matrix_b[4];
  for (int ax1_0_3_init = 0; ax1_0_3_init < 3; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 2; ++ax2_0_3_init) {
      nvcuda::wmma::fill_fragment(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_init * 2) + ax2_0_3_init)], 0.000000e+00f);
    }
  }
  for (int64_t ax1_ax2_fused_0_0_0 = 0; ax1_ax2_fused_0_0_0 < (int64_t)3; ++ax1_ax2_fused_0_0_0) {
    uint4 condval;
    if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0 * (int64_t)64)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
      condval = *(uint4*)(lv9 + (((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0 * (int64_t)655360)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
    }
    lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0] = condval;
  }
  for (int ax1_ax2_fused_0_0_0_1 = 0; ax1_ax2_fused_0_0_0_1 < 2; ++ax1_ax2_fused_0_0_0_1) {
    lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_1] = *(uint4*)(lv3 + (((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_1) * (int64_t)655360)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)));
  }
  for (int ax1_ax2_fused_0_0_0_2 = 0; ax1_ax2_fused_0_0_0_2 < 3; ++ax1_ax2_fused_0_0_0_2) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((((((int64_t)ax1_ax2_fused_0_0_0_2) * (int64_t)4608) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_2];
  }
  for (int ax1_ax2_fused_0_0_0_3 = 0; ax1_ax2_fused_0_0_0_3 < 2; ++ax1_ax2_fused_0_0_0_3) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int64_t)ax1_ax2_fused_0_0_0_3) * (int64_t)4608) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_3];
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456) + (int64_t)18432)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456) + (int64_t)19584)])), 72);
  nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456) + (int64_t)20736)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)2304)])), 72);
  nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)2304) + (int64_t)1152)])), 72);
  for (int64_t ax3_0_0 = 0; ax3_0_0 < (int64_t)159; ++ax3_0_0) {
    for (int64_t ax1_ax2_fused_0_0_0_4 = 0; ax1_ax2_fused_0_0_0_4 < (int64_t)3; ++ax1_ax2_fused_0_0_0_4) {
      uint4 condval_1;
      if ((((((((int64_t)blockIdx.x) * (int64_t)192) + (ax1_ax2_fused_0_0_0_4 * (int64_t)64)) + (((int64_t)threadIdx.y) * (int64_t)4)) + (((int64_t)threadIdx.x) >> (int64_t)3)) < n)) {
        condval_1 = *(uint4*)(lv9 + (((((((((int64_t)blockIdx.x) * (int64_t)1966080) + (ax1_ax2_fused_0_0_0_4 * (int64_t)655360)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      } else {
        condval_1 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
      lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_4] = condval_1;
    }
    for (int ax1_ax2_fused_0_0_0_5 = 0; ax1_ax2_fused_0_0_0_5 < 2; ++ax1_ax2_fused_0_0_0_5) {
      lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_5] = *(uint4*)(lv3 + (((((((((int64_t)blockIdx.y) * (int64_t)1310720) + (((int64_t)ax1_ax2_fused_0_0_0_5) * (int64_t)655360)) + (((int64_t)threadIdx.y) * (int64_t)40960)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)10240)) + (ax3_0_0 * (int64_t)64)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
    }
    for (int ax3_0_1 = 0; ax3_0_1 < 3; ++ax3_0_1) {
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 + 1) & 1) * 3)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)18448)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 3) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)19600)])), 72);
      nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1 + 1) & 1) * 3) + 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)20752)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)16)])), 72);
      nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)2304)) + (((int64_t)ax3_0_1) * (int64_t)16)) + (int64_t)1168)])), 72);
      for (int ax1_0_3 = 0; ax1_0_3 < 3; ++ax1_0_3) {
        for (int ax2_0_3 = 0; ax2_0_3 < 2; ++ax2_0_3) {
          nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 2) + ax2_0_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1 & 1) * 3) + ax1_0_3)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1 & 1) * 2) + ax2_0_3)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3 * 2) + ax2_0_3)]);
        }
      }
    }
    __syncthreads();
    for (int ax1_ax2_fused_0_0_0_6 = 0; ax1_ax2_fused_0_0_0_6 < 3; ++ax1_ax2_fused_0_0_0_6) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + (((int64_t)ax1_ax2_fused_0_0_0_6) * (int64_t)4608)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8)) + (int64_t)18432)) = lv9_reindex_pad_shared_dyn_local[ax1_ax2_fused_0_0_0_6];
    }
    for (int ax1_ax2_fused_0_0_0_7 = 0; ax1_ax2_fused_0_0_0_7 < 2; ++ax1_ax2_fused_0_0_0_7) {
      *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + (((int64_t)ax1_ax2_fused_0_0_0_7) * (int64_t)4608)) + (((int64_t)threadIdx.y) * (int64_t)288)) + ((((int64_t)threadIdx.x) >> (int64_t)3) * (int64_t)72)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)8))) = lv3_reindex_shared_dyn_local[ax1_ax2_fused_0_0_0_7];
    }
    __syncthreads();
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456)) + (int64_t)18432)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456)) + (int64_t)19584)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)13824) + ((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456)) + (int64_t)20736)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)2304))])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((ax3_0_0 + (int64_t)1) & (int64_t)1) * (int64_t)9216) + ((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)2304)) + (int64_t)1152)])), 72);
    for (int ax1_0_3_1 = 0; ax1_0_3_1 < 3; ++ax1_0_3_1) {
      for (int ax2_0_3_1 = 0; ax2_0_3_1 < 2; ++ax2_0_3_1) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 2) + ax2_0_3_1)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_1 + 3)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_1 + 2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_1 * 2) + ax2_0_3_1)]);
      }
    }
  }
  for (int ax3_0_1_1 = 0; ax3_0_1_1 < 3; ++ax3_0_1_1) {
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 + 1) & 1) * 3)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)32272)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 3) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)33424)])), 72);
    nvcuda::wmma::load_matrix_sync(lv9_reindex_pad_shared_dyn_wmma_matrix_a[((((ax3_0_1_1 + 1) & 1) * 3) + 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)3456) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)34576)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 + 1) & 1) * 2)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)9232)])), 72);
    nvcuda::wmma::load_matrix_sync(lv3_reindex_shared_dyn_wmma_matrix_b[((((ax3_0_1_1 + 1) & 1) * 2) + 1)], (&(((half*)buf_dyn_shmem)[((((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)2304) + (((int64_t)ax3_0_1_1) * (int64_t)16)) + (int64_t)10384)])), 72);
    for (int ax1_0_3_2 = 0; ax1_0_3_2 < 3; ++ax1_0_3_2) {
      for (int ax2_0_3_2 = 0; ax2_0_3_2 < 2; ++ax2_0_3_2) {
        nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 2) + ax2_0_3_2)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(((ax3_0_1_1 & 1) * 3) + ax1_0_3_2)], lv3_reindex_shared_dyn_wmma_matrix_b[(((ax3_0_1_1 & 1) * 2) + ax2_0_3_2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_2 * 2) + ax2_0_3_2)]);
      }
    }
  }
  for (int ax1_0_3_3 = 0; ax1_0_3_3 < 3; ++ax1_0_3_3) {
    for (int ax2_0_3_3 = 0; ax2_0_3_3 < 2; ++ax2_0_3_3) {
      nvcuda::wmma::mma_sync(var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 2) + ax2_0_3_3)], lv9_reindex_pad_shared_dyn_wmma_matrix_a[(ax1_0_3_3 + 3)], lv3_reindex_shared_dyn_wmma_matrix_b[(ax2_0_3_3 + 2)], var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax1_0_3_3 * 2) + ax2_0_3_3)]);
    }
  }
  __syncthreads();
  for (int ax0_0 = 0; ax0_0 < 3; ++ax0_0) {
    for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0) {
      nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)6144) + (((int64_t)ax0_0) * (int64_t)2048)) + ((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)32)) + (((int64_t)ax1_0) * (int64_t)16)) + (int64_t)18432)])), var_NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[((ax0_0 * 2) + ax1_0)], 128, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int64_t ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < (int64_t)6; ++ax0_ax1_fused_0) {
    if (((((((int64_t)blockIdx.x) * (int64_t)192) + ((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)48)) + (ax0_ax1_fused_0 * (int64_t)8)) + (((int64_t)threadIdx.x) >> (int64_t)2)) < n) {
      uint4 __1;
        uint4 v_ = *(uint4*)(((half*)buf_dyn_shmem) + (((((((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)6144) + (ax0_ax1_fused_0 * (int64_t)1024)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)) + (int64_t)18432));
        uint4 v__1 = *(uint4*)(lv4 + (((((int64_t)blockIdx.y) * (int64_t)128) + ((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(v_.x)))->x+((half2*)(&(v__1.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(v_.x)))->y+((half2*)(&(v__1.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(v_.y)))->x+((half2*)(&(v__1.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(v_.y)))->y+((half2*)(&(v__1.y)))->y);
        ((half2*)(&(__1.z)))->x = (((half2*)(&(v_.z)))->x+((half2*)(&(v__1.z)))->x);
        ((half2*)(&(__1.z)))->y = (((half2*)(&(v_.z)))->y+((half2*)(&(v__1.z)))->y);
        ((half2*)(&(__1.w)))->x = (((half2*)(&(v_.w)))->x+((half2*)(&(v__1.w)))->x);
        ((half2*)(&(__1.w)))->y = (((half2*)(&(v_.w)))->y+((half2*)(&(v__1.w)))->y);
      *(uint4*)(var_T_add_intermediate + (((((((((int64_t)blockIdx.x) * (int64_t)491520) + ((((int64_t)threadIdx.y) & (int64_t)3) * (int64_t)122880)) + (ax0_ax1_fused_0 * (int64_t)20480)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)2560)) + (((int64_t)blockIdx.y) * (int64_t)128)) + ((((int64_t)threadIdx.y) >> (int64_t)2) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8))) = __1;
    }
  }
}

