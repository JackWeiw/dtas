#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#include <cuda_fp16.h>
__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}
#else

typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

#define TVM_FORCE_INLINE inline __attribute__((always_inline))
#define TVM_XINLINE TVM_FORCE_INLINE __device__ __host__
#define TVM_ALIGNED(x) __attribute__ ((aligned(x)))
#define TVM_HALF_OPERATOR(RTYPE, OP)                              \
  TVM_XINLINE RTYPE operator OP (half a, half b) {                \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (half a, T b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (T a, half b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }

#define TVM_HALF_ASSIGNOP(AOP, OP)                                \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const T& a) {                    \
    return *this = half(float(*this) OP float(a));                \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const volatile T& a) volatile {  \
    return *this = half(float(*this) OP float(a));                \
  }

class TVM_ALIGNED(2) half {
 public:
  uint16_t half_;

  static TVM_XINLINE half Binary(uint16_t value) {
    half res;
    res.half_ = value;
    return res;
  }

  TVM_XINLINE half() {}

  TVM_XINLINE half(const float& value) { constructor(value); }
  TVM_XINLINE explicit half(const double& value) { constructor(value); }
  TVM_XINLINE explicit half(const int8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const int32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const long long& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint64_t& value) { constructor(value); }

  TVM_XINLINE operator float() const {                          \
    return float(half2float(half_));                            \
  }                                                             \
  TVM_XINLINE operator float() const volatile {                 \
    return float(half2float(half_));                            \
  }


  TVM_HALF_ASSIGNOP(+=, +)
  TVM_HALF_ASSIGNOP(-=, -)
  TVM_HALF_ASSIGNOP(*=, *)
  TVM_HALF_ASSIGNOP(/=, /)

  TVM_XINLINE half operator+() {
    return *this;
  }

  TVM_XINLINE half operator-() {
    return half(-float(*this));
  }

  TVM_XINLINE half operator=(const half& a) {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) {
    return *this = half(a);
  }

  TVM_XINLINE half operator=(const half& a) volatile {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) volatile {
    return *this = half(a);
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static int const fp16FractionBits = 10;
  static int const fp32FractionBits = 23;
  static int32_t const fp32FractionMask = ~(~0u << fp32FractionBits);   // == 0x7fffff
  static int32_t const fp32HiddenBit = 1 << fp32FractionBits;   // == 0x800000
  static int const shift = fp32FractionBits - fp16FractionBits;   // == 13
  static int const shiftSign = 16;
  static int32_t const expAdjust = 127 - 15;   // exp32-127 = exp16-15, so exp16 = exp32 - (127-15)

  static int32_t const infN = 0x7F800000;   // flt32 infinity
  static int32_t const maxN = 0x477FFFFF;   // max flt32 that's a flt16 normal after >> by shift
  static int32_t const minN = 0x38800000;   // min flt16 normal as a flt32
  static int32_t const maxZ = 0x33000000;   // max fp32 number that's still rounded to zero in fp16
  static int32_t const signN = 0x80000000;  // flt32 sign bit

  static int32_t const infC = infN >> shift;
  static int32_t const nanN = (infC + 1) << shift;   // minimum flt16 nan as a flt32
  static int32_t const maxC = maxN >> shift;
  static int32_t const minC = minN >> shift;
  static int32_t const signC = signN >> shiftSign;  // flt16 sign bit

  static int32_t const mulN = 0x52000000;  // (1 << 23) / minN
  static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))

  static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted
  static int32_t const norC = 0x00400;  // min flt32 normal down shifted

  static int32_t const maxD = infC - maxC - 1;
  static int32_t const minD = minC - subC - 1;

  TVM_XINLINE uint16_t float2half(const float& value) const {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  // Same as above routine, except for addition of volatile keyword
  TVM_XINLINE uint16_t float2half(
    const volatile float& value) const volatile {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  TVM_XINLINE float half2float(const uint16_t& value) const {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  TVM_XINLINE float half2float(
    const volatile uint16_t& value) const volatile {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  template<typename T>
  TVM_XINLINE void constructor(const T& value) {
    half_ = float2half(float(value));
  }
};

TVM_HALF_OPERATOR(half, +)
TVM_HALF_OPERATOR(half, -)
TVM_HALF_OPERATOR(half, *)
TVM_HALF_OPERATOR(half, /)
TVM_HALF_OPERATOR(bool, >)
TVM_HALF_OPERATOR(bool, <)
TVM_HALF_OPERATOR(bool, >=)
TVM_HALF_OPERATOR(bool, <=)

TVM_XINLINE half __float2half_rn(const float a) {
  return half(a);
}
#endif


// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x, half y) {   \
  float tmp_x = __half2float(x);                                          \
  float tmp_y = __half2float(y);                                          \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2half(result);                                            \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x) {          \
  float tmp_x = __half2float(x);                                         \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2half(result);                                           \
}

// Some fp16 math functions are not supported in cuda_fp16.h,
// so we define them here to make sure the generated CUDA code
// is valid.
#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ >= 530)
CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)
#else
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hexp, exp)
#endif
#endif

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#define __shfl_sync(mask, var, lane, width) \
        __shfl((var), (lane), (width))

#define __shfl_down_sync(mask, var, offset, width) \
        __shfl_down((var), (offset), (width))

#define __shfl_up_sync(mask, var, offset, width) \
        __shfl_up((var), (offset), (width))
#endif


#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
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
extern "C" __global__ void __launch_bounds__(256) main_kernel(float* __restrict__ lv38, half* __restrict__ var_compute_intermediate, int64_t m, int64_t n);
extern "C" __global__ void __launch_bounds__(256) main_kernel(float* __restrict__ lv38, half* __restrict__ var_compute_intermediate, int64_t m, int64_t n) {
  extern __shared__ float lv38_shared_dyn[];
  float in_thread_T_softmax_maxelem_shared[1];
  __shared__ float red_result[1];
  __shared__ float T_softmax_maxelem_shared[1];
  float in_thread_T_softmax_expsum_shared[1];
  __shared__ float red_result_1[1];
  __shared__ float T_softmax_expsum_shared[1];
  for (int64_t ax3_fused_0 = 0; ax3_fused_0 < (((max((int64_t)1, (((m + (int64_t)255) >> (int64_t)8) * (int64_t)256)) + (int64_t)1023) - min((int64_t)0, ((((m + (int64_t)255) >> (int64_t)8) * (int64_t)256) - (int64_t)1))) >> (int64_t)10); ++ax3_fused_0) {
    for (int64_t ax3_fused_2_s = 0; ax3_fused_2_s < (int64_t)4; ++ax3_fused_2_s) {
      if ((((int64_t)0 <= ((((ax3_fused_0 * (int64_t)1024) + (((int64_t)threadIdx.x) * (int64_t)4)) + min((int64_t)0, ((((m + (int64_t)255) >> (int64_t)8) * (int64_t)256) - (int64_t)1))) + ax3_fused_2_s)) && (((((ax3_fused_0 * (int64_t)1024) + (((int64_t)threadIdx.x) * (int64_t)4)) + min((int64_t)0, ((((m + (int64_t)255) >> (int64_t)8) * (int64_t)256) - (int64_t)1))) + ax3_fused_2_s) < m)) && ((((ax3_fused_0 * (int64_t)1024) + (((int64_t)threadIdx.x) * (int64_t)4)) + ax3_fused_2_s) < (max((int64_t)1, (((m + (int64_t)255) >> (int64_t)8) * (int64_t)256)) - min((int64_t)0, ((((m + (int64_t)255) >> (int64_t)8) * (int64_t)256) - (int64_t)1))))) {
        lv38_shared_dyn[(((((ax3_fused_0 * (int64_t)1024) + (((int64_t)threadIdx.x) * (int64_t)4)) + (min((int64_t)0, ((((m + (int64_t)255) >> (int64_t)8) * (int64_t)256) - (int64_t)1)) * (int64_t)4)) + (max((int64_t)0, ((int64_t)1 - (((m + (int64_t)255) >> (int64_t)8) * (int64_t)256))) * (int64_t)3)) + ax3_fused_2_s)] = lv38[((((((ax3_fused_0 * (int64_t)1024) + (((int64_t)threadIdx.x) * (int64_t)4)) + (min((int64_t)0, ((((m + (int64_t)255) >> (int64_t)8) * (int64_t)256) - (int64_t)1)) * (int64_t)4)) + (max((int64_t)0, ((int64_t)1 - (((m + (int64_t)255) >> (int64_t)8) * (int64_t)256))) * (int64_t)3)) + (((int64_t)blockIdx.x) * m)) + ax3_fused_2_s)];
      }
    }
  }
  in_thread_T_softmax_maxelem_shared[0] = -3.402823e+38f;
  __syncthreads();
  for (int64_t ax2_fused_0 = 0; ax2_fused_0 < ((m + (int64_t)255) >> (int64_t)8); ++ax2_fused_0) {
    if (((ax2_fused_0 * (int64_t)256) + ((int64_t)threadIdx.x)) < m) {
      in_thread_T_softmax_maxelem_shared[0] = max(in_thread_T_softmax_maxelem_shared[0], lv38_shared_dyn[((ax2_fused_0 * (int64_t)256) + ((int64_t)threadIdx.x))]);
    }
  }
  float red_buf0[1];
  uint mask[1];
  float t0[1];
  float red_buf0_1[1];
  uint mask_1[1];
  float t0_1[1];
  __shared__ float red_buf_staging[8];
  red_buf0_1[0] = in_thread_T_softmax_maxelem_shared[0];
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
    T_softmax_maxelem_shared[0] = ((volatile float*)red_result)[0];
  }
  in_thread_T_softmax_expsum_shared[0] = 0.000000e+00f;
  __syncthreads();
  for (int64_t ax2_fused_0_1 = 0; ax2_fused_0_1 < ((m + (int64_t)255) >> (int64_t)8); ++ax2_fused_0_1) {
    if (((ax2_fused_0_1 * (int64_t)256) + ((int64_t)threadIdx.x)) < m) {
      in_thread_T_softmax_expsum_shared[0] = (in_thread_T_softmax_expsum_shared[0] + __expf((lv38_shared_dyn[((ax2_fused_0_1 * (int64_t)256) + ((int64_t)threadIdx.x))] - T_softmax_maxelem_shared[0])));
    }
  }
  float red_buf0_2[1];
  uint mask_2[1];
  float t0_2[1];
  float red_buf0_3[1];
  uint mask_3[1];
  float t0_3[1];
  __shared__ float red_buf_staging_1[8];
  red_buf0_3[0] = in_thread_T_softmax_expsum_shared[0];
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
    T_softmax_expsum_shared[0] = ((volatile float*)red_result_1)[0];
  }
  __syncthreads();
  for (int64_t ax2_0 = 0; ax2_0 < ((m + (int64_t)255) >> (int64_t)8); ++ax2_0) {
    if (((ax2_0 * (int64_t)256) + ((int64_t)threadIdx.x)) < m) {
      var_compute_intermediate[(((ax2_0 * (int64_t)256) + (((int64_t)blockIdx.x) * m)) + ((int64_t)threadIdx.x))] = ((half)(__expf((lv38_shared_dyn[((ax2_0 * (int64_t)256) + ((int64_t)threadIdx.x))] - T_softmax_maxelem_shared[0])) / T_softmax_expsum_shared[0]));
    }
  }
}

