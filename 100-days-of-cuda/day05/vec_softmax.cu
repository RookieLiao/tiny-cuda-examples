#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

template <int NUM>
inline __device__ float warpReduceMax(__half *val, int thread_group = 32) {
#pragma unroll
  for (int i = 0; i < NUM; ++i) {
#pragma unroll
    for (int lane_mask = thread_group / 2; lane_mask > 0; lane_mask >>= 1) {
      val[i] = __hmax(
          val[i], __shfl_xor_sync(0xFFFFFFFF, val[i], lane_mask, thread_group));
    }
  }
  return 0.0f;
}

template <int NUM>
inline __device__ float warpReduceSum(__half *val, int thread_group = 32) {
#pragma unroll
  for (int i = 0; i < NUM; ++i) {
#pragma unroll
    for (int lane_mask = thread_group / 2; lane_mask > 0; lane_mask >>= 1) {
      val[i] += __shfl_xor_sync(0xFFFFFFFF, val[i], lane_mask, thread_group);
    }
  }
}

struct __align__(8) half4 {
  __half x = 0;
  __half y = 0;
  __half z = 0;
  __half w = 0;

  __device__ __host__ half4() {}

  __device__ __host__ half4(__half _x, __half _y, __half _z, __half _w) {
    x = _x;
    y = _y;
    z = _z;
    w = _w;
  }
};

__global__ void vec_softmax_kernel(const half4 *input, half4 *output, int m, int n) {
}

int main() {

  int m = 1;
  int n = 128;
  int cols_per_thread = 4;

  half *input = (half *)malloc(sizeof(half) * m * n);
  half *output = (half *)malloc(sizeof(half) * m * n);

  // initialize input with random values
  for (int i = 0; i < m * n; ++i) {
    float val = rand() / (float)RAND_MAX;
    input[i] = __float2half(val);
  }


  return 0;
}
