#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

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

__global__ void vec_softmax_kernel(const half4 *input, half4 *output, int m,
                                   int n) {

  // int m_idx =
}

int main() {

  int m = 1;
  int n = 128;

  size_t mem_size = sizeof(half) * m * n;

  half *input_h = (half *)malloc(mem_size);
  half *output_h = (half *)malloc(mem_size);

  // initialize input with random values
  for (int i = 0; i < m * n; ++i) {
    float val = rand() / (float)RAND_MAX;
    input_h[i] = __float2half(val);
  }

  half4 *input_d;
  half4 *output_d;
  cudaMalloc(&input_d, mem_size);
  cudaMalloc(&output_d, mem_size);

  cudaMemcpy(input_d, input_h, mem_size, cudaMemcpyHostToDevice);

  // 128 threads per block, parallel 4 rows, 1 warp per row
  dim3 block_size(32, 4);
  dim3 grid_size((m + block_size.y - 1) / block_size.y);

  vec_softmax_kernel<<<grid_size, block_size>>>(input_d, output_d, m, n);

  cudaMemcpy(output_h, output_d, mem_size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < m * n; ++i) {
    printf("%f ", __half2float(output_h[i]));
  }
  printf("\n");

  return 0;
}
