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
inline __device__ float warpReduceSum(float *val, int thread_group = 32) {
#pragma unroll
  for (int i = 0; i < NUM; ++i) {
#pragma unroll
    for (int lane_mask = thread_group / 2; lane_mask > 0; lane_mask >>= 1) {
      val[i] += __shfl_xor_sync(0xFFFFFFFF, val[i], lane_mask, thread_group);
    }
  }
  return 0.0f;
}

// block_size.x=32, block_size.y=4 (32, 4)
// grid_size.x = m / block_size.y (m / block_size.y,)
template <int block_size_y>
__global__ void vec_softmax_kernel(const half4 *input, half4 *output, int m,
                                   int n) {
  const int cols_per_thread = 4;
  int m_idx = blockIdx.x * blockDim.y + threadIdx.y;
  if (m_idx >= m) {
    return;
  }

  int row_offset = m_idx * n / cols_per_thread;
  const half4 *row_x = input + row_offset;
  half4 *row_y = output + row_offset;

  __half local_max[1] = {__float2half(-INFINITY)};

  for (int i = 0; i < n / cols_per_thread; i += blockDim.x) {
    int col_idx = i + threadIdx.x;
    if (col_idx * cols_per_thread < n) {
      half4 x = row_x[col_idx];
      local_max[0] =
          __hmax(local_max[0], __hmax(x.x, __hmax(x.y, __hmax(x.z, x.w))));
    }
  }
  warpReduceMax<1>(local_max);

  __shared__ float buf[block_size_y][2];
  if (threadIdx.x == 0) {
    buf[threadIdx.y][0] = local_max[0];
  }
  __syncthreads();

  float local_sum[1] = {0.0f};
  for (int i = 0; i < n / cols_per_thread; i += blockDim.x) {
    int col_idx = i + threadIdx.x;
    if (col_idx * cols_per_thread < n) {
      float max_val = buf[threadIdx.y][0];
      half4 x = row_x[col_idx];
      local_sum[0] += expf(__half2float(x.x) - max_val) +
                      expf(__half2float(x.y) - max_val) +
                      expf(__half2float(x.z) - max_val) +
                      expf(__half2float(x.w) - max_val);
    }
  }
  warpReduceSum<1>(local_sum);

  if (threadIdx.x == 0) {
    buf[threadIdx.y][1] = local_sum[0];
  }
  __syncthreads();

  for (int i = 0; i < n / cols_per_thread; i += blockDim.x) {
    int col_idx = i + threadIdx.x;
    if (col_idx * cols_per_thread < n) {
      float max_val = buf[threadIdx.y][0];
      float sum_val = buf[threadIdx.y][1];
      half4 x = row_x[col_idx];
      row_y[col_idx] = {
          __float2half(expf(__half2float(x.x) - max_val) / sum_val),
          __float2half(expf(__half2float(x.y) - max_val) / sum_val),
          __float2half(expf(__half2float(x.z) - max_val) / sum_val),
          __float2half(expf(__half2float(x.w) - max_val) / sum_val),
      };
    }
  }
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
  const int block_size_y = 4;
  dim3 block_size(32, block_size_y);
  dim3 grid_size((m + block_size.y - 1) / block_size.y);

  vec_softmax_kernel<block_size_y>
      <<<grid_size, block_size>>>(input_d, output_d, m, n);

  cudaMemcpy(output_h, output_d, mem_size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < m * n; ++i) {
    printf("%f ", __half2float(output_h[i]));
  }
  printf("\n");

  return 0;
}
