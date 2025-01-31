#include <cuda_runtime.h>
#include <limits.h>
#include <stdio.h>

inline __device__ float warpReduceMax(float *val, int thread_group = 32) {
#pragma unroll
  for (int lane_mask = thread_group / 2; lane_mask > 0; lane_mask >>= 1) {
    val[0] = fmaxf(
        val[0], __shfl_xor_sync(0xFFFFFFFF, val[0], lane_mask, thread_group));
  }
  return 0.0f;
}

inline __device__ float warpReduceSum(float *val, int thread_group = 32) {
  float sum = 0.0f;
#pragma unroll
  for (int lane_mask = thread_group / 2; lane_mask > 0; lane_mask >>= 1) {
    sum += __shfl_xor_sync(0xFFFFFFFF, val[0], lane_mask, thread_group);
  }
  return sum;
}

__global__ void wrap_softmax_kernel(const float *input, float *output, size_t m,
                                    size_t n) {
  int m_idx = blockDim.y * blockIdx.x + threadIdx.y;
  if (m_idx >= m)
    return;

  int row_offset = m_idx * n;
  const float *row_x = input + row_offset;
  float *row_y = output + row_offset;
  const int num_packs = (n + blockDim.x - 1) / blockDim.x;

  // first pass: find max value
  float local_max[1] = {-INFINITY};
  for (int pack_id = 0; pack_id < num_packs; pack_id++) {
    const int col_idx = pack_id * blockDim.x + threadIdx.x;
    if (col_idx < n) {
      local_max[0] = fmaxf(local_max[0], row_x[col_idx]);
    }
  }
  warpReduceMax(local_max);

  // second pass: compute denominator
  float local_sum[1] = {0.0f};
  for (int pack_id = 0; pack_id < num_packs; pack_id++) {
    const int col_idx = pack_id * blockDim.x + threadIdx.x;
    if (col_idx < n) {
      local_sum[0] += expf(row_x[col_idx] - local_max[0]);
    }
  }
  warpReduceSum(local_sum);

  // third pass: compute softmax
  for (int pack_id = 0; pack_id < num_packs; pack_id++) {
    const int col_idx = pack_id * blockDim.x + threadIdx.x;
    if (col_idx < n) {
      row_y[col_idx] = expf(row_x[col_idx] - local_max[0]) / local_sum[0];
    }
  }
}

int main() {
  int M = 5;   // rows
  int N = 128; // cols

  int num_elements = N * M;
  float *input_h = (float *)malloc(num_elements * sizeof(float));
  float *output_h = (float *)malloc(num_elements * sizeof(float));

  // initialize input with random values
  for (int i = 0; i < num_elements; i++) {
    input_h[i] = rand() / (float)RAND_MAX;
  }

  float *input_d;
  cudaMalloc(&input_d, num_elements * sizeof(float));
  cudaMemcpy(input_d, input_h, num_elements * sizeof(float),
             cudaMemcpyHostToDevice);

  float *output_d;
  cudaMalloc(&output_d, num_elements * sizeof(float));

  dim3 block_size(32, 4);
  int num_blocks = (M + block_size.y - 1) / block_size.y;
  dim3 grid_size(num_blocks);
  wrap_softmax_kernel<<<grid_size, block_size>>>(input_d, output_d, M, N);

  cudaMemcpy(output_h, output_d, num_elements * sizeof(float),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < num_elements; i++) {
    printf("row: %d, col: %d, value: %f\n", i / N, i % N, output_h[i]);
  }
}
