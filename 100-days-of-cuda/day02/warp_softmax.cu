#include <torch/extension.h>
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
#pragma unroll
  for (int lane_mask = thread_group / 2; lane_mask > 0; lane_mask >>= 1) {
    val[0] += __shfl_xor_sync(0xFFFFFFFF, val[0], lane_mask, thread_group);
  }
  return 0.0f;
}

template<int block_size_y>
__global__ void warp_softmax_kernel(const float *input, float *output, size_t m,
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

  __shared__ float smem_val[block_size_y][2]; // [0]: local_max, [1]: local_sum
  if (threadIdx.x == 0) {
    smem_val[threadIdx.y][0] = local_max[0];
  }
  __syncthreads();

  // second pass: compute denominator
  float local_sum[1] = {0.0f};
  for (int pack_id = 0; pack_id < num_packs; pack_id++) {
    const int col_idx = pack_id * blockDim.x + threadIdx.x;
    if (col_idx < n) {
      local_sum[0] += expf(row_x[col_idx] - smem_val[threadIdx.y][0]);
    }
  }
  warpReduceSum(local_sum);
  if (threadIdx.x == 0) {
    smem_val[threadIdx.y][1] = local_sum[0];
  }
  __syncthreads();

  // third pass: compute softmax
  for (int pack_id = 0; pack_id < num_packs; pack_id++) {
    const int col_idx = pack_id * blockDim.x + threadIdx.x;
    if (col_idx < n) {
      row_y[col_idx] = expf(row_x[col_idx] - smem_val[threadIdx.y][0]) / smem_val[threadIdx.y][1];
    }
  }
}

void warp_softmax_launcher(
  torch::Tensor input,
  torch::Tensor output
) {
  size_t m = input.size(0);
  size_t n = input.size(1);

  constexpr int block_size_y = 4;
  dim3 block_size(32, block_size_y);
  int num_blocks = (m + block_size_y - 1) / block_size_y;
  dim3 grid_size(num_blocks);

  warp_softmax_kernel<block_size_y><<< grid_size, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), m, n);
}
