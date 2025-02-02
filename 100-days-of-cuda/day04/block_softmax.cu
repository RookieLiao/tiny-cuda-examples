#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/extension.h>

inline __device__ float warpReduceMax(float *val, int thread_group = 32) {
#pragma unroll
  for (int lane_mask = thread_group / 2; lane_mask > 0; lane_mask >>= 1) {
    // perform warp shuffle using lane_id from 0 to 31 which is global_id %
    // warp_size
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

inline __device__ float blockReduceMax(float *val, int thread_group = 32) {
  __shared__ float shared_val[33];
  int lane = threadIdx.x & 0x1f; // threadIdx.x % warp_size
  int wid = threadIdx.x >> 5;    // threadIdx.x / warp_size

  warpReduceMax(val, thread_group);
  int tid = threadIdx.x;
  if (lane == 0) {
    shared_val[wid] = val[0];
  }
  __syncthreads();

  val[0] = tid < (blockDim.x / 32.f) ? shared_val[tid] : -INFINITY;

  // only perform on the first warp
  if (wid == 0) {
    warpReduceMax(val, 32);
  }
  return 0.0f;
}

inline __device__ float blockReduceSum(float *val, int thread_group = 32) {
  __shared__ float shared_val[33];
  int lane = threadIdx.x & 0x1f; // threadIdx.x % warp_size
  int wid = threadIdx.x >> 5;    // threadIdx.x / warp_size

  warpReduceSum(val, thread_group);
  int tid = threadIdx.x;
  if (lane == 0) {
    shared_val[wid] = val[0];
  }
  __syncthreads();

  val[0] = tid < (blockDim.x / 32.f) ? shared_val[tid] : 0.0f;

  if (wid == 0) {
    warpReduceSum(val, 32);
  }
  return 0.0f;
}

__global__ void block_softmax_kernel(const float *input, float *output,
                                     size_t m, size_t n) {
  int row_offset = blockIdx.x;

  if (row_offset >= m)
    return;

  const float *row_input = input + row_offset * n;
  float *row_output = output + row_offset * n;

  float local_max[1] = {-INFINITY};
  for (int i = 0; i < n; i += blockDim.x) {
    int col_idx = i + threadIdx.x;
    if (col_idx < n) {
      local_max[0] = fmax(local_max[0], row_input[col_idx]);
    }
  }
  blockReduceMax(local_max, 1);

  __shared__ float shared_val[2];
  if (threadIdx.x == 0) {
    shared_val[0] = local_max[0];
  }
  __syncthreads();

  float local_sum[1] = {0.0f};
  for (int i = 0; i < n; i += blockDim.x) {
    int col_idx = i + threadIdx.x;
    if (col_idx < n) {
      local_sum[0] += expf(row_input[col_idx] - shared_val[0]);
    }
  }
  blockReduceSum(local_sum, 32);

  if (threadIdx.x == 0) {
    shared_val[1] = local_sum[0];
  }
  __syncthreads();

  for (int i = 0; i < n; i += blockDim.x) {
    int col_idx = i + threadIdx.x;
    if (col_idx < n) {
      row_output[col_idx] =
          expf(row_input[col_idx] - shared_val[0]) / shared_val[1];
    }
  }
}

// New PyTorch launcher function
void block_softmax_kernel_launcher(torch::Tensor input, torch::Tensor output) {
  // Get tensor dimensions
  const int rows = input.size(0);
  const int cols = input.size(1);

  // Ensure inputs are on GPU and contiguous
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
  TORCH_CHECK(output.is_cuda(), "Output tensor must be on CUDA device");
  TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "Output tensor must be contiguous");
  TORCH_CHECK(input.dim() == 2, "Input must be a 2D tensor");
  TORCH_CHECK(output.dim() == 2, "Output must be a 2D tensor");

  // Launch configuration
  const dim3 grid_size(rows); // One block per row
  const dim3 block_size(512); // 512 threads per block as in original

  // Launch kernel
  block_softmax_kernel<<<grid_size, block_size>>>(
      input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
}

#ifdef STANDALONE_TEST
int main() {
  int m = 1;
  int n = 1024;

  float *input_h = (float *)malloc(m * n * sizeof(float));
  float *output_h = (float *)malloc(m * n * sizeof(float));

  // initialize input
  for (int i = 0; i < m * n; i++) {
    input_h[i] = 1.0f;
  }

  float *input_d;
  cudaMalloc(&input_d, m * n * sizeof(float));
  float *output_d;
  cudaMalloc(&output_d, m * n * sizeof(float));

  cudaMemcpy(input_d, input_h, m * n * sizeof(float), cudaMemcpyHostToDevice);

  dim3 grid_size(m);
  dim3 block_size(512);

  block_softmax_kernel<<<grid_size, block_size>>>(input_d, output_d, m, n);

  cudaMemcpy(output_h, output_d, m * n * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < m * n; i++) {
    printf("%f ", output_h[i]);
  }
  printf("\n");
}
#endif
