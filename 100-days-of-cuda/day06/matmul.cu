#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void naiveMatmulKernel(const float *A, const float *B, float *C,
                                  int m, int n, int k) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int col = bx * blockDim.x + tx;
  int row = by * blockDim.y + ty;

  if ((col < n) && (row < m)) {
    float sum = 0.f;
    for (int i = 0; i < k; ++i) {
      sum += (A[row * k + i] * B[i * n + col]);
    }
    C[row * n + col] = sum;
  }
}

void naive_matmul_launcher(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C) {

    const int m = A.size(0);
    const int k = A.size(1);
    const int n = B.size(1);

    dim3 block_size(16, 16);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);

    naiveMatmulKernel<<<grid_size, block_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        m, n, k);
}
