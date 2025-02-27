#include <cuda_runtime.h>

#define TILE_SIZE 16

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

__global__ void tiledMatmulKernel(const float *A, const float *B, float *C,
                                  int m, int n, int k) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int col = bx * blockDim.x + tx;
  int row = by * blockDim.y + ty;

  int num_stages = (k + TILE_SIZE - 1) / TILE_SIZE;

  __shared__ float smem_A[TILE_SIZE][TILE_SIZE];
  __shared__ float smem_B[TILE_SIZE][TILE_SIZE];

  float sum = 0.f;
  for (int stage = 0; stage < num_stages; ++stage) {
    int k_offset = stage * TILE_SIZE;

    if (row < m && tx + k_offset < k) {
      smem_A[ty][tx] = A[row * k + tx + k_offset];
    }
    if (col < n && ty + k_offset < k) {
      smem_B[ty][tx] = B[(ty + k_offset) * n + col];
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i) {
      if (k_offset + i < k) {
        sum += smem_A[ty][i] * smem_B[i][tx];
      }
    }
    __syncthreads();
  }

  if ((col < n) && (row < m)) {
    C[row * n + col] = sum;
  }
}
