#include <cuda_runtime.h>
#include <stdio.h>

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
  }
  if ((col < n) && (row < m)) {
    C[row * n + col] = sum;
  }
}

int main() {
  int m = 33;
  int n = 129;
  int k = 257;

  float *A_h = (float *)malloc(m * k * sizeof(float));
  float *B_h = (float *)malloc(k * n * sizeof(float));
  float *C_h = (float *)malloc(m * n * sizeof(float));
  float *C_h_naive = (float *)malloc(m * n * sizeof(float));

  for (int i = 0; i < m * k; i++) {
    A_h[i] = rand() / (float)RAND_MAX;
  }
  for (int i = 0; i < k * n; i++) {
    B_h[i] = rand() / (float)RAND_MAX;
  }

  float *A_d, *B_d, *C_d, *C_d_naive;
  cudaMalloc(&A_d, m * k * sizeof(float));
  cudaMalloc(&B_d, k * n * sizeof(float));
  cudaMalloc(&C_d, m * n * sizeof(float));
  cudaMalloc(&C_d_naive, m * n * sizeof(float));

  cudaMemcpy(A_d, A_h, m * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, k * n * sizeof(float), cudaMemcpyHostToDevice);

  dim3 block_size(TILE_SIZE, TILE_SIZE);
  dim3 grid_size((n + TILE_SIZE - 1) / TILE_SIZE,
                 (m + TILE_SIZE - 1) / TILE_SIZE);

  tiledMatmulKernel<<<grid_size, block_size>>>(A_d, B_d, C_d, m, n, k);
  cudaMemcpy(C_h, C_d, m * n * sizeof(float), cudaMemcpyDeviceToHost);
  printf("Tiled matmul done\n");
  naiveMatmulKernel<<<grid_size, block_size>>>(A_d, B_d, C_d_naive, m, n, k);
  cudaMemcpy(C_h_naive, C_d_naive, m * n * sizeof(float),
             cudaMemcpyDeviceToHost);
  printf("Naive matmul done\n");

  for (int i = 0; i < m * n; i++) {
    if (C_h[i] != C_h_naive[i]) {
      printf("Error at %d: %f != %f\n", i, C_h[i], C_h_naive[i]);
    }
  }

  free(A_h);
  free(B_h);
  free(C_h);
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}
