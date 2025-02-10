#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16

__global__ void naiveMatmulKernel(const float *A, const float *B, float *C,
                                  int m, int n, int k, int lda, int ldb,
                                  int ldc) {
  // operands A, B, C: row-major format
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int x = bx * blockDim.x + tx;
  int y = by * blockDim.y + ty;

  if (x >= n || y >= m) {
    return;
  }

    float sum = 0.f;
    for (int i = 0; i < k; ++i) {
    sum += (A[y * lda + i] * B[i * ldb + x]);
  }
  C[y * ldc + x] = sum;
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

// Function: runCorrectnessCheck
// Description: Launches the tiled and naive matrix multiplication kernels,
//              copies the results back to the host, and then compares
//              element-wise.
void runCorrectnessCheck(const float *A_d, const float *B_d, float *C_d,
                         float *C_d_naive, float *C_h, float *C_h_naive, int m,
                         int n, int k, dim3 grid_size, dim3 block_size) {
  // Launch tiled matmul kernel and copy result to host.
  tiledMatmulKernel<<<grid_size, block_size>>>(A_d, B_d, C_d, m, n, k);
  cudaMemcpy(C_h, C_d, m * n * sizeof(float), cudaMemcpyDeviceToHost);

  // Launch naive matmul kernel and copy its result to host.
  naiveMatmulKernel<<<grid_size, block_size>>>(A_d, B_d, C_d_naive, m, n, k);
  cudaMemcpy(C_h_naive, C_d_naive, m * n * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Compare the results element-wise.
  for (int i = 0; i < m * n; i++) {
    if (C_h[i] != C_h_naive[i]) {
      printf("Error at %d: %f != %f\n", i, C_h[i], C_h_naive[i]);
    }
  }
}

// Function: profileTiledMatmulKernel
// Description: Profiles the tiledMatmulKernel over a given number of iterations
// using CUDA events.
void profileMatmulKernel(const float *A_d, const float *B_d, float *C_d, int m,
                         int n, int k, dim3 grid_size, dim3 block_size,
                         int iterations) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record the start event.
  cudaEventRecord(start, 0);
  for (int i = 0; i < iterations; ++i) {
    // tiledMatmulKernel<<<grid_size, block_size>>>(A_d, B_d, C_d, m, n, k);
    naiveMatmulKernel<<<grid_size, block_size>>>(A_d, B_d, C_d, m, n, k);
  }
  // Record the stop event.
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsed_time = 0.f;
  cudaEventElapsedTime(&elapsed_time, start, stop);
  printf("Tiled matmul average execution time: %f ms over %d iterations\n",
         elapsed_time / iterations, iterations);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

int main() {
  int m = 1024;
  int n = 4096;
  int k = 2048;

  float *A_h = (float *)malloc(m * k * sizeof(float));
  float *B_h = (float *)malloc(k * n * sizeof(float));
  float *C_h = (float *)malloc(m * n * sizeof(float));
  float *C_h_naive = (float *)malloc(m * n * sizeof(float));

  // Initialize matrices A_h and B_h.
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

  // Perform the correctness check.
  runCorrectnessCheck(A_d, B_d, C_d, C_d_naive, C_h, C_h_naive, m, n, k,
                      grid_size, block_size);

  // Profile the performance of the tiled matrix multiplication over 10
  // iterations.
  profileMatmulKernel(A_d, B_d, C_d, m, n, k, grid_size, block_size, 10);

  free(A_h);
  free(B_h);
  free(C_h);
  free(C_h_naive);
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  cudaFree(C_d_naive);

  return 0;
}
