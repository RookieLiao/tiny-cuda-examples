#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16
#define K_TILE_SIZE 8
#define MN_TILE_SIZE 128

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
                                  int m, int n, int k, int lda, int ldb,
                                  int ldc) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int col_offset = bx * MN_TILE_SIZE;
  int row_offset = by * MN_TILE_SIZE;

  const int num_ks = (k + K_TILE_SIZE - 1) / K_TILE_SIZE;

  __shared__ float smem_A[MN_TILE_SIZE][K_TILE_SIZE];
  __shared__ float smem_B[K_TILE_SIZE][MN_TILE_SIZE];

  float partial_sum[8][8]{0.0f};
  for (int i = 0; i < num_ks; ++i) {
    int k_offset = i * K_TILE_SIZE;

    // load to A_smem
    /*
    128 * 8

    global A:
    method1:
    t0 t0 t0 t0 t1 t1 t1 t1
    t2 t2 t2 t2 t3 t3 t3 t3
    ...

    method2:
    t0 t1 t2 t3 t4 t5 t6 t7
    t0 t1 t2 t3 t4 t5 t6 t7
    t0 t1 t2 t3 t4 t5 t6 t7
    t0 t1 t2 t3 t4 t5 t6 t7
    t8 t9 t10 t11 t12 t13 t14 t15
    t8 t9 t10 t11 t12 t13 t14 t15
    t8 t9 t10 t11 t12 t13 t14 t15
    t8 t9 t10 t11 t12 t13 t14 t15
    ...
    */

    int a_row_idx = (tx / 8) * 4;
    int a_col_idx = tx % 8;

    if (row_offset + a_row_idx < m && k_offset + a_col_idx < k) {
      smem_A[a_row_idx][a_col_idx] =
          A[(row_offset + a_row_idx) * k + k_offset + a_col_idx];
      smem_A[a_row_idx + 1][a_col_idx] =
          A[(row_offset + a_row_idx + 1) * k + k_offset + a_col_idx];
      smem_A[a_row_idx + 2][a_col_idx] =
          A[(row_offset + a_row_idx + 2) * k + k_offset + a_col_idx];
      smem_A[a_row_idx + 3][a_col_idx] =
          A[(row_offset + a_row_idx + 3) * k + k_offset + a_col_idx];
    }

    // load to B_smem
    /*
    8 * 128

    global B:

    t0 t0 t0 t0 t8 t8 t8 t8
    t1 t1 t1 t1 t9 t9 t9 t9
    t2 t2 t2 t2 t10 t10 t10 t10
    t3 t3 t3 t3 t11 t11 t11 t11
    t4 t4 t4 t4 t12 t12 t12 t12
    t5 t5 t5 t5 t13 t13 t13 t13
    t6 t6 t6 t6 t14 t14 t14 t14
    t7 t7 t7 t7 t15 t15 t15 t15

    t0 t1 t2 t3 t4 t5 t6 t7 ...
    t0 t1 t2 t3 t4 t5 t6 t7
    t0 t1 t2 t3 t4 t5 t6 t7
    t0 t1 t2 t3 t4 t5 t6 t7
    t128 t129 t130 t131 t132 t133 t134 ...
    t128 t129 t130 t131 t132 t133 t134
    t128 t129 t130 t131 t132 t133 t134
    t128 t129 t130 t131 t132 t133 t134
    ...
    */

    // load to smem
    int b_row_idx = (tx / 128) * 4;
    int b_col_idx = tx % 128;

    if (col_offset + b_col_idx < n && k_offset + b_row_idx < k) {
      smem_B[b_row_idx][b_col_idx] =
          B[(k_offset + b_row_idx) * n + col_offset + b_col_idx];
      smem_B[b_row_idx + 1][b_col_idx] =
          B[(k_offset + b_row_idx + 1) * n + col_offset + b_col_idx];
      smem_B[b_row_idx + 2][b_col_idx] =
          B[(k_offset + b_row_idx + 2) * n + col_offset + b_col_idx];
      smem_B[b_row_idx + 3][b_col_idx] =
          B[(k_offset + b_row_idx + 3) * n + col_offset + b_col_idx];
    }
    __syncthreads();

    for (int warp_k = 0; warp_k < K_TILE_SIZE; ++warp_k) {
      if (k_offset + warp_k < k) {
        int warp_idx = (tx / 32);
        int lane_id = tx % 32;
        int row_smem_a = (warp_idx / 2) * 32;
        int col_smem_b = warp_idx % 2;
        for (int l = 0; l < 8; l++) {
          for (int t = 0; t < 8; t++) {
            int warp_col_off = lane_id / 8 * 2 + lane_id % 2;
            int warp_row_off = (lane_id - lane_id / 8 * 8) / 2;
            partial_sum[t][l] +=
                smem_A[row_smem_a + (warp_idx / 2) * 32 + t * 4 + warp_row_off]
                      [warp_k] *
                smem_B[warp_k][col_smem_b + l * 8 + warp_col_off];
          }
        }
      }
    }
    __syncthreads();
  }

  if (col_offset + tx % 128 < n && row_offset + tx / 8 * 4 < m) {
    int warp_idx = tx / 32;
    int lane_id = tx % 32;
    for (int l = 0; l < 8; ++l) {
      for (int t = 0; t < 8; ++t) {
        int warp_col_off = lane_id / 8 * 2 + lane_id % 2;
        int warp_row_off = (lane_id - lane_id / 8 * 8) / 2;
        C[(row_offset + (warp_idx / 2) * 32 + t * 4 + warp_row_off) * ldc +
          col_offset + l * 8 + warp_col_off] = partial_sum[t][l];
      }
    }
  }
}

// Function: runCorrectnessCheck
// Description: Launches the tiled and naive matrix multiplication kernels,
//              copies the results back to the host, and then compares
//              element-wise.
void runCorrectnessCheck(const float *A_d, const float *B_d, float *C_d,
                         float *C_d_naive, float *C_h, float *C_h_naive, int m,
                         int n, int k) {

  // Launch tiled matmul kernel and copy result to host.
  dim3 tile_block_size(256);
  dim3 tile_grid_size((n + MN_TILE_SIZE - 1) / MN_TILE_SIZE,
                      (m + MN_TILE_SIZE - 1) / MN_TILE_SIZE);
  tiledMatmulKernel<<<tile_grid_size, tile_block_size>>>(A_d, B_d, C_d, m, n, k,
                                                         k, n, n);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch tiledMatmulKernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  cudaMemcpy(C_h, C_d, m * n * sizeof(float), cudaMemcpyDeviceToHost);

  // Launch naive matmul kernel and copy its result to host.
  dim3 naive_block_size(TILE_SIZE, TILE_SIZE);
  dim3 naive_grid_size((n + TILE_SIZE - 1) / TILE_SIZE,
                       (m + TILE_SIZE - 1) / TILE_SIZE);
  naiveMatmulKernel<<<naive_grid_size, naive_block_size>>>(A_d, B_d, C_d_naive,
                                                           m, n, k, k, n, n);
  cudaMemcpy(C_h_naive, C_d_naive, m * n * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Compare the results element-wise.
  for (int i = 0; i < m * n; i++) {
    if (C_h[i] != C_h_naive[i]) {
      printf("Error at %d: %f != %f\n", i, C_h[i], C_h_naive[i]);
      break;
    }
  }
}

// Function: profileTiledMatmulKernel
// Description: Profiles the tiledMatmulKernel over a given number of
// iterations using CUDA events.
void profileMatmulKernel(const float *A_d, const float *B_d, float *C_d, int m,
                         int n, int k, int iterations) {
  // dim3 grid_size((n + TILE_SIZE - 1) / TILE_SIZE,
  //                (m + TILE_SIZE - 1) / TILE_SIZE);
  // dim3 block_size(TILE_SIZE, TILE_SIZE);
  dim3 grid_size((n + MN_TILE_SIZE - 1) / MN_TILE_SIZE,
                 (m + MN_TILE_SIZE - 1) / MN_TILE_SIZE);
  dim3 block_size(256);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record the start event.
  cudaEventRecord(start);
  for (int i = 0; i < iterations; ++i) {
    tiledMatmulKernel<<<grid_size, block_size>>>(A_d, B_d, C_d, m, n, k, k, n, n);
    // naiveMatmulKernel<<<grid_size, block_size>>>(A_d, B_d, C_d, m, n, k, k, n,
    //                                              n);
  }
  // Record the stop event.
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);

  float elapsed_time = 0.0f;
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

  // Perform the correctness check.
  runCorrectnessCheck(A_d, B_d, C_d, C_d_naive, C_h, C_h_naive, m, n, k);

  // Profile the performance of the tiled matrix multiplication over 10
  // iterations.
  profileMatmulKernel(A_d, B_d, C_d, m, n, k, 10);

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
