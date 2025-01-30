#include <stdio.h>
#include <float.h>

/* naive safe softmax kernel */
__global__ void softmax_kernel(float* input, float* output, size_t rows, size_t cols) {
  int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // boundary check
  if (g_idx >= rows) return;

  // first pass: find max value
  float max_val = -FLT_MAX;
  for (int i = 0; i < cols; ++i) {
    max_val = fmaxf(max_val, input[g_idx * cols + i]);
  }

  // second pass: compute denominator
  float denom = 0.0f;
  for (int i = 0; i < cols; ++i) {
    denom += expf(input[g_idx * cols + i] - max_val);
  }

  // third pass: compute softmax
  for (int i = 0; i < cols; ++i) {
    output[g_idx * cols + i] = expf(input[g_idx * cols + i] - max_val) / denom;
  }
}


int main() {
  int N = 128; // rows
  int M = 2048; // cols

  int num_elements = N * M;
  float* input_h = (float*)malloc(num_elements * sizeof(float));
  float* output_h = (float*)malloc(num_elements * sizeof(float));

  // initialize input with random values
  for (int i = 0; i < num_elements; i++) {
    input_h[i] = rand() / (float)RAND_MAX;
  }

  float* input_d;
  cudaMalloc(&input_d, num_elements * sizeof(float));
  cudaMemcpy(input_d, input_h, num_elements * sizeof(float), cudaMemcpyHostToDevice);

  float* output_d;
  cudaMalloc(&output_d, num_elements * sizeof(float));

  dim3 block_size(128);
  dim3 grid_size(ceil(float(N) / block_size.x));
  softmax_kernel<<<grid_size, block_size>>>(input_d, output_d, N, M);

  cudaMemcpy(output_h, output_d, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < num_elements; i++) {
    printf("%f\n", output_h[i]);
  }
}
