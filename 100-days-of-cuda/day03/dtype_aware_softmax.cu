#include <cuda_runtime.h>
#include <limits.h>
#include <stdio.h>
#include <torch/extension.h>

template <typename T> inline __device__ float to_float(T val)
{
  return static_cast<float>(val);
}

template <> inline __device__ float to_float<__half>(__half val)
{
  return __half2float(val);
}

template <> inline __device__ float to_float<__nv_bfloat16>(__nv_bfloat16 val)
{
  return __bfloat162float(val);
}

template <typename T> inline __device__ T convert_float_to(float val)
{
  return static_cast<T>(val);
}

template <> inline __device__ __half convert_float_to<__half>(float val)
{
  return __float2half(val);
}

template <>
inline __device__ __nv_bfloat16 convert_float_to<__nv_bfloat16>(float val)
{
  return __float2bfloat16(val);
}

template <typename T> inline __device__ T max(T a, T b)
{
  return a > b ? a : b;
}

template <> inline __device__ __half max<__half>(__half a, __half b)
{
  return __hmax(a, b);
}

template <>
inline __device__ __nv_bfloat16 max<__nv_bfloat16>(__nv_bfloat16 a,
                                                   __nv_bfloat16 b)
{
  float max_val = fmaxf(__bfloat162float(a), __bfloat162float(b));
  return __float2bfloat16(max_val);
}

template <typename T>
inline __device__ void warpReduceMax(T* val, int thread_group = 32)
{
#pragma unroll
  for (int lane_mask = thread_group / 2; lane_mask > 0; lane_mask >>= 1) {
    val[0] = max(val[0],
                 __shfl_xor_sync(0xFFFFFFFF, val[0], lane_mask, thread_group));
  }
}

template <typename T>
inline __device__ void warpReduceSum(T* val, int thread_group = 32)
{
#pragma unroll
  for (int lane_mask = thread_group / 2; lane_mask > 0; lane_mask >>= 1) {
    val[0] += __shfl_xor_sync(0xFFFFFFFF, val[0], lane_mask, thread_group);
  }
}

template <typename T, int block_size_y>
__global__ void dtype_aware_softmax_kernel(const T* input, T* output, size_t m, size_t n)
{
  int m_idx = blockDim.y * blockIdx.x + threadIdx.y;
  if (m_idx >= m)
    return;

  int       row_offset = m_idx * n;
  const T*  row_x = input + row_offset;
  T*        row_y = output + row_offset;
  const int num_packs = (n + blockDim.x - 1) / blockDim.x;

  // first pass: find max value
  T local_max[1] = {convert_float_to<T>(-INFINITY)};
  for (int pack_id = 0; pack_id < num_packs; pack_id++) {
    const int col_idx = pack_id * blockDim.x + threadIdx.x;
    if (col_idx < n) {
      local_max[0] = max(local_max[0], row_x[col_idx]);
    }
  }
  warpReduceMax<T>(local_max);

  __shared__ float smem_val[block_size_y][2]; // [0]: local_max, [1]: local_sum
  if (threadIdx.x == 0) {
    smem_val[threadIdx.y][0] = to_float(local_max[0]);
  }
  __syncthreads();

  // second pass: compute denominator
  float local_sum[1] = {0.0f};
  for (int pack_id = 0; pack_id < num_packs; pack_id++) {
    const int col_idx = pack_id * blockDim.x + threadIdx.x;
    if (col_idx < n) {
      local_sum[0] += expf(to_float(row_x[col_idx]) - smem_val[threadIdx.y][0]);
    }
  }
  warpReduceSum<float>(local_sum);
  if (threadIdx.x == 0) {
    smem_val[threadIdx.y][1] = float(local_sum[0]);
  }
  __syncthreads();

  // third pass: compute softmax
  for (int pack_id = 0; pack_id < num_packs; pack_id++) {
    const int col_idx = pack_id * blockDim.x + threadIdx.x;
    if (col_idx < n) {
      float out = expf(to_float(row_x[col_idx]) - smem_val[threadIdx.y][0])
                  / smem_val[threadIdx.y][1];
      row_y[col_idx] = convert_float_to<T>(out);
    }
  }
}

void dtype_aware_softmax_launcher(torch::Tensor input, torch::Tensor output)
{
  size_t m = input.size(0);
  size_t n = input.size(1);
  auto   dtype = input.dtype().toScalarType();

  constexpr int block_size_y = 4;
  dim3          block_size(32, block_size_y);
  int           num_blocks = (m + block_size_y - 1) / block_size_y;
  dim3          grid_size(num_blocks);

  switch (dtype) {
  case torch::kFloat32:
    dtype_aware_softmax_kernel<float, block_size_y><<<grid_size, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), m, n);
    break;
  case torch::kFloat16:
    dtype_aware_softmax_kernel<__half, block_size_y><<<grid_size, block_size>>>(
        reinterpret_cast<__half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        m,
        n);
    break;
  case torch::kBFloat16:
    dtype_aware_softmax_kernel<__nv_bfloat16, block_size_y><<<grid_size, block_size>>>(
        reinterpret_cast<__nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        m,
        n);
    break;
  default:
    throw std::runtime_error("Unsupported dtype");
  }
}
