#include <cuda_runtime.h>
#include <stdint.h>

#define CVTA_TO_SHARED_PTX(addr, smem_ptr)                                     \
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(addr) : "l"(smem_ptr));

#define LDG32_GUARD_PTX(reg, ptr, guard)                                       \
  {                                                                            \
    asm volatile("{.reg .pred p;\n\t"                                          \
                 "setp.ne.u32 p, %2, 0;\n\t"                                   \
                 "@p ld.global.f32 %0, [%1];}\n\t"                             \
                 : "=f"(reg)                                                   \
                 : "l"(ptr), "r"(guard));                                      \
  }

#define LDG32_GUARD_MOV0_PTX(reg, ptr, guard)                                  \
  {                                                                            \
    asm volatile("{.reg .pred p;\n\t"                                          \
                 "setp.ne.u32 p, %2, 0;\n\t"                                   \
                 "@!p mov.b32 %0, 0;\n\t"                                      \
                 "@p ld.global.f32 %0, [%1];}\n\t"                             \
                 : "=f"(reg)                                                   \
                 : "l"(ptr), "r"(guard));                                      \
  }

#define STS128_PTX(reg0, reg1, reg2, reg3, addr)                               \
  {                                                                            \
    asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n\t"                \
                 :                                                             \
                 : "l"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3));     \
  }

#define STS32_PTX(reg, addr)                                                   \
  {                                                                            \
    asm volatile("st.shared.f32 [%0], %1;\n" : : "l"(addr), "f"(reg));         \
  }

__global__ void sgemm_128x128x8(int m, int n, int k, const float *A, int lda,
                                const float *B, int ldb, float *C, int ldc) {
  // Operands A, B, C: row-major format

  // Modern NVIDIA GPUs perform best when shared-memory accesses are naturally
  // aligned to (at least) 128‑ or 256‑byte boundaries.
  const int smem_a_padding = 256;
  const int smem_a_size = smem_a_padding * 8;
  const int smem_a_ld = 132; // leading dim 128 + 4 for avoiding bank conflict
  const int smem_b_padding = 128;
  const int smem_b_size = smem_b_padding * 8;
  const int smem_b_ld = 128;
  __shared__ float smem_ptr[smem_a_size + smem_b_size];

  // Registers for (global memory -> shared memory) transfers
  float ldg_a_buffer[4]; // ldg - load global
  float ldg_b_buffer[4];

  float *smem_a_ptr = smem_ptr;
  float *smem_b_ptr = smem_ptr + smem_a_size;

  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;

  int ldg_a_start_x = threadIdx.x % 8;
  int ldg_a_start_y = blockIdx.y * 128 + 4 * (threadIdx.x / 8);
  int ldg_a_start = ldg_a_start_y * lda + ldg_a_start_x;

  const float *ldg_a_ptr = A + ldg_a_start;

  int ldg_b_start_x = threadIdx.x % 32 + blockIdx.x * 128;
  int ldg_b_start_y = threadIdx.x / 32;
  int ldg_b_start = ldg_b_start_y * ldb + ldg_b_start_x;

  const float *ldg_b_ptr = B + ldg_b_start;

  // check bounds of A and B
  unsigned ldg_a_bitmask = 0x0;
  unsigned ldg_b_bitmask = 0x0;

  int ldg_a_offsets_y[4];
  int ldg_a_offsets[4];
  int ldg_b_offsets[4];

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    ldg_a_offsets_y[i] = i;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    ldg_a_offsets[i] = ldg_a_offsets_y[i] * lda;
    ldg_b_offsets[i] = i * 32;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int m_idx = ldg_a_start_y + ldg_a_offsets_y[i];
    if (m_idx < m) {
      ldg_a_bitmask ^= (0x1 << i);
    }

    int n_idx = ldg_b_start_x + ldg_b_offsets[i];
    if (n_idx < n) {
      ldg_b_bitmask ^= (0x1 << i);
    }
  }

  int sts_a_start_x = 4 * (threadIdx.x / 8);
  int sts_a_start_y = threadIdx.x % 8;
  int sts_a_start = sts_a_start_y * smem_a_ld + sts_a_start_x;
  float *sts_a_ptr = smem_a_ptr + sts_a_start;

  int sts_b_start_x = threadIdx.x % 32;
  int sts_b_start_y = threadIdx.x / 32;
  int sts_b_start = sts_b_start_y * smem_b_ld + sts_b_start_x;
  float *sts_b_ptr = smem_b_ptr + sts_b_start;
  int sts_b_offsets[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    sts_b_offsets[i] = i * 32;
  }

  uint64_t sts_a_addr;
  uint64_t sts_b_addr;

  // Convert from generic to .shared state space
  CVTA_TO_SHARED_PTX(sts_a_addr, sts_a_ptr);
  CVTA_TO_SHARED_PTX(sts_b_addr, sts_b_ptr);

  int n_block_k = (k + 7) / 8;
  for (int block_k = 0; block_k < n_block_k; ++block_k) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      bool guard_m = (ldg_a_bitmask & (0x1 << i));
      LDG32_GUARD_MOV0_PTX(ldg_a_buffer[i], ldg_a_ptr + ldg_a_offsets[i],
                           (unsigned)guard_m);
    }
    STS128_PTX(ldg_a_buffer[0], ldg_a_buffer[1], ldg_a_buffer[2],
               ldg_a_buffer[3], sts_a_addr);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      bool guard_n = (ldg_b_bitmask & (0x1 << i));
      LDG32_GUARD_MOV0_PTX(ldg_b_buffer[i], ldg_b_ptr + ldg_b_offsets[i],
                           (unsigned)guard_n);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      STS32_PTX(ldg_b_buffer[i], sts_b_addr + sts_b_offsets[i]);
    }
    __syncthreads();
  }
}
