#include <cuda_runtime.h>

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

#define LDG32_GUARD_MOV0_PTX(reg, ptr, guard)          \
    {                                                  \
        asm volatile("{.reg .pred p;\n\t"              \
                     "setp.ne.u32 p, %2, 0;\n\t"       \
                     "@!p mov.b32 %0, 0;\n\t"          \
                     "@p ld.global.f32 %0, [%1];}\n\t" \
                     : "=f"(reg)                       \
                     : "l"(ptr), "r"(guard));          \
    }

#define STS128_PTX(reg0, reg1, reg2, reg3, addr)                               \
  {                                                                            \
    asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n\t"                \
                 :                                                             \
                 : "l"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3));     \
  }

__global__ void sgemm_128x128x8(int m, int n, int k, const float *A, int lda,
                                const float *B, int ldb, float *C, int ldc) {
  // Operands A, B, C: row-major format

  const int smem_a_padding = 128;
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

  // check bounds of A and B
  unsigned ldg_a_bitmask = 0x0;
  unsigned ldg_b_bitmask = 0x0;

  int ldg_a_offsets_y[4];
  int ldg_a_offsets[4];

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    ldg_a_offsets_y[i] = i;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    ldg_a_offsets[i] = ldg_a_offsets_y[i] * lda;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int m_idx = ldg_a_start_y + ldg_a_offsets_y[i];
    if (m_idx < m) {
      ldg_a_bitmask ^= (0x1 << i);
    }
  }

  int sts_a_start_x = 4 * (threadIdx.x / 8);
  int sts_a_start_y = threadIdx.x % 8;
  int sts_a_start = sts_a_start_y * smem_a_ld + sts_a_start_x;
  float *sts_a_ptr = smem_a_ptr + sts_a_start;

  uint64_t sts_a_addr;

  CVTA_TO_SHARED_PTX(sts_a_addr, sts_a_ptr);

  int n_block_k = (k + 7) / 8;

  for (int block_k = 0; block_k < n_block_k; ++block_k) {
    for (int i = 0; i < 4; ++i) {
      bool guard_m = (ldg_a_bitmask & (0x1 << i));
      LDG32_GUARD_MOV0_PTX(ldg_a_buffer[i], ldg_a_ptr + ldg_a_offsets[i],
                      (unsigned)guard_m);
    }
    STS128_PTX(ldg_a_buffer[0], ldg_a_buffer[1], ldg_a_buffer[2],
               ldg_a_buffer[3], sts_a_addr);
    __syncthreads();
    printf("sts_a_ptr: %f, %f, %f, %f\n", sts_a_ptr[0], sts_a_ptr[1],
           sts_a_ptr[2], sts_a_ptr[3]);
  }
}
