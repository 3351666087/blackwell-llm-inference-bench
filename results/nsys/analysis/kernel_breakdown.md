| capture | window ms | GPU busy | gemm_fp8_block | gemm | attn_prefill | attn_decode | norm_act_rope | sampling_logits | memcpy_comm | other |
|---|---|---|---|---|---|---|---|---|---|---|
| trt_q8b_fp8_chat_c64 | 5096 | 0.41 | 59.8% | 7.8% | 0.8% | 0.3% | 4.2% | 0.1% | 0.3% | 26.6% |
| vllm_q8b_fp8_chat_c64 | 5099 | 0.89 | 64.7% | 29.4% | 0.0% | 0.0% | 2.4% | 0.3% | 1.6% | 1.6% |
| vllm_q32b_fp8_chat_c256 | 3496 | 0.99 | 48.8% | 47.5% | 0.0% | 0.0% | 2.1% | 0.2% | 0.5% | 0.9% |
| sglang_q32b_fp8_chat_c256 | 3990 | 0.99 | 0.0% | 71.0% | 23.3% | 0.0% | 2.4% | 0.3% | 0.7% | 2.3% |

### trt_q8b_fp8_chat_c64 — top kernels

| kernel | category | ms | count | share |
|---|---|---|---|---|
| `void cutlass::device_kernel<sm120_blockscaled_gemm::SM120BlockScaledKernel<sm120_blockscal` | gemm_fp8_block | 1257.1 | 27389 | 59.8% |
| `kernel_mha` | other | 470.6 | 6847 | 22.4% |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_tn_align8>` | gemm | 164.6 | 190 | 7.8% |
| `_per_token_quant_and_transform_kernel` | other | 84.0 | 27389 | 4.0% |
| `void flashinfer::norm::FusedAddRMSNormKernel<(unsigned int)8, __nv_bfloat16>(T2 *, T2 *, T` | norm_act_rope | 36.8 | 13694 | 1.8% |
| `void tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel<(int)128, (bool)0>(__nv_bfloat16 *,` | norm_act_rope | 23.0 | 6848 | 1.1% |
| `fmha_v2_flash_attention_bf16_64_32_S_qkv_128_causal_sm120_kernel_nl` | attn_prefill | 17.7 | 1764 | 0.8% |
| `void tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2<__nv_bfloat16, __nv_bfloat16` | norm_act_rope | 16.0 | 8611 | 0.8% |
| `silu_and_mul_kernel` | norm_act_rope | 11.7 | 6847 | 0.6% |
| `void tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets<__nv_bfloat16, (int)256>(tens` | attn_decode | 6.4 | 1764 | 0.3% |

### vllm_q8b_fp8_chat_c64 — top kernels

| kernel | category | ms | count | share |
|---|---|---|---|---|
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<(unsigned int)0, (unsigned int)24576, (unsign` | gemm_fp8_block | 1292.1 | 14750 | 23.3% |
| `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<(int)128, (int)64, (int)64, (` | gemm | 900.8 | 12014 | 16.2% |
| `void deep_gemm::sm120_split_k_reduce_impl<cutlass::bfloat16_t, (unsigned int)4, deep_gemm:` | gemm_fp8_block | 733.4 | 26836 | 13.2% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<(unsigned int)0, (unsigned int)4096, (unsigne` | gemm_fp8_block | 650.6 | 14750 | 11.7% |
| `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<(int)128, (int)64, (int)128, ` | gemm | 379.1 | 2736 | 6.8% |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_tn_align8>` | gemm | 353.3 | 409 | 6.4% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<(unsigned int)0, (unsigned int)6144, (unsigne` | gemm_fp8_block | 340.6 | 14750 | 6.1% |
| `void deep_gemm::sm120_split_k_reduce_impl<cutlass::bfloat16_t, (unsigned int)2, deep_gemm:` | gemm_fp8_block | 294.7 | 14102 | 5.3% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<(unsigned int)0, (unsigned int)4096, (unsigne` | gemm_fp8_block | 254.0 | 14750 | 4.6% |
| `void per_token_group_quant_8bit_packed_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, ` | other | 84.7 | 59000 | 1.5% |

### vllm_q32b_fp8_chat_c256 — top kernels

| kernel | category | ms | count | share |
|---|---|---|---|---|
| `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<(int)128, (int)64, (int)128, ` | gemm | 1441.8 | 1579 | 39.4% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<(unsigned int)0, (unsigned int)51200, (unsign` | gemm_fp8_block | 773.4 | 2602 | 21.1% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<(unsigned int)0, (unsigned int)5120, (unsigne` | gemm_fp8_block | 476.5 | 2602 | 13.0% |
| `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<(int)128, (int)64, (int)64, (` | gemm | 244.5 | 1024 | 6.7% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<(unsigned int)0, (unsigned int)10240, (unsign` | gemm_fp8_block | 191.9 | 2603 | 5.2% |
| `void deep_gemm::sm120_split_k_reduce_impl<cutlass::bfloat16_t, (unsigned int)2, deep_gemm:` | gemm_fp8_block | 188.4 | 2048 | 5.2% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<(unsigned int)0, (unsigned int)5120, (unsigne` | gemm_fp8_block | 153.1 | 2602 | 4.2% |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8` | gemm | 51.4 | 40 | 1.4% |
| `triton_poi_fused_mul_per_token_group_fp8_quant_packed_silu_slice_1` | norm_act_rope | 40.3 | 2602 | 1.1% |
| `void per_token_group_quant_8bit_packed_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, ` | other | 23.2 | 7807 | 0.6% |

### sglang_q32b_fp8_chat_c256 — top kernels

| kernel | category | ms | count | share |
|---|---|---|---|---|
| `void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute::tuple<int, int, int` | gemm | 2642.1 | 22183 | 67.1% |
| `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` | attn_prefill | 894.4 | 5546 | 22.7% |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8` | gemm | 71.5 | 55 | 1.8% |
| `void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::ActivationKind)0, (bool)1, (bool)0` | norm_act_rope | 57.8 | 5545 | 1.5% |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a` | gemm | 48.3 | 11091 | 1.2% |
| `void sglang::per_token_group_quant_flat_kernel<sglang::QuantTrait<__nv_bfloat16, __nv_fp8_` | other | 43.9 | 22183 | 1.1% |
| `void flashinfer::PersistentVariableLengthMergeStatesKernel<(unsigned int)8, (unsigned int)` | other | 31.5 | 3712 | 0.8% |
| `void flashinfer::BatchPrefillWithRaggedKVCacheKernel<flashinfer::KernelTraits<(flashinfer:` | attn_prefill | 23.8 | 2026 | 0.6% |
| `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bflo` | gemm | 20.3 | 19 | 0.5% |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` | memcpy_comm | 19.9 | 5546 | 0.5% |
