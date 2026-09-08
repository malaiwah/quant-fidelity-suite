/*
 * Derived from cuda_recurrent_gated_delta_rule_kernel_128<false, V_SPLIT, true>
 * in exllamav3/exllamav3_ext/gdn.cu, release v1.4.8, commit
 * 6ff3a17ea7f3d0026b273d43239398d57f71b788.
 * This is a CHANGED RUNTIME, not the stock extension. The two shared atomicAdd
 * reductions become indexed SUBK partials, combined in ascending SUBK order.
 * Redundant identical shared writes now have one writer. All arithmetic within
 * a partial, q/k normalization, decay, BF16 conversion and slot layout are kept.
 * No Graph or other exllamav3 extension symbols are referenced.
 *
 * MIT License
 * Copyright (c) 2025 Turboderp
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <limits>

using bfloat16 = __nv_bfloat16;
constexpr int SUBK = 4;

namespace {

template <int V_SPLIT>
__global__ __launch_bounds__(128 * SUBK)
void ordered_channelwise_128(
    const bfloat16* __restrict__ mixed_qkv,
    const float* __restrict__ g,
    const bfloat16* __restrict__ beta,
    float* __restrict__ recurrent_state,
    bfloat16* __restrict__ core_attn_out,
    const int seqlen,
    const int num_k_heads,
    const int num_v_heads,
    const float scale,
    const int* __restrict__ slots,
    const int64_t history_stride)
{
    constexpr int HEAD_DIM = 128;
    constexpr int V_CHUNK_DIM = HEAD_DIM / V_SPLIT;
    constexpr int BTS = HEAD_DIM / SUBK;
    constexpr size_t HEAD_STATE_SIZE = HEAD_DIM * HEAD_DIM;
    const int group = num_v_heads / num_k_heads;
    const size_t state_size = num_v_heads * HEAD_STATE_SIZE;
    const size_t slot_size = history_stride * state_size;
    const int bi = blockIdx.x;
    const size_t qkv_stride = (2 * size_t(num_k_heads) + num_v_heads) * HEAD_DIM;
    mixed_qkv += size_t(bi) * seqlen * qkv_stride;
    g += size_t(bi) * seqlen * num_v_heads * HEAD_DIM;
    beta += size_t(bi) * seqlen * num_v_heads;
    const int state_slot = slots ? slots[bi] : bi;
    float* final_state = recurrent_state + size_t(state_slot) * slot_size;
    core_attn_out += size_t(bi) * seqlen * num_v_heads * HEAD_DIM;

    const int t = threadIdx.x;
    const int bt = threadIdx.y;
    const int lane = t % 32;
    const int warp = t / 32;
    const int head = blockIdx.y;
    const int k_head = head / group;
    const int v_start = blockIdx.z * V_CHUNK_DIM;

    __shared__ float sh_red[2][HEAD_DIM / 32];
    __shared__ float sh_k[HEAD_DIM];
    __shared__ float sh_q[HEAD_DIM];
    __shared__ float sh_g[HEAD_DIM];
    __shared__ float sh_dot1[V_CHUNK_DIM];
    __shared__ float partial1[SUBK][V_CHUNK_DIM];
    __shared__ float partial2[SUBK][V_CHUNK_DIM];

    for (int s = 0; s < seqlen; ++s)
    {
        const bfloat16* gl_q = mixed_qkv + k_head * HEAD_DIM;
        const bfloat16* gl_k = mixed_qkv + (num_k_heads + k_head) * HEAD_DIM;
        const bfloat16* gl_v = mixed_qkv + 2 * size_t(num_k_heads) * HEAD_DIM + head * HEAD_DIM + v_start;
        bfloat16* out = core_attn_out + head * HEAD_DIM + v_start;
        float* gl_rs = final_state + head * HEAD_STATE_SIZE;

        // Only bt=0 publishes q/k/g; the original kernel wrote identical values
        // from all four bt groups. Warp topology and normalization order match.
        float q = 0.0f, k = 0.0f;
        if (bt == 0)
        {
            q = __bfloat162float(gl_q[t]);
            k = __bfloat162float(gl_k[t]);
            float sumq = q * q;
            float sumk = k * k;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
            {
                sumq += __shfl_xor_sync(0xffffffff, sumq, offset);
                sumk += __shfl_xor_sync(0xffffffff, sumk, offset);
            }
            if (lane == 0)
            {
                sh_red[0][warp] = sumq;
                sh_red[1][warp] = sumk;
            }
        }
        __syncthreads();
        if (bt == 0)
        {
            float sumq = lane < HEAD_DIM / 32 ? sh_red[0][lane] : 0.0f;
            float sumk = lane < HEAD_DIM / 32 ? sh_red[1][lane] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
            {
                sumq += __shfl_xor_sync(0xffffffff, sumq, offset);
                sumk += __shfl_xor_sync(0xffffffff, sumk, offset);
            }
            q = q * rsqrtf(sumq + 1e-6f);
            k = k * rsqrtf(sumk + 1e-6f);
            sh_k[t] = k;
            sh_q[t] = q;
            sh_g[t] = __expf(g[head * HEAD_DIM + t]);
        }
        __syncthreads();

        if (t < V_CHUNK_DIM)
        {
            float sum = 0.0f;
            const float* sh_k_rd = sh_k + bt * BTS;
            const float* sh_g_rd = sh_g + bt * BTS;
            const float* rs_rd = gl_rs + v_start + t + bt * BTS * HEAD_DIM;
            #pragma unroll
            for (int i = 0; i < HEAD_DIM / 8 / SUBK; ++i)
            {
                #pragma unroll
                for (int j = 0; j < 8; ++j, rs_rd += HEAD_DIM, sh_k_rd++, sh_g_rd++)
                    sum = sum + *sh_k_rd * *sh_g_rd * *rs_rd;
            }
            partial1[bt][t] = sum;
        }
        __syncthreads();
        if (t < V_CHUNK_DIM && bt == 0)
        {
            float sum = 0.0f;
            #pragma unroll
            for (int part = 0; part < SUBK; ++part)
                sum = __fadd_rn(sum, partial1[part][t]);
            sh_dot1[t] = sum;
        }
        __syncthreads();

        if (t < V_CHUNK_DIM)
        {
            const float beta_h = __bfloat162float(beta[head]);
            const float v = __bfloat162float(gl_v[t]) - sh_dot1[t];
            float v_out = 0.0f;
            const float* sh_k_rd = sh_k + bt * BTS;
            const float* sh_g_rd = sh_g + bt * BTS;
            const float* sh_q_rd = sh_q + bt * BTS;
            float* rs = gl_rs + v_start + t + bt * BTS * HEAD_DIM;
            #pragma unroll
            for (int i = 0; i < HEAD_DIM / 8 / SUBK; ++i)
            {
                #pragma unroll
                for (int j = 0; j < 8; ++j, rs += HEAD_DIM, sh_k_rd++, sh_g_rd++, sh_q_rd++)
                {
                    float state = *rs;
                    state = state * *sh_g_rd + *sh_k_rd * v * beta_h;
                    *rs = state;
                    v_out = v_out + *sh_q_rd * state;
                }
            }
            partial2[bt][t] = v_out;
        }
        __syncthreads();
        if (t < V_CHUNK_DIM && bt == 0)
        {
            float v_out = 0.0f;
            #pragma unroll
            for (int part = 0; part < SUBK; ++part)
                v_out = __fadd_rn(v_out, partial2[part][t]);
            out[t] = __float2bfloat16_rz(v_out * scale);
        }
        // Every thread reaches every barrier. Each state element has one owner
        // (slot, head, v_chunk, bt, t, k); distinct batch slots are checked below.
        __syncthreads();
        mixed_qkv += qkv_stride;
        g += size_t(num_v_heads) * HEAD_DIM;
        beta += num_v_heads;
        core_attn_out += size_t(num_v_heads) * HEAD_DIM;
    }
}

void check_tensor(const at::Tensor& tensor, const at::Tensor& input,
                  at::ScalarType dtype, const char* name)
{
    TORCH_CHECK(tensor.defined() && tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.device() == input.device(), name, " must match mixed_qkv device");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has unsupported dtype");
    TORCH_CHECK(tensor.layout() == at::kStrided && tensor.is_contiguous(), name, " must be contiguous strided");
}

bool overlaps(const at::Tensor& a, const at::Tensor& b)
{
    const auto ap = reinterpret_cast<uintptr_t>(a.const_data_ptr());
    const auto bp = reinterpret_cast<uintptr_t>(b.const_data_ptr());
    return ap < bp + b.nbytes() && bp < ap + a.nbytes();
}

void ordered_rule(const at::Tensor& mixed_qkv, const at::Tensor& g,
                  const at::Tensor& beta, at::Tensor recurrent_state,
                  at::Tensor out, int num_k_heads, int num_v_heads,
                  int k_head_dim, int v_head_dim,
                  const c10::optional<at::Tensor>& recurrent_slots, bool history)
{
    TORCH_CHECK(!history, "ordered KDA does not support history");
    TORCH_CHECK(k_head_dim == 128 && v_head_dim == 128, "ordered KDA requires 128x128 heads");
    TORCH_CHECK(num_k_heads > 0 && num_v_heads > 0 && num_v_heads <= 65535 &&
                num_v_heads % num_k_heads == 0, "invalid KDA head geometry");
    check_tensor(mixed_qkv, mixed_qkv, at::kBFloat16, "mixed_qkv");
    check_tensor(g, mixed_qkv, at::kFloat, "g");
    check_tensor(beta, mixed_qkv, at::kBFloat16, "beta");
    check_tensor(recurrent_state, mixed_qkv, at::kFloat, "recurrent_state");
    check_tensor(out, mixed_qkv, at::kBFloat16, "out");
    TORCH_CHECK(mixed_qkv.dim() == 3, "mixed_qkv must have rank 3");
    const int64_t bsz = mixed_qkv.size(0), seqlen = mixed_qkv.size(1);
    TORCH_CHECK(bsz > 0 && bsz <= std::numeric_limits<int>::max() &&
                seqlen > 0 && seqlen <= std::numeric_limits<int>::max(), "invalid batch/sequence geometry");
    TORCH_CHECK(mixed_qkv.size(2) == (2LL * num_k_heads + num_v_heads) * 128,
                "mixed_qkv packed width mismatch");
    TORCH_CHECK(g.sizes() == at::IntArrayRef({bsz, seqlen, num_v_heads, 128}),
                "g must be channelwise [batch, sequence, num_v_heads, 128]");
    TORCH_CHECK(beta.sizes() == at::IntArrayRef({bsz, seqlen, num_v_heads}), "beta shape mismatch");
    TORCH_CHECK(out.sizes() == at::IntArrayRef({bsz, seqlen, num_v_heads, 128}), "out shape mismatch");
    TORCH_CHECK(recurrent_state.dim() == 5 && recurrent_state.size(0) > 0 &&
                recurrent_state.size(1) >= 1 && recurrent_state.size(2) == num_v_heads &&
                recurrent_state.size(3) == 128 && recurrent_state.size(4) == 128,
                "state must be [num_slots, history_stride>=1, num_v_heads, 128, 128]");
    for (const auto* input : {&mixed_qkv, &g, &beta})
        TORCH_CHECK(!overlaps(recurrent_state, *input) && !overlaps(out, *input),
                    "KDA writable tensors must not alias inputs");
    TORCH_CHECK(!overlaps(recurrent_state, out), "KDA state and output must not overlap");

    const c10::cuda::CUDAGuard device_guard(mixed_qkv.device());
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    cudaStreamCaptureStatus capture;
    C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "ordered KDA does not support CUDA graph capture");
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(properties->major >= 8, "ordered KDA requires NVIDIA compute capability >= 8.0");
    const int* slots = nullptr;
    if (recurrent_slots.has_value())
    {
        const auto& slot_tensor = recurrent_slots.value();
        check_tensor(slot_tensor, mixed_qkv, at::kInt, "recurrent_slots");
        TORCH_CHECK(slot_tensor.dim() == 1 && slot_tensor.size(0) == bsz, "slots must be [batch]");
        TORCH_CHECK(!overlaps(recurrent_state, slot_tensor) && !overlaps(out, slot_tensor),
                    "slots must not alias writable tensors");
        // Synchronous validation only; all recurrence arithmetic remains CUDA.
        // Duplicate batch slots would introduce a cross-block state-update race.
        const auto host_slots = slot_tensor.to(at::kCPU);
        const auto* host = host_slots.data_ptr<int>();
        for (int64_t i = 0; i < bsz; ++i)
        {
            TORCH_CHECK(host[i] >= 0 && host[i] < recurrent_state.size(0), "slot index out of bounds");
            for (int64_t j = 0; j < i; ++j)
                TORCH_CHECK(host[i] != host[j], "duplicate batch slots are unsupported");
        }
        slots = slot_tensor.data_ptr<int>();
    }
    else
        TORCH_CHECK(recurrent_state.size(0) >= bsz, "insufficient implicit state slots");

    const int v_split = (bsz == 1 && num_v_heads <= 64) ? 4 : 1;
    const dim3 blocks(bsz, num_v_heads, v_split), threads(128, SUBK);
    const float scale = 1.0f / sqrtf(128.0f);
    #define ORDERED_ARGS \
        reinterpret_cast<const bfloat16*>(mixed_qkv.data_ptr()), g.data_ptr<float>(), \
        reinterpret_cast<const bfloat16*>(beta.data_ptr()), recurrent_state.data_ptr<float>(), \
        reinterpret_cast<bfloat16*>(out.data_ptr()), int(seqlen), num_k_heads, num_v_heads, \
        scale, slots, recurrent_state.size(1)
    if (v_split == 4)
        ordered_channelwise_128<4><<<blocks, threads, 0, stream>>>(ORDERED_ARGS);
    else
        ordered_channelwise_128<1><<<blocks, threads, 0, stream>>>(ORDERED_ARGS);
    #undef ORDERED_ARGS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

pybind11::dict runtime_identity()
{
    int runtime = 0, driver = 0;
    C10_CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    C10_CUDA_CHECK(cudaDriverGetVersion(&driver));
    pybind11::dict result;
    result["cuda_runtime_version"] = runtime;
    result["cuda_driver_version"] = driver;
    result["cuda_headers_version"] = CUDART_VERSION;
    return result;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cuda_recurrent_gated_delta_rule", &ordered_rule,
          pybind11::arg("mixed_qkv"), pybind11::arg("g"), pybind11::arg("beta"),
          pybind11::arg("recurrent_state"), pybind11::arg("out"),
          pybind11::arg("num_k_heads"), pybind11::arg("num_v_heads"),
          pybind11::arg("k_head_dim"), pybind11::arg("v_head_dim"),
          pybind11::arg("recurrent_slots"), pybind11::arg("history"));
    m.def("runtime_identity", &runtime_identity);
}
