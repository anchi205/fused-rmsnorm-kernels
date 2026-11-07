// Fused residual-add + RMSNorm: hand-written CUDA forward and backward.
//
//   s = x + residual
//   y = round(s * rstd) * w,    rstd = rsqrt(mean(s^2) + eps)
//
// Optimizations, each of which can be seen in Nsight Compute (see profile/):
//   1. Coalesced, vectorized global access. Thread t loads 16-byte vector t,
//      then t + blockDim, and so on. A warp reads 512 contiguous bytes per
//      instruction (LDG.E.128), so every 32-byte DRAM sector fetched is used.
//   2. Register row cache. Each row is read from DRAM exactly once. s stays in
//      registers between the reduction and the normalize/scale pass.
//   3. Warp-level reductions. __shfl_xor_sync butterfly within a warp, then
//      one shared-memory slot per warp, then a final warp shuffle. That is
//      2 __syncthreads per reduction instead of log2(blockDim).
//   4. Fused backward with a persistent grid. dw is accumulated in registers
//      across rows and written as [G, N] fp32 partials, with no atomics. A
//      small coalesced column-sum kernel finishes the job.
//
// fwd_naive_kernel is a deliberately bad baseline (strided per-thread chunks,
// scalar loads, shared-memory tree reduction, re-reading from global) so the
// effect of these changes can be measured, not just claimed.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

// Each thread caches ITERS * VEC fp32 elements of its row in registers. ITERS
// is a template parameter chosen at launch from N (the smallest value that
// fits the row in one block), so register use scales with the real row width,
// not the worst case. v1 used a fixed 32-element cache; under
// __launch_bounds__(1024), i.e. 64 regs/thread, ptxas spilled ~470 B/thread
// in the backward. See README, "the register-spill bug".
constexpr int kFwdMaxThreads = 1024;  // forward: ~3 live floats/elem
constexpr int kBwdMaxThreads = 512;   // backward: 3 live arrays -> allow 128 regs/thread
constexpr int kMaxElemsPerThread = 32;
constexpr int kMaxHidden = kFwdMaxThreads * kMaxElemsPerThread;  // 32768

template <typename T, int VEC>
struct alignas(sizeof(T) * VEC) Vec {
  T v[VEC];
};

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Block-wide sum; blockDim.x must be a multiple of 32. Result broadcast to all.
__device__ __forceinline__ float block_sum(float v, float* smem /*[32]*/) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int nwarps = blockDim.x >> 5;
  v = warp_sum(v);
  if (lane == 0) smem[wid] = v;
  __syncthreads();
  if (wid == 0) {
    v = (lane < nwarps) ? smem[lane] : 0.f;
    v = warp_sum(v);
    if (lane == 0) smem[0] = v;
  }
  __syncthreads();
  v = smem[0];
  __syncthreads();  // smem may be reused by the caller's next reduction
  return v;
}

// ---------------------------------------------------------------------------
// Forward: one block per row.
// ---------------------------------------------------------------------------
template <typename T, int VEC, int ITERS>
__global__ void __launch_bounds__(kFwdMaxThreads)
fwd_kernel(const T* __restrict__ x, const T* __restrict__ res,
           const T* __restrict__ w, T* __restrict__ y, T* __restrict__ s_out,
           float* __restrict__ rstd_out, int N, float eps) {
  using V = Vec<T, VEC>;
  __shared__ float red[32];

  const int64_t row = blockIdx.x;
  const int nvec = N / VEC;
  const V* xv = reinterpret_cast<const V*>(x + row * N);
  const V* rv = reinterpret_cast<const V*>(res + row * N);
  const V* wv = reinterpret_cast<const V*>(w);
  V* sv = reinterpret_cast<V*>(s_out + row * N);
  V* yv = reinterpret_cast<V*>(y + row * N);

  float sreg[ITERS][VEC];
  float sumsq = 0.f;

#pragma unroll
  for (int it = 0; it < ITERS; ++it) {
    const int i = threadIdx.x + it * blockDim.x;
    if (i < nvec) {
      V a = xv[i], b = rv[i], o;
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        // Round to T first. Eager PyTorch materializes s in T.
        T sk = static_cast<T>(static_cast<float>(a.v[k]) + static_cast<float>(b.v[k]));
        o.v[k] = sk;
        const float f = static_cast<float>(sk);
        sreg[it][k] = f;
        sumsq += f * f;
      }
      sv[i] = o;
    }
  }

  const float rstd = rsqrtf(block_sum(sumsq, red) / N + eps);
  if (threadIdx.x == 0) rstd_out[row] = rstd;

#pragma unroll
  for (int it = 0; it < ITERS; ++it) {
    const int i = threadIdx.x + it * blockDim.x;
    if (i < nvec) {
      V wk = wv[i], o;
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        const float n = static_cast<float>(static_cast<T>(sreg[it][k] * rstd));
        o.v[k] = static_cast<T>(n * static_cast<float>(wk.v[k]));
      }
      yv[i] = o;
    }
  }
}

// Naive baseline: each thread owns a *contiguous chunk*, so at every step the
// 32 lanes of a warp touch 32 different cache lines (uncoalesced). Scalar
// loads; shared-memory tree reduction with a __syncthreads per level; s is
// re-read from global memory in the second pass.
template <typename T>
__global__ void fwd_naive_kernel(const T* __restrict__ x, const T* __restrict__ res,
                                 const T* __restrict__ w, T* __restrict__ y,
                                 T* __restrict__ s_out, float* __restrict__ rstd_out,
                                 int N, float eps) {
  extern __shared__ float sh[];
  const int64_t base = (int64_t)blockIdx.x * N;
  const int chunk = (N + blockDim.x - 1) / blockDim.x;
  const int start = threadIdx.x * chunk;
  const int end = min(N, start + chunk);
  float ss = 0.f;
  for (int j = start; j < end; ++j) {
    T sk = static_cast<T>(static_cast<float>(x[base + j]) + static_cast<float>(res[base + j]));
    s_out[base + j] = sk;
    const float f = static_cast<float>(sk);
    ss += f * f;
  }
  sh[threadIdx.x] = ss;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
    __syncthreads();
  }
  const float rstd = rsqrtf(sh[0] / N + eps);
  if (threadIdx.x == 0) rstd_out[blockIdx.x] = rstd;
  for (int j = start; j < end; ++j) {
    const float n = static_cast<float>(static_cast<T>(static_cast<float>(s_out[base + j]) * rstd));
    y[base + j] = static_cast<T>(n * static_cast<float>(w[j]));
  }
}

// ---------------------------------------------------------------------------
// Backward: persistent grid, each block loops over rows blockIdx.x + k*gridDim.x
//   g = dy * w;  dot = sum(g * s)
//   dx = rstd * g - s * rstd^3 * dot / N  (+ ds_out)
//   dw_partial[block] += dy * s * rstd
// ---------------------------------------------------------------------------
template <typename T, int VEC, int ITERS, bool HAS_DSO>
__global__ void __launch_bounds__(kBwdMaxThreads)
bwd_kernel(const T* __restrict__ dy, const T* __restrict__ dso,
           const T* __restrict__ s, const T* __restrict__ w,
           const float* __restrict__ rstd_in, T* __restrict__ dx,
           float* __restrict__ dw_partial, int M, int N) {
  using V = Vec<T, VEC>;
  __shared__ float red[32];
  const int nvec = N / VEC;
  const V* wv = reinterpret_cast<const V*>(w);

  // dw accumulates across every row this block handles. w itself is re-read
  // each row rather than pinned in registers: it is only N elements, so it
  // stays hot in L1/L2, and that frees ITERS*VEC registers.
  float dwacc[ITERS][VEC];
#pragma unroll
  for (int it = 0; it < ITERS; ++it)
#pragma unroll
    for (int k = 0; k < VEC; ++k) dwacc[it][k] = 0.f;

  for (int64_t row = blockIdx.x; row < M; row += gridDim.x) {
    const V* dyv = reinterpret_cast<const V*>(dy + row * N);
    const V* sv = reinterpret_cast<const V*>(s + row * N);
    const float rstd = rstd_in[row];
    float greg[ITERS][VEC], sreg[ITERS][VEC];
    float dot = 0.f;

#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      const int i = threadIdx.x + it * blockDim.x;
      if (i < nvec) {
        V a = dyv[i], b = sv[i], wk = wv[i];
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
          const float d = static_cast<float>(a.v[k]);
          const float sk = static_cast<float>(b.v[k]);
          const float g = d * static_cast<float>(wk.v[k]);
          greg[it][k] = g;
          sreg[it][k] = sk;
          dot += g * sk;
          dwacc[it][k] += d * sk * rstd;
        }
      }
    }

    dot = block_sum(dot, red);
    const float c = rstd * rstd * rstd * dot / N;

    V* dxv = reinterpret_cast<V*>(dx + row * N);
    const V* dsov = HAS_DSO ? reinterpret_cast<const V*>(dso + row * N) : nullptr;
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      const int i = threadIdx.x + it * blockDim.x;
      if (i < nvec) {
        V o, e;
        if (HAS_DSO) e = dsov[i];
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
          float v = rstd * greg[it][k] - c * sreg[it][k];
          if (HAS_DSO) v += static_cast<float>(e.v[k]);
          o.v[k] = static_cast<T>(v);
        }
        dxv[i] = o;
      }
    }
  }

  float* out = dw_partial + (int64_t)blockIdx.x * N;
#pragma unroll
  for (int it = 0; it < ITERS; ++it) {
    const int i = threadIdx.x + it * blockDim.x;
    if (i < nvec) {
#pragma unroll
      for (int k = 0; k < VEC; ++k) out[i * VEC + k] = dwacc[it][k];
    }
  }
}

// dw[c] = sum_g partial[g, c]. Adjacent threads read adjacent columns (coalesced).
template <typename T>
__global__ void dw_reduce_kernel(const float* __restrict__ partial, T* __restrict__ dw,
                                 int G, int N) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= N) return;
  float acc = 0.f;
  for (int g = 0; g < G; ++g) acc += partial[(int64_t)g * N + c];
  dw[c] = static_cast<T>(acc);
}

template <int I> using IC = std::integral_constant<int, I>;

// Pick the smallest ITERS (vectors per thread) with nvec <= MAXT * ITERS, and
// call f(IC<ITERS>{}, threads). Fewer iterations means fewer live registers.
template <int VEC, int MAXT, int MAXE, typename F>
void dispatch_iters(int nvec, F&& f) {
  auto go = [&](auto iters) {
    constexpr int I = decltype(iters)::value;
    if constexpr (I * VEC <= MAXE) {
      const int per = (nvec + I - 1) / I;
      f(iters, std::max(32, (per + 31) / 32 * 32));
    } else {
      TORCH_CHECK(false, "hidden size too large for this vector width");
    }
  };
  if (nvec <= MAXT * 1) go(IC<1>{});
  else if (nvec <= MAXT * 2) go(IC<2>{});
  else if (nvec <= MAXT * 4) go(IC<4>{});
  else if (nvec <= MAXT * 8) go(IC<8>{});
  else if (nvec <= MAXT * 16) go(IC<16>{});
  else if (nvec <= MAXT * 32) go(IC<32>{});
  else if (nvec <= MAXT * 64) go(IC<64>{});
  else TORCH_CHECK(false, "hidden size too large");
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

void check_inputs(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == at::kFloat || t.scalar_type() == at::kHalf ||
                  t.scalar_type() == at::kBFloat16,
              name, " must be float32, float16 or bfloat16");
}

}  // namespace

// ---------------------------------------------------------------------------
// Host entry points
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> fused_add_rmsnorm_fwd(torch::Tensor x, torch::Tensor residual,
                                                 torch::Tensor weight, double eps,
                                                 bool naive) {
  check_inputs(x, "x");
  check_inputs(residual, "residual");
  check_inputs(weight, "weight");
  TORCH_CHECK(x.sizes() == residual.sizes(), "x and residual shape mismatch");
  TORCH_CHECK(x.scalar_type() == residual.scalar_type() &&
                  x.scalar_type() == weight.scalar_type(),
              "dtype mismatch");
  const int N = x.size(-1);
  TORCH_CHECK(weight.numel() == N, "weight must have shape [hidden]");
  TORCH_CHECK(N <= kMaxHidden, "hidden size too large: ", N);
  const int64_t M = x.numel() / N;

  const at::cuda::CUDAGuard guard(x.device());
  auto y = torch::empty_like(x);
  auto s = torch::empty_like(x);
  auto rstd = torch::empty({M}, x.options().dtype(at::kFloat));
  if (M == 0) return {y, s, rstd};
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "fused_add_rmsnorm_fwd", [&] {
    const auto* xp = x.data_ptr<scalar_t>();
    const auto* rp = residual.data_ptr<scalar_t>();
    const auto* wp = weight.data_ptr<scalar_t>();
    auto* yp = y.data_ptr<scalar_t>();
    auto* sp = s.data_ptr<scalar_t>();
    auto* rs = rstd.data_ptr<float>();
    if (naive) {
      const int threads = 256;
      fwd_naive_kernel<scalar_t><<<M, threads, threads * sizeof(float), stream>>>(
          xp, rp, wp, yp, sp, rs, N, (float)eps);
      return;
    }
    constexpr int VEC = 16 / sizeof(scalar_t);
    const bool vec_ok = (N % VEC == 0) && aligned16(xp) && aligned16(rp) && aligned16(wp) &&
                        aligned16(yp) && aligned16(sp);
    auto launch = [&](auto vec_tag) {
      constexpr int V = decltype(vec_tag)::value;
      dispatch_iters<V, kFwdMaxThreads, kMaxElemsPerThread>(N / V, [&](auto iters, int threads) {
        fwd_kernel<scalar_t, V, decltype(iters)::value><<<M, threads, 0, stream>>>(
            xp, rp, wp, yp, sp, rs, N, (float)eps);
      });
    };
    vec_ok ? launch(IC<VEC>{}) : launch(IC<1>{});
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, s, rstd};
}

std::vector<torch::Tensor> fused_add_rmsnorm_bwd(torch::Tensor dy, c10::optional<torch::Tensor> ds_out,
                                                 torch::Tensor s, torch::Tensor weight,
                                                 torch::Tensor rstd) {
  check_inputs(dy, "dy");
  check_inputs(s, "s");
  check_inputs(weight, "weight");
  TORCH_CHECK(rstd.scalar_type() == at::kFloat && rstd.is_contiguous(), "rstd must be contiguous fp32");
  const bool has_dso = ds_out.has_value() && ds_out->defined();
  if (has_dso) check_inputs(*ds_out, "ds_out");
  const int N = s.size(-1);
  const int64_t M = s.numel() / N;
  TORCH_CHECK(N <= kMaxHidden, "hidden size too large: ", N);

  const at::cuda::CUDAGuard guard(s.device());
  auto dx = torch::empty_like(s);
  auto dw = torch::empty_like(weight);
  if (M == 0) return {dx, dw.zero_()};
  auto stream = at::cuda::getCurrentCUDAStream();
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  // ~1 block per SM, >= 8 rows per block, >= 16 blocks (see bwd_grid() in
  // triton_kernels.py and bench/bwd_grid_sweep.py). Each extra block costs an
  // [N] fp32 partial write + re-read.
  int64_t G64 = std::min<int64_t>({M, (int64_t)sms, std::max<int64_t>(16, (M + 7) / 8)});
  if (const char* e = std::getenv("FUSED_NORM_BWD_GRID"))  // override for sweeps
    G64 = std::max<int64_t>(1, std::min<int64_t>(M, std::atoi(e)));
  const int G = (int)G64;
  auto partial = torch::empty({G, N}, s.options().dtype(at::kFloat));

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, s.scalar_type(), "fused_add_rmsnorm_bwd", [&] {
    const auto* dyp = dy.data_ptr<scalar_t>();
    const scalar_t* dsop = has_dso ? ds_out->data_ptr<scalar_t>() : nullptr;
    const auto* sp = s.data_ptr<scalar_t>();
    const auto* wp = weight.data_ptr<scalar_t>();
    auto* dxp = dx.data_ptr<scalar_t>();
    constexpr int VEC = 16 / sizeof(scalar_t);
    const bool vec_ok = (N % VEC == 0) && aligned16(dyp) && aligned16(sp) && aligned16(wp) &&
                        aligned16(dxp) && (!has_dso || aligned16(dsop));
    auto launch = [&](auto vec_tag, auto dso_tag) {
      constexpr int V = decltype(vec_tag)::value;
      constexpr bool D = decltype(dso_tag)::value;
      // Rows wider than 512*32 elements fall back to 64 elems/thread (some spilling).
      dispatch_iters<V, kBwdMaxThreads, 2 * kMaxElemsPerThread>(N / V, [&](auto iters, int threads) {
        bwd_kernel<scalar_t, V, decltype(iters)::value, D><<<G, threads, 0, stream>>>(
            dyp, dsop, sp, wp, rstd.data_ptr<float>(), dxp, partial.data_ptr<float>(), (int)M, N);
      });
    };
    if (vec_ok) {
      has_dso ? launch(IC<VEC>{}, std::true_type{}) : launch(IC<VEC>{}, std::false_type{});
    } else {
      has_dso ? launch(IC<1>{}, std::true_type{}) : launch(IC<1>{}, std::false_type{});
    }
    dw_reduce_kernel<scalar_t><<<(N + 255) / 256, 256, 0, stream>>>(
        partial.data_ptr<float>(), dw.data_ptr<scalar_t>(), G, N);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx, dw};
}
