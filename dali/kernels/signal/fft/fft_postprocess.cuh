// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_KERNELS_SIGNAL_FFT_FFT_POSTPROCESS_CUH_
#define DALI_KERNELS_SIGNAL_FFT_FFT_POSTPROCESS_CUH_

#include <cuda_runtime.h>
#include <math.h>
#include <memory>
#include "dali/core/geom/vec.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/signal/fft/fft_common.h"

namespace dali {
namespace kernels {
namespace signal {

namespace fft_postprocess {

using namespace fft;  // NOLINT

template <typename Out, typename In>
struct SampleDesc {
  Out *out;
  const In *in;
  ptrdiff_t out_stride;
  ptrdiff_t in_stride;
};

struct BlockDesc {
  int sample_idx;
  ivec2 start, end;
};

constexpr int kBlock = 32;

struct norm2square {
  DALI_HOST_DEV DALI_FORCEINLINE
  auto operator()(float2 c) const {
    return c.x * c.x + c.y * c.y;
  }

  DALI_FORCEINLINE
  auto operator()(complexf c) const {
    return (*this)(float2{c.real(), c.imag()});
  }
};

struct norm2 {
  DALI_HOST_DEV DALI_FORCEINLINE
  auto operator()(float2 c) const {
    return sqrtf(c.x * c.x + c.y * c.y);
  }

  DALI_FORCEINLINE
  auto operator()(complexf c) const {
    return (*this)(float2{c.real(), c.imag()});
  }
};

#if defined(__clang__) && defined(__CUDA__)

template <typename T>
__host__ const T& hostdev_max(const T& l, const  T& r) {
  return std::max(l, r);
}

template <typename T>
__device__ const T hostdev_max(const T& l, const  T& r) {
  return ::max(l, r);
}

#else

template <typename T>
__host__ __device__ const T hostdev_max(const T& l, const T& r) {
  return ::max(l, r);
}

#endif

struct power_dB {
  power_dB() = default;
  explicit power_dB(float cutoff_dB) {
    cutoff = pow(10, cutoff_dB / 10);
  }

  float mul = 3.01029995664f;  // log10(2)
  float cutoff = 1e-8;         // -80 dB

  DALI_HOST_DEV DALI_FORCEINLINE
  auto operator()(float2 c) const {
    return mul * log2f(hostdev_max(c.x * c.x + c.y * c.y, cutoff));
  }

  DALI_HOST DALI_FORCEINLINE
  auto operator()(complexf c) const {
    return (*this)(float2{c.real(), c.imag()});
  }
};


template <typename Out, typename In, typename Convert>
__global__ void ConvertTimeMajorSpectrogram(
      Out *out, int out_stride,
      const In *in, int in_stride, int nfft, int64_t nwindows,
      Convert convert = {}) {
  int64_t wnd = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (wnd >= nwindows)
    return;
  out += wnd * out_stride;
  in += wnd * in_stride;

  for (int i = threadIdx.x; i < nfft; i += 32)
    out[i] = convert(in[i]);
}

template <typename Out, typename In, typename Convert>
__global__ void ConvertTimeMajorSpectrogram_InPlaceDiffTypeSize(
      Out *out, int out_stride,
      const In *in, int in_stride, int nfft, int64_t nwindows,
      Convert convert = {}) {
  // A warp processes a whole row (transform) to ensure sequential processing
  // and thus enable in-place operation.

  int64_t wnd = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (wnd >= nwindows)
    return;
  out += wnd * out_stride;
  in += wnd * in_stride;

  // The loop starts at 0 (not threadIdx.x) to ensure there's no divergence which would
  // cause undefined behavior in __syncwarp() (or require complex mask calculation).
  for (int i = 0; i < nfft; i += 32) {
    int j = i + threadIdx.x;
    Out v = j < nfft ? convert(in[j]) : Out();
    __syncwarp();  // memory barrier for in-place execution with non-trivial aliasing
    if (j < nfft)
      out[j] = v;
  }
}

template <typename Out, typename In, typename Convert>
__global__ void ConvertTimeMajorSpectrogram_Flat(
      Out *out, const In *in, int64_t n,
      Convert convert = {}) {
  int64_t offset = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t i = offset; i < n; i += step)
    out[i] = convert(in[i]);
}

template <typename Out, typename In, typename Convert = identity>
__global__ void
__launch_bounds__(kBlock*kBlock)
TransposeBatch(
    const SampleDesc<Out, In> *samples,
    const BlockDesc *blocks,
    Convert convert = {}) {
  BlockDesc block = blocks[blockIdx.x];
  SampleDesc<Out, In> sample = samples[block.sample_idx];

  // Use double buffering to cut the use of __syncthreads by half.
  // Without the double buffering, we'd need to wait for tmp to be consumed
  // before being able to populate it again.
  __shared__ Out tmp[2][kBlock][kBlock + 1];

  ivec2 blk, in_pos, out_pos;
  int page = 0;

  for (blk.y = block.start.y; blk.y < block.end.y; blk.y += kBlock) {
    in_pos.y = blk.y + threadIdx.y;
    out_pos.x = blk.y + threadIdx.x;
    for (blk.x = block.start.x; blk.x < block.end.x; blk.x += kBlock) {
      in_pos.x = blk.x + threadIdx.x;
      out_pos.y = blk.x + threadIdx.y;

      if (all_coords(in_pos < block.end)) {
        Out v = convert(sample.in[in_pos.y * sample.in_stride + in_pos.x]);
        tmp[page][threadIdx.y][threadIdx.x] = v;
      }

      __syncthreads();
      if (out_pos.y < block.end.x && out_pos.x < block.end.y)
        sample.out[out_pos.y * sample.out_stride + out_pos.x] = tmp[page][threadIdx.x][threadIdx.y];

      page = 1-page;
    }
  }
}

template <typename Out, typename In>
class FFTPostprocess {
 public:
  virtual KernelRequirements Setup(KernelContext &ctx, const TensorListShape<2> &in_shape) = 0;

  virtual void Run(KernelContext &ctx,
                   const OutListGPU<Out, 2> &out,
                   const InListGPU<In, 2> &in) = 0;

  virtual ~FFTPostprocess() = default;
};

template <typename Out, typename In, typename Convert = identity>
class ConvertTimeMajorSpectrum : public FFTPostprocess<Out, In> {
 public:
  ConvertTimeMajorSpectrum() = default;
  explicit ConvertTimeMajorSpectrum(Convert convert) : convert_(convert) {}

  KernelRequirements Setup(KernelContext &ctx, const TensorListShape<2> &in_shape) override {
    KernelRequirements req;
    req.output_shapes = { in_shape };
    return req;
  }

  void Run(KernelContext &ctx, const OutListGPU<Out, 2> &out, const InListGPU<In, 2> &in) override {
    DALI_ENFORCE(out.num_samples() == in.num_samples(),
                 "Input and output must have the same number of samples.");

    int N = in.num_samples();
    if (N == 0)
      return;

    for (int i = 0; i < N; i++) {
      DALI_ENFORCE(in.shape[i][0] == out.shape[i][0],
                   "Number of transforms must match for corresponding input/output samples.");
      DALI_ENFORCE(in.shape[i][1] == in.shape[0][1],
                   "All input tensors must have the same width");
      DALI_ENFORCE(out.shape[i][1] == out.shape[0][1],
                   "All output tensors must have the same width");
    }

    int nfft = std::min(out.shape[0][1], in.shape[0][1]);

    auto launch_kernel = [&](Out *out_ptr, int64_t out_stride, const In *in_ptr, int64_t in_stride,
                             int64_t nwindows, int nfft) {
      dim3 blocks(div_ceil(nwindows, 8));
      dim3 threads(32, 8);

      if (static_cast<const void*>(out_ptr) == static_cast<const void *>(in_ptr) &&
          sizeof(Out) != sizeof(In)) {
        ConvertTimeMajorSpectrogram_InPlaceDiffTypeSize<<<blocks, threads, 0, ctx.gpu.stream>>>(
            out_ptr, out_stride, in_ptr, in_stride, nfft, nwindows, convert_);
      } else if (out_stride == in_stride) {
        int64_t n = nfft * nwindows;
        int block, grid;
        CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(
            &grid, &block,
            reinterpret_cast<void const *>(ConvertTimeMajorSpectrogram_Flat<Out, In, Convert>),
            0,  // shm_size,
            256));
        grid = std::min(static_cast<int>(div_ceil(n, block)), grid);
        ConvertTimeMajorSpectrogram_Flat<<<grid, block, 0, ctx.gpu.stream>>>(
            out_ptr, in_ptr, n, convert_);
      } else {
        ConvertTimeMajorSpectrogram<<<blocks, threads, 0, ctx.gpu.stream>>>(
            out_ptr, out_stride, in_ptr, in_stride, nfft, nwindows, convert_);
      }
      CUDA_CALL(cudaGetLastError());
    };

    if (in.is_contiguous() && out.is_contiguous()) {
      int64_t nwindows = 0;
      for (int i = 0; i < N; i++) {
        nwindows += in.shape[i][0];
      }
      launch_kernel(out.data[0], out.shape[0][1], in.data[0], in.shape[0][1], nwindows, nfft);
    } else {
      for (int i = 0; i < N; i++) {
        int nwindows = in.shape[i][0];
        launch_kernel(out.data[i], out.shape[i][1], in.data[i], in.shape[i][1], nwindows, nfft);
      }
    }
  }

 private:
  Convert convert_;
};

/**
 * A specialized kernel that tranposes a frame-major spectrogrum to frequency-major one,
 * with an option to apply some pointwise transform on the data (e.g. complex magnitude).
 *
 * Constraints: all input samples must have the same number of spectrum bins
 */
template <typename Out, typename In = Out, typename Convert = identity>
class ToFreqMajorSpectrum : public FFTPostprocess<Out, In> {
 public:
  using SampleDesc = fft_postprocess::SampleDesc<Out, In>;

  ToFreqMajorSpectrum() = default;
  explicit ToFreqMajorSpectrum(Convert convert) : convert_(convert) {}

  KernelRequirements Setup(KernelContext &ctx, const TensorListShape<2> &in_shape) override {
    KernelRequirements req;
    req.output_shapes.resize(1);
    auto &out_shape = req.output_shapes[0];

    int N = in_shape.num_samples();
    if (N == 0) {
      nblocks_ = 0;
      return req;
    }
    int nblocks = 0;
    int nfft = in_shape[0][1];

    int64_t total_windows = 0;
    out_shape.resize(N, 2);
    for (int i = 0; i < N; i++) {
      TensorShape<2> sample_shape = in_shape[i];
      DALI_ENFORCE(sample_shape[1] == nfft, "All inputs must have the same number of FFT bins");
      out_shape.set_tensor_shape(i, { sample_shape[1], sample_shape[0] });

      total_windows += sample_shape[0];
    }

    // if the data is big, use larger blocks
    block_size_ = kBlock * (1 + total_windows / (100000 * kBlock));

    for (int i = 0; i < N; i++) {
      int nwindows = in_shape[i][0];
      nblocks += div_ceil(nwindows, block_size_);
    }
    nblocks_ = nblocks;
    return req;
  }

  void Run(KernelContext &ctx, const OutListGPU<Out, 2> &out, const InListGPU<In, 2> &in) override {
    int N = in.num_samples();
    if (!N)
      return;

    SampleDesc *cpu_samples = ctx.scratchpad->AllocateHost<SampleDesc>(N);
    BlockDesc *cpu_blocks = ctx.scratchpad->AllocateHost<BlockDesc>(nblocks_);

    int nfft = in.shape[0][1];

    int b = 0;
    for (int i = 0; i < N; i++) {
      TensorShape<2> sample_shape = in.shape[i];
      assert(sample_shape[1] == nfft);
      int nwindows = sample_shape[0];
      cpu_samples[i] = {
        out.data[i],
        in.data[i],
        nwindows,  // output stride - a row contains all windows
        nfft       // input stride  - a row contains all frequencies
      };
      for (int start = 0; start < nwindows; start += block_size_) {
        int end = std::min(start + block_size_, nwindows);
        assert(b < nblocks_);
        cpu_blocks[b++] = { i, ivec2(0, start), ivec2(nfft, end) };
      }
    }
    assert(b == nblocks_);

    SampleDesc *gpu_samples;
    BlockDesc *gpu_blocks;

    std::tie(gpu_samples, gpu_blocks) = ctx.scratchpad->ToContiguousGPU(
        ctx.gpu.stream, make_span(cpu_samples, N), make_span(cpu_blocks, nblocks_));

    dim3 blockDim(kBlock, kBlock, 1);
    TransposeBatch<<<nblocks_, blockDim, 0, ctx.gpu.stream>>>(gpu_samples, gpu_blocks, convert_);
    CUDA_CALL(cudaGetLastError());
  }

 private:
  Convert convert_;
  int nblocks_ = 0;
  int block_size_ = kBlock;
};

using ToFreqMajorComplexSpectrum = ToFreqMajorSpectrum<float2, float2>;
using ToFreqMajorPowerSpectrum = ToFreqMajorSpectrum<float, float2, norm2square>;
using ToFreqMajorAmplitudeSpectrum = ToFreqMajorSpectrum<float, float2, norm2>;
using ToFreqMajorDecibelSpectrum = ToFreqMajorSpectrum<float, float2, power_dB>;

inline std::unique_ptr<FFTPostprocess<float2, float2>> GetSTFTPostprocessor(bool time_major) {
  if (time_major)
    return std::make_unique<ConvertTimeMajorSpectrum<float2, float2>>();
  else
    return std::make_unique<ToFreqMajorComplexSpectrum>();
}

template <typename Convert>
std::unique_ptr<FFTPostprocess<float, float2>> GetSpectrogramPostprocessor(
    bool time_major) {
  if (time_major)
    return std::make_unique<ConvertTimeMajorSpectrum<float, float2, Convert>>();
  else
    return std::make_unique<ToFreqMajorSpectrum<float, float2, Convert>>();
}

inline std::unique_ptr<FFTPostprocess<float, float2>> GetSpectrogramPostprocessor(
    bool time_major,
    FftSpectrumType type) {
  switch (type) {
    case FFT_SPECTRUM_MAGNITUDE:
      return GetSpectrogramPostprocessor<norm2>(time_major);
    case FFT_SPECTRUM_POWER:
      return GetSpectrogramPostprocessor<norm2square>(time_major);
    case FFT_SPECTRUM_POWER_DECIBELS:
      return GetSpectrogramPostprocessor<power_dB>(time_major);
    default:
      assert(!"Unsupported spectrum type");
      return nullptr;
  }
}

}  // namespace fft_postprocess

}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_FFT_FFT_POSTPROCESS_CUH_
