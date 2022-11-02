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

#include <vector>
#include <memory>
#include "dali/kernels/audio/mel_scale/mel_filter_bank_gpu.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/core/cuda_rt_utils.h"
#include "dali/core/static_switch.h"

namespace dali {
namespace kernels {
namespace audio {

const int kBlockDim2 = 32;

template <typename T>
struct BlockDesc {
  BlockDesc() = default;
  BlockDesc(int64_t out_offset, int64_t in_offset, int64_t start_window, int64_t nwindows) {
    this->out_offset = out_offset;
    this->in_offset  = in_offset;
    this->out_frame  = nullptr;
    this->in_frame   = nullptr;

    this->start_window = start_window;
    this->frame_nwindows = nwindows;
  }

  void SetBasePointers(T *out, const T *in) {
    out_frame = out + out_offset;
    in_frame  = in  + in_offset;
  }

  int64_t   out_offset;
  int64_t   in_offset;
  T        *out_frame;
  const T  *in_frame;

  int64_t start_window;
  int64_t frame_nwindows;
};

template <typename T>
__device__ T calcMel(const T* in_frame, int mel_bin,
                     const T *weights_down, const int *interval_ends,
                     int fft_stride, int fft_shift,
                     T norm_factor) {
  T out = 0;
  int fftbin = interval_ends[mel_bin];
  int fftbin_end = interval_ends[mel_bin + 1];
  const T *in =  in_frame + fftbin * fft_stride + fft_shift;
  for (; fftbin < fftbin_end; ++fftbin, in += fft_stride) {
    auto weight_up = T(1) - weights_down[fftbin];
    weight_up *= norm_factor;
    out += *in * weight_up;
  }
  fftbin_end = interval_ends[mel_bin + 2];
  in =  in_frame + fftbin * fft_stride + fft_shift;
  for (; fftbin < fftbin_end; ++fftbin, in += fft_stride) {
    auto weight_down = weights_down[fftbin];
    weight_down *= norm_factor;
    out += *in * weight_down;
  }
  return out;
}

// For layouts where the frequency is not the innermost dimension, data is flattened into
// 3 dimensions - frame, frequency, time
// Every frame is treated as independent two-dimensional sample
template <typename T>
__global__ void MelFilterBankKernel(const BlockDesc<T> *block_desc,
                                    const T *weights_down, const int *interval_ends,
                                    bool normalize, const T *norm_factors,
                                    int nmel) {
  auto block_id = blockIdx.x;
  const T *in_frame = block_desc[block_id].in_frame;
  T *out_frame = block_desc[block_id].out_frame;
  int mel_bin = blockIdx.y * kBlockDim2 + threadIdx.y;

  if (mel_bin >= nmel)
    return;

  int64_t window = block_desc[block_id].start_window + threadIdx.x;
  int64_t nwindows = block_desc[block_id].frame_nwindows;

  if (window >= nwindows)
    return;

  T *out = out_frame + mel_bin * nwindows + window;
  T norm_factor = (normalize) ? norm_factors[mel_bin] : 1;
  *out = calcMel(in_frame, mel_bin,
                 weights_down, interval_ends,
                 nwindows, window, norm_factor);
}

/**
 * Mel filter bank for inner FFT
 *
 * The algorithm processes the data in shared memory.
 * Shared memory layout:
 *
 * ```
 * input:
 * win[0]:  f0 f1 f2 ... fF <pad>
 * win[1]:  f0 f1 f2 ... fF <pad>
 * ...
 * win[H]:  f0 f1 f2 ... fF <pad>
 * output:
 * win[0]:  m0 m1 ... mM <pad>
 * win[1]:  m0 m1 ... mM <pad>
 * ...
 * win[H]:  m0 m1 ... mM <pad>
 * ```
 *
 * where:
 * F = max. number of frequencies; fixed to 48
 * M = number of Mel filters
 * H = height of shared memory block
 *
 * Each CUDA block processes H windows and all frequencies.
 * The value H is a power of two from 1 to 32, depending on the number of Mel filter banks.
 * The maximum number of Mel filters is around 10000 for fp32 and about half of that for fp64.
 * For common numbers of Mel filters (40-100), the H is at its maximum value (32).
 *
 * The kernel traverses the data vertically with stride H and processes blocks of windows.
 * Each block is subdivided into horizontal blocks of frequencies, which are loaded from
 * global to shared memory.
 * After a block is loaded, the thread indexing is tranposed and now each thread in a warp
 * processes a single frequency. There's a mapping from FFT bins to Mel bins, which determines
 * to which Mel bins given frequency will contribute. Because multiple frequencies can contribute
 * to one Mel bin, it's possible that multiple warps will try to accumulate the value to the
 * same Mel bin. To avoid race conditions, shared memory atomicAdd is used.
 *
 * The maximum number of frequencies is set to 48 to avoid a situation where we have, for example,
 * 257 FFT bins and the last horizontal block would be very thin. Now it's done so that if the
 * number of remaining FFT bins is <= 48, it's processed at once. Otherise, 32 bins are taken and
 * the reaminder is processed in the next iteration.
 *
 */
namespace mel_inner_fft {

static constexpr int kMaxInnerFftFreqs = 48;

template <typename T>
DALI_HOST_DEV
constexpr int shm_in_size(int shm_height) {
  const int alignment = 32 / sizeof(T);
  const int sh_in_stride = kMaxInnerFftFreqs + 1;
  return align_up(sh_in_stride * shm_height, alignment) * sizeof(T);
}

template <typename T>
DALI_HOST_DEV
constexpr int shm_out_stride(int mel) {
  const int alignment = 32 / sizeof(T);
  return align_up(mel, alignment) + 1;
}

template <typename T>
DALI_HOST_DEV
constexpr int shm_out_size(int shm_height, int mel) {
  return shm_out_stride<T>(mel) * shm_height * sizeof(T);
}

/**
 * @brief Implements MelFilterbank with inner FFT
 *
 * @tparam shm_height the number of data rows stored in shared memory; must be 1, 2, 4, 8, 16 or 32
 * @tparam T the data type (float or double)
 */
template <int shm_height, typename T>
struct MelFilterBankInnerFft {
  BlockDesc<T> block_desc;
  const T *__restrict__ weights_down;
  const T *__restrict__ weights_up;
  const int *__restrict__ bin_down;
  int fft_lo, fft_hi;
  int nmel, nfft;
  char *shm;

  static const int max_freqs = kMaxInnerFftFreqs;
  static const int sh_in_stride = max_freqs + 1;
  int sh_out_stride = 0;

  __device__ T &shm_in(int window, int freq) const {
    return reinterpret_cast<T*>(shm)[sh_in_stride * window + freq];
  }

  __device__ T &shm_out(int window, int mel) const {
    const int shm_out_offset = shm_in_size<T>(shm_height);
    return reinterpret_cast<T*>(shm + shm_out_offset)[window * sh_out_stride + mel];
  }

  __device__ void Run() {
    sh_out_stride = shm_out_stride<T>(nmel);

    bool first = true;

    for (int start_window = 0;
         start_window < block_desc.frame_nwindows;
         start_window += shm_height) {
      if (first)
        first = false;
      else
        __syncthreads();

      int fft_end;
      // The inner loop accumulates Mel coefficients from FFT spectrum
      ClearMel();
      for (int fft_start = fft_lo; fft_start < fft_hi; fft_start = fft_end) {
        // avoid last thin slice - use up to max_freqs witdth, if it fits
        if (fft_hi - fft_start <= max_freqs)
          fft_end = fft_hi;
        else
          fft_end = fft_start + 32;
        // Load a block of FFT coefficients into shared memory
        LoadFrequencyBlock(start_window, fft_start, fft_end);
        __syncthreads();
        // Go over FFT coefficients and accumulate Mel values in shared memory
        ProcessFrequencyBlock(fft_start, fft_end);
        __syncthreads();
      }
      // Store the final Mel values in global memory
      StoreMel(start_window);
    }
  }

  __device__ void LoadFrequencyBlock(int start_window, int fft_start, int fft_end) {
    int bh = blockDim.y;
    const T *in = block_desc.in_frame;
    // block height might be smaller than shm_height, so we need to iterate
    for (int y = threadIdx.y; y < shm_height; y += bh) {
      int window = start_window + y;
      if (window < block_desc.frame_nwindows) {
        for (int x = threadIdx.x; x < fft_end - fft_start; x += 32)
          shm_in(y, x) = in[window * nfft + x + fft_start];
      } else {
        for (int x = threadIdx.x; x < fft_end - fft_start; x += 32)
          shm_in(y, x) = 0;
      }
    }
  }

  __device__ void ClearMel() {
    for (int y = threadIdx.y; y < shm_height; y += blockDim.y) {
      for (int x = threadIdx.x; x < nmel; x += blockDim.x)
        shm_out(y, x) = 0;
    }
  }

  __device__ void ProcessFrequencyBlock(int fft_start, int fft_end) {
    const int cols_per_warp = (32 / shm_height);
    int bh = blockDim.y * cols_per_warp;
    int wnd = threadIdx.x % shm_height;
    for (int fft = fft_start + threadIdx.y * cols_per_warp + threadIdx.x / shm_height;
         fft < fft_end;
         fft += bh) {
      int bin0 = bin_down[fft];
      int bin1 = bin0 + 1;
      T value = shm_in(wnd, fft - fft_start);

      // bin0 is accumulated along the downward slope, bin1 along the upward slope.
      // Because of that, the very first bin starts with invalid bin0 (-1), because we start
      // with upward slope.
      if (bin0 >= 0)
        atomicAdd(&shm_out(wnd, bin0), value * weights_down[fft]);

      // Analogously, the last value of bin1 is invalid (nmel), because we will accumulate with
      // the downward slop only.
      if (bin1 < nmel)
        atomicAdd(&shm_out(wnd, bin1), value * weights_up[fft]);
    }
  }

  __device__ void StoreMel(int start_window) {
    int end_window = cuda_min<int>(start_window + shm_height, block_desc.frame_nwindows);
    int n = (end_window - start_window) * nmel;
    T *out = block_desc.out_frame;
    // Use flattened indices to have contiguous stores - there can be very few Mel bins, so
    // we can't affort to waste a significant part of warp's bandwidth.
    for (int idx = blockDim.x * threadIdx.y + threadIdx.x;
         idx < n;
         idx += blockDim.x * blockDim.y) {
      // idx is 32-bit, so div/mod isn't that bad - no need for fast_div
      int w = idx / nmel;
      int m = idx % nmel;
      out[start_window * nmel + idx] = shm_out(w, m);
    }
  }
};

}  // namespace mel_inner_fft

template <int shm_height, typename T>
__global__ void MelFilterBankKernelInnerFft(const BlockDesc<T> *block_descs,
                                            const T *__restrict__ weights_down,
                                            const T *__restrict__ weights_up,
                                            const int *__restrict__ bin_down,
                                            int fft_lo, int fft_hi,
                                            int nmel, int nfft) {
  extern __shared__ char shm_arena[];
  mel_inner_fft::MelFilterBankInnerFft<shm_height, T> fb = {
    block_descs[blockIdx.x],
    weights_down,
    weights_up,
    bin_down,
    fft_lo, fft_hi,
    nmel, nfft,
    shm_arena,
  };
  fb.Run();
}


template <typename T>
class MelFilterBankGpu<T>::Impl : public MelFilterImplBase<T> {
 public:
  template <typename MelScale>
  Impl(MelScale mel_scale, const MelFilterBankArgs &args) :
      MelFilterImplBase<T>(mel_scale, args),
      interval_ends_(args.nfilter + 2) {
    double mel = mel_low_ + mel_delta_;
    interval_ends_[0] = fftbin_start_;
    interval_ends_[args.nfilter + 1] = fftbin_end_;

    for (int interval = 1; interval < args_.nfilter + 1; interval++, mel += mel_delta_) {
      double freq = mel_scale.mel_to_hz(mel);
      interval_ends_[interval] = std::ceil(freq / hz_step_);
    }

    weights_down_norm_.clear();
    weights_down_norm_.resize(fftbin_size_);
    weights_up_norm_.clear();
    weights_up_norm_.resize(fftbin_size_);
    fft2mel_.clear();
    fft2mel_.resize(fftbin_size_, -1);

    for (int f = fftbin_start_, interval = -1; f < fftbin_end_; f++) {
      while (interval + 2 < static_cast<int>(interval_ends_.size()) &&
             interval_ends_[interval + 2] <= f)
        interval++;
      assert(interval >= -1 && interval <= args.nfilter);
      bool first = interval == -1;
      bool last = interval == args.nfilter;

      fft2mel_[f] = interval;

      double w = weights_down_[f];
      if (args.normalize) {
        if (!first)
          weights_down_norm_[f] = w * norm_factors_[interval];
        if (!last)
          weights_up_norm_[f] = (1 - w) * norm_factors_[interval + 1];
      } else {
        if (!first)
          weights_down_norm_[f] = w;
        if (!last)
          weights_up_norm_[f] = (1 - w);
      }
    }
    for (int f = fftbin_end_; f < fftbin_size_; f++)
      fft2mel_[f] = args_.nfilter;
  }

  void Setup(ScratchpadEstimator &se, const TensorListShape<> &in_shape) {
    inner_fft_ = true;
    for (int s = 0; s < in_shape.size(); s++) {
      inner_fft_ &= volume(in_shape.tensor_shape_span(s).begin() + args_.axis + 1,
                           in_shape.tensor_shape_span(s).end()) == 1;
    }
    nfft_ = in_shape.tensor_shape_span(0)[args_.axis];
    if (inner_fft_) {
      se.add<mm::memory_kind::device, int>(fft2mel_.size());
      se.add<mm::memory_kind::device, T>(weights_down_norm_.size());
      se.add<mm::memory_kind::device, T>(weights_up_norm_.size());

      SetupBlockDescsInnerFft(se, in_shape);
    } else {
      se.add<mm::memory_kind::device, int>(interval_ends_.size());
      se.add<mm::memory_kind::device, T>(weights_down_.size());
      if (args_.normalize)
        se.add<mm::memory_kind::device, T>(norm_factors_.size());
      SetupBlockDescsOuterFft(se, in_shape);
    }
  }

  void Compute(T * const*out_list, const T *const *in_list, int ndim,
               Scratchpad *scratchpad, cudaStream_t stream) {
    if (inner_fft_) {
      FillBlockDescsInnerFft(out_list, in_list);

      BlockDesc<T> *block_descs = nullptr;
      T *weights_down = nullptr, *weights_up = nullptr;
      int *fft2mel = nullptr;
      std::tie(block_descs, weights_down, weights_up, fft2mel) =
            scratchpad->ToContiguousGPU(stream, block_descs_,
                                        weights_down_norm_, weights_up_norm_,
                                        fft2mel_);

      int blockHeight = std::min(8, shm_height_);
      VALUE_SWITCH(shm_height_, ShmHeight, (1, 2, 4, 8, 16, 32), (
          MelFilterBankKernelInnerFft<ShmHeight>
              <<<block_descs_.size(), dim3(32, blockHeight), shm_size_, stream>>>
                (block_descs, weights_down, weights_up, fft2mel,
                fftbin_start_, fftbin_end_,
                args_.nfilter, nfft_);
        ), (DALI_FAIL(make_string("Unreachable code: invalid value of shm_height_: ", shm_height_)  // NOLINT
      )));  // NOLINT
    } else {
      FillBlockDescsOuterFft(out_list, in_list);
      BlockDesc<T> *block_descs = nullptr;
      int *interval_ends = nullptr;
      T *weights_down = nullptr, *norm_factors = nullptr;
      std::tie(block_descs, interval_ends, weights_down, norm_factors) =
            scratchpad->ToContiguousGPU(stream, block_descs_, interval_ends_,
                                        weights_down_, norm_factors_);

      dim3 block(kBlockDim2, std::min(args_.nfilter, kBlockDim2));
      dim3 grid(block_descs_.size(), div_ceil(args_.nfilter, kBlockDim2));
      MelFilterBankKernel
        <<<grid, block, 0, stream>>>(block_descs, weights_down, interval_ends,
                                     args_.normalize, norm_factors, args_.nfilter);
    }
    CUDA_CALL(cudaGetLastError());
  }

  using MelFilterImplBase<T>::Args;

 private:
  void SetupBlockDescsOuterFft(ScratchpadEstimator &se, const TensorListShape<> &in_shape) {
    nframes_.clear();
    nwindows_.clear();
    block_descs_.clear();
    auto batch_size = in_shape.num_samples();
    for (int64_t ti = 0; ti < batch_size; ++ti) {
      const auto &tshape = in_shape.tensor_shape(ti);
      nframes_.push_back(volume(tshape.begin(), tshape.begin() + args_.axis));
      nwindows_.push_back(volume(tshape.begin() + args_.axis + 1, tshape.end()));
      for (int64_t s = 0; s < nframes_.back(); ++s) {
        auto nblocks = div_ceil(nwindows_.back(), kBlockDim2);
        for (int64_t b = 0; b < nblocks; ++b) {
          block_descs_.emplace_back(0, 0, b * kBlockDim2, nwindows_.back());
        }
      }
    }
    se.add<mm::memory_kind::device, BlockDesc<T>>(block_descs_.size());
  }

  void SetupBlockDescsInnerFft(ScratchpadEstimator &se, const TensorListShape<> &in_shape) {
    nframes_.clear();
    nwindows_.clear();
    block_descs_.clear();
    block2sample_.clear();
    auto batch_size = in_shape.num_samples();

    int max_shm_size = GetSharedMemPerBlock();
    for (shm_height_ = 32; ; shm_height_ >>= 1) {
      int shm_in = mel_inner_fft::shm_in_size<T>(shm_height_);
      int shm_out = mel_inner_fft::shm_out_size<T>(shm_height_, args_.nfilter);
      shm_size_ = shm_in + shm_out;
      if (shm_size_ <= max_shm_size)
        break;
    }

    if (shm_height_ < 1)
      throw std::out_of_range(make_string("Too many Mel filters ", args_.nfilter));


    for (int64_t ti = 0; ti < batch_size; ++ti) {
      const auto &tshape = in_shape.tensor_shape(ti);
      auto nwindows = volume(tshape.begin(), tshape.begin() + args_.axis);

      int windows_per_block = 512;  // TODO(michalz): some smarter way of determining this
      int nblocks = div_ceil(nwindows, windows_per_block);

      block2sample_.resize(block2sample_.size() + nblocks, ti);

      for (int b = 0; b < nblocks; ++b) {
        int64_t start = b * windows_per_block;
        int64_t count = std::min<int64_t>(nwindows - start, windows_per_block);
        assert(start >= 0 && start < nwindows);
        assert(count > 0 && count <= windows_per_block);
        block_descs_.emplace_back(start * args_.nfilter, start * nfft_, 0, count);
      }
    }
    se.add<mm::memory_kind::device, BlockDesc<T>>(block_descs_.size());
  }

  void FillBlockDescsOuterFft(T *const *out_list, const T *const *in_list) {
    int64_t block_id = 0;
    for (uint64_t ti = 0; ti < nframes_.size(); ++ti) {
      const T *in = in_list[ti];
      T *out = out_list[ti];
      for (int64_t s = 0; s < nframes_[ti]; ++s) {
        auto nblocks = div_ceil(nwindows_[ti], kBlockDim2);
        for (int b = 0; b < nblocks; ++b) {
          block_descs_[block_id].in_frame = in;
          block_descs_[block_id].out_frame = out;
          ++block_id;
        }
        in += nwindows_[ti] * nfft_;
        out += nwindows_[ti] * args_.nfilter;
      }
    }
  }

  void FillBlockDescsInnerFft(T *const *out_list, const T *const *in_list) {
    for (size_t block_id = 0; block_id < block_descs_.size(); block_id++) {
      int sample = block2sample_[block_id];
      block_descs_[block_id].SetBasePointers(out_list[sample], in_list[sample]);
    }
  }

  std::vector<int> interval_ends_;
  std::vector<int> fft2mel_;
  std::vector<T> weights_down_norm_, weights_up_norm_;
  std::vector<int64_t> nframes_;
  std::vector<int64_t> nwindows_;
  std::vector<BlockDesc<T>> block_descs_;
  std::vector<int> block2sample_;
  int64_t nfft_ = 0;
  int shm_height_ = 32;
  int shm_size_ = 32 << 10;
  bool inner_fft_ = false;
  USE_MEL_FILTER_IMPL_MEMBERS(T);
};

template <typename T>
KernelRequirements MelFilterBankGpu<T>::Setup(KernelContext &context,
                                              const InListGPU<T> &in,
                                              const MelFilterBankArgs &original_args) {
  auto args = original_args;
  args.axis = args.axis >= 0 ? args.axis : in.sample_dim() - 2;
  TensorListShape<> out_shape = in.shape;
  for (int64_t s = 0; s < out_shape.num_samples(); ++s) {
    out_shape.tensor_shape_span(s)[args.axis] = args.nfilter;
  }
  KernelRequirements req;
  req.output_shapes = {out_shape};

  auto nfft = in.shape.tensor_shape_span(0)[args.axis];
  for (int s = 1; s < in.shape.num_samples(); ++s) {
    DALI_ENFORCE(in.shape.tensor_shape_span(s)[args.axis] == nfft,
        "All samples should have the same FFT dimension");
  }
  ScratchpadEstimator se;
  args.nfft = args.nfft > 0 ? args.nfft : 2 * (in.shape[0][args.axis] - 1);
  args.freq_high = args.freq_high > 0 ? args.freq_high : args.sample_rate / 2;
  if (!impl_ || impl_->Args() != args) {
    impl_.reset();
    switch (args.mel_formula) {
      case MelScaleFormula::HTK:
        impl_ = std::make_unique<Impl>(HtkMelScale<T>(), args);
        break;
      case MelScaleFormula::Slaney:
      default:
        impl_ = std::make_unique<Impl>(SlaneyMelScale<T>(), args);
        break;
    }
  }
  impl_->Setup(se, in.shape);
  req.scratch_sizes = se.sizes;
  return req;
}

template <typename T>
void MelFilterBankGpu<T>::Run(KernelContext &context, OutListGPU<T> &out, const InListGPU<T> &in) {
  assert(impl_ != nullptr);
  impl_->Compute(out.data.data(), in.data.data(), in.sample_dim(), context.scratchpad,
                 context.gpu.stream);
}

template <typename T>
MelFilterBankGpu<T>::MelFilterBankGpu() {}

template <typename T>
MelFilterBankGpu<T>::~MelFilterBankGpu() {}


template class MelFilterBankGpu<float>;
template class MelFilterBankGpu<double>;

}  // namespace audio
}  // namespace kernels
}  // namespace dali
