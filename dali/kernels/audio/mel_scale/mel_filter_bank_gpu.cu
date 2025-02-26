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
#include "dali/core/fast_div.h"

namespace dali {
namespace kernels {
namespace audio {

const int kBlockDim2 = 32;

template <typename T>
struct BlockDescOuter {
  BlockDescOuter() = default;
  BlockDescOuter(int64_t start_window, int64_t nwindows) {
    this->out_frame  = nullptr;
    this->in_frame   = nullptr;

    this->start_window = start_window;
    this->frame_nwindows = nwindows;
  }

  void SetBasePointers(T *out, const T *in) {
    out_frame = out;
    in_frame  = in;
  }

  T        *out_frame;
  const T  *in_frame;

  int64_t start_window;
  int64_t frame_nwindows;
};

template <typename T>
struct BlockDescInner {
  BlockDescInner() = default;

  BlockDescInner(int start_window, int end_window) {
    this->out_frame  = nullptr;
    this->in_frame   = nullptr;

    this->start_window = start_window;
    this->end_window = end_window;
  }

  void SetBasePointers(T *out, const T *in) {
    out_frame = out;
    in_frame  = in;
  }

  T        *out_frame;
  const T  *in_frame;

  int start_window;
  int end_window;
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
__global__ void MelFilterBankKernel(const BlockDescOuter<T> *block_desc,
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
 * @brief Mel filter bank for used when FFT / Mel coefficients are in the innermost dimension
 *
 * The algorithm:
 * A block processes a range of windows. Each range is futher subdivided into segments
 * that fit into shared memory.
 * First, the range of input values is estimated and then it is flattened (as much as possible)
 * and loaded into shared memory. When not all FFT coefficients the unused ones are skipped and
 * the used ones are densely packed in shared memory.
 * After the data has been loaded, it is flattened and each thread processes a single Mel bin,
 * loading the data from shared memory. The result is accumulated in a register and stored
 * directly to global memory.
 *
 * Because each block starts at the beginning of a window, we know that the offset from the
 * beginning of the block is a multiple of a window size. This fact is utilized to allow
 * the use of floating point multiplication instead of the expensive integer division.
 * The necessary reciprocals are calculated on host.
 */
namespace mel_inner_fft {

template <bool use_full_spectrum, typename T>
struct MelFilterBankInnerFft {
  BlockDescInner<T> block;
  const T *__restrict__ weights_down;
  const T *__restrict__ weights_up;
  const int *__restrict__ interval_ends;
  int shm_height;  // number of input rows stored in shared memory
  int fft_lo, fft_hi, fft_used, fft_total;
  int nmel;
  float rnfft_used;  // 1/fft_used, rounded up
  float rnmel;       // 1/nmel, rounded up
  T *shm;

  __device__ void Run() {
    for (int w = block.start_window; w < block.end_window; w += shm_height) {
      if (w != block.start_window)
        __syncthreads();
      int end_w = cuda_min(w + shm_height, block.end_window);
      Load(w, end_w);
      __syncthreads();
      Process(w, end_w);
    }
  }

  __device__ void Load(int begin_w, int end_w) {
    assert(end_w - begin_w <= shm_height);
    if (!use_full_spectrum) {
      // If the number of used FFT coefficients is different than the total number of FFTs
      // we reindex, to save bandwidth (and shared memory)
      const T *in = block.in_frame;
      for (int x = threadIdx.x; x < (end_w - begin_w) * fft_used; x += blockDim.x) {
        // Compute the window within this sub-range.
        //
        // The reciprocal is rounded up, so we never end up with a number that's too small
        // - this is helpful, because we can just round down the result.
        int w = __float2int_rd(x * rnfft_used);
        // Compute the fft bin (again, within this block) - only relevant frequencies count.
        int f = x - w * fft_used;
        // The actual FFT needs to be offset by fft_lo, to skip the unused frequencies
        int fft = f + fft_lo;
        int window = w + begin_w;
        // Calculate the offset of the source value and load
        int64_t offset = static_cast<int64_t>(window) * fft_total + fft;
        shm[x] = in[offset];
      }
    } else {
      // If we use all coefficients, we can just load the data linearly
      // - no reindexing is necessary
      const T *in = &block.in_frame[static_cast<int64_t>(begin_w) * fft_total];
      for (int x = threadIdx.x; x < (end_w - begin_w) * fft_total; x += blockDim.x) {
        shm[x] = in[x];
      }
    }
  }

  __device__ void Process(int begin_w, int end_w) {
    T *out = &block.out_frame[begin_w * nmel];
    // Linear loop over output values
    for (int x = threadIdx.x; x < (end_w - begin_w) * nmel; x += blockDim.x) {
      // Reconstitute window and mel bin
      // Use the multiplication by reciprocal to speed up things.
      int w = __float2int_rd(x * rnmel);
      assert(w < shm_height);
      int mel = x - w * nmel;
      assert(mel >= 0 && mel < nmel);
      int f0 = interval_ends[mel];
      int f1 = interval_ends[mel + 1];
      int f2 = interval_ends[mel + 2];
      assert(f0 >= fft_lo && f0 <= fft_hi);
      assert(f1 >= fft_lo && f1 <= fft_hi);
      assert(f2 >= fft_lo && f2 <= fft_hi);

      //  this pointer is out of range by -fft_lo, but the indices will compensate
      T *shm_in = &shm[w * fft_used - fft_lo];

      int f;
      T result = 0;
      for (f = f0; f < f1; f++) {
        result = fma(weights_up[f], shm_in[f], result);
      }
      for (; f < f2; f++) {
        result = fma(weights_down[f], shm_in[f], result);
      }
      out[x] = result;
    }
  }
};

}  // namespace mel_inner_fft

template <bool use_full_spectrum, typename T>
__global__ void MelFilterBankKernelInnerFft(const BlockDescInner<T> *block_descs,
                                            const T *__restrict__ weights_down,
                                            const T *__restrict__ weights_up,
                                            const int *__restrict__ interval_ends,
                                            int shm_height,
                                            int fft_lo, int fft_hi, int fft_total, float rnfft_used,
                                            int nmel, float rnmel) {
  extern __shared__ char shm_arena[];
  mel_inner_fft::MelFilterBankInnerFft<use_full_spectrum, T> fb = {
    block_descs[blockIdx.x],
    weights_down,
    weights_up,
    interval_ends,
    shm_height,
    fft_lo, fft_hi, fft_hi - fft_lo, fft_total,
    nmel,
    rnfft_used,
    rnmel,
    reinterpret_cast<T*>(shm_arena)
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

    for (int f = fftbin_start_, interval = -1; f < fftbin_end_; f++) {
      while (interval + 2 < static_cast<int>(interval_ends_.size()) &&
             interval_ends_[interval + 2] <= f)
        interval++;
      assert(interval >= -1 && interval <= args.nfilter);
      bool first = interval == -1;
      bool last = interval == args.nfilter;

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
  }

  void Setup(const TensorListShape<> &in_shape) {
    inner_fft_ = true;
    for (int s = 0; s < in_shape.size(); s++) {
      inner_fft_ &= volume(in_shape.tensor_shape_span(s).begin() + args_.axis + 1,
                           in_shape.tensor_shape_span(s).end()) == 1;
    }
    nfft_ = in_shape.tensor_shape_span(0)[args_.axis];
    if (inner_fft_) {
      SetupBlockDescsInnerFft(in_shape);
    } else {
      SetupBlockDescsOuterFft(in_shape);
    }
  }

  void Compute(T * const*out_list, const T *const *in_list, int ndim,
               Scratchpad *scratchpad, cudaStream_t stream) {
    if (inner_fft_) {
      FillBlockDescsInnerFft(out_list, in_list);

      BlockDescInner<T> *block_descs = nullptr;
      T *weights_down = nullptr, *weights_up = nullptr;
      int *interval_ends = nullptr;
      std::tie(block_descs, weights_down, weights_up, interval_ends) =
            scratchpad->ToContiguousGPU(stream, block_descs_inner_,
                                        weights_down_norm_, weights_up_norm_,
                                        interval_ends_);

      int max_block_size = shm_height_ * args_.nfilter;
      int block_size = std::min(128, max_block_size);
      if (max_block_size / block_size < 2)
        block_size = align_up(max_block_size / 2, 32);

      int fft_lo, fft_hi;
      if (load_full_spectrum_) {
        fft_lo = 0;
        fft_hi = nfft_;
      } else {
        fft_lo = fftbin_start_;
        fft_hi = fftbin_end_;
      }

      // reciprocal, rounded up
      auto rcp_ru = [](float denom) {
        assert(denom > 0);
        float r = 1 / denom;
        while (r * denom < 1)
          r = std::nextafter(r, r + 1);
        return r;
      };

      float rcp_fft_used = rcp_ru(fft_hi - fft_lo);
      float rcp_mel = rcp_ru(args_.nfilter);

      BOOL_SWITCH(load_full_spectrum_, UseFullSpectrum, (
        MelFilterBankKernelInnerFft<UseFullSpectrum>
          <<<block_descs_inner_.size(), dim3(block_size), shm_size_, stream>>>
                (block_descs, weights_down, weights_up, interval_ends,
                shm_height_,
                fft_lo, fft_hi, nfft_, rcp_fft_used,
                args_.nfilter, rcp_mel)));
    } else {
      FillBlockDescsOuterFft(out_list, in_list);
      BlockDescOuter<T> *block_descs = nullptr;
      int *interval_ends = nullptr;
      T *weights_down = nullptr, *norm_factors = nullptr;
      std::tie(block_descs, interval_ends, weights_down, norm_factors) =
            scratchpad->ToContiguousGPU(stream, block_descs_outer_, interval_ends_,
                                        weights_down_, norm_factors_);

      dim3 block(kBlockDim2, std::min(args_.nfilter, kBlockDim2));
      dim3 grid(block_descs_outer_.size(), div_ceil(args_.nfilter, kBlockDim2));
      MelFilterBankKernel
        <<<grid, block, 0, stream>>>(block_descs, weights_down, interval_ends,
                                     args_.normalize, norm_factors, args_.nfilter);
    }
    CUDA_CALL(cudaGetLastError());
  }

  using MelFilterImplBase<T>::Args;

 private:
  void SetupBlockDescsOuterFft(const TensorListShape<> &in_shape) {
    nframes_.clear();
    nwindows_.clear();
    block_descs_outer_.clear();
    auto batch_size = in_shape.num_samples();
    for (int64_t ti = 0; ti < batch_size; ++ti) {
      const auto &tshape = in_shape.tensor_shape(ti);
      nframes_.push_back(volume(tshape.begin(), tshape.begin() + args_.axis));
      nwindows_.push_back(volume(tshape.begin() + args_.axis + 1, tshape.end()));
      for (int64_t s = 0; s < nframes_.back(); ++s) {
        auto nblocks = div_ceil(nwindows_.back(), kBlockDim2);
        for (int64_t b = 0; b < nblocks; ++b) {
          block_descs_outer_.emplace_back(b * kBlockDim2, nwindows_.back());
        }
      }
    }
  }

  void SetupBlockDescsInnerFft(const TensorListShape<> &in_shape) {
    nframes_.clear();
    nwindows_.clear();
    block_descs_inner_.clear();
    block2sample_.clear();
    auto batch_size = in_shape.num_samples();

    int total_windows = 0;
    for (int64_t ti = 0; ti < batch_size; ++ti) {
      const auto &tshape = in_shape.tensor_shape(ti);
      int windows = volume(tshape.begin(), tshape.begin() + args_.axis);
      total_windows += windows;
    }

    int max_windows_per_block = std::min(512, total_windows / (8 * GetSmCount()));
    if (max_windows_per_block < 1)
      max_windows_per_block = 1;

    // Compute the shared memory size
    if (max_shm_size_ < 0) {
      cudaFuncAttributes attr = {};
      CUDA_CALL(cudaFuncGetAttributes(&attr, &MelFilterBankKernelInnerFft<false, T>));
      max_shm_size_ = attr.maxDynamicSharedSizeBytes;
    }

    int fft_used = fftbin_end_ - fftbin_start_;
    load_full_spectrum_ = (nfft_ - fft_used) * sizeof(T) < 32;
    if (load_full_spectrum_)
      fft_used = nfft_;
    int max_height = max_shm_size_ / (sizeof(T) * fft_used);

    // Avoid problems with the precision of fake division
    max_height = std::min<int>(max_height, 1e+6 / fft_used);

    // try to get more active blocks
    if (max_height > 16)
      max_height = prev_pow2(max_height / 3);
    else if (max_height > 8)
      max_height = prev_pow2(max_height / 2);

    if (max_height > max_windows_per_block)
      max_height = max_windows_per_block;

    shm_height_ = max_height;

    if (shm_height_ < 1) {
      // This is a simplification - we only care about the FFTs actually used
      // but the error message would be rather confusing.
      throw std::out_of_range(make_string("Too large FFT: ", nfft_));
    }

    shm_size_ = shm_height_ * fft_used * sizeof(T);

    for (int64_t ti = 0; ti < batch_size; ++ti) {
      const auto &tshape = in_shape.tensor_shape(ti);
      auto nwindows = volume(tshape.begin(), tshape.begin() + args_.axis);

      int64_t windows_per_block =
          std::min(max_windows_per_block, prev_pow2((1 << 20) / args_.nfilter));

      int nblocks = div_ceil(nwindows, windows_per_block);

      windows_per_block = div_ceil(nwindows, nblocks);

      block2sample_.resize(block2sample_.size() + nblocks, ti);

      for (int b = 0; b < nblocks; ++b) {
        int64_t start = b * windows_per_block;
        int64_t count = std::min<int64_t>(nwindows - start, windows_per_block);
        assert(start >= 0 && start < nwindows);
        assert(count > 0 && count <= windows_per_block);
        assert(start + count <= nwindows);
        block_descs_inner_.emplace_back(start, start + count);
      }
    }
  }

  void FillBlockDescsOuterFft(T *const *out_list, const T *const *in_list) {
    int64_t block_id = 0;
    for (uint64_t ti = 0; ti < nframes_.size(); ++ti) {
      const T *in = in_list[ti];
      T *out = out_list[ti];
      for (int64_t s = 0; s < nframes_[ti]; ++s) {
        auto nblocks = div_ceil(nwindows_[ti], kBlockDim2);
        for (int b = 0; b < nblocks; ++b) {
          block_descs_outer_[block_id].in_frame = in;
          block_descs_outer_[block_id].out_frame = out;
          ++block_id;
        }
        in += nwindows_[ti] * nfft_;
        out += nwindows_[ti] * args_.nfilter;
      }
    }
  }

  void FillBlockDescsInnerFft(T *const *out_list, const T *const *in_list) {
    for (size_t block_id = 0; block_id < block_descs_inner_.size(); block_id++) {
      int sample = block2sample_[block_id];
      block_descs_inner_[block_id].SetBasePointers(out_list[sample], in_list[sample]);
    }
  }

  std::vector<int> interval_ends_;
  std::vector<T> weights_down_norm_, weights_up_norm_;
  std::vector<int64_t> nframes_;
  std::vector<int64_t> nwindows_;
  std::vector<BlockDescOuter<T>> block_descs_outer_;
  std::vector<BlockDescInner<T>> block_descs_inner_;
  std::vector<int> block2sample_;
  int64_t nfft_ = 0;
  int shm_height_ = -1;
  int shm_size_ = -1, max_shm_size_ = -1;
  bool load_full_spectrum_ = false;
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
  impl_->Setup(in.shape);
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
