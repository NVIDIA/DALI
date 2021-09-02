// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace dali {
namespace kernels {
namespace audio {

const int kBlockDim2 = 32;
const int kBlockDim1 = 1024;

template <typename T>
struct BlockDesc {
  const T *in_frame;
  T *out_frame;
  union {
    struct {  // outer-dim fft
      int64_t start_window;
      int64_t frame_nwindows;
    };
    struct {  // inner-dim fft
      int64_t block_start;
      int64_t out_frame_size;
    };
  };
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
                                    int mel_bins) {
  auto block_id = blockIdx.x;
  const T *in_frame = block_desc[block_id].in_frame;
  T *out_frame = block_desc[block_id].out_frame;
  int mel_bin = blockIdx.y * kBlockDim2 + threadIdx.y;

  if (mel_bin >= mel_bins)
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

// For layouts with the innermost frequency dimension, data is flattened
// to two dimensions - time, frequency
template <typename T>
__global__ void MelFilterBankKernelInnerFft(const BlockDesc<T> *block_desc,
                                            const T *weights_down, const int *interval_ends,
                                            bool normalize, const T *norm_factors,
                                            int mel_bins, int64_t fftdim) {
  auto block_id = blockIdx.x;
  auto idx = block_desc[block_id].block_start + threadIdx.x;

  if (idx >= block_desc[block_id].out_frame_size)
    return;

  auto window = idx / mel_bins;
  auto mel_bin = idx % mel_bins;
  const T *in = block_desc[block_id].in_frame;
  T *out =  block_desc[block_id].out_frame;
  T norm_factor = (normalize) ? norm_factors[mel_bin] : 1;
  *(out + idx) = calcMel(in + window * fftdim, mel_bin,
                         weights_down, interval_ends, 1, 0, norm_factor);
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
    interval_ends_[args.nfilter + 1] = fftbin_end_ + 1;
    for (int interval = 1; interval < args_.nfilter + 1; interval++, mel += mel_delta_) {
      double freq = mel_scale.mel_to_hz(mel);
      interval_ends_[interval] = std::ceil(freq / hz_step_);
    }
  }

  void Setup(ScratchpadEstimator &se, const TensorListShape<> &in_shape) {
    se.add<mm::memory_kind::device, int>(interval_ends_.size());
    se.add<mm::memory_kind::device, T>(weights_down_.size());
    if (args_.normalize)
      se.add<mm::memory_kind::device, T>(norm_factors_.size());

    inner_fft_ = true;
    for (int s = 0; s < in_shape.size(); s++) {
      inner_fft_ &= volume(in_shape.tensor_shape_span(s).begin() + args_.axis + 1,
                           in_shape.tensor_shape_span(s).end()) == 1;
    }
    fft_dim_ = in_shape.tensor_shape_span(0)[args_.axis];
    if (inner_fft_) {
      SetupBlockDescsInnerFft(se, in_shape);
    } else {
      SetupBlockDescsOuterFft(se, in_shape);
    }
  }

  void Compute(const T* const* in_list, T **out_list, int ndim,
               Scratchpad *scratchpad, cudaStream_t stream) {
    if (inner_fft_) {
      FillBlockDescsInnerFft(in_list, out_list);
    } else {
      FillBlockDescsOuterFft(in_list, out_list);
    }
    auto block_descs = scratchpad->ToGPU(stream, block_descs_);
    auto interval_ends = scratchpad->ToGPU(stream, interval_ends_);
    auto weights_down = scratchpad->ToGPU(stream, weights_down_);
    T *norm_factors = nullptr;
    if (args_.normalize)
      norm_factors = scratchpad->ToGPU(stream, norm_factors_);
    if (inner_fft_) {
      MelFilterBankKernelInnerFft
          <<<block_descs_.size(), kBlockDim1, 0, stream>>>
            (block_descs, weights_down, interval_ends, args_.normalize,
             norm_factors, args_.nfilter, fft_dim_);
    } else {
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
          block_descs_.push_back(BlockDesc<T>{nullptr, nullptr,
                                              {b * kBlockDim2, nwindows_.back()}});
        }
      }
    }
    se.add<mm::memory_kind::device, BlockDesc<T>>(block_descs_.size());
  }

  void SetupBlockDescsInnerFft(ScratchpadEstimator &se, const TensorListShape<> &in_shape) {
    nframes_.clear();
    nwindows_.clear();
    block_descs_.clear();
    auto batch_size = in_shape.num_samples();
    for (int64_t ti = 0; ti < batch_size; ++ti) {
      const auto &tshape = in_shape.tensor_shape(ti);
      auto sample_size = volume(tshape.begin(), tshape.begin() + args_.axis) * args_.nfilter;
      auto nblocks = div_ceil(sample_size, kBlockDim1);
      for (int b = 0; b < nblocks; ++b) {
        block_descs_.push_back(BlockDesc<T>{nullptr, nullptr, {b * kBlockDim1, sample_size}});
      }
    }
    se.add<mm::memory_kind::device, BlockDesc<T>>(block_descs_.size());
  }

  void FillBlockDescsOuterFft(const T* const* in_list, T **out_list) {
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
        in += nwindows_[ti] * fft_dim_;
        out += nwindows_[ti] * args_.nfilter;
      }
    }
  }

  void FillBlockDescsInnerFft(const T* const* in_list, T **out_list) {
    int block_id = 0;
    int sample = 0;
    while (block_id < static_cast<int>(block_descs_.size())) {
      auto sample_size = block_descs_[block_id].out_frame_size;
      auto nblocks = div_ceil(sample_size, kBlockDim1);
      for (int i = 0; i < nblocks; ++i, ++block_id) {
        block_descs_[block_id].in_frame = in_list[sample];
        block_descs_[block_id].out_frame = out_list[sample];
      }
      ++sample;
    }
  }

  std::vector<int> interval_ends_;
  std::vector<int64_t> nframes_;
  std::vector<int64_t> nwindows_;
  std::vector<BlockDesc<T>> block_descs_;
  int64_t fft_dim_ = 0;
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

  auto fftdim = in.shape.tensor_shape_span(0)[args.axis];
  for (int s = 1; s < in.shape.num_samples(); ++s) {
    DALI_ENFORCE(in.shape.tensor_shape_span(s)[args.axis] == fftdim,
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
  impl_->Compute(in.data.data(), out.data.data(), in.sample_dim(), context.scratchpad,
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
