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
  const T *in_frame;
  T *out_frame;
  struct {
    int64_t start_window;
    int64_t frame_nwindows;
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

// For layouts with the innermost frequency dimension, data is flattened
// to two dimensions - time, frequency
/*template <typename T>
__global__ void MelFilterBankKernelInnerFft(const BlockDesc<T> *block_desc,
                                            const T *weights_down, const int *interval_ends,
                                            bool normalize, const T *norm_factors,
                                            int nmel, int64_t nfft) {
  auto block_id = blockIdx.x;
  auto idx = block_desc[block_id].block_start + threadIdx.x;

  if (idx >= block_desc[block_id].out_frame_size)
    return;

  auto window = idx / nmel;
  auto mel_bin = idx % nmel;
  const T *in = block_desc[block_id].in_frame;
  T *out =  block_desc[block_id].out_frame;
  T norm_factor = (normalize) ? norm_factors[mel_bin] : 1;
  *(out + idx) = calcMel(in + window * nfft, mel_bin,
                         weights_down, interval_ends, 1, 0, norm_factor);
}*/

static constexpr int kMaxInnerFftFreqs = 48;

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
  T *shm;

  static const int max_freqs = kMaxInnerFftFreqs;
  static const int sh_in_stride = max_freqs + 1;
  int sh_out_stride = 0;

  __device__ T &shm_in(int window, int freq) const {
    return shm[sh_in_stride * window + freq];
  }

  __device__ T &shm_out(int window, int mel) const {
    const int shm_out_offset = sh_in_stride * shm_height;
    return shm[shm_out_offset + window * sh_out_stride + mel];
  }

  __device__ void Run() {
    sh_out_stride = align_up(nmel, 32) + 1;

    bool first = true;

    for (int64_t start_window = block_desc.start_window;
         start_window < block_desc.frame_nwindows;
         start_window += shm_height) {
      if (first)
        first = false;
      else
        __syncthreads();

      int fft_end;
      for (int fft_start = fft_lo; fft_start < fft_hi; fft_start = fft_end) {
        if (fft_hi - fft_start <= max_freqs)
          fft_end = fft_hi;
        else
          fft_end = fft_hi + 32;

        LoadFrequencyBlock(start_window, fft_start, fft_end);
        ClearMel();
        __syncthreads();
        ProcessFrequencyBlock(fft_start, fft_end);
        __syncthreads();
      }
      StoreMel(start_window);
    }
  }

  __device__ void LoadFrequencyBlock(int64_t start_window, int fft_start, int fft_end) {
    int bh = blockDim.y;
    const T *in = block_desc.in_frame;
    for (int y = threadIdx.y; y + bh < shm_height; y += bh) {
      int window = start_window + y;
      if (window < block_desc.frame_nwindows) {
        for (int x = threadIdx.x; x < fft_end - fft_start; x += 32)
          shm_in(y, x) = in[window * nfft + x];
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
    for (int f0 = fft_start + threadIdx.y * cols_per_warp; f0 < fft_end; f0 += bh) {
      int fft = f0 + threadIdx.x / shm_height;
      if (fft >= fft_end)
        break;
      int wnd = threadIdx.x % shm_height;
      int bin0 = bin_down[fft];
      int bin1 = bin0 + 1;
      if (bin0 >= 0)
        atomicAdd(&shm_out(wnd, bin0), weights_down[fft]);
      if (bin1 < nmel)
        atomicAdd(&shm_out(wnd, bin1), weights_up[fft]);
    }
  }

  __device__ void StoreMel(int64_t start_window) {
    int end_window = cuda_min(start_window + shm_height, block_desc.frame_nwindows);
    int nwindows = end_window - start_window;
    T *out = block_desc.out_frame;
    for (int y = threadIdx.y; y < nwindows; y += blockDim.y) {
      int64_t window = start_window + y;
      for (int mel = threadIdx.x; mel < nmel; mel += blockDim.x)
        out[window * nmel + mel] = shm_out(y, mel);
    }
  }
};

template <int shm_height, typename T>
__global__ void MelFilterBankKernelInnerFft(const BlockDesc<T> *block_descs,
                                            const T *__restrict__ weights_down,
                                            const T *__restrict__ weights_up,
                                            const int *__restrict__ bin_down,
                                            int fft_lo, int fft_hi,
                                            int nmel, int nfft) {
  extern __shared__ char shm_arena[];
  MelFilterBankInnerFft<shm_height, T> fb = {
    block_descs[blockIdx.x],
    weights_down,
    weights_up,
    bin_down,
    fft_lo, fft_hi,
    nmel, nfft,
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
    interval_ends_[args.nfilter + 1] = fftbin_end_ + 1;

    for (int interval = 1; interval < args_.nfilter + 1; interval++, mel += mel_delta_) {
      double freq = mel_scale.mel_to_hz(mel);
      interval_ends_[interval] = std::ceil(freq / hz_step_);
    }

    weights_down_norm_.resize(fftbin_size_);
    weights_up_norm_.resize(fftbin_size_);
    fft2mel_.resize(fftbin_size_);

    for (int f = fftbin_start_, interval = -1; f <= fftbin_end_; f++) {
      while (interval_ends_[interval + 1] <= f)
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

  void Compute(const T* const* in_list, T **out_list, int ndim,
               Scratchpad *scratchpad, cudaStream_t stream) {
    if (inner_fft_) {
      FillBlockDescsInnerFft(in_list, out_list);

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
                fftbin_start_, fftbin_end_ + 1,
                args_.nfilter, nfft_);
        ), (DALI_FAIL(make_string("Unreachable code: invalid value of shm_height_: ", shm_height_)  // NOLINT
      )));  // NOLINT

      /*MelFilterBankKernelInnerFft
          <<<block_descs_.size(), kBlockDim1, 0, stream>>>
            (block_descs, weights_down, interval_ends, args_.normalize,
             norm_factors, args_.nfilter, nfft_);*/
    } else {
      FillBlockDescsOuterFft(in_list, out_list);
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
    block2sample_.clear();
    auto batch_size = in_shape.num_samples();

    int shm_size = GetSharedMemPerBlock();
    int shm_width = ((kMaxInnerFftFreqs + 1) + (align_up(args_.nfilter, 32) + 1)) * sizeof(T);
    int shm_height = shm_size / shm_width;
    if (shm_height_ < 1)
      throw std::out_of_range(make_string("Too many Mel filters ", args_.nfilter));

    shm_height_ = prev_pow2(std::min(shm_height, 32));
    shm_size_ = shm_height_ * shm_width;

    for (int64_t ti = 0; ti < batch_size; ++ti) {
      const auto &tshape = in_shape.tensor_shape(ti);
      auto nwindows = volume(tshape.begin(), tshape.begin() + args_.axis);

      int windows_per_block = 512;  // TODO(michalz): some smarter way of determining this
      int nblocks = div_ceil(nwindows, windows_per_block);

      block2sample_.resize(block2sample_.size() + nblocks, ti);

      for (int b = 0; b < nblocks; ++b) {
        int64_t start = b * windows_per_block;
        int64_t count = std::min<int64_t>(nwindows - start, windows_per_block);
        block_descs_.push_back(BlockDesc<T>{nullptr, nullptr, {start, count}});
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
        in += nwindows_[ti] * nfft_;
        out += nwindows_[ti] * args_.nfilter;
      }
    }
  }

  void FillBlockDescsInnerFft(const T* const* in_list, T **out_list) {
    for (size_t block_id = 0; block_id < block_descs_.size(); block_id++) {
      int sample = block2sample_[block_id];
      block_descs_[block_id].in_frame = in_list[sample];
      block_descs_[block_id].out_frame = out_list[sample];
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
