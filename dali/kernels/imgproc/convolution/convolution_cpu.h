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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_CONVOLUTION_CPU_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_CONVOLUTION_CPU_H_

#include "dali/core/boundary.h"
#include "dali/core/convert.h"
#include "dali/core/format.h"
#include "dali/core/tensor_view.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/kernel.h"
#include "dali/pipeline/util/operator_impl_utils.h"

namespace dali {
namespace kernels {
/**
 * @brief Cyclic buffer used for storing input window for convolution.
 *
 * Keeps `num_lanes` of each window element, intended when there is stride between
 * window elements, but the same convolution can be applied to neighbouring innermost elements.
 *
 * Wraps a pointer to a buffer of type T, with size equal to length * num_lanes.
 */
template <typename T, int max_lanes = 16>
class CyclicWindowWrapper {
 public:
  CyclicWindowWrapper(T* __restrict__ ptr, int length, int num_lanes = 1)
      : data_(ptr), start_(0), end_(0), elements_(0), length_(length), num_lanes_(num_lanes) {
    DALI_ENFORCE(num_lanes <= max_lanes, "Static lanes limit exceeded.");
  }

  /**
   * @brief Drop one element consisting of `NumLanes()` lanes from the buffer, moving the start
   *        element.
   */
  void PopFront() {
    assert(elements_ > 0);
    elements_--;
    start_++;
    WrapPosition(start_);
  }

  /**
   * @brief Add a window element, that has `NumLanes()` consecutive lanes and starting at `input`,
   * to the buffer.
   */
  void PushBack(const T* __restrict__ input) {
    assert(elements_ < length_);
    memcpy(data_ + end_ * NumLanes(), input, sizeof(T) * NumLanes());
    elements_++;
    end_++;
    WrapPosition(end_);
  }

  /**
   * @brief Get pointer to the start of pixel `idx` in the buffer, taking into account the cyclic
   * wrapping
   */
  T* GetElementOffset(int idx) const {
    assert(idx < elements_);
    if (start_ + idx < length_) {
      return data_ + (start_ + idx) * NumLanes();
    } else {
      return data_ + (start_ + idx - length_) * NumLanes();
    }
  }

  /**
   * @brief Calculate dot product with 1-channel `window` length equal to the one specified
   * at construction. The result is stored in pixel stored at `accum`.
   */
  template <typename W>
  void CalculateDot(W* __restrict__ accum, const W* __restrict__ window) const {
    assert(elements_ == length_);
    for (int c = 0; c < NumLanes(); c++) {
      accum[c] = 0;
    }
    int window_idx = 0;
    for (int buf_idx = start_; buf_idx < length_; buf_idx++, window_idx++) {
      auto data_offset = buf_idx * NumLanes();
      auto w = window[window_idx];
      for (int c = 0; c < NumLanes(); c++) {
        accum[c] += w * data_[data_offset + c];
      }
    }
    for (int buf_idx = 0; buf_idx < end_; buf_idx++, window_idx++) {
      auto data_offset = buf_idx * NumLanes();
      auto w = window[window_idx];
      for (int c = 0; c < NumLanes(); c++) {
        accum[c] += w * data_[data_offset + c];
      }
    }
  }

  template <typename U, typename W>
  void CalculateDot(U* __restrict__ output, const W* __restrict__ window, float scale) const {
    std::array<W, max_lanes> tmp;
    CalculateDot(tmp.data(), window);
    for (int c = 0; c < NumLanes(); c++) {
      output[c] = ConvertSat<U>(tmp[c] * scale);
    }
  }

  int Size() const {
    return elements_;
  }

  bool Empty() const {
    return elements_ == 0;
  }

 private:
  void WrapPosition(int& pos) const {
    if (pos == length_) {
      pos = 0;
    }
  }

  int NumLanes() const {
    return std::min(num_lanes_, max_lanes);
  }

  T* __restrict__ data_ = nullptr;
  int start_ = 0;
  int end_ = 0;  ///< next empty element
  int elements_ = 0;
  int length_ = 0;
  int num_lanes_ = 0;
};

template <typename T, int max_lanes>
void load_pixel_with_border(CyclicWindowWrapper<T, max_lanes>& cww, const T* in_ptr, int in_idx,
                            int stride, int axis_size) {
  cww.PushBack(in_ptr + boundary::idx_reflect_101(in_idx, axis_size) * stride);
}

template <typename T, int max_lanes>
void reload_pixel_with_border(CyclicWindowWrapper<T, max_lanes>& cww, const T* in_ptr, int in_idx,
                              int out_idx, int stride, int axis_size, int radius) {
  // out_idx is currently the center of the window
  // we won't look further than radius elements, so we will pick something from first half
  // of the cyclic window.
  auto input_idx_in_bounds = boundary::idx_reflect_101(in_idx, axis_size);
  // remap from input_idx in image coordinate to window coordinates so we can take it from
  // cyclic window buffer.
  auto window_start_idx = out_idx - radius;
  auto in_window_idx = input_idx_in_bounds - window_start_idx;
  cww.PushBack(cww.GetElementOffset(in_window_idx));
}

template <typename T, int max_lanes>
void load_pixel_no_border(CyclicWindowWrapper<T, max_lanes>& cww, const T* in_ptr, int in_idx,
                          int stride) {
  cww.PushBack(in_ptr + in_idx * stride);
}

template <bool has_channels, typename Out, typename In, typename W, int ndim>
void ConvolveInnerDim(Out* out, const In* in, const W* window, int window_size,
                      const TensorShape<ndim>& shape, const TensorShape<ndim>& strides,
                      float scale) {
  constexpr int last_dim = has_channels ? ndim - 2 : ndim - 1;
  int channels = has_channels ? strides[last_dim] : 1;
  int64_t outer_elements = volume(&shape[0], &shape[last_dim]);
  int64_t axis_size = shape[last_dim];
  int64_t axis_stride = last_dim > 0 ? strides[last_dim - 1] : 0;
  int radius = (window_size - 1) / 2;
  // N.B. this can be negative for window_size > axis_size + 1
  int64_t flat_x_limit = (axis_size - window_size + 1) * channels;
  for (int64_t o = 0; o < outer_elements; o++) {
    int64_t x0 = -radius;
    int64_t xout = 0;
    Out* out_axis = &out[o * axis_stride];
    const In* in_axis = &in[o * axis_stride];
    // Left border
    for (; x0 < 0 && xout < axis_size; x0++, xout++) {
      for (int c = 0; c < channels; c++) {
        float acc = 0;
        for (int k = 0; k < window_size; k++) {
          int x = boundary::idx_reflect_101(x0 + k, axis_size);
          acc += in_axis[x * channels + c] * window[k];
        }
        out_axis[xout * channels + c] = ConvertSat<Out>(acc * scale);
      }
    }
    int64_t flat_x = x0 * channels;
    int64_t flat_xout = xout * channels;
    // This loop won't execute if the window_size > axis_size
    for (; flat_x < flat_x_limit; flat_x++, flat_xout++) {
      float acc = 0;
      for (int k = 0; k < window_size; k++) {
        acc += in_axis[flat_x + k * channels] * window[k];
      }
      out_axis[flat_xout] = ConvertSat<Out>(acc * scale);
    }
    // get back from flat coordinates
    x0 = flat_x / channels;
    xout = flat_xout / channels;
    // Right border
    for (; xout < axis_size; x0++, xout++) {
      for (int c = 0; c < channels; c++) {
        float acc = 0;
        for (int k = 0; k < window_size; k++) {
          int x = boundary::idx_reflect_101(x0 + k, axis_size);
          acc += in_axis[x * channels + c] * window[k];
        }
        out_axis[xout * channels + c] = ConvertSat<Out>(acc * scale);
      }
    }
  }
}

template <int axis, bool has_channels, int max_lanes, typename Out, typename In, typename W,
          int ndim>
void ConvolveInplaceAxisLoop(Out* out, const In* in, const W* window,
                             const TensorShape<ndim>& shape, const TensorShape<ndim>& strides,
                             int diameter, int64_t offset, In* input_window_buffer, float scale,
                             int num_lanes) {
  auto axis_stride = strides[axis];
  auto axis_size = shape[axis];
  int radius = (diameter - 1) / 2;
  // offset <- start of current axis
  auto* out_ptr = out + offset;
  auto* in_ptr = in + offset;

  CyclicWindowWrapper<In, max_lanes> input_window(input_window_buffer, diameter, num_lanes);
  int in_idx = -radius, out_idx = 0;
  for (in_idx = -radius; in_idx < 0; in_idx++) {
    load_pixel_with_border(input_window, in_ptr, in_idx, axis_stride, axis_size);
  }
  // we load the window without the last element
  for (; in_idx < radius && in_idx < axis_size; in_idx++) {
    load_pixel_no_border(input_window, in_ptr, in_idx, axis_stride);
  }
  // if we already went out of the window, fill the rest
  for (; in_idx < radius; in_idx++) {
    load_pixel_with_border(input_window, in_ptr, in_idx, axis_stride, axis_size);
  }
  // if we're still in the window, use version without border, till the in_idx goes out
  for (; out_idx < axis_size && in_idx < axis_size; out_idx++, in_idx++) {
    // we load last element of the input window corresponding to the out_idx
    load_pixel_no_border(input_window, in_ptr, in_idx, axis_stride);
    input_window.CalculateDot(out_ptr + out_idx * axis_stride, window, scale);
    // remove one element, to make space for next out_idx and in_idx
    input_window.PopFront();
  }
  // finish the rest of output
  for (; out_idx < axis_size; out_idx++, in_idx++) {
    // To process in-place, we need to pick the vales back from the CyclicBuffer,
    // as it may happen that we already stored the element.u
    reload_pixel_with_border(input_window, in_ptr, in_idx, out_idx, axis_stride, axis_size, radius);
    input_window.CalculateDot(out_ptr + out_idx * axis_stride, window, scale);
    input_window.PopFront();
  }
}

template <int axis, bool has_channels, int max_lanes, typename Out, typename In, typename W,
          int ndim>
void ConvolveInplaceOuterLoop(Out* out, const In* in, const W* window,
                              const TensorShape<ndim>& shape, const TensorShape<ndim>& strides,
                              int diameter, In* input_window_buffer, float scale = 1.f) {
  int64_t outer_elements = volume(&shape[0], &shape[axis]);
  int64_t axis_elements = shape[axis];
  int64_t inner_elements = volume(&shape[axis + 1], &shape[ndim]);
  assert(strides[axis] == inner_elements);

  int64_t strip_size = max_lanes;
  if (has_channels && axis == ndim - 2) {
    strip_size = shape[ndim - 1];
  } else if (!has_channels && axis == ndim - 1) {
    strip_size = 1;
  }
  // TODO(klecki): to handle border fill, one must keep track of how inner_idx maps to
  // pixel/channels, and prepare a fill window starting with appropriate channel
  for (int64_t outer_idx = 0; outer_idx < outer_elements; outer_idx++) {
    for (int64_t inner_idx = 0; inner_idx < inner_elements; inner_idx += strip_size) {
      int64_t offset = outer_idx * (axis > 0 ? strides[axis - 1] : 0) + inner_idx;
      int num_lanes = std::min(inner_elements - inner_idx, strip_size);
      ConvolveInplaceAxisLoop<axis, has_channels, max_lanes>(
          out, in, window, shape, strides, diameter, offset, input_window_buffer, scale, num_lanes);
    }
  }
}

/**
 * @brief Apply convolution with 1-channel `window` in specified axis.
 *
 * The innermost dimension performed _not_ in-place uses implementation that will be faster
 * than in-place one that requires additional copy.
 *
 * For non-innermost convolution a sliding window (using a cyclic buffer) over several lanes is used
 * (can be comprised of several pixels, one channel is one lane). Can be safely performed in-place.
 *
 * The same implementation is used for in-place innermost convolution.
 */
template <typename Out, typename In, typename W, int ndim, int axis, bool has_channels = true>
struct ConvolutionCpu {
  // This can be ballanced between additional memory required and speed,
  // it will request memory for a cyclic helper buffer of kStripSize * window_size.
  static constexpr int kStripSize = 64;

  KernelRequirements Setup(KernelContext& ctx, const TensorShape<ndim>& in_shape, int window_size) {
    KernelRequirements req;
    ScratchpadEstimator se;
    DALI_ENFORCE(window_size % 2 == 1,
                 make_string("Kernel window should have odd length, got: ", window_size, "."));
    se.add<mm::memory_kind::host, In>(GetInputWindowBufSize(in_shape, window_size));
    req.scratch_sizes = se.sizes;
    req.output_shapes.push_back(uniform_list_shape<ndim>(1, in_shape));
    return req;
  }

  void Run(KernelContext& ctx, const TensorView<StorageCPU, Out, ndim> &out,
           const TensorView<StorageCPU, const In, ndim>& in,
           const TensorView<StorageCPU, const W, 1>& window, float scale = 1) {
    auto diameter = window.num_elements();
    int input_window_buf_size = GetInputWindowBufSize(in.shape, diameter);
    auto* input_window_buffer = ctx.scratchpad->AllocateHost<In>(input_window_buf_size);
    auto strides = GetStrides(in.shape);

    if (axis == ndim - has_channels - 1 &&
        static_cast<const void*>(out.data) != static_cast<const void*>(in.data)) {
      ConvolveInnerDim<has_channels>(out.data, in.data, window.data, diameter, in.shape, strides,
                                     scale);
    } else {
      ConvolveInplaceOuterLoop<axis, has_channels, kStripSize, Out, In, W, ndim>(
          out.data, in.data, window.data, in.shape, strides, diameter, input_window_buffer, scale);
    }
  }

 private:
  static_assert(0 <= axis && axis < (has_channels ? ndim - 1 : ndim),
                "Selected axis must be in [0, ndim) when there is no channel axis, or in [0, ndim "
                "- 1) for channel-last input");

  int GetInputWindowBufSize(const TensorShape<ndim>& in_shape, int window_size) {
    if (axis == ndim - has_channels - 1) {
      int num_channels = has_channels ? in_shape[ndim - 1] : 1;
      return num_channels * window_size;
    } else {
      return kStripSize * window_size;
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_CONVOLUTION_CPU_H_
