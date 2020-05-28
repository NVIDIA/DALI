// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 * Wraps a pointer to a buffer of type T, with size equal to length * num_channels
 * or just length if has_channels = false.
 */
template <typename T, bool has_channels = true>
class CyclicPixelWrapper {
 public:
  CyclicPixelWrapper(T* ptr, int length, int num_channels = 1)
      : data_(ptr),
        start_(0),
        end_(0),
        elements_(0),
        length_(length),
        num_channels_(num_channels) {}

  /**
   * @brief Drop one pixel from the buffer, moving the start element.
   */
  void PopPixel() {
    assert(elements_ > 0);
    elements_--;
    start_++;
    WrapPosition(start_);
  }

  /**
   * @brief Add a pixel, represented by contiguous `NumChannels()` elements and starting at `input`,
   * to the buffer.
   */
  void PushPixel(const T* input) {
    assert(elements_ < length_);
    for (int c = 0; c < NumChannels(); c++) {
      data_[end_ * NumChannels() + c] = *input;
      input++;
    }
    elements_++;
    end_++;
    WrapPosition(end_);
  }

  /**
   * @brief Add a pixel to the buffer
   */
  void PushPixel(span<const T> input) {
    assert(elements_ < length_);
    for (int c = 0; c < NumChannels(); c++) {
      data_[end_ * NumChannels() + c] = input[c];
    }
    elements_++;
    end_++;
    WrapPosition(end_);
  }

  /**
   * @brief Get pointer to the start of pixel `idx` in the buffer, taking into account the cyclic
   * wrapping
   */
  T* GetPixelOffset(int idx) {
    assert(idx < elements_);
    if (start_ + idx < length_) {
      return data_ + (start_ + idx) * NumChannels();
    } else {
      return data_ + (start_ + idx - length_) * NumChannels();
    }
  }

  /**
   * @brief Calculate dot product with 1-channel `window` length equal to the one specified
   * at construction. The result is stored in pixel stored at `accum`.
   */
  template <typename W>
  void CalculateDot(W* accum, const W* window) {
    assert(elements_ == length_);
    for (int c = 0; c < NumChannels(); c++) {
      accum[c] = 0;
    }
    // TODO(klecki): the if from GetPixelOffset can be factored above the loop
    for (int idx = 0; idx < length_; idx++) {
      const auto* pixel = GetPixelOffset(idx);
      for (int c = 0; c < NumChannels(); c++) {
        accum[c] += window[idx] * pixel[c];
      }
    }
  }

  int Size() {
    return elements_;
  }

  bool Empty() {
    return elements_ == 0;
  }

 private:
  void WrapPosition(int& pos) {
    if (pos == length_) {
      pos = 0;
    }
  }

  template <bool has_channels_ = has_channels>
  std::enable_if_t<has_channels_, int> NumChannels() {
    return num_channels_;
  }

  template <bool has_channels_ = has_channels>
  std::enable_if_t<!has_channels_, int> NumChannels() {
    return 1;
  }

  T* data_ = nullptr;
  int start_ = 0;
  int end_ = 0;  ///< next empty element
  int elements_ = 0;
  int length_ = 0;
  int num_channels_ = 0;
};

template <typename T, bool has_channels>
void load_pixel_with_border(CyclicPixelWrapper<T, has_channels>& cpw, const T* in_ptr, int in_idx,
                            int stride, int axis_size, span<const T> fill_value) {
  cpw.PushPixel(in_ptr + boundary::idx_reflect_101(in_idx, axis_size) * stride);
}

template <typename T, bool has_channels>
void load_pixel_no_border(CyclicPixelWrapper<T, has_channels>& cpw, const T* in_ptr, int in_idx,
                          int stride) {
  cpw.PushPixel(in_ptr + in_idx * stride);
}

constexpr bool is_convolution_inner_loop(int dim, int ndim, bool has_channels) {
  if (has_channels) {
    return dim == ndim - 1;
  } else {
    return dim == ndim;
  }
}
template <int dim, int ndim, bool has_channels>
using is_convolution_inner = std::enable_if_t<is_convolution_inner_loop(dim, ndim, has_channels)>;

template <int dim, int ndim, bool has_channels>
using is_convolution_outer = std::enable_if_t<!is_convolution_inner_loop(dim, ndim, has_channels)>;

// we're in channel dim
template <int axis, bool has_channels, int dim = 0, typename Out, typename In, typename W, int ndim>
is_convolution_inner<dim, ndim, has_channels> ConvolutionCpuImpl(
    Out* out, const In* in, const W* window, const TensorShape<ndim>& shape,
    const TensorShape<ndim>& strides, int d, int64_t offset, span<const In> border_fill,
    In* input_window_buffer, span<W> pixel_tmp, W scale = 1) {
  auto pixel_stride = strides[axis];
  auto axis_size = shape[axis];
  auto num_channels = has_channels ? shape[ndim - 1] : 1;  // channel-last is assumed
  int r = (d - 1) / 2;                                     // radius = (diameter - 1) / 2
  // offset <- start of current axis
  auto* out_ptr = out + offset;
  auto* in_ptr = in + offset;

  CyclicPixelWrapper<In, has_channels> input_window(input_window_buffer, d, num_channels);

  int in_idx = -r, out_idx = 0;
  for (in_idx = -r; in_idx < 0; in_idx++) {
    load_pixel_with_border(input_window, in_ptr, in_idx, pixel_stride, axis_size, border_fill);
  }
  if (r < axis_size) {
    // we load the window without the last element
    for (; in_idx < r; in_idx++) {
      load_pixel_no_border(input_window, in_ptr, in_idx, pixel_stride);
    }
    for (; out_idx < axis_size - r; out_idx++, in_idx++) {
      // we load last element of the input window corresponding to the out_idx
      load_pixel_no_border(input_window, in_ptr, in_idx, pixel_stride);
      // we have both windows as almost-contiguous buffers
      input_window.CalculateDot(pixel_tmp.data(), window);
      for (int c = 0; c < num_channels; c++) {
        out_ptr[out_idx * pixel_stride + c] = ConvertSat<Out>(pixel_tmp[c] * scale);
      }
      // remove one pixel, to make space for next out_idx and in_idx
      input_window.PopPixel();
    }
  } else {
    // we need to load the rest of the window, just handle all with border condition for simplicity
    for (; in_idx < r; in_idx++) {
      load_pixel_with_border(input_window, in_ptr, in_idx, pixel_stride, axis_size, border_fill);
    }
  }
  // we need write out the rest of the outputs, the input window is full of data
  for (; out_idx < axis_size; out_idx++, in_idx++) {
    load_pixel_with_border(input_window, in_ptr, in_idx, pixel_stride, axis_size, border_fill);
    input_window.CalculateDot(pixel_tmp.data(), window);
    for (int c = 0; c < num_channels; c++) {
      out_ptr[out_idx * pixel_stride + c] = ConvertSat<Out>(pixel_tmp[c] * scale);
    }
    input_window.PopPixel();
  }
}

template <int axis, bool has_channels, int dim = 0, typename Out, typename In, typename W, int ndim>
is_convolution_outer<dim, ndim, has_channels> ConvolutionCpuImpl(
    Out* out, const In* in, const W* window, const TensorShape<ndim>& shape,
    const TensorShape<ndim>& strides, int d, int64_t offset, span<const In> border_fill,
    In* input_window_buffer, span<W> pixel_tmp, W scale = 1) {
  if (dim == axis) {
    ConvolutionCpuImpl<axis, has_channels, dim + 1>(out, in, window, shape, strides, d, offset,
                                                    border_fill, input_window_buffer, pixel_tmp,
                                                    scale);
  } else if (dim != axis) {
    for (int64_t i = 0; i < shape[dim]; i++) {
      ConvolutionCpuImpl<axis, has_channels, dim + 1>(out, in, window, shape, strides, d, offset,
                                                      border_fill, input_window_buffer, pixel_tmp,
                                                      scale);
      offset += strides[dim];
    }
  }
}

/**
 * @brief Apply convolution with 1-channel `window` in specified axis.
 *
 * Cyclic sliding window is used when accessing the input, so when the pixels in given axis
 * have big stride the convolution is calculated with basically contiguous buffers
 */
template <typename Out, typename In, typename W, int ndim, int axis, bool has_channels = true>
struct ConvolutionCpu {
  KernelRequirements Setup(KernelContext& ctx, const InTensorCPU<In, ndim>& in,
                           const TensorView<StorageCPU, const W, 1>& window) {
    KernelRequirements req;
    ScratchpadEstimator se;
    DALI_ENFORCE(
        window.num_elements() % 2 == 1,
        make_string("Kernel window should have odd length, got: ", window.num_elements(), "."));
    se.add<In>(AllocType::Host, GetInputWindowBufSize(in, window));
    se.add<In>(AllocType::Host, GetPixelSize(in));  // fill value
    se.add<W>(AllocType::Host, GetPixelSize(in));   // tmp result
    req.scratch_sizes = se.sizes;
    req.output_shapes.push_back(uniform_list_shape<ndim>(1, in.shape));
    return req;
  }

  void Run(KernelContext& ctx, const TensorView<StorageCPU, Out, ndim> out,
           const TensorView<StorageCPU, const In, ndim>& in,
           const TensorView<StorageCPU, const W, 1>& window, W scale = 1) {
    int num_channels = GetPixelSize(in);
    int input_window_buf_size = GetInputWindowBufSize(in, window);
    auto* input_window_buffer =
        ctx.scratchpad->Allocate<In>(AllocType::Host, input_window_buf_size);
    auto* border_fill_buf = ctx.scratchpad->Allocate<In>(AllocType::Host, num_channels);
    auto* pixel_tmp_buf = ctx.scratchpad->Allocate<W>(AllocType::Host, num_channels);
    auto strides = GetStrides(in.shape);
    auto diameter = window.num_elements();

    auto border_fill = make_span(border_fill_buf, num_channels);
    for (int c = 0; c < num_channels; c++) {
      border_fill[c] = 0;
    }
    auto pixel_tmp = make_span(pixel_tmp_buf, num_channels);

    ConvolutionCpuImpl<axis, has_channels, 0, Out, In, W, ndim>(
        out.data, in.data, window.data, in.shape, strides, diameter, 0, border_fill,
        input_window_buffer, pixel_tmp, scale);
  }

 private:
  static_assert(0 <= axis && axis < (has_channels ? ndim - 1 : ndim),
                "Selected axis must be in [0, ndim) when there is no channel axis, or in [0, ndim "
                "- 1) for channel-last input");

  int GetInputWindowBufSize(const TensorView<StorageCPU, const In, ndim>& in,
                            const TensorView<StorageCPU, const W, 1>& window) {
    return GetPixelSize(in) * window.num_elements();
  }
  int GetPixelSize(const TensorView<StorageCPU, const In, ndim>& in) {
    return has_channels ? in.shape[ndim - 1] : 1;
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_CONVOLUTION_CPU_H_
