// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_IMGCODEC_UTIL_CONVERT_H_
#define DALI_IMGCODEC_UTIL_CONVERT_H_

#include <utility>
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/static_switch.h"
#include "dali/core/geom/vec.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/sample_view.h"
#include "dali/kernels/imgproc/color_manipulation/color_space_conversion_impl.h"

#define DALI_IMGCODEC_SUPPORTED_IMAGE_TYPES \
  (DALI_RGB, DALI_GRAY, DALI_YCbCr, DALI_BGR, DALI_ANY_DATA)

namespace dali {
namespace imgcodec {

/**
 * @brief Applies a conversion function `func` to the input data
 *
 * The data is strided - even the innermost dimension can have a non-unit stride.
 * `func` takes a pointer to output and input pointers; it can have some context (but not state),
 * to facilitate color space conversion with strided channels.
 */
template <int static_ndim = -1, typename Out, typename In, typename ConvertFunc>
void Convert(Out *out, const int64_t *out_strides,
             const In *in, const int64_t *in_strides,
             const int64_t *size, int ndim,
             ConvertFunc &&func) {
  if constexpr (static_ndim < 0) {
    VALUE_SWITCH(ndim, NDim, (0, 1, 2, 3, 4),
      (Convert<NDim>(out, out_strides, in, in_strides, size, NDim,
                     std::forward<ConvertFunc>(func));
      return;), ()
    );  // NOLINT
  }

  int64_t extent = size[0];
  int64_t in_stride = in_strides[0];
  int64_t out_stride = out_strides[0];

  if constexpr (static_ndim == 0) {
    func(out, in);
  } else if constexpr (static_ndim == 1) {  // NOLINT - if constexpr not recognized
    for (int64_t i = 0; i < extent; i++) {
      func(out + i * out_stride, in + i * in_stride);
    }
  } else {
    assert(ndim != 1 && "This should go with the static ndim codepath");
    for (int64_t i = 0; i < extent; i++) {
      const int next_ndim = static_ndim < 0 ? -1 : static_ndim - 1;
      Convert<next_ndim>(out + i * out_stride, out_strides + 1,
                         in + i * in_stride, in_strides + 1,
                         size + 1, ndim - 1,
                         std::forward<ConvertFunc>(func));
    }
  }
}

/**
 * @brief Converts a data type of a vector.
 */
template <typename Out, typename In, int N>
inline vec<N, Out> ConvertSatNormVec(const vec<N, In> &in) {
  vec<N, Out> out = {};
  for (int i = 0; i < N; i++)
    out[i] = ConvertSatNorm<Out>(in[i]);
  return out;
}

/**
 * @brief A functor for converting between color spaces.
 *
 * It reads the input data from memory, passes it to chosen color conversion function, and then
 * stores the result. Both the argument and return value of the conversion function can either be
 * a scalar or a vector.
 */
template <typename Out, typename In, DALIImageType OutFormat, DALIImageType InFormat>
struct ConvertColorSpace {
  void operator()(Out *out_mem, const In *in_mem) {
    ptrdiff_t out_channel_stride = out_channel_stride_, in_channel_stride = in_channel_stride_;
    ptrdiff_t out_offset = 0, in_offset = 0;

    // For BGR data we will use RGB conversion functions, but we will access channels in the pixel
    // in opposite order.
    if constexpr (InFormat == DALI_BGR) {
      in_offset = 2 * in_channel_stride_;
      in_channel_stride = -in_channel_stride_;
    }
    if constexpr (OutFormat == DALI_BGR) {
      out_offset = 2 * out_channel_stride_;
      out_channel_stride = -out_channel_stride_;
    }

    auto f = GetConversionFunction();
    typename FunctionInfo<decltype(f)>::arg_type input;

    if constexpr (is_vec<decltype(input)>::value) {
      for (int i = 0; i < input.size(); i++)
        input[i] = in_mem[i * in_channel_stride + in_offset];
    } else {
      input = *in_mem;
    }

    auto output = f(input);

    if constexpr (is_vec<decltype(output)>::value) {
      for (int i = 0; i < output.size(); i++)
        out_mem[i * out_channel_stride + out_offset] = output[i];
    } else {
      *out_mem = output;
    }
  }

  static constexpr auto GetConversionFunction() {
    // BGR conversions will use the RGB conversion functions, but we will call them with negative
    // strides to access the colors in reverse order (see the constructor)
    constexpr bool InRgbOrBgr = (InFormat == DALI_RGB || InFormat == DALI_BGR);
    constexpr bool OutRgbOrBgr = (OutFormat == DALI_RGB || OutFormat == DALI_BGR);

    if constexpr (InRgbOrBgr && OutRgbOrBgr) {
      return ConvertSatNormVec<Out, In, 3>;
    } else if constexpr (InRgbOrBgr && OutFormat == DALI_GRAY) {
      return kernels::color::rgb_to_gray<Out, In>;
    } else if constexpr (InRgbOrBgr && OutFormat == DALI_YCbCr) {
      return kernels::color::itu_r_bt_601::rgb_to_ycbcr<Out, In>;
    } else if constexpr (InFormat == DALI_GRAY && OutRgbOrBgr) {
      return kernels::color::gray_to_rgb<Out, In>;
    } else if constexpr (InFormat == DALI_GRAY && OutFormat == DALI_YCbCr) {
      return kernels::color::itu_r_bt_601::gray_to_ycbcr<Out, In>;
    } else if constexpr (InFormat == DALI_YCbCr && OutRgbOrBgr) {
      return kernels::color::itu_r_bt_601::ycbcr_to_rgb<Out, In>;
    } else if constexpr (InFormat == DALI_YCbCr && OutFormat == DALI_GRAY) {
      return kernels::color::itu_r_bt_601::ycbcr_to_gray<Out, In>;
    } else {
      return ConversionErrorFunction;
    }
  }

  static constexpr Out ConversionErrorFunction(In) {
    throw std::logic_error(make_string("Not implemented: conversion from ", to_string(InFormat),
                                       " to ", to_string(OutFormat), " is not supported"));
  }

  template <typename F> struct FunctionInfo;

  template <typename FuncRet, typename FuncArg>
  struct FunctionInfo<FuncRet(*)(FuncArg)> {
    using arg_type = typename std::remove_const<
                        typename std::remove_reference<FuncArg>::type>::type;
    using ret_type = FuncRet;
  };

  ptrdiff_t out_channel_stride_, in_channel_stride_;
};

/**
 * @brief Converts a data type of a single-channel value.
 */
template <typename Out, typename In>
inline void ConvertDType(Out *out, const In *in) {
  *out = ConvertSatNorm<Out>(*in);
}

/**
 * @brief Converts an image stored in `in` and stores it in `out`.
 *
 * This is a wrapper for more generic variant of Convert. Based on image format and strides it
 * chooses an appropriate conversion function and runs the generic Convert with it.
 */
template <typename Out, typename In>
void Convert(Out *out, const int64_t *out_strides, int out_channel_dim, DALIImageType out_format,
             const In *in, const int64_t *in_strides, int in_channel_dim, DALIImageType in_format,
             const int64_t *size, int ndim) {
  // TODO(skarpinski) Support other layouts
  DALI_ENFORCE(out_channel_dim == ndim - 1 && in_channel_dim == ndim - 1,
    "Not implemented: currently only channels-last layout is supported");

  VALUE_SWITCH(out_format, OutFormat, DALI_IMGCODEC_SUPPORTED_IMAGE_TYPES, (
    VALUE_SWITCH(in_format, InFormat, DALI_IMGCODEC_SUPPORTED_IMAGE_TYPES, (
      if constexpr (OutFormat == InFormat || OutFormat == DALI_ANY_DATA) {
        Convert(out, out_strides, in, in_strides, size, ndim, &ConvertDType<Out, In>);
      } else {
        // If the color conversion will be needed, we strip the last (channel) dimension to let
        // the conversion function work on whole pixels and not single values.
        auto func = ConvertColorSpace<Out, In, OutFormat, InFormat>{1, 1};
        Convert(out, out_strides, in, in_strides, size, ndim - 1, func);
      }
    ), throw std::logic_error(  // NOLINT
        make_string("Unsupported input format" , to_string(in_format))););  // NOLINT
  ), throw std::logic_error(  // NOLINT
      make_string("Unsupported output format " , to_string(out_format))););  // NOLINT
}

/**
 * @brief Converts an image stored in `in` and stores it in `out`.
 *
 * The function converts data type (normalizing) and color space.
 * When roi_start or roi_end is empty, it is assumed to be the lower bound and upport bound
 * of the spatial extent. Channel dimension must not be included in ROI specification.
 */
void DLL_PUBLIC Convert(
    SampleView<CPUBackend> out, TensorLayout out_layout, DALIImageType out_format,
    ConstSampleView<CPUBackend> in, TensorLayout in_layout, DALIImageType in_format,
    TensorShape<> roi_start, TensorShape<> roi_end);


}  // namespace imgcodec
}  // namespace dali


#endif  // DALI_IMGCODEC_UTIL_CONVERT_H_
