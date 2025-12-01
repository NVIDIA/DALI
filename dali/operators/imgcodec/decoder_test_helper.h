// Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMGCODEC_DECODER_TEST_HELPER_H_
#define DALI_OPERATORS_IMGCODEC_DECODER_TEST_HELPER_H_

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include "dali/core/mm/memory.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/kernels/slice/slice_cpu.h"
#include "dali/operators/imgcodec/util/convert.h"
#include "dali/pipeline/data/views.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/util/file.h"
#include "dali/util/numpy.h"

namespace dali {
namespace imgcodec {
namespace test {

inline cv::Mat rgb2bgr(const cv::Mat &img) {
  cv::Mat bgr;
  cv::cvtColor(img, bgr, cv::COLOR_RGB2BGR);
  return bgr;
}

inline cv::Mat bgr2rgb(const cv::Mat &img) {
  return rgb2bgr(img);
}

/**
 * @brief Extends ROI with channel dimension of given shape.
 */
inline ROI ExtendRoi(ROI roi, const TensorShape<> &shape, const TensorLayout &layout) {
  int channel_dim = ImageLayoutInfo::ChannelDimIndex(layout);
  if (channel_dim == -1)
    channel_dim = shape.size() - 1;

  int ndim = shape.sample_dim();
  if (roi.begin.size() == ndim - 1) {
    roi.begin = shape_cat(shape_cat(roi.begin.first(channel_dim), 0),
                          roi.begin.last(roi.begin.size() - channel_dim));
  }
  if (roi.end.size() == ndim - 1) {
    roi.end = shape_cat(shape_cat(roi.end.first(channel_dim), shape[channel_dim]),
                        roi.end.last(roi.end.size() - channel_dim));
  }
  return roi;
}

inline TensorShape<> AdjustToRoi(const TensorShape<> &shape, const ROI &roi) {
  if (roi) {
    auto result = roi.shape();
    int ndim = shape.sample_dim();
    if (roi.shape().sample_dim() != ndim) {
      assert(roi.shape().sample_dim() + 1 == ndim);
      result.resize(ndim);
      result[ndim - 1] = shape[ndim - 1];
    }
    return result;
  } else {
    return shape;
  }
}

/**
 * @brief Crops a tensor to specified roi_shape, anchored at roi_begin.
 * Does not support padding.
 */
template <typename T, int ndim>
void Crop(const TensorView<StorageCPU, T, ndim> &output,
          const TensorView<StorageCPU, const T, ndim> &input, const ROI &requested_roi,
          const TensorLayout &layout = "HWC") {
  auto roi = ExtendRoi(requested_roi, input.shape, layout);

  static_assert(ndim >= 0, "expected static ndim");
  ASSERT_TRUE(output.shape == roi.shape());  // output should have the desired shape

  kernels::SliceCPU<T, T, ndim> kernel;
  kernels::SliceArgs<T, ndim> args;
  args.anchor = roi.begin;
  args.shape = roi.shape();
  kernels::KernelContext ctx;
  // no need to run Setup (we already know the output shape)
  kernel.Run(ctx, output, input, args);
}

template <typename T, int ndim>
Tensor<CPUBackend> Crop(const TensorView<StorageCPU, const T, ndim> &input,
                        const ROI &requested_roi, const TensorLayout &layout = "HWC") {
  auto roi = ExtendRoi(requested_roi, input.shape, layout);
  auto num_dims = input.shape.sample_dim();
  assert(roi.shape().sample_dim() == num_dims);
  Tensor<CPUBackend> output;
  output.Resize(roi.shape(), type2id<T>::value);

  auto out_view = view<T, ndim>(output);
  Crop(out_view, input, roi);

  return output;
}

static Tensor<CPUBackend> Crop(const Tensor<CPUBackend> &input, const ROI &roi) {
  int ndim = input.shape().sample_dim();
  VALUE_SWITCH(ndim, Dims, (2, 3, 4), (
    TYPE_SWITCH(input.type(), type2id, InputType, (IMGCODEC_TYPES, double), (
      return Crop(view<const InputType, Dims>(input), roi, input.GetLayout());
    ), DALI_FAIL(make_string("Unsupported type ", input.type())););  // NOLINT
  ), DALI_FAIL(make_string("Unsupported number of dimensions: ", ndim)););  // NOLINT
}

/**
 * @brief Checks if the image and the reference are equal
 */
template <typename T>
void AssertEqual(const TensorView<StorageCPU, const T> &img,
                 const TensorView<StorageCPU, const T> &ref) {
  Check(img, ref);
}

/**
 * @brief Checks if the image and the reference are equal after converting the reference
 * with ConvertSatNorm
 */
template <typename T, typename RefType>
void AssertEqualSatNorm(const TensorView<StorageCPU, const T> &img,
                        const TensorView<StorageCPU, const RefType> &ref) {
  Check(img, ref, EqualConvertSatNorm());
}

template <typename T>
void AssertEqualSatNorm(const TensorView<StorageCPU, const T> &img, const Tensor<CPUBackend> &ref) {
  TYPE_SWITCH(ref.type(), type2id, RefType, NUMPY_ALLOWED_TYPES, (
    Check(img, view<const RefType>(ref), EqualConvertSatNorm());
  ), DALI_FAIL(make_string("Unsupported reference type: ", ref.type())));  // NOLINT
}

static void AssertEqualSatNorm(const Tensor<CPUBackend> &img, const Tensor<CPUBackend> &ref) {
  TYPE_SWITCH(img.type(), type2id, T, (IMGCODEC_TYPES), (
    AssertEqualSatNorm<T>(view<const T>(img), ref);
  ), DALI_FAIL(make_string("Unsupported type: ", img.type())));  // NOLINT
}


/**
 * @brief Checks that the image is similar to the reference based on mean square error
 *
 * Unlike AssertClose, this check allows for a rather large max. error, but has a much lower
 * limit on mean square error. It's used for different JPEG decoders which have differences in
 * chroma upsampling or apply deblocking filters.
 */
template <typename OutputType, typename RefType>
void AssertSimilar(const TensorView<StorageCPU, const OutputType> &img,
                   const TensorView<StorageCPU, const RefType> &ref) {
  float eps = ConvertSatNorm<OutputType>(0.3);
  Check(img, ref, EqualConvertNorm(eps));

  double mean_square_error = 0;
  uint64_t size = volume(img.shape);
  for (size_t i = 0; i < size; i++) {
    double img_value = ConvertSatNorm<double>(img.data[i]);
    double ref_value = ConvertSatNorm<double>(ref.data[i]);
    double error = img_value - ref_value;
    mean_square_error += error * error;
  }
  mean_square_error = sqrt(mean_square_error / size);
  if (mean_square_error >= 0.04 && img.shape[2] == 3) {
    cv::Mat img_mat(img.shape[0], img.shape[1], CV_8UC3,
                    const_cast<void *>(reinterpret_cast<const void *>(img.data)));
    cv::Mat ref_mat(ref.shape[0], ref.shape[1], CV_8UC3,
                    const_cast<void *>(reinterpret_cast<const void *>(ref.data)));
    cv::Mat diff;
    cv::absdiff(img_mat, ref_mat, diff);

    cv::imwrite("out.bmp", rgb2bgr(img_mat));
    cv::imwrite("ref.bmp", rgb2bgr(ref_mat));
    cv::imwrite("diff.bmp", rgb2bgr(diff));
  }
  EXPECT_LT(mean_square_error, 0.04);
}


template <typename T>
void AssertSimilar(const TensorView<StorageCPU, const T> &img, const Tensor<CPUBackend> &ref) {
  TYPE_SWITCH(ref.type(), type2id, RefType, NUMPY_ALLOWED_TYPES, (
    AssertSimilar(img, view<const RefType>(ref));
  ), DALI_FAIL(make_string("Unsupported reference type: ", ref.type())));  // NOLINT
}

static void AssertSimilar(const Tensor<CPUBackend> &img, const Tensor<CPUBackend> &ref) {
  TYPE_SWITCH(img.type(), type2id, T, (IMGCODEC_TYPES), (
    AssertSimilar<T>(view<const T>(img), ref);
  ), DALI_FAIL(make_string("Unsupported type: ", img.type())));  // NOLINT
}


/**
 * @brief Checks if an image is close to a reference
 *
 * The eps parameter should be specified in the dynamic range of the image.
 */
template <typename T, typename RefType>
void AssertClose(const TensorView<StorageCPU, const T> &img,
                 const TensorView<StorageCPU, const RefType> &ref, float eps) {
  if (std::is_integral<T>::value)
    eps /= max_value<T>();
  Check(img, ref, EqualConvertNorm(eps));
}

template <typename T>
void AssertClose(const TensorView<StorageCPU, const T> &img, const Tensor<CPUBackend> &ref,
                 float eps) {
  TYPE_SWITCH(ref.type(), type2id, RefType, NUMPY_ALLOWED_TYPES, (
    AssertClose(img, view<const RefType>(ref), eps);
  ), DALI_FAIL(make_string("Unsupported reference type: ", ref.type())));  // NOLINT
}

inline void AssertClose(const Tensor<CPUBackend> &img, const Tensor<CPUBackend> &ref, float eps) {
  TYPE_SWITCH(img.type(), type2id, T, (IMGCODEC_TYPES), (
    AssertClose<T>(view<const T>(img), ref, eps);
  ), DALI_FAIL(make_string("Unsupported type: ", img.type())));  // NOLINT
}

inline Tensor<CPUBackend> ReadReference(InputStream *src, TensorLayout layout = "HWC") {
  auto tensor = numpy::ReadTensor(src, false);
  tensor.SetLayout(layout);
  return tensor;
}

/**
 * @brief Reads the reference image from specified path and returns it as a tensor.
 */
inline Tensor<CPUBackend> ReadReferenceFrom(const std::string &reference_path,
                                            TensorLayout layout = "HWC") {
  auto src = FileStream::Open(reference_path);
  return ReadReference(src.get(), layout);
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_OPERATORS_IMGCODEC_DECODER_TEST_HELPER_H_
