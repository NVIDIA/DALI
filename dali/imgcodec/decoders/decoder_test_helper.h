// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_IMGCODEC_DECODERS_DECODER_TEST_HELPER_H_
#define DALI_IMGCODEC_DECODERS_DECODER_TEST_HELPER_H_

#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include "dali/core/convert.h"
#include "dali/core/cuda_event_pool.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/static_switch.h"
#include "dali/core/stream.h"
#include "dali/imgcodec/image_decoder_interfaces.h"
#include "dali/imgcodec/image_format.h"
#include "dali/imgcodec/util/convert.h"
#include "dali/imgcodec/util/output_shape.h"
#include "dali/kernels/slice/slice_cpu.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/test/dali_test.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"
#include "dali/util/file.h"
#include "dali/util/numpy.h"

namespace dali {
namespace imgcodec {
namespace test {

inline std::vector<uint8_t> read_file(const std::string &filename) {
  std::ifstream stream(filename, std::ios::binary);
  return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

struct ImageBuffer {
  std::vector<uint8_t> buffer;
  ImageSource src;

  explicit ImageBuffer(const std::string &filename)
    : buffer(read_file(filename))
    , src(ImageSource::FromHostMem(buffer.data(), buffer.size(), filename)) {
  }
};

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
void AssertEqualSatNorm(const TensorView<StorageCPU, const T> &img,
                        const Tensor<CPUBackend> &ref) {
  TYPE_SWITCH(ref.type(), type2id, RefType, NUMPY_ALLOWED_TYPES, (
    Check(img, view<const RefType>(ref), EqualConvertSatNorm());
  ), DALI_FAIL(make_string("Unsupported reference type: ", ref.type())));  // NOLINT
}

static void AssertEqualSatNorm(const Tensor<CPUBackend> &img,
                               const Tensor<CPUBackend> &ref) {
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
  EXPECT_LT(mean_square_error, 0.04);
}


template <typename T>
void AssertSimilar(const TensorView<StorageCPU, const T> &img,
                   const Tensor<CPUBackend> &ref) {
  TYPE_SWITCH(ref.type(), type2id, RefType, NUMPY_ALLOWED_TYPES, (
    AssertSimilar(img, view<const RefType>(ref));
  ), DALI_FAIL(make_string("Unsupported reference type: ", ref.type())));  // NOLINT
}

static void AssertSimilar(const Tensor<CPUBackend> &img,
                          const Tensor<CPUBackend> &ref) {
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
                 const TensorView<StorageCPU, const RefType> &ref,
                 float eps) {
  if (std::is_integral<T>::value)
    eps /= max_value<T>();
  Check(img, ref, EqualConvertNorm(eps));
}

template <typename T>
void AssertClose(const TensorView<StorageCPU, const T> &img,
                 const Tensor<CPUBackend> &ref,
                 float eps) {
  TYPE_SWITCH(ref.type(), type2id, RefType, NUMPY_ALLOWED_TYPES, (
    AssertClose(img, view<const RefType>(ref), eps);
  ), DALI_FAIL(make_string("Unsupported reference type: ", ref.type())));  // NOLINT
}

static void AssertClose(const Tensor<CPUBackend> &img,
                        const Tensor<CPUBackend> &ref,
                        float eps) {
  TYPE_SWITCH(img.type(), type2id, T, (IMGCODEC_TYPES), (
    AssertClose<T>(view<const T>(img), ref, eps);
  ), DALI_FAIL(make_string("Unsupported type: ", img.type())));  // NOLINT
}


/**
 * @brief Extends ROI with channel dimension of given shape.
 */
static ROI ExtendRoi(ROI roi, const TensorShape<> &shape, const TensorLayout &layout) {
  int channel_dim = ImageLayoutInfo::ChannelDimIndex(layout);
  if (channel_dim == -1) channel_dim = shape.size() - 1;

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

static TensorShape<> AdjustToRoi(const TensorShape<> &shape, const ROI &roi) {
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
          const TensorView<StorageCPU, const T, ndim> &input,
          const ROI &requested_roi, const TensorLayout &layout = "HWC") {
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
* @brief Base class template for tests comparing decoder's results with reference images.
*
* @tparam OutputType Type, to which the image should be decoded.
*/
template<typename Backend, typename OutputType>
class DecoderTestBase : public ::testing::Test {
 public:
  explicit DecoderTestBase(int threads_cnt = 4)
    : tp_(threads_cnt, GetDeviceId(), false, "Decoder test") {}

  /**
  * @brief Decodes an image and returns the result as a CPU tensor.
  */
  TensorView<StorageCPU, const OutputType> Decode(ImageSource *src, const DecodeParams &opts = {},
                                                  const ROI &roi = {}) {
    EXPECT_TRUE(Parser()->CanParse(src));
    DecodeContext ctx;
    ctx.tp = &tp_;
    EXPECT_TRUE(Decoder()->CanDecode(ctx, src, opts));

    ImageInfo info = Parser()->GetInfo(src);
    TensorShape<> shape;
    OutputShape(shape, info, opts, roi);

    output_.reshape({{shape}});

    if (GetDeviceId() == CPU_ONLY_DEVICE_ID) {
      auto tv = output_.cpu()[0];
      SampleView<CPUBackend> view(tv.data, tv.shape, type2id<OutputType>::value);
      DecodeResult decode_result = Decoder()->Decode(ctx, view, src, opts, roi);
      EXPECT_TRUE(decode_result.success);
      return tv;
    } else {  // GPU
      auto tv = output_.gpu()[0];
      SampleView<GPUBackend> view(tv.data, tv.shape, type2id<OutputType>::value);
      auto stream_lease = CUDAStreamPool::instance().Get(GetDeviceId());
      ctx.stream = stream_lease;
      auto decode_result = Decoder()->Decode(ctx, view, src, opts, roi);
      EXPECT_TRUE(decode_result.success);
      auto out_tv = output_.cpu(ctx.stream)[0];
      CUDA_CALL(cudaStreamSynchronize(ctx.stream));
      return out_tv;
    }
  }

  /**
   * @brief Decodes a batch of images, invoking the batch version of ImageDecoder::Decode
   */
  TensorListView<StorageCPU, const OutputType> Decode(cspan<ImageSource *> in,
                                                      const DecodeParams &opts = {},
                                                      cspan<ROI> rois = {}) {
    int n = in.size();
    std::vector<TensorShape<>> shape(n);

    DecodeContext ctx;
    ctx.tp = &tp_;

    for (int i = 0; i < n; i++) {
      EXPECT_TRUE(Parser()->CanParse(in[i]));
      EXPECT_TRUE(Decoder()->CanDecode(ctx, in[i], opts));
      ImageInfo info = Parser()->GetInfo(in[i]);
      OutputShape(shape[i], info, opts, rois.empty() ? ROI{} : rois[i]);
    }

    output_.reshape(TensorListShape{shape});

    if (GetDeviceId() == CPU_ONLY_DEVICE_ID) {
      auto tlv = output_.cpu();
      std::vector<SampleView<CPUBackend>> view(n);
      for (int i = 0; i < n; i++)
        view[i] = {tlv[i].data, tlv[i].shape, type2id<OutputType>::value};
      auto res = Decoder()->Decode(ctx, make_span(view), in, opts, rois);
      for (auto decode_result : res)
        EXPECT_TRUE(decode_result.success);
      return tlv;
    } else {  // GPU
      auto tlv = output_.gpu();
      std::vector<SampleView<GPUBackend>> view(n);
      for (int i = 0; i < n; i++)
        view[i] = {tlv[i].data, tlv[i].shape, type2id<OutputType>::value};
      auto stream = CUDAStreamPool::instance().Get(GetDeviceId());
      ctx.stream = stream;
      auto res = Decoder()->Decode(ctx, make_span(view), in, opts, rois);
      for (auto decode_result : res)
        EXPECT_TRUE(decode_result.success);
      auto out_tlv = output_.cpu(ctx.stream);
      CUDA_CALL(cudaStreamSynchronize(ctx.stream));
      return out_tlv;
    }
  }

  /**
  * @brief Returns the parser used.
  */
  std::shared_ptr<ImageParser> Parser() {
    if (!parser_) parser_ = CreateParser();
    return parser_;
  }

  /**
  * @brief Returns the decoder used.
  */
  std::shared_ptr<ImageDecoderInstance> Decoder() {
    if (!decoder_)
      decoder_ = CreateDecoder();
    return decoder_;
  }

  /**
  * @brief Reads the reference image from specified path and returns it as a tensor.
  */
  Tensor<CPUBackend> ReadReferenceFrom(const std::string &reference_path) {
    auto src = FileStream::Open(reference_path, false, false);
    return ReadReference(src.get());
  }

  /**
   * @brief Get device_id for the Backend
   */
  int GetDeviceId() {
    if constexpr (std::is_same<Backend, CPUBackend>::value) {
      return CPU_ONLY_DEVICE_ID;
    } else {
      static_assert(std::is_same<Backend, GPUBackend>::value);
      int device_id;
      CUDA_CALL(cudaGetDevice(&device_id));
      return device_id;
    }
  }

  /**
  * @brief Reads the reference image from specified stream.
  */
  virtual Tensor<CPUBackend> ReadReference(InputStream *src) = 0;

  using Type = OutputType;

 protected:
  DecodeContext Context() {
    DecodeContext ctx;
    ctx.tp = &tp_;
    return ctx;
  }

  /**
  * @brief Creates a decoder instance, working on a specified thread pool.
  */
  virtual std::shared_ptr<ImageDecoderInstance> CreateDecoder() = 0;

  /**
  * @brief Creates a parser to be used.
  */
  virtual std::shared_ptr<ImageParser> CreateParser() = 0;

 private:
  ThreadPool tp_;  // we want the thread pool to outlive the decoder instance
  std::shared_ptr<ImageDecoderInstance> decoder_ = nullptr;
  std::shared_ptr<ImageParser> parser_ = nullptr;
  kernels::TestTensorList<OutputType> output_;
};


/**
* @brief Base class template for tests comparing decoder's results with numpy files.
*
* @tparam OutputType Type, to which the image should be decoded.
*/
template<typename Backend, typename OutputType>
class NumpyDecoderTestBase : public DecoderTestBase<Backend, OutputType> {
 public:
  explicit NumpyDecoderTestBase(int threads_cnt = 4)
  : DecoderTestBase<Backend, OutputType>(threads_cnt) {}

  Tensor<CPUBackend> ReadReference(InputStream *src) override {
    return numpy::ReadTensor(src);
  }
};

}  // namespace test
}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_DECODER_TEST_HELPER_H_
