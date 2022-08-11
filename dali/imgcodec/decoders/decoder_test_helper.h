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

#ifndef DALI_IMGCODEC_DECODERS_DECODER_TEST_HELPER_H_
#define DALI_IMGCODEC_DECODERS_DECODER_TEST_HELPER_H_

#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include "dali/pipeline/data/tensor.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"
#include "dali/imgcodec/image_format.h"
#include "dali/imgcodec/image_decoder.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/test/dali_test.h"
#include "dali/test/test_tensors.h"
#include "dali/kernels/slice/slice_cpu.h"
#include "dali/core/stream.h"
#include "dali/util/file.h"
#include "dali/util/numpy.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/core/cuda_event_pool.h"
#include "dali/core/cuda_stream_pool.h"

namespace dali {
namespace imgcodec {
namespace test {

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
  TensorView<StorageCPU, OutputType> Decode(ImageSource *src, const DecodeParams &opts = {},
                                            const ROI &roi = {}) {
    EXPECT_TRUE(Parser()->CanParse(src));
    EXPECT_TRUE(Decoder()->CanDecode(src, opts));

    ImageInfo info = Parser()->GetInfo(src);
    auto shape = AdjustToRoi(info.shape, roi);
    output_.reshape({{shape}});

    if (GetDeviceId() == CPU_ONLY_DEVICE_ID) {
      auto tv = output_.cpu()[0];
      SampleView<CPUBackend> view(tv.data, tv.shape, type2id<OutputType>::value);
      DecodeResult decode_result = Decoder()->Decode(view, src, opts, roi);
      EXPECT_TRUE(decode_result.success);
      return tv;
    } else {  // GPU
      auto tv = output_.gpu()[0];
      SampleView<GPUBackend> view(tv.data, tv.shape, type2id<OutputType>::value);
      auto stream = CUDAStreamPool::instance().Get(GetDeviceId());
      auto decode_result = Decoder()->Decode(stream, view, src, opts, roi);
      EXPECT_TRUE(decode_result.success);
      CUDA_CALL(cudaStreamSynchronize(stream));
      return output_.cpu()[0];
    }
  }

  /**
   * @brief Decodes a batch of images, invoking the batch version of ImageDecoder::Decode
   */
  TensorListView<StorageCPU, OutputType> Decode(cspan<ImageSource *> in,
                                                const DecodeParams &opts = {},
                                                cspan<ROI> rois = {}) {
    int n = in.size();
    std::vector<TensorShape<>> shape(n);

    for (int i = 0; i < n; i++) {
      EXPECT_TRUE(Parser()->CanParse(in[i]));
      EXPECT_TRUE(Decoder()->CanDecode(in[i], opts));
      ImageInfo info = Parser()->GetInfo(in[i]);
      shape[i] = AdjustToRoi(info.shape, rois.empty() ? ROI{} : rois[i]);
    }

    output_.reshape(TensorListShape{shape});

    if (GetDeviceId() == CPU_ONLY_DEVICE_ID) {
      auto tlv = output_.cpu();
      std::vector<SampleView<CPUBackend>> view(n);
      for (int i = 0; i < n; i++)
        view[i] = {tlv[i].data, tlv[i].shape, type2id<OutputType>::value};
      auto res = Decoder()->Decode(make_span(view), in, opts, rois);
      for (auto decode_result : res)
        EXPECT_TRUE(decode_result.success);
      return tlv;
    } else {  // GPU
      auto tlv = output_.gpu();
      std::vector<SampleView<GPUBackend>> view(n);
      for (int i = 0; i < n; i++)
        view[i] = {tlv[i].data, tlv[i].shape, type2id<OutputType>::value};
      auto stream = CUDAStreamPool::instance().Get(GetDeviceId());
      auto res = Decoder()->Decode(stream, make_span(view), in, opts, rois);
      for (auto decode_result : res)
        EXPECT_TRUE(decode_result.success);
      CUDA_CALL(cudaStreamSynchronize(stream));
      return output_.cpu();
    }
  }

  /**
  * @brief Checks if the image and the reference are equal
  */
  void AssertEqual(const TensorView<StorageCPU, OutputType> &img,
                   const TensorView<StorageCPU, OutputType> &ref) {
    Check(img, ref);
  }

  /**
  * @brief Checks if the image and the reference are equal after converting the reference
  * with ConvertSatNorm
  */
  template <typename RefType>
  void AssertEqualSatNorm(const TensorView<StorageCPU, OutputType> &img,
                          const TensorView<StorageCPU, RefType> &ref) {
    Check(img, ref, EqualConvertSatNorm());
  }

  void AssertEqualSatNorm(const TensorView<StorageCPU, OutputType> &img,
                          const Tensor<CPUBackend> &ref) {
    TYPE_SWITCH(ref.type(), type2id, RefType, NUMPY_ALLOWED_TYPES, (
      Check(img, view<const RefType>(ref), EqualConvertSatNorm());
    ), DALI_FAIL(make_string("Unsupported reference type: ", ref.type())));  // NOLINT
  }

  void AssertEqualSatNorm(const Tensor<CPUBackend> &img,
                          const Tensor<CPUBackend> &ref) {
    AssertEqualSatNorm(view<const OutputType>(img), ref);
  }

  /**
  * @brief Checks if an image is close to a reference
  *
  * The eps parameter shound be specified in the dynamic range of the image.
  */
  template <typename RefType>
  void AssertClose(const TensorView<StorageCPU, OutputType> &img,
                   const TensorView<StorageCPU, RefType> &ref,
                   float eps) {
    if (std::is_integral<OutputType>::value)
      eps /= max_value<OutputType>();
    Check(img, ref, EqualConvertNorm(eps));
  }

  void AssertClose(const TensorView<StorageCPU, OutputType> &img,
                   const Tensor<CPUBackend> &ref,
                   float eps) {
    TYPE_SWITCH(ref.type(), type2id, RefType, NUMPY_ALLOWED_TYPES, (
      Check(img, view<const RefType>(ref), EqualConvertNorm(eps));
    ), DALI_FAIL(make_string("Unsupported reference type: ", ref.type())));  // NOLINT
  }

  void AssertClose(const Tensor<CPUBackend> &img,
                   const Tensor<CPUBackend> &ref,
                   float eps) {
    TYPE_SWITCH(ref.type(), type2id, RefType, NUMPY_ALLOWED_TYPES, (
      Check(view<const OutputType>(img), view<const RefType>(ref), EqualConvertNorm(eps));
    ), DALI_FAIL(make_string("Unsupported reference type: ", ref.type())));  // NOLINT
  }


  /**
  * @brief Crops a tensor to specified roi_shape, anchored at roi_begin.
  * Does not support padding.
  */
  template <typename T, int ndim>
  void Crop(const TensorView<StorageCPU, T, ndim> &output,
            const TensorView<StorageCPU, T, ndim> &input,
            const ROI &roi) {
    static_assert(ndim >= 0, "expected static ndim");
    ASSERT_TRUE(output.shape() == roi.shape());  // output should have the desired shape

    kernels::SliceCPU<T, T, ndim> kernel;
    kernels::SliceArgs<T, ndim> args;
    args.anchor = roi.begin;
    args.shape = roi.shape();
    kernels::KernelContext ctx;
    // no need to run Setup (we already know the output shape)
    kernel.Run(ctx, output, input, args);
  }

  template <typename T, int ndim>
  Tensor<CPUBackend> Crop(const TensorView<StorageCPU, T, ndim> &input,
                          const ROI &roi) {
    auto num_dims = input.shape.sample_dim();
    assert(roi.shape().sample_dim() == num_dims);
    Tensor<CPUBackend> output;
    output.Resize(roi.shape(), type2id<OutputType>::value);

    VALUE_SWITCH(num_dims, Dims, (2, 3, 4), (
      auto out_view = view<OutputType, Dims>(output);
      Crop(out_view, input, roi);
    ), DALI_FAIL(make_string("Unsupported number of dimensions: ", ndim)););  // NOLINT

    return output;
  }

  Tensor<CPUBackend> Crop(const Tensor<CPUBackend> &input, const ROI &roi) {
    return Crop(view<const OutputType>(input), roi);
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
    if (!decoder_) decoder_ = CreateDecoder(tp_);
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

 protected:
  /**
  * @brief Creates a decoder instance, working on a specified thread pool.
  */
  virtual std::shared_ptr<ImageDecoderInstance> CreateDecoder(ThreadPool &tp) = 0;

  /**
  * @brief Creates a parser to be used.
  */
  virtual std::shared_ptr<ImageParser> CreateParser() = 0;

 private:
  TensorShape<> AdjustToRoi(const TensorShape<> &shape, const ROI &roi) {
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

  std::shared_ptr<ImageDecoderInstance> decoder_ = nullptr;
  std::shared_ptr<ImageParser> parser_ = nullptr;
  kernels::TestTensorList<OutputType> output_;
  ThreadPool tp_;
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
