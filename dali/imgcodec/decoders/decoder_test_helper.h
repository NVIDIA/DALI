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

#include <string>
#include <memory>
#include <vector>
#include "dali/pipeline/data/tensor.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"
#include "dali/imgcodec/image_format.h"
#include "dali/imgcodec/image_decoder.h"
#include "dali/test/dali_test.h"
#include "dali/kernels/slice/slice_cpu.h"
#include "dali/core/stream.h"
#include "dali/util/file.h"
#include "dali/util/numpy.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {
namespace imgcodec {
namespace test {

/**
* @brief Base class template for tests comparing decoder's results with reference images.
*
* @tparam OutputType Type, to which the image should be decoded.
*/
template<typename OutputType>
class CpuDecoderTestBase : public ::testing::Test {
 public:
  CpuDecoderTestBase() : tp_(4, CPU_ONLY_DEVICE_ID, false, "Decoder test") {}

  /**
  * @brief Decodes an image and returns the result as a tensor.
  */
  Tensor<CPUBackend> Decode(ImageSource *src, const DecodeParams &opts = {}, const ROI &roi = {}) {
    EXPECT_TRUE(Parser()->CanParse(src));
    ImageInfo info = Parser()->GetInfo(src);

    Tensor<CPUBackend> result;
    EXPECT_TRUE(Decoder()->CanDecode(src, opts));
    TensorShape<> shape = (roi.use_roi() ? roi.shape() : info.shape);
    result.Resize(shape, type2id<OutputType>::value);

    SampleView<CPUBackend> view(result.raw_mutable_data(), result.shape(), result.type());
    DecodeResult decode_result = Decoder()->Decode(view, src, opts, roi);
    EXPECT_TRUE(decode_result.success);

    return result;
  }

  /**
  * @brief Checks if two tensors are equal.
  */
  void AssertEqual(const Tensor<CPUBackend> &img, const Tensor<CPUBackend> &ref) {
    Check(view<const OutputType>(img), view<const OutputType>(ref));
  }

  /**
  * @brief Checks if two tensors are equal after converting the second tensor with ConvertSatNorm
  */
  void AssertEqualSatNorm(const Tensor<CPUBackend> &img, const Tensor<CPUBackend> &ref) {
    TYPE_SWITCH(ref.type(), type2id, RefType, NUMPY_ALLOWED_TYPES, (
      Check(view<const OutputType>(img), view<const RefType>(ref), EqualConvertSatNorm());
    ), DALI_FAIL(make_string("Unsupported reference type: ", ref.type())));  // NOLINT
  }

  /**
  * @brief Crops a tensor to specified roi_shape, anchored at roi_begin.
  * Does not support padding.
  */
  Tensor<CPUBackend> Crop(const Tensor<CPUBackend> &input, const ROI &roi) {
    int ndim = input.shape().sample_dim();
    Tensor<CPUBackend> output;
    output.Resize(roi.shape(), type2id<OutputType>::value);

    VALUE_SWITCH(ndim, Dims, (2, 3, 4), (
      auto out_view = view<OutputType, Dims>(output);
      auto in_view = view<const OutputType, Dims>(input);
      kernels::SliceCPU<OutputType, OutputType, Dims> kernel;
      kernels::SliceArgs<OutputType, Dims> args;
      args.anchor = roi.begin;
      args.shape = roi.shape();
      kernels::KernelContext ctx;
      // no need to run Setup (we already know the output shape)
      kernel.Schedule(ctx, out_view, in_view, args, tp_);
      tp_.RunAll();
    ), DALI_FAIL(make_string("Unsupported number of dimensions: ", ndim)););  // NOLINT

    return output;
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
  std::shared_ptr<ImageDecoderInstance> decoder_ = nullptr;
  std::shared_ptr<ImageParser> parser_ = nullptr;
  ThreadPool tp_;
};


/**
* @brief Base class template for tests comparing decoder's results with numpy files.
*
* @tparam OutputType Type, to which the image should be decoded.
*/
template<typename OutputType>
class NumpyDecoderTestBase : public CpuDecoderTestBase<OutputType> {
 private:
  /**
  * @brief Reads a tensor from numpy file.
  */
  Tensor<CPUBackend> ReadNumpy(InputStream *src) {
    numpy::HeaderData header;
    numpy::ParseHeader(header, src);
    src->SeekRead(header.data_offset, SEEK_SET);

    Tensor<CPUBackend> data;
    data.Resize(header.shape, header.type());
    Index ret = src->Read(static_cast<uint8_t*>(data.raw_mutable_data()), header.nbytes());
    EXPECT_EQ(ret, (Index)header.nbytes()) << "Failed to read reference numpy file";

    if (header.fortran_order) {
      Tensor<CPUBackend> transposed;
      transposed.Resize(data.shape(), data.type());
      SampleView<CPUBackend> input(data.raw_mutable_data(), data.shape(), data.type());
      SampleView<CPUBackend> output(transposed.raw_mutable_data(), transposed.shape(),
                                    transposed.type());
      numpy::FromFortranOrder(output, input);
      return transposed;
    }
    return data;
  }

 public:
  Tensor<CPUBackend> ReadReference(InputStream *src) override {
    return ReadNumpy(src);
  }
};

}  // namespace test
}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_DECODER_TEST_HELPER_H_
