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

#ifndef DALI_IMGCODEC_DECODERS_DECODER_TEST_H_
#define DALI_IMGCODEC_DECODERS_DECODER_TEST_H_

#include "dali/pipeline/data/tensor.h"
#include "dali/operators/reader/loader/numpy_loader.h"
#include "dali/kernels/transpose/transpose.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"
#include "dali/imgcodec/decoders/test/numpy_helper.h"
#include "dali/imgcodec/image_format.h"
#include "dali/imgcodec/image_decoder.h"
#include "dali/test/dali_test.h"
#include "dali/kernels/slice/slice_cpu.h"

namespace dali {
namespace imgcodec {
namespace test {

template<typename OutputType>
class CpuDecoderTestBase : public ::testing::Test {
 public:
  CpuDecoderTestBase() : tp_(4, CPU_ONLY_DEVICE_ID, false, "Decoder test") {}

  Tensor<CPUBackend> Decode(ImageSource *src, const DecodeParams &opts = {}) {
    Init();

    EXPECT_TRUE(parser_->CanParse(src));
    ImageInfo info = parser_->GetInfo(src);

    Tensor<CPUBackend> result;
    EXPECT_TRUE(decoder_->CanDecode(src, opts));
    TensorShape<> shape = (opts.use_roi ? opts.roi.shape() : info.shape);
    result.Resize(shape, type2id<OutputType>::value);

    SampleView<CPUBackend> view(result.raw_mutable_data(), result.shape(), result.type());
    DecodeResult decode_result = decoder_->Decode(view, src, opts);
    EXPECT_TRUE(decode_result.success);
    
    return result;
  }

  void AssertEqual(const Tensor<CPUBackend> &img, const Tensor<CPUBackend> &ref) {
    EXPECT_EQ(img.shape(), ref.shape()) << "Different shapes";
    auto img_view = view<const OutputType>(img), ref_view = view<const OutputType>(ref);
    for (int i = 0; i < volume(img.shape()); i++) {
      // TODO(skarpinski) Pretty-print position on error
      ASSERT_EQ(img_view.data[i], ref_view.data[i]) << "Pixel " << i;
    }
  }

  Tensor<CPUBackend> Crop(const Tensor<CPUBackend> &input,
                          const TensorShape<> &roi_begin, const TensorShape<> &roi_shape) {
    int ndim = input.shape().sample_dim();
    Tensor<CPUBackend> output;
    output.Resize(roi_shape, type2id<OutputType>::value);

    VALUE_SWITCH(ndim, Dims, (2, 3, 4), (
      auto out_view = view<OutputType, Dims>(output);
      auto in_view = view<const OutputType, Dims>(input);
      kernels::SliceCPU<OutputType, OutputType, Dims> kernel;
      kernels::SliceArgs<OutputType, Dims> args;
      args.anchor = roi_begin;
      args.shape = roi_shape;
      kernels::KernelContext ctx;
      // no need to run Setup (we already know the output shape)
      kernel.Schedule(ctx, out_view, in_view, args, tp_);
      tp_.RunAll();
    ), DALI_FAIL(make_string("Unsupported number of dimensions: ", ndim)););  // NOLINT

    return output;
  }

  ImageInfo GetInfo(ImageSource *src) {
    Init();
    return parser_->GetInfo(src);
  }

  virtual Tensor<CPUBackend> ReadReference(const std::string &reference_path) = 0;

 protected:
  virtual std::shared_ptr<ImageDecoderInstance> CreateDecoder(ThreadPool &tp) = 0;
  virtual std::shared_ptr<ImageParser> GetParser() = 0;

 private:
  void Init() {
    if (!parser_) parser_ = GetParser();
    if (!decoder_) decoder_ = CreateDecoder(tp_);
  }

  std::shared_ptr<ImageDecoderInstance> decoder_ = nullptr;
  std::shared_ptr<ImageParser> parser_ = nullptr;
  ThreadPool tp_;
};


template<typename OutputType>
class NumpyDecoderTestBase : public CpuDecoderTestBase<OutputType> {
 private:
  template<typename InputType>
  Tensor<CPUBackend> Convert(const Tensor<CPUBackend> &in) {
    Tensor<CPUBackend> out;
    out.Resize(in.shape(), type2id<OutputType>::value);
    auto in_view = view<const InputType>(in), out_view = view<OutputType>(out);
    for (int i = 0; i < volume(in.shape()); i++) {
      out_view.data[i] = ConvertSatNorm<OutputType>(in_view.data[i]);
    }
    return out;
  }
 public:
  Tensor<CPUBackend> ReadReference(const std::string &reference_path) override {
    Tensor<CPUBackend> ref = ReadNumpy(reference_path);

    DALIDataType output_type = type2id<OutputType>::value;
    TYPE_SWITCH(ref.type(), type2id, InputType, NUMPY_ALLOWED_TYPES, (
      return Convert<InputType>(ref);
    ), DALI_FAIL(make_string("Unsupported numpy reference type: ", ref.type())));  // NOLINT
  }
};

}  // namespace test
}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_DECODER_TEST_H_
