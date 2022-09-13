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

#include "dali/imgcodec/image_decoder.h"
#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include "dali/core/cuda_event_pool.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/stream.h"
#include "dali/imgcodec/decoders/decoder_test_helper.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/test/dali_test_config.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"
#include "dali/util/file.h"
#include "dali/util/numpy.h"
#include "dali/imgcodec/decoders/opencv_fallback.h"

namespace dali {
namespace imgcodec {

namespace test {
namespace {
template<typename... Args>
std::string join(Args... args) {
  return make_string_delim('/', args...);
}

std::vector<uint8_t> read_file(const std::string &filename) {
    std::ifstream stream(filename, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

struct ImageBuffer {
  std::vector<uint8_t> buffer;
  ImageSource src;
  explicit ImageBuffer(const std::string &filename)
  : buffer(read_file(filename))
  , src(ImageSource::FromHostMem(buffer.data(), buffer.size())) {}
};

struct test_sample {
  test_sample(std::string img_path, std::string npy_ref_path)
      : image(img_path),
        ref(numpy::ReadTensor(FileStream::Open(npy_ref_path, false, false).get())) {}

  ImageBuffer image;
  Tensor<CPUBackend> ref;
};

struct TestData {
  void Init() {
    const auto jpeg_dir = join(dali::testing::dali_extra_path(), "db/single/jpeg");
    const auto jpeg_ref_dir = join(dali::testing::dali_extra_path(), "db/single/reference/jpeg");
    test_samples["JPEG"].emplace_back(join(jpeg_dir, "134/site-1534685_1280.jpg"),
                                      join(jpeg_ref_dir, "site-1534685_1280.npy"));
    test_samples["JPEG"].emplace_back(join(jpeg_dir, "113/snail-4291306_1280.jpg"),
                                      join(jpeg_ref_dir, "snail-4291306_1280.npy"));
    test_samples["JPEG"].emplace_back(join(jpeg_dir, "100/swan-3584559_640.jpg"),
                                      join(jpeg_ref_dir, "swan-3584559_640.npy"));

    const auto tiff_dir = join(dali::testing::dali_extra_path(), "db/single/tiff");
    const auto tiff_ref_dir = join(dali::testing::dali_extra_path(), "db/single/reference/tiff");
    test_samples["TIFF"].emplace_back(join(tiff_dir, "0/cat-3504008_640.tiff"),
                                      join(tiff_ref_dir, "0/cat-3504008_640.tiff.npy"));
    test_samples["TIFF"].emplace_back(join(tiff_dir, "0/cat-3449999_640.tiff"),
                                      join(tiff_ref_dir, "0/cat-3449999_640.tiff.npy"));
    test_samples["TIFF"].emplace_back(join(tiff_dir, "0/cat-111793_640.tiff"),
                                      join(tiff_ref_dir, "0/cat-111793_640.tiff.npy"));

    const auto jpeg2000_dir = join(dali::testing::dali_extra_path(), "db/single/jpeg2k");
    const auto jpeg2000_ref_dir =
        join(dali::testing::dali_extra_path(), "db/single/reference/jpeg2k");
    test_samples["JPEG2000"].emplace_back(join(jpeg2000_dir, "0/cat-1245673_640.jp2"),
                                          join(jpeg2000_ref_dir, "0/cat-1245673_640.npy"));
    test_samples["JPEG2000"].emplace_back(join(jpeg2000_dir, "0/cat-2184682_640.jp2"),
                                          join(jpeg2000_ref_dir, "0/cat-2184682_640.npy"));
    test_samples["JPEG2000"].emplace_back(join(jpeg2000_dir, "0/cat-300572_640.jp2"),
                                          join(jpeg2000_ref_dir, "0/cat-300572_640.npy"));

    // TODO(janton): Cover more formats (?)
  }

  void Destroy() {
    test_samples.clear();
  }

  bool empty() const {
    return test_samples.empty();
  }

  span<test_sample> get(const std::string &fmt) {
    return make_span(test_samples[fmt]);
  }

 private:
  std::map<std::string, std::vector<test_sample>> test_samples;
};

static TestData data;  // will initialize once for the whole suite

}  // namespace

template <typename T>
struct DecodeSampleOutput {
  DecodeResult res;
  TensorView<StorageCPU, const T> view;
};

template <typename T>
struct DecodeBatchOutput {
  std::vector<DecodeResult> res;
  TensorListView<StorageCPU, const T> view;
};

void AssertSuccess(const DecodeResult& res) {
  ASSERT_TRUE(res.success);
  ASSERT_NO_THROW(
    if (res.exception)
      std::rethrow_exception(res.exception);
  );  // NOLINT
}

void AssertSuccess(const std::vector<DecodeResult>& res) {
  for (auto &r : res)
    AssertSuccess(r);
}


TEST(ImageDecoderTest, GetInfo) {
  ImageDecoder dec(CPU_ONLY_DEVICE_ID, true);

  auto filename = testing::dali_extra_path() + "/db/single/jpeg/100/swan-3584559_640.jpg";
  ImageSource src = ImageSource::FromFilename(filename);

  auto info = dec.GetInfo(&src);

  EXPECT_EQ(info.shape, TensorShape<>(408, 640, 3));
  EXPECT_FALSE(info.orientation.flip_x);
  EXPECT_FALSE(info.orientation.flip_y);
  EXPECT_EQ(info.orientation.rotate, 0);
}

template<typename Backend, typename OutputType>
class ImageDecoderTest : public ::testing::Test {
 public:
  static const auto dtype = type2id<OutputType>::value;

  explicit ImageDecoderTest(int threads_cnt = 4)
      : tp_(threads_cnt, GetDeviceId(), false, "Decoder test"),
        decoder_(
            std::make_unique<ImageDecoder>(GetDeviceId(), false, std::map<std::string, any>{})) {}

  static void SetUpTestSuite() {
    // Avoid reallocating static objects if called in subclasses
    if (data.empty()) {
      data.Init();
    }
  }

  static void TearDownTestSuite() {
    data.Destroy();
  }

  ImageDecoder& Decoder() {
    return *decoder_;
  }

  void SetDecoder(std::unique_ptr<ImageDecoder>&& decoder) {
    decoder_ = std::move(decoder);
  }

  /**
  * @brief Decodes an image and returns the result as a CPU tensor.
  */
  DecodeSampleOutput<OutputType> Decode(ImageSource *src, const DecodeParams &opts = {},
                                        const ROI &roi = {}) {
    DecodeContext ctx;
    ctx.tp = &tp_;

    EXPECT_TRUE(Decoder().CanDecode(ctx, src, opts));

    ImageInfo info = Decoder().GetInfo(src);
    auto shape = AdjustToRoi(info.shape, roi);

    // Number of channels can be different than input's due to color conversion
    // TODO(skarpinski) Don't assume channel-last layout here
    *(shape.end() - 1) = NumberOfChannels(opts.format, *(info.shape.end() - 1));

    output_.reshape({{shape}});

    if (GetDeviceId() == CPU_ONLY_DEVICE_ID) {
      auto tv = output_.cpu()[0];
      SampleView<CPUBackend> view(tv.data, tv.shape, type2id<OutputType>::value);
      DecodeResult decode_result = Decoder().Decode(ctx, view, src, opts, roi);
      return {decode_result, tv};
    } else {  // GPU
      auto tv = output_.gpu()[0];
      SampleView<GPUBackend> view(tv.data, tv.shape, type2id<OutputType>::value);
      auto stream_lease = CUDAStreamPool::instance().Get(GetDeviceId());
      ctx.stream = stream_lease;
      auto decode_result = Decoder().Decode(ctx, view, src, opts, roi);
      CUDA_CALL(cudaStreamSynchronize(ctx.stream));
      return {decode_result, output_.cpu()[0]};
    }
  }

  /**
   * @brief Decodes a batch of images, invoking the batch version of ImageDecoder::Decode
   */
  DecodeBatchOutput<OutputType> Decode(cspan<ImageSource *> in, const DecodeParams &opts = {},
                                       cspan<ROI> rois = {}) {
    int n = in.size();
    std::vector<TensorShape<>> shape(n);

    DecodeContext ctx;
    ctx.tp = &tp_;

    for (int i = 0; i < n; i++) {
      EXPECT_TRUE(Decoder().CanDecode(ctx, in[i], opts));
      ImageInfo info = Decoder().GetInfo(in[i]);
      shape[i] = AdjustToRoi(info.shape, rois.empty() ? ROI{} : rois[i]);
    }

    output_.reshape(TensorListShape{shape});

    if (GetDeviceId() == CPU_ONLY_DEVICE_ID) {
      auto tlv = output_.cpu();
      std::vector<SampleView<CPUBackend>> view(n);
      for (int i = 0; i < n; i++)
        view[i] = {tlv[i].data, tlv[i].shape, type2id<OutputType>::value};
      auto res = Decoder().Decode(ctx, make_span(view), in, opts, rois);
      for (auto decode_result : res)
        EXPECT_TRUE(decode_result.success);
      return {res, tlv};
    } else {  // GPU
      auto tlv = output_.gpu();
      std::vector<SampleView<GPUBackend>> view(n);
      for (int i = 0; i < n; i++)
        view[i] = {tlv[i].data, tlv[i].shape, type2id<OutputType>::value};
      auto stream = CUDAStreamPool::instance().Get(GetDeviceId());
      ctx.stream = stream;
      auto res = Decoder().Decode(ctx, make_span(view), in, opts, rois);
      for (auto decode_result : res)
        EXPECT_TRUE(decode_result.success);
      CUDA_CALL(cudaStreamSynchronize(stream));
      return {res, output_.cpu()};
    }
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

  DecodeParams GetParams() {
    DecodeParams opts{};
    opts.dtype = dtype;
    return opts;
  }

  span<test_sample> GetData(const std::string& fmt) {
    return data.get(fmt);
  }

 protected:
  DecodeContext Context() {
    DecodeContext ctx;
    ctx.tp = &tp_;
    return ctx;
  }

 private:
  ThreadPool tp_;  // we want the thread pool to outlive the decoder instance
  std::unique_ptr<ImageDecoder> decoder_;
  kernels::TestTensorList<OutputType> output_;
};

template<typename OutputType>
class ImageDecoderTest_CPU : public ImageDecoderTest<CPUBackend, OutputType> {
};

using DecodeOutputTypes = ::testing::Types<uint8_t, int16_t, float>;
TYPED_TEST_SUITE(ImageDecoderTest_CPU, DecodeOutputTypes);

TYPED_TEST(ImageDecoderTest_CPU, DecodeSample_JPEG) {
  auto samples = this->GetData("JPEG");
  auto out = this->Decode(&samples[0].image.src, this->GetParams());
  AssertSuccess(out.res);
  AssertEqualSatNorm(out.view, samples[0].ref);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_JPEG) {
  auto samples = this->GetData("JPEG");
  std::vector<ImageSource*> srcs = {
    &samples[0].image.src, &samples[1].image.src, &samples[2].image.src
  };
  auto out = this->Decode(make_span(srcs), this->GetParams());
  AssertSuccess(out.res);
  for (int i = 0; i < out.view.size(); i++) {
    AssertEqualSatNorm(out.view[i], samples[i].ref);
  }
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeSample_TIFF) {
  auto samples = this->GetData("TIFF");
  auto out = this->Decode(&samples[0].image.src, this->GetParams());
  AssertSuccess(out.res);
  AssertEqualSatNorm(out.view, samples[0].ref);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_TIFF) {
  auto samples = this->GetData("TIFF");
  std::vector<ImageSource*> srcs = {
    &samples[0].image.src, &samples[1].image.src, &samples[2].image.src
  };
  auto out = this->Decode(make_span(srcs), this->GetParams());
  AssertSuccess(out.res);
  for (int i = 0; i < out.view.size(); i++) {
    AssertEqualSatNorm(out.view[i], samples[i].ref);
  }
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeSample_JPEG2000) {
  auto samples = this->GetData("JPEG2000");
  auto out = this->Decode(&samples[0].image.src, this->GetParams());
  AssertSuccess(out.res);
  AssertEqualSatNorm(out.view, samples[0].ref);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_JPEG2000) {
  auto samples = this->GetData("JPEG2000");
  std::vector<ImageSource*> srcs = {
    &samples[0].image.src, &samples[1].image.src, &samples[2].image.src
  };
  auto out = this->Decode(make_span(srcs), this->GetParams());
  AssertSuccess(out.res);
  for (int i = 0; i < out.view.size(); i++) {
    AssertEqualSatNorm(out.view[i], samples[i].ref);
  }
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_MultiFormat) {
  auto jpeg_samples = this->GetData("JPEG");
  auto tiff_samples = this->GetData("TIFF");
  auto jpeg2000_samples = this->GetData("JPEG2000");
  std::vector<ImageSource*> srcs = {
    &jpeg_samples[0].image.src,
    &tiff_samples[1].image.src,
    &tiff_samples[0].image.src,
    &jpeg_samples[1].image.src,
    &jpeg2000_samples[0].image.src,
    &jpeg2000_samples[1].image.src,
    &jpeg2000_samples[2].image.src,
    &jpeg_samples[2].image.src,
  };
  auto out = this->Decode(make_span(srcs), this->GetParams());
  AssertSuccess(out.res);
  int i = 0;
  AssertEqualSatNorm(out.view[i++], jpeg_samples[0].ref);
  AssertEqualSatNorm(out.view[i++], tiff_samples[1].ref);
  AssertEqualSatNorm(out.view[i++], tiff_samples[0].ref);
  AssertEqualSatNorm(out.view[i++], jpeg_samples[1].ref);
  AssertEqualSatNorm(out.view[i++], jpeg2000_samples[0].ref);
  AssertEqualSatNorm(out.view[i++], jpeg2000_samples[1].ref);
  AssertEqualSatNorm(out.view[i++], jpeg2000_samples[2].ref);
  AssertEqualSatNorm(out.view[i++], jpeg_samples[2].ref);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_NoFallback) {
  // making sure that we don't pick the fallback implementation
  auto filter = [](ImageDecoderFactory *factory) {
    if (dynamic_cast<OpenCVDecoderFactory*>(factory) != nullptr) {
      return false;
    }
    return true;
  };
  this->SetDecoder(std::make_unique<ImageDecoder>(this->GetDeviceId(), false,
                                                  std::map<std::string, any>{}, filter));
  auto jpeg_samples = this->GetData("JPEG");
  auto tiff_samples = this->GetData("TIFF");
  auto jpeg2000_samples = this->GetData("JPEG2000");

  std::vector<ImageSource*> srcs = {
    &jpeg_samples[0].image.src,
    &tiff_samples[1].image.src,
    &tiff_samples[0].image.src,
    &jpeg_samples[1].image.src,
    &jpeg2000_samples[0].image.src,
    &jpeg2000_samples[1].image.src,
    &jpeg2000_samples[2].image.src,
    &jpeg_samples[2].image.src,
  };
  auto out = this->Decode(make_span(srcs), this->GetParams());
  AssertSuccess(out.res);
  int i = 0;
  AssertEqualSatNorm(out.view[i++], jpeg_samples[0].ref);
  AssertEqualSatNorm(out.view[i++], tiff_samples[1].ref);
  AssertEqualSatNorm(out.view[i++], tiff_samples[0].ref);
  AssertEqualSatNorm(out.view[i++], jpeg_samples[1].ref);
  AssertEqualSatNorm(out.view[i++], jpeg2000_samples[0].ref);
  AssertEqualSatNorm(out.view[i++], jpeg2000_samples[1].ref);
  AssertEqualSatNorm(out.view[i++], jpeg2000_samples[2].ref);
  AssertEqualSatNorm(out.view[i++], jpeg_samples[2].ref);
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
