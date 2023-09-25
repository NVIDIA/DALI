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

#if NVJPEG_ENABLED
#include "dali/imgcodec/decoders/nvjpeg/nvjpeg.h"
#endif

#if NVJPEG2K_ENABLED
#include "dali/imgcodec/decoders/nvjpeg2k/nvjpeg2k.h"
#endif

#if defined(DALI_USE_JPEG_TURBO)
#include "dali/imgcodec/decoders/libjpeg_turbo.h"
#endif

#if LIBTIFF_ENABLED
#include "dali/imgcodec/decoders/libtiff/tiff_libtiff.h"
#endif

namespace dali {
namespace imgcodec {

namespace test {
namespace {
template<typename... Args>
std::string join(Args... args) {
  return make_string_delim('/', args...);
}

struct test_sample {
  test_sample(std::string img_path, std::string npy_path, std::string npy_ycbcr_path,
              std::string npy_gray_path)
      : path(img_path),
        image(path),
        ref(numpy::ReadTensor(FileStream::Open(npy_path, false, false).get())),
        ref_ycbcr(numpy::ReadTensor(FileStream::Open(npy_ycbcr_path, false, false).get())),
        ref_gray(numpy::ReadTensor(FileStream::Open(npy_gray_path, false, false).get())) {}

  const Tensor<CPUBackend> &GetRef(DALIImageType format) const {
    if (format == DALI_YCbCr) {
      return ref_ycbcr;
    } else if (format == DALI_GRAY) {
      return ref_gray;
    } else {
      assert(format == DALI_RGB || format == DALI_ANY_DATA);
      return ref;
    }
  }

  std::string path;
  ImageBuffer image;
  Tensor<CPUBackend> ref;
  Tensor<CPUBackend> ref_ycbcr;
  Tensor<CPUBackend> ref_gray;
};


struct TestData {
  void Init() {
    const auto jpeg_dir = join(dali::testing::dali_extra_path(), "db/single/jpeg");
    const auto jpeg_ref_dir = join(dali::testing::dali_extra_path(), "db/single/reference/jpeg");
    auto &samples_jpeg = test_samples["JPEG"];
    samples_jpeg.emplace_back(join(jpeg_dir, "134", "site-1534685_1280.jpg"),
                              join(jpeg_ref_dir, "site-1534685_1280.npy"),
                              join(jpeg_ref_dir, "site-1534685_1280_ycbcr.npy"),
                              join(jpeg_ref_dir, "site-1534685_1280_gray.npy"));
    samples_jpeg.emplace_back(join(jpeg_dir, "113", "snail-4291306_1280.jpg"),
                              join(jpeg_ref_dir, "snail-4291306_1280.npy"),
                              join(jpeg_ref_dir, "snail-4291306_1280_ycbcr.npy"),
                              join(jpeg_ref_dir, "snail-4291306_1280_gray.npy"));
    samples_jpeg.emplace_back(join(jpeg_dir, "100", "swan-3584559_640.jpg"),
                              join(jpeg_ref_dir, "swan-3584559_640.npy"),
                              join(jpeg_ref_dir, "swan-3584559_640_ycbcr.npy"),
                              join(jpeg_ref_dir, "swan-3584559_640_gray.npy"));

    const auto tiff_dir = join(dali::testing::dali_extra_path(), "db/single/tiff");
    const auto tiff_ref_dir = join(dali::testing::dali_extra_path(), "db/single/reference/tiff");
    auto &samples_tiff = test_samples["TIFF"];
    samples_tiff.emplace_back(join(tiff_dir, "0/cat-3504008_640.tiff"),
                              join(tiff_ref_dir, "0", "cat-3504008_640.tiff.npy"),
                              join(tiff_ref_dir, "0", "cat-3504008_640_ycbcr.tiff.npy"),
                              join(tiff_ref_dir, "0", "cat-3504008_640_gray.tiff.npy"));
    samples_tiff.emplace_back(join(tiff_dir, "0", "cat-3449999_640.tiff"),
                              join(tiff_ref_dir, "0", "cat-3449999_640.tiff.npy"),
                              join(tiff_ref_dir, "0", "cat-3449999_640_ycbcr.tiff.npy"),
                              join(tiff_ref_dir, "0", "cat-3449999_640_gray.tiff.npy"));
    samples_tiff.emplace_back(join(tiff_dir, "0", "cat-111793_640.tiff"),
                              join(tiff_ref_dir, "0", "cat-111793_640.tiff.npy"),
                              join(tiff_ref_dir, "0", "cat-111793_640_ycbcr.tiff.npy"),
                              join(tiff_ref_dir, "0", "cat-111793_640_gray.tiff.npy"));

    const auto jpeg2000_dir = join(dali::testing::dali_extra_path(), "db", "single", "jpeg2k");
    const auto jpeg2000_ref_dir =
        join(dali::testing::dali_extra_path(), "db", "single", "reference", "jpeg2k");
    auto &samples_jpeg2000 = test_samples["JPEG2000"];
    samples_jpeg2000.emplace_back(join(jpeg2000_dir, "0", "cat-1245673_640.jp2"),
                                  join(jpeg2000_ref_dir, "0", "cat-1245673_640.npy"),
                                  join(jpeg2000_ref_dir, "0", "cat-1245673_640_ycbcr.npy"),
                                  join(jpeg2000_ref_dir, "0", "cat-1245673_640_gray.npy"));
    samples_jpeg2000.emplace_back(join(jpeg2000_dir, "0", "cat-2184682_640.jp2"),
                                  join(jpeg2000_ref_dir, "0", "cat-2184682_640.npy"),
                                  join(jpeg2000_ref_dir, "0", "cat-2184682_640_ycbcr.npy"),
                                  join(jpeg2000_ref_dir, "0", "cat-2184682_640_gray.npy"));
    samples_jpeg2000.emplace_back(join(jpeg2000_dir, "0", "cat-300572_640.jp2"),
                                  join(jpeg2000_ref_dir, "0", "cat-300572_640.npy"),
                                  join(jpeg2000_ref_dir, "0", "cat-300572_640_ycbcr.npy"),
                                  join(jpeg2000_ref_dir, "0", "cat-300572_640_gray.npy"));

    const auto bmp_dir = join(dali::testing::dali_extra_path(), "db", "single", "bmp");
    const auto bmp_ref_dir =
        join(dali::testing::dali_extra_path(), "db", "single", "reference", "bmp");
    auto &samples_bmp = test_samples["BMP"];
    samples_bmp.emplace_back(join(bmp_dir, "0", "cat-1046544_640.bmp"),
                             join(bmp_ref_dir, "cat-1046544_640.npy"),
                             join(bmp_ref_dir, "cat-1046544_640_ycbcr.npy"),
                             join(bmp_ref_dir, "cat-1046544_640_gray.npy"));
    samples_bmp.emplace_back(join(bmp_dir, "0", "cat-1046544_640.bmp"),
                             join(bmp_ref_dir, "cat-1046544_640.npy"),
                             join(bmp_ref_dir, "cat-1046544_640_ycbcr.npy"),
                             join(bmp_ref_dir, "cat-1046544_640_gray.npy"));
    samples_bmp.emplace_back(join(bmp_dir, "0", "cat-1245673_640.bmp"),
                             join(bmp_ref_dir, "cat-1245673_640.npy"),
                             join(bmp_ref_dir, "cat-1245673_640_ycbcr.npy"),
                             join(bmp_ref_dir, "cat-1245673_640_gray.npy"));

    const auto png_dir = join(dali::testing::dali_extra_path(), "db", "single", "png");
    const auto png_ref_dir =
        join(dali::testing::dali_extra_path(), "db", "single", "reference", "png");
    auto &samples_png = test_samples["PNG"];
    samples_png.emplace_back(join(png_dir, "0", "cat-3449999_640.png"),
                             join(png_ref_dir, "cat-3449999_640.npy"),
                             join(png_ref_dir, "cat-3449999_640_ycbcr.npy"),
                             join(png_ref_dir, "cat-3449999_640_gray.npy"));
    samples_png.emplace_back(join(png_dir, "0", "cat-1046544_640.png"),
                             join(png_ref_dir, "cat-1046544_640.npy"),
                             join(png_ref_dir, "cat-1046544_640_ycbcr.npy"),
                             join(png_ref_dir, "cat-1046544_640_gray.npy"));
    samples_png.emplace_back(join(png_dir, "0", "cat-1245673_640.png"),
                             join(png_ref_dir, "cat-1245673_640.npy"),
                             join(png_ref_dir, "cat-1245673_640_ycbcr.npy"),
                             join(png_ref_dir, "cat-1245673_640_gray.npy"));

    const auto pnm_dir = join(dali::testing::dali_extra_path(), "db", "single", "pnm");
    const auto pnm_ref_dir =
        join(dali::testing::dali_extra_path(), "db", "single", "reference", "pnm");
    auto &samples_pnm = test_samples["PNM"];
    samples_pnm.emplace_back(join(pnm_dir, "0", "cat-1046544_640.pnm"),
                             join(pnm_ref_dir, "cat-1046544_640.npy"),
                             join(pnm_ref_dir, "cat-1046544_640_ycbcr.npy"),
                             join(pnm_ref_dir, "cat-1046544_640_gray.npy"));
    samples_pnm.emplace_back(join(pnm_dir, "0", "cat-111793_640.ppm"),
                             join(pnm_ref_dir, "cat-111793_640.npy"),
                             join(pnm_ref_dir, "cat-111793_640_ycbcr.npy"),
                             join(pnm_ref_dir, "cat-111793_640_gray.npy"));
    samples_pnm.emplace_back(join(pnm_dir, "0", "domestic-cat-726989_640.pnm"),
                             join(pnm_ref_dir, "domestic-cat-726989_640.npy"),
                             join(pnm_ref_dir, "domestic-cat-726989_640_ycbcr.npy"),
                             join(pnm_ref_dir, "domestic-cat-726989_640_gray.npy"));

    const auto webp_dir = join(dali::testing::dali_extra_path(), "db", "single", "webp");
    const auto webp_ref_dir =
        join(dali::testing::dali_extra_path(), "db", "single", "reference", "webp");
    auto &samples_webp = test_samples["WEBP"];
    samples_webp.emplace_back(join(webp_dir, "lossless", "cat-3449999_640.webp"),
                             join(webp_ref_dir, "cat-3449999_640.npy"),
                             join(webp_ref_dir, "cat-3449999_640_ycbcr.npy"),
                             join(webp_ref_dir, "cat-3449999_640_gray.npy"));
    samples_webp.emplace_back(join(webp_dir, "lossy", "cat-1046544_640.webp"),
                             join(webp_ref_dir, "cat-1046544_640.npy"),
                             join(webp_ref_dir, "cat-1046544_640_ycbcr.npy"),
                             join(webp_ref_dir, "cat-1046544_640_gray.npy"));
    samples_webp.emplace_back(join(webp_dir, "lossless", "cat-1245673_640.webp"),
                             join(webp_ref_dir, "cat-1245673_640.npy"),
                             join(webp_ref_dir, "cat-1245673_640_ycbcr.npy"),
                             join(webp_ref_dir, "cat-1245673_640_gray.npy"));
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

void ExpectSuccess(const DecodeResult& res) {
  EXPECT_TRUE(res.success);
  ASSERT_NO_THROW(
    if (res.exception)
      std::rethrow_exception(res.exception);
  );  // NOLINT
}

void ExpectSuccess(const std::vector<DecodeResult>& res) {
  for (auto &r : res)
    ExpectSuccess(r);
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

template <typename B, typename O, DALIImageType t>
struct ImageDecoderTestParams {
  using Backend = B;
  using OutputType = O;
  static constexpr DALIImageType image_type() { return t; }
};

template <typename TestParams>
class ImageDecoderTest;

template <typename Backend, typename OutputType, DALIImageType color_fmt>
class ImageDecoderTest<ImageDecoderTestParams<Backend, OutputType, color_fmt>>
    : public ::testing::Test {
 public:
  static const auto dtype = type2id<OutputType>::value;

  explicit ImageDecoderTest(int threads_cnt = 4)
      : tp_(threads_cnt, GetDeviceId(), false, "Decoder test") {}

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
    if (!decoder_)
      decoder_ = std::make_unique<ImageDecoder>(GetDeviceId(), false);
    return *decoder_;
  }

  void SetDecoder(std::unique_ptr<ImageDecoder>&& decoder) {
    decoder_ = std::move(decoder);
  }

  /**
  * @brief Decodes an image and returns the result as a CPU tensor.
  */
  DecodeSampleOutput<OutputType> Decode(ImageSource *src, const DecodeParams &opts = {},
                                        const ROI &roi = {}, bool require_success = true) {
    DecodeContext ctx;
    ctx.tp = &tp_;

    EXPECT_TRUE(Decoder().CanDecode(ctx, src, opts));

    ImageInfo info = Decoder().GetInfo(src);
    TensorShape<> shape;
    OutputShape(shape, info, opts, roi);

    output_.reshape({{shape}});

    if (GetDeviceId() == CPU_ONLY_DEVICE_ID) {
      auto tv = output_.cpu()[0];
      SampleView<CPUBackend> view(tv.data, tv.shape, type2id<OutputType>::value);
      DecodeResult decode_result = Decoder().Decode(ctx, view, src, opts, roi);
      if (require_success)
        ExpectSuccess(decode_result);
      return {decode_result, tv};
    } else {  // GPU
      auto tv = output_.gpu()[0];
      SampleView<GPUBackend> view(tv.data, tv.shape, type2id<OutputType>::value);
      auto stream_lease = CUDAStreamPool::instance().Get(GetDeviceId());
      ctx.stream = stream_lease;
      auto decode_result = Decoder().Decode(ctx, view, src, opts, roi);
      if (require_success)
        ExpectSuccess(decode_result);
      DecodeSampleOutput<OutputType> res{decode_result, output_.cpu(ctx.stream)[0]};
      CUDA_CALL(cudaStreamSynchronize(ctx.stream));
      return res;
    }
  }

  /**
   * @brief Decodes a batch of images, invoking the batch version of ImageDecoder::Decode
   */
  DecodeBatchOutput<OutputType> Decode(cspan<ImageSource *> in, const DecodeParams &opts = {},
                                       cspan<ROI> rois = {}, bool require_success = true) {
    int n = in.size();
    std::vector<TensorShape<>> shape(n);

    DecodeContext ctx;
    ctx.tp = &tp_;

    for (int i = 0; i < n; i++) {
      bool can_decode = Decoder().CanDecode(ctx, in[i], opts);
      if (require_success) {
        EXPECT_TRUE(can_decode);
      } else if (!can_decode) {
        continue;
      }

      ImageInfo info = Decoder().GetInfo(in[i]);
      OutputShape(shape[i], info, opts, rois.empty() ? ROI() : rois[i]);
    }

    output_.reshape(TensorListShape{shape});

    if (GetDeviceId() == CPU_ONLY_DEVICE_ID) {
      auto tlv = output_.cpu();
      std::vector<SampleView<CPUBackend>> view(n);
      for (int i = 0; i < n; i++)
        view[i] = {tlv[i].data, tlv[i].shape, type2id<OutputType>::value};
      auto res = Decoder().Decode(ctx, make_span(view), in, opts, rois);
      if (require_success)
        ExpectSuccess(res);
      return {res, tlv};
    } else {  // GPU
      auto tlv = output_.gpu();
      std::vector<SampleView<GPUBackend>> view(n);
      for (int i = 0; i < n; i++)
        view[i] = {tlv[i].data, tlv[i].shape, type2id<OutputType>::value};
      auto stream = CUDAStreamPool::instance().Get(GetDeviceId());
      ctx.stream = stream;
      auto decode_result = Decoder().Decode(ctx, make_span(view), in, opts, rois);
      if (require_success)
        ExpectSuccess(decode_result);
      DecodeBatchOutput<OutputType> res{decode_result, output_.cpu(stream)};
      CUDA_CALL(cudaStreamSynchronize(stream));
      return res;
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

  DecodeParams GetParams(DALIImageType format) {
    DecodeParams opts{};
    opts.dtype = dtype;
    opts.format = format;
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

  void FilterDecoder(std::function<bool(ImageDecoderFactory *)> decoder_filter) {
    this->SetDecoder(std::make_unique<ImageDecoder>(this->GetDeviceId(), false,
                                                    std::map<std::string, std::any>{},
                                                    decoder_filter));
  }

  void DisableFallback() {
    // making sure that we don't pick the fallback implementation
    auto filter = [](ImageDecoderFactory *factory) {
      return dynamic_cast<OpenCVDecoderFactory *>(factory) == nullptr;
    };
    FilterDecoder(filter);
  }


  float GetEps() {
    float eps = 0.01f;
    if (!std::is_floating_point_v<OutputType>) {
      // Adjusting the epsilon to OutputType
      eps *= max_value<OutputType>();
    }
    return eps;
  }

  void CompareData(const TensorView<StorageCPU, const OutputType> &data,
                   const test_sample &sample) {
    if (std::is_same<Backend, GPUBackend>::value &&
        sample.path.find(".jpg") != std::string::npos) {
      AssertSimilar(data, sample.GetRef(color_fmt));
    } else {
      if (color_fmt == DALI_YCbCr || color_fmt == DALI_GRAY)
        AssertClose(data, sample.GetRef(color_fmt), this->GetEps());
      else
        AssertEqualSatNorm(data, sample.GetRef(color_fmt));
    }
  }

  void TestSingleFormatDecodeSample(const std::string& file_fmt) {
    auto samples = this->GetData(file_fmt);
    auto &sample = samples[0];
    auto out = this->Decode(&sample.image.src, this->GetParams(color_fmt));
    CompareData(out.view, sample);
  }

  void TestSingleFormatDecodeBatch(const std::string& file_fmt) {
    auto samples = this->GetData(file_fmt);
    std::vector<ImageSource *> srcs = {&samples[0].image.src, &samples[1].image.src,
                                       &samples[2].image.src};
    auto out = this->Decode(make_span(srcs), this->GetParams(color_fmt));

    for (int i = 0; i < out.view.size(); i++) {
      CompareData(out.view[i], samples[i]);
    }
  }

  DALIImageType color_format() const {
    return color_fmt;
  }

  bool IsGPUBackend() const {
    return std::is_same<GPUBackend, Backend>::value;
  }

 private:
  ThreadPool tp_;  // we want the thread pool to outlive the decoder instance
  std::unique_ptr<ImageDecoder> decoder_;
  kernels::TestTensorList<OutputType> output_;
};

///////////////////////////////////////////////////////////////////////////////
// CPU tests
///////////////////////////////////////////////////////////////////////////////

template <typename TestParams>
class ImageDecoderTest_Basic : public ImageDecoderTest<TestParams> {};

using Params_Basic = ::testing::Types<
  ImageDecoderTestParams<CPUBackend, uint8_t, DALI_RGB>,
  ImageDecoderTestParams<CPUBackend, uint8_t, DALI_GRAY>,
  ImageDecoderTestParams<CPUBackend, uint8_t, DALI_YCbCr>,
  ImageDecoderTestParams<CPUBackend, uint8_t, DALI_ANY_DATA>,
  ImageDecoderTestParams<CPUBackend, int16_t, DALI_RGB>,
  ImageDecoderTestParams<CPUBackend, int16_t, DALI_GRAY>,
  ImageDecoderTestParams<CPUBackend, int16_t, DALI_YCbCr>,
  ImageDecoderTestParams<CPUBackend, int16_t, DALI_ANY_DATA>,
  ImageDecoderTestParams<CPUBackend, float, DALI_RGB>,
  ImageDecoderTestParams<CPUBackend, float, DALI_GRAY>,
  ImageDecoderTestParams<CPUBackend, float, DALI_YCbCr>,
  ImageDecoderTestParams<CPUBackend, float, DALI_ANY_DATA>,
  ImageDecoderTestParams<GPUBackend, uint8_t, DALI_RGB>,
  ImageDecoderTestParams<GPUBackend, uint8_t, DALI_GRAY>,
  ImageDecoderTestParams<GPUBackend, uint8_t, DALI_YCbCr>,
  ImageDecoderTestParams<GPUBackend, uint8_t, DALI_ANY_DATA>,
  ImageDecoderTestParams<GPUBackend, int16_t, DALI_RGB>,
  ImageDecoderTestParams<GPUBackend, int16_t, DALI_GRAY>,
  ImageDecoderTestParams<GPUBackend, int16_t, DALI_YCbCr>,
  ImageDecoderTestParams<GPUBackend, int16_t, DALI_ANY_DATA>,
  ImageDecoderTestParams<GPUBackend, float, DALI_RGB>,
  ImageDecoderTestParams<GPUBackend, float, DALI_GRAY>,
  ImageDecoderTestParams<GPUBackend, float, DALI_YCbCr>,
  ImageDecoderTestParams<GPUBackend, float, DALI_ANY_DATA>
>;

template <typename TestParams>
class ImageDecoderTest_OnlyCPU : public ImageDecoderTest<TestParams> {};

using Params_CPU = ::testing::Types<
  ImageDecoderTestParams<CPUBackend, uint8_t, DALI_RGB>,
  ImageDecoderTestParams<CPUBackend, uint8_t, DALI_GRAY>,
  ImageDecoderTestParams<CPUBackend, uint8_t, DALI_YCbCr>,
  ImageDecoderTestParams<CPUBackend, uint8_t, DALI_ANY_DATA>,
  ImageDecoderTestParams<CPUBackend, int16_t, DALI_RGB>,
  ImageDecoderTestParams<CPUBackend, int16_t, DALI_GRAY>,
  ImageDecoderTestParams<CPUBackend, int16_t, DALI_YCbCr>,
  ImageDecoderTestParams<CPUBackend, int16_t, DALI_ANY_DATA>,
  ImageDecoderTestParams<CPUBackend, float, DALI_RGB>,
  ImageDecoderTestParams<CPUBackend, float, DALI_GRAY>,
  ImageDecoderTestParams<CPUBackend, float, DALI_YCbCr>,
  ImageDecoderTestParams<CPUBackend, float, DALI_ANY_DATA>
>;

template <typename TestParams>
class ImageDecoderTest_OnlyGPU : public ImageDecoderTest<TestParams> {};

using Params_GPU = ::testing::Types<
  ImageDecoderTestParams<GPUBackend, uint8_t, DALI_RGB>,
  ImageDecoderTestParams<GPUBackend, uint8_t, DALI_GRAY>,
  ImageDecoderTestParams<GPUBackend, uint8_t, DALI_YCbCr>,
  ImageDecoderTestParams<GPUBackend, uint8_t, DALI_ANY_DATA>,
  ImageDecoderTestParams<GPUBackend, int16_t, DALI_RGB>,
  ImageDecoderTestParams<GPUBackend, int16_t, DALI_GRAY>,
  ImageDecoderTestParams<GPUBackend, int16_t, DALI_YCbCr>,
  ImageDecoderTestParams<GPUBackend, int16_t, DALI_ANY_DATA>,
  ImageDecoderTestParams<GPUBackend, float, DALI_RGB>,
  ImageDecoderTestParams<GPUBackend, float, DALI_GRAY>,
  ImageDecoderTestParams<GPUBackend, float, DALI_YCbCr>,
  ImageDecoderTestParams<GPUBackend, float, DALI_ANY_DATA>
>;

template <typename TestParams>
class ImageDecoderTest_CorruptedData : public ImageDecoderTest<TestParams> {};

using Params_CorruptedData = ::testing::Types<
  ImageDecoderTestParams<CPUBackend, uint8_t, DALI_RGB>,
  ImageDecoderTestParams<GPUBackend, uint8_t, DALI_RGB>
>;


TYPED_TEST_SUITE(ImageDecoderTest_Basic, Params_Basic);
TYPED_TEST_SUITE(ImageDecoderTest_OnlyCPU, Params_CPU);
TYPED_TEST_SUITE(ImageDecoderTest_OnlyGPU, Params_GPU);
TYPED_TEST_SUITE(ImageDecoderTest_CorruptedData, Params_CorruptedData);

TYPED_TEST(ImageDecoderTest_Basic, DecodeSample_JPEG) {
  this->TestSingleFormatDecodeSample("JPEG");
}

TYPED_TEST(ImageDecoderTest_Basic, DecodeBatch_JPEG) {
  this->TestSingleFormatDecodeBatch("JPEG");
}

TYPED_TEST(ImageDecoderTest_Basic, DecodeSample_TIFF) {
  this->TestSingleFormatDecodeSample("TIFF");
}

TYPED_TEST(ImageDecoderTest_Basic, DecodeBatch_TIFF) {
  this->TestSingleFormatDecodeBatch("TIFF");
}

TYPED_TEST(ImageDecoderTest_Basic, DecodeSample_JPEG2000) {
  this->TestSingleFormatDecodeSample("JPEG2000");
}

TYPED_TEST(ImageDecoderTest_Basic, DecodeBatch_JPEG2000) {
  this->TestSingleFormatDecodeBatch("JPEG2000");
}

TYPED_TEST(ImageDecoderTest_Basic, DecodeSample_BMP) {
  this->TestSingleFormatDecodeSample("BMP");
}

TYPED_TEST(ImageDecoderTest_Basic, DecodeBatch_BMP) {
  this->TestSingleFormatDecodeBatch("BMP");
}

TYPED_TEST(ImageDecoderTest_Basic, DecodeSample_PNM) {
  this->TestSingleFormatDecodeSample("PNM");
}

TYPED_TEST(ImageDecoderTest_Basic, DecodeBatch_PNM) {
  this->TestSingleFormatDecodeBatch("PNM");
}

TYPED_TEST(ImageDecoderTest_Basic, DecodeSample_PNG) {
  this->TestSingleFormatDecodeSample("PNG");
}

TYPED_TEST(ImageDecoderTest_Basic, DecodeBatch_PNG) {
  this->TestSingleFormatDecodeBatch("PNG");
}

TYPED_TEST(ImageDecoderTest_Basic, DecodeBatch_Multiformat) {
  auto jpeg_samples = this->GetData("JPEG");
  auto tiff_samples = this->GetData("TIFF");
  auto jpeg2000_samples = this->GetData("JPEG2000");
  std::vector<ImageSource *> srcs = {
      &jpeg_samples[0].image.src,     &tiff_samples[1].image.src,
      &tiff_samples[0].image.src,     &jpeg_samples[1].image.src,
      &jpeg2000_samples[0].image.src, &jpeg2000_samples[1].image.src,
      &jpeg2000_samples[2].image.src, &jpeg_samples[2].image.src,
  };
  auto out = this->Decode(make_span(srcs), this->GetParams(this->color_format()));
  int i = 0;
  this->CompareData(out.view[i++], jpeg_samples[0]);
  this->CompareData(out.view[i++], tiff_samples[1]);
  this->CompareData(out.view[i++], tiff_samples[0]);
  this->CompareData(out.view[i++], jpeg_samples[1]);
  this->CompareData(out.view[i++], jpeg2000_samples[0]);
  this->CompareData(out.view[i++], jpeg2000_samples[1]);
  this->CompareData(out.view[i++], jpeg2000_samples[2]);
  this->CompareData(out.view[i++], jpeg_samples[2]);
}

TYPED_TEST(ImageDecoderTest_Basic, DecodeBatch_Multiformat_NoFallback) {
  if (this->IsGPUBackend()) {
#if NVJPEG_ENABLED && NVJPEG2K_ENABLED && LIBTIFF_ENABLED
    this->FilterDecoder([](ImageDecoderFactory *factory) {
      return dynamic_cast<NvJpegDecoderFactory *>(factory) != nullptr ||
             dynamic_cast<NvJpeg2000DecoderFactory *>(factory) != nullptr ||
             dynamic_cast<LibTiffDecoderFactory *>(factory) != nullptr;
    });
#else
    GTEST_SKIP();
#endif
  } else {
#if defined(DALI_USE_JPEG_TURBO) && LIBTIFF_ENABLED
    this->FilterDecoder([](ImageDecoderFactory *factory) {
      return dynamic_cast<LibJpegTurboDecoderFactory *>(factory) != nullptr ||
             dynamic_cast<LibTiffDecoderFactory *>(factory) != nullptr;
    });
#else
    GTEST_SKIP();
#endif
  }
  auto jpeg_samples = this->GetData("JPEG");
  auto tiff_samples = this->GetData("TIFF");
  auto jpeg2000_samples = this->GetData("JPEG2000");  // Won't be supported because of the filter

  std::vector<ImageSource *> srcs = {
      &jpeg_samples[0].image.src,     &tiff_samples[1].image.src,
      &tiff_samples[0].image.src,     &jpeg_samples[1].image.src,
      &jpeg2000_samples[0].image.src, &jpeg2000_samples[1].image.src,
      &jpeg2000_samples[2].image.src, &jpeg_samples[2].image.src,
  };
  auto out = this->Decode(make_span(srcs), this->GetParams(this->color_format()), {}, false);
  int i = 0;
  ExpectSuccess(out.res[i]);
  this->CompareData(out.view[i++], jpeg_samples[0]);
  ExpectSuccess(out.res[i]);
  this->CompareData(out.view[i++], tiff_samples[1]);
  ExpectSuccess(out.res[i]);
  this->CompareData(out.view[i++], tiff_samples[0]);
  ExpectSuccess(out.res[i]);
  this->CompareData(out.view[i++], jpeg_samples[1]);

  if (this->IsGPUBackend()) {
    ExpectSuccess(out.res[i]);
    this->CompareData(out.view[i++], jpeg2000_samples[0]);
    ExpectSuccess(out.res[i]);
    this->CompareData(out.view[i++], jpeg2000_samples[1]);
    ExpectSuccess(out.res[i]);
    this->CompareData(out.view[i++], jpeg2000_samples[2]);
  } else {
    // we didn't enable jpeg2000 for CPUBackend in this test
    EXPECT_FALSE(out.res[i++].success);
    EXPECT_FALSE(out.res[i++].success);
    EXPECT_FALSE(out.res[i++].success);
  }

  ExpectSuccess(out.res[i]);
  this->CompareData(out.view[i++], jpeg_samples[2]);
}

TYPED_TEST(ImageDecoderTest_OnlyCPU, DecodeSample_JPEG_SingleDecoder_LibjpegTurbo) {
#if defined(DALI_USE_JPEG_TURBO)
  this->FilterDecoder(
    [](ImageDecoderFactory *factory) {
      return dynamic_cast<LibJpegTurboDecoderFactory *>(factory) != nullptr;
    });
#else
  GTEST_SKIP();
#endif
  this->TestSingleFormatDecodeSample("JPEG");
}

TYPED_TEST(ImageDecoderTest_OnlyCPU, DecodeSample_JPEG_SingleDecoder_OpenCV) {
  this->FilterDecoder(
    [](ImageDecoderFactory *factory) {
      return dynamic_cast<OpenCVDecoderFactory *>(factory) != nullptr;
    });
  this->TestSingleFormatDecodeSample("JPEG");
}

TYPED_TEST(ImageDecoderTest_OnlyCPU, DecodeSample_TIFF_SingleDecoder_Libtiff) {
#if LIBTIFF_ENABLED
  this->FilterDecoder(
    [](ImageDecoderFactory *factory) {
      return dynamic_cast<LibTiffDecoderFactory *>(factory) != nullptr;
    });
#else
  GTEST_SKIP();
#endif
  this->TestSingleFormatDecodeSample("TIFF");
}

TYPED_TEST(ImageDecoderTest_OnlyCPU, DecodeSample_TIFF_SingleDecoder_OpenCV) {
  this->FilterDecoder(
    [](ImageDecoderFactory *factory) {
      return dynamic_cast<OpenCVDecoderFactory *>(factory) != nullptr;
    });
  this->TestSingleFormatDecodeSample("TIFF");
}

TYPED_TEST(ImageDecoderTest_OnlyGPU, DecodeSample_JPEG_SingleDecoder_NvJpeg) {
#if NVJPEG_ENABLED
  this->FilterDecoder(
    [](ImageDecoderFactory *factory) {
      return dynamic_cast<NvJpegDecoderFactory *>(factory) != nullptr;
    });
#else
  GTEST_SKIP();
#endif
  this->TestSingleFormatDecodeSample("JPEG");
}

TYPED_TEST(ImageDecoderTest_OnlyGPU, DecodeSample_JPEG_SingleDecoder_NvJpeg2000) {
#if NVJPEG2K_ENABLED
  this->FilterDecoder(
    [](ImageDecoderFactory *factory) {
      return dynamic_cast<NvJpeg2000DecoderFactory *>(factory) != nullptr;
    });
#else
  GTEST_SKIP();
#endif
  this->TestSingleFormatDecodeSample("JPEG2000");
}

TYPED_TEST(ImageDecoderTest_CorruptedData, DecodeSample_JPEG) {
  auto samples = this->GetData("JPEG");
  auto corrupted_sample =
      ImageSource::FromHostMem(samples[0].image.src.RawData(), samples[0].image.src.Size() / 10);
  auto out = this->Decode(&corrupted_sample, this->GetParams(this->color_format()), {}, false);
  // OpenCV actually silently succeeds and produce a half-decoded image
  ASSERT_TRUE(out.res.success);
}

TYPED_TEST(ImageDecoderTest_CorruptedData, DecodeSample_TIFF) {
  auto samples = this->GetData("TIFF");

  std::vector<uint8_t> corrupted_tiff_data(samples[0].image.src.Size());
  auto *ptr = static_cast<const uint8_t*>(samples[0].image.src.RawData());
  for (size_t i = 0; i < corrupted_tiff_data.size(); i++) {
    corrupted_tiff_data[i] = ptr[i];
  }
  for (size_t i = 1000; i <= 2000; i++)
    corrupted_tiff_data[i] = 0x00;
  auto corrupted_sample =
      ImageSource::FromHostMem(corrupted_tiff_data.data(), corrupted_tiff_data.size());

  auto out = this->Decode(&corrupted_sample, this->GetParams(this->color_format()), {}, false);
  // OpenCV actually silently succeeds and produce a half-decoded image
  ASSERT_TRUE(out.res.success);
}

TYPED_TEST(ImageDecoderTest_CorruptedData, DecodeBatch) {
  auto jpeg_samples = this->GetData("JPEG");
  auto tiff_samples = this->GetData("TIFF");

  std::vector<uint8_t> corrupted_tiff_data(tiff_samples[0].image.src.Size());
  // corrupting data
  {
    auto *ptr = static_cast<const uint8_t*>(tiff_samples[0].image.src.RawData());
    for (size_t i = 0; i < corrupted_tiff_data.size(); i++) {
      corrupted_tiff_data[i] = ptr[i];
    }
    ASSERT_GT(tiff_samples[0].image.src.Size(), 2000);
    for (size_t i = 1000; i < 2000; i++)
      corrupted_tiff_data[i] = 0x00;
  }
  auto tiff_sample1_corrupted =
      ImageSource::FromHostMem(corrupted_tiff_data.data(), corrupted_tiff_data.size());

  // Truncating the file
  auto jpeg_sample1_corrupted = ImageSource::FromHostMem(jpeg_samples[1].image.src.RawData(),
                                                         jpeg_samples[1].image.src.Size() / 10);

  std::vector<ImageSource*> srcs = {
    &jpeg_samples[0].image.src,
    &tiff_sample1_corrupted,
    &tiff_samples[0].image.src,
    &jpeg_sample1_corrupted,
    &jpeg_samples[2].image.src,
  };

  auto out = this->Decode(make_span(srcs), this->GetParams(this->color_format()), {}, false);
  ExpectSuccess(out.res[0]);
  ASSERT_TRUE(out.res[1].success);  // OpenCV silently succeeds (and produces incomplete data)
  ExpectSuccess(out.res[2]);
  ASSERT_TRUE(out.res[3].success);  // OpenCV silently succeeds (and produces incomplete data)
  ExpectSuccess(out.res[4]);
}


}  // namespace test
}  // namespace imgcodec
}  // namespace dali
