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
#include "dali/imgcodec/decoders/nvjpeg/nvjpeg.h"

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
  , src(ImageSource::FromHostMem(buffer.data(), buffer.size(), filename)) {}
};

struct test_sample {
  test_sample(std::string img_path, std::string npy_path, std::string npy_ycbcr_path,
              std::string npy_gray_path)
      : image(img_path),
        ref(numpy::ReadTensor(FileStream::Open(npy_path, false, false).get())),
        ref_ycbcr(numpy::ReadTensor(FileStream::Open(npy_ycbcr_path, false, false).get())),
        ref_gray(numpy::ReadTensor(FileStream::Open(npy_gray_path, false, false).get())) {}

  const Tensor<CPUBackend> &GetRef(DALIImageType format) {
    if (format == DALI_YCbCr) {
      return ref_ycbcr;
    } else if (format == DALI_GRAY) {
      return ref_gray;
    } else {
      assert(format == DALI_RGB);
      return ref;
    }
  }

  ImageBuffer image;
  Tensor<CPUBackend> ref;
  Tensor<CPUBackend> ref_ycbcr;
  Tensor<CPUBackend> ref_gray;
};

cv::Mat rgb2bgr(const cv::Mat &img) {
  cv::Mat bgr;
  cv::cvtColor(img, bgr, cv::COLOR_RGB2BGR);
  return bgr;
}

template <typename T>
cv::Mat GetCvMat(TensorView<StorageCPU, const T> v, DALIImageType color_fmt) {
  static_assert(std::is_same<T, uint8_t>::value, "Only uint8 images supported for now");
  cv::Mat v_mat(v.shape[0], v.shape[1], color_fmt == DALI_GRAY ? CV_8UC1 : CV_8UC3,
                (void *)v.data);  // NOLINT
  return rgb2bgr(v_mat);
}

cv::Mat GetRefCvMat(test_sample& sample, DALIImageType color_fmt) {
  void *refdata = nullptr;
  TensorShape<> ref_sh;
  if (color_fmt == DALI_YCbCr) {
    refdata = sample.ref_ycbcr.raw_mutable_data();
    ref_sh = sample.ref_ycbcr.shape();
  } else if (color_fmt == DALI_GRAY) {
    refdata = sample.ref_gray.raw_mutable_data();
    ref_sh = sample.ref_gray.shape();
  } else {
    refdata = sample.ref.raw_mutable_data();
    ref_sh = sample.ref.shape();
  }
  cv::Mat vref_mat(ref_sh[0], ref_sh[1], color_fmt == DALI_GRAY ? CV_8UC1 : CV_8UC3, refdata);
  return rgb2bgr(vref_mat);
}

void DumpImages(TensorView<StorageCPU, const uint8_t> v,
                test_sample& sample,
                DALIImageType color_fmt) {
  auto out = GetCvMat<uint8_t>(v, color_fmt);
  auto ref = GetRefCvMat(sample, color_fmt);
  cv::Mat diff;
  cv::absdiff(out, ref, diff);
  cv::imwrite("/tmp/img_out.bmp", out);
  cv::imwrite("/tmp/img_ref.bmp", ref);
  cv::imwrite("/tmp/img_diff.bmp", diff);
}

struct TestData {
  void Init() {
    const auto jpeg_dir = join(dali::testing::dali_extra_path(), "db/single/jpeg");
    const auto jpeg_ref_dir = join(dali::testing::dali_extra_path(), "db/single/reference/jpeg");
    auto &samples_jpeg = test_samples["JPEG"];
    samples_jpeg.emplace_back(join(jpeg_dir, "134/site-1534685_1280.jpg"),
                              join(jpeg_ref_dir, "site-1534685_1280.npy"),
                              join(jpeg_ref_dir, "site-1534685_1280_ycbcr.npy"),
                              join(jpeg_ref_dir, "site-1534685_1280_gray.npy"));
    samples_jpeg.emplace_back(join(jpeg_dir, "113/snail-4291306_1280.jpg"),
                              join(jpeg_ref_dir, "snail-4291306_1280.npy"),
                              join(jpeg_ref_dir, "snail-4291306_1280_ycbcr.npy"),
                              join(jpeg_ref_dir, "snail-4291306_1280_gray.npy"));
    samples_jpeg.emplace_back(join(jpeg_dir, "100/swan-3584559_640.jpg"),
                              join(jpeg_ref_dir, "swan-3584559_640.npy"),
                              join(jpeg_ref_dir, "swan-3584559_640_ycbcr.npy"),
                              join(jpeg_ref_dir, "swan-3584559_640_gray.npy"));

    const auto tiff_dir = join(dali::testing::dali_extra_path(), "db/single/tiff");
    const auto tiff_ref_dir = join(dali::testing::dali_extra_path(), "db/single/reference/tiff");
    auto &samples_tiff = test_samples["TIFF"];
    samples_tiff.emplace_back(join(tiff_dir, "0/cat-3504008_640.tiff"),
                              join(tiff_ref_dir, "0/cat-3504008_640.tiff.npy"),
                              join(tiff_ref_dir, "0/cat-3504008_640_ycbcr.tiff.npy"),
                              join(tiff_ref_dir, "0/cat-3504008_640_gray.tiff.npy"));
    samples_tiff.emplace_back(join(tiff_dir, "0/cat-3449999_640.tiff"),
                              join(tiff_ref_dir, "0/cat-3449999_640.tiff.npy"),
                              join(tiff_ref_dir, "0/cat-3449999_640_ycbcr.tiff.npy"),
                              join(tiff_ref_dir, "0/cat-3449999_640_gray.tiff.npy"));
    samples_tiff.emplace_back(join(tiff_dir, "0/cat-111793_640.tiff"),
                              join(tiff_ref_dir, "0/cat-111793_640.tiff.npy"),
                              join(tiff_ref_dir, "0/cat-111793_640_ycbcr.tiff.npy"),
                              join(tiff_ref_dir, "0/cat-111793_640_gray.tiff.npy"));

    const auto jpeg2000_dir = join(dali::testing::dali_extra_path(), "db/single/jpeg2k");
    const auto jpeg2000_ref_dir =
        join(dali::testing::dali_extra_path(), "db/single/reference/jpeg2k");
    auto &samples_jpeg2000 = test_samples["JPEG2000"];
    samples_jpeg2000.emplace_back(join(jpeg2000_dir, "0/cat-1245673_640.jp2"),
                                  join(jpeg2000_ref_dir, "0/cat-1245673_640.npy"),
                                  join(jpeg2000_ref_dir, "0/cat-1245673_640_ycbcr.npy"),
                                  join(jpeg2000_ref_dir, "0/cat-1245673_640_gray.npy"));
    samples_jpeg2000.emplace_back(join(jpeg2000_dir, "0/cat-2184682_640.jp2"),
                                  join(jpeg2000_ref_dir, "0/cat-2184682_640.npy"),
                                  join(jpeg2000_ref_dir, "0/cat-2184682_640_ycbcr.npy"),
                                  join(jpeg2000_ref_dir, "0/cat-2184682_640_gray.npy"));
    samples_jpeg2000.emplace_back(join(jpeg2000_dir, "0/cat-300572_640.jp2"),
                                  join(jpeg2000_ref_dir, "0/cat-300572_640.npy"),
                                  join(jpeg2000_ref_dir, "0/cat-300572_640_ycbcr.npy"),
                                  join(jpeg2000_ref_dir, "0/cat-300572_640_gray.npy"));

    const auto bmp_dir = join(dali::testing::dali_extra_path(), "db/single/bmp");
    const auto bmp_ref_dir =
        join(dali::testing::dali_extra_path(), "db/single/reference/bmp");
    auto &samples_bmp = test_samples["BMP"];
    samples_bmp.emplace_back(join(bmp_dir, "0/cat-1046544_640.bmp"),
                             join(bmp_ref_dir, "cat-1046544_640.npy"),
                             join(bmp_ref_dir, "cat-1046544_640_ycbcr.npy"),
                             join(bmp_ref_dir, "cat-1046544_640_gray.npy"));
    samples_bmp.emplace_back(join(bmp_dir, "0/cat-1046544_640.bmp"),
                             join(bmp_ref_dir, "cat-1046544_640.npy"),
                             join(bmp_ref_dir, "cat-1046544_640_ycbcr.npy"),
                             join(bmp_ref_dir, "cat-1046544_640_gray.npy"));
    samples_bmp.emplace_back(join(bmp_dir, "0/cat-1245673_640.bmp"),
                             join(bmp_ref_dir, "cat-1245673_640.npy"),
                             join(bmp_ref_dir, "cat-1245673_640_ycbcr.npy"),
                             join(bmp_ref_dir, "cat-1245673_640_gray.npy"));

    const auto png_dir = join(dali::testing::dali_extra_path(), "db/single/png");
    const auto png_ref_dir =
        join(dali::testing::dali_extra_path(), "db/single/reference/png");
    auto &samples_png = test_samples["PNG"];
    samples_png.emplace_back(join(png_dir, "0/cat-3449999_640.png"),
                             join(png_ref_dir, "cat-3449999_640.npy"),
                             join(png_ref_dir, "cat-3449999_640_ycbcr.npy"),
                             join(png_ref_dir, "cat-3449999_640_gray.npy"));
    samples_png.emplace_back(join(png_dir, "0/cat-1046544_640.png"),
                             join(png_ref_dir, "cat-1046544_640.npy"),
                             join(png_ref_dir, "cat-1046544_640_ycbcr.npy"),
                             join(png_ref_dir, "cat-1046544_640_gray.npy"));
    samples_png.emplace_back(join(png_dir, "0/cat-1245673_640.png"),
                             join(png_ref_dir, "cat-1245673_640.npy"),
                             join(png_ref_dir, "cat-1245673_640_ycbcr.npy"),
                             join(png_ref_dir, "cat-1245673_640_gray.npy"));

    const auto pnm_dir = join(dali::testing::dali_extra_path(), "db/single/pnm");
    const auto pnm_ref_dir =
        join(dali::testing::dali_extra_path(), "db/single/reference/pnm");
    auto &samples_pnm = test_samples["PNM"];
    samples_pnm.emplace_back(join(pnm_dir, "0/cat-1046544_640.pnm"),
                             join(pnm_ref_dir, "cat-1046544_640.npy"),
                             join(pnm_ref_dir, "cat-1046544_640_ycbcr.npy"),
                             join(pnm_ref_dir, "cat-1046544_640_gray.npy"));
    samples_pnm.emplace_back(join(pnm_dir, "0/cat-111793_640.ppm"),
                             join(pnm_ref_dir, "cat-111793_640.npy"),
                             join(pnm_ref_dir, "cat-111793_640_ycbcr.npy"),
                             join(pnm_ref_dir, "cat-111793_640_gray.npy"));
    samples_pnm.emplace_back(join(pnm_dir, "0/domestic-cat-726989_640.pnm"),
                             join(pnm_ref_dir, "domestic-cat-726989_640.npy"),
                             join(pnm_ref_dir, "domestic-cat-726989_640_ycbcr.npy"),
                             join(pnm_ref_dir, "domestic-cat-726989_640_gray.npy"));

    const auto webp_dir = join(dali::testing::dali_extra_path(), "db/single/webp");
    const auto webp_ref_dir =
        join(dali::testing::dali_extra_path(), "db/single/reference/webp");
    auto &samples_webp = test_samples["WEBP"];
    samples_webp.emplace_back(join(webp_dir, "lossless/cat-3449999_640.webp"),
                             join(webp_ref_dir, "cat-3449999_640.npy"),
                             join(webp_ref_dir, "cat-3449999_640_ycbcr.npy"),
                             join(webp_ref_dir, "cat-3449999_640_gray.npy"));
    samples_webp.emplace_back(join(webp_dir, "lossy/cat-1046544_640.webp"),
                             join(webp_ref_dir, "cat-1046544_640.npy"),
                             join(webp_ref_dir, "cat-1046544_640_ycbcr.npy"),
                             join(webp_ref_dir, "cat-1046544_640_gray.npy"));
    samples_webp.emplace_back(join(webp_dir, "lossless/cat-1245673_640.webp"),
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

template<typename Backend, typename OutputType>
class ImageDecoderTest : public ::testing::Test {
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

  void DisableFallback() {
    // making sure that we don't pick the fallback implementation
    auto filter = [](ImageDecoderFactory *factory) {
      if (dynamic_cast<OpenCVDecoderFactory *>(factory) != nullptr) {
        return false;
      }
      return true;
    };
    this->SetDecoder(std::make_unique<ImageDecoder>(this->GetDeviceId(), false,
                                                    std::map<std::string, any>{}, filter));
  }

  float GetEps() {
    float eps = 0.01f;
    if (!std::is_floating_point_v<OutputType>) {
      // Adjusting the epsilon to OutputType
      eps *= max_value<OutputType>();
    }
    return eps;
  }

  void CompareData(const TensorView<StorageCPU, const OutputType> &data, const test_sample &sample,
                   DALIImageType color_fmt) {
    if constexpr (std::is_same<Backend, GPUBackend>::value) {
      if (color_fmt == DALI_YCbCr) {
        AssertSimilar(data, sample.ref_ycbcr);
      } else if (color_fmt == DALI_GRAY) {
        AssertSimilar(data, sample.ref_gray);
      } else {
        assert(color_fmt == DALI_RGB);
        AssertSimilar(data, sample.ref);
      }
    } else {
      if (color_fmt == DALI_YCbCr) {
        AssertClose(data, sample.ref_ycbcr, this->GetEps());
      } else if (color_fmt == DALI_GRAY) {
        AssertClose(data, sample.ref_gray, this->GetEps());
      } else {
        assert(color_fmt == DALI_RGB);
        AssertEqualSatNorm(data, sample.ref);
      }
    }
  }

  void TestSingleFormatDecodeSample(const std::string& file_fmt, DALIImageType color_fmt) {
    auto samples = this->GetData(file_fmt);
    auto &sample = samples[0];
    auto out = this->Decode(&sample.image.src, this->GetParams(color_fmt));
    CompareData(out.view, sample, color_fmt);
  }

  void TestSingleFormatDecodeBatch(const std::string& file_fmt, DALIImageType color_fmt) {
    auto samples = this->GetData(file_fmt);
    std::vector<ImageSource *> srcs = {&samples[0].image.src, &samples[1].image.src,
                                       &samples[2].image.src};
    auto out = this->Decode(make_span(srcs), this->GetParams(color_fmt));

    for (int i = 0; i < out.view.size(); i++) {
      CompareData(out.view[i], samples[i], color_fmt);
    }
  }

 private:
  ThreadPool tp_;  // we want the thread pool to outlive the decoder instance
  std::unique_ptr<ImageDecoder> decoder_;
  kernels::TestTensorList<OutputType> output_;
};

///////////////////////////////////////////////////////////////////////////////
// CPU tests
///////////////////////////////////////////////////////////////////////////////

template<typename OutputType>
class ImageDecoderTest_CPU : public ImageDecoderTest<CPUBackend, OutputType> {
};

using DecodeOutputTypes = ::testing::Types<uint8_t, int16_t, float>;
TYPED_TEST_SUITE(ImageDecoderTest_CPU, DecodeOutputTypes);

TYPED_TEST(ImageDecoderTest_CPU, DecodeSample_JPEG) {
  this->TestSingleFormatDecodeSample("JPEG", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeSample_JPEG_YCbCr) {
  this->TestSingleFormatDecodeSample("JPEG", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeSample_JPEG_GRAY) {
  this->TestSingleFormatDecodeSample("JPEG", DALI_GRAY);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_JPEG) {
  this->TestSingleFormatDecodeBatch("JPEG", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_JPEG_YCbCr) {
  this->TestSingleFormatDecodeBatch("JPEG", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_JPEG_GRAY) {
  this->TestSingleFormatDecodeBatch("JPEG", DALI_GRAY);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeSample_TIFF) {
  this->TestSingleFormatDecodeSample("TIFF", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeSample_TIFF_YCbCr) {
  this->TestSingleFormatDecodeSample("TIFF", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeSample_TIFF_GRAY) {
  this->TestSingleFormatDecodeSample("TIFF", DALI_GRAY);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_TIFF) {
  this->TestSingleFormatDecodeBatch("TIFF", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_TIFF_YCbCr) {
  this->TestSingleFormatDecodeBatch("TIFF", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_TIFF_GRAY) {
  this->TestSingleFormatDecodeBatch("TIFF", DALI_GRAY);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeSample_JPEG2000) {
  this->TestSingleFormatDecodeSample("JPEG2000", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeSample_JPEG2000_YCbCr) {
  this->TestSingleFormatDecodeSample("JPEG2000", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeSample_JPEG2000_GRAY) {
  this->TestSingleFormatDecodeSample("JPEG2000", DALI_GRAY);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_JPEG2000) {
  this->TestSingleFormatDecodeBatch("JPEG2000", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_JPEG2000_YCbCr) {
  this->TestSingleFormatDecodeBatch("JPEG2000", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_JPEG2000_GRAY) {
  this->TestSingleFormatDecodeBatch("JPEG2000", DALI_GRAY);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeSample_BMP) {
  this->TestSingleFormatDecodeSample("BMP", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeSample_BMP_YCbCr) {
  this->TestSingleFormatDecodeSample("BMP", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeSample_BMP_GRAY) {
  this->TestSingleFormatDecodeSample("BMP", DALI_GRAY);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_BMP) {
  this->TestSingleFormatDecodeBatch("BMP", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_BMP_YCbCr) {
  this->TestSingleFormatDecodeBatch("BMP", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_BMP_GRAY) {
  this->TestSingleFormatDecodeBatch("BMP", DALI_GRAY);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeSample_PNM) {
  this->TestSingleFormatDecodeSample("PNM", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeSample_PNM_YCbCr) {
  this->TestSingleFormatDecodeSample("PNM", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeSample_PNM_GRAY) {
  this->TestSingleFormatDecodeSample("PNM", DALI_GRAY);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_PNM) {
  this->TestSingleFormatDecodeBatch("PNM", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_PNM_YCbCr) {
  this->TestSingleFormatDecodeBatch("PNM", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_PNM_GRAY) {
  this->TestSingleFormatDecodeBatch("PNM", DALI_GRAY);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeSample_PNG) {
  this->TestSingleFormatDecodeSample("PNG", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeSample_PNG_YCbCr) {
  this->TestSingleFormatDecodeSample("PNG", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeSample_PNG_GRAY) {
  this->TestSingleFormatDecodeSample("PNG", DALI_GRAY);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_PNG) {
  this->TestSingleFormatDecodeBatch("PNG", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_PNG_YCbCr) {
  this->TestSingleFormatDecodeBatch("PNG", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_PNG_GRAY) {
  this->TestSingleFormatDecodeBatch("PNG", DALI_GRAY);
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
  auto out = this->Decode(make_span(srcs), this->GetParams(DALI_RGB));
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
  this->DisableFallback();
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
  auto out = this->Decode(make_span(srcs), this->GetParams(DALI_RGB), {}, false);
  int i = 0;
  ExpectSuccess(out.res[i]);
  AssertEqualSatNorm(out.view[i++], jpeg_samples[0].ref);
  ExpectSuccess(out.res[i]);
  AssertEqualSatNorm(out.view[i++], tiff_samples[1].ref);
  ExpectSuccess(out.res[i]);
  AssertEqualSatNorm(out.view[i++], tiff_samples[0].ref);
  ExpectSuccess(out.res[i]);
  AssertEqualSatNorm(out.view[i++], jpeg_samples[1].ref);

  EXPECT_FALSE(out.res[i++].success);
  EXPECT_FALSE(out.res[i++].success);
  EXPECT_FALSE(out.res[i++].success);

  ExpectSuccess(out.res[i]);
  AssertEqualSatNorm(out.view[i++], jpeg_samples[2].ref);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeSample_CorruptedData_JPEG) {
  this->DisableFallback();
  auto samples = this->GetData("JPEG");
  auto corrupted_sample =
      ImageSource::FromHostMem(samples[0].image.src.RawData(), samples[0].image.src.Size() / 10);
  auto out = this->Decode(&corrupted_sample, this->GetParams(DALI_RGB), {}, false);
  ASSERT_FALSE(out.res.success);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeSample_CorruptedData_TIFF) {
  this->DisableFallback();
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

  auto out = this->Decode(&corrupted_sample, this->GetParams(DALI_RGB), {}, false);
  ASSERT_FALSE(out.res.success);
}

TYPED_TEST(ImageDecoderTest_CPU, DecodeBatch_CorruptedData) {
  this->DisableFallback();
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

  auto out = this->Decode(make_span(srcs), this->GetParams(DALI_RGB), {}, false);
  ExpectSuccess(out.res[0]);
  ASSERT_FALSE(out.res[1].success);
  ExpectSuccess(out.res[2]);
  ASSERT_FALSE(out.res[3].success);
  ExpectSuccess(out.res[4]);
}

///////////////////////////////////////////////////////////////////////////////
// GPU tests
///////////////////////////////////////////////////////////////////////////////

template<typename OutputType>
class ImageDecoderTest_GPU : public ImageDecoderTest<GPUBackend, OutputType> {
};

TYPED_TEST_SUITE(ImageDecoderTest_GPU, DecodeOutputTypes);

TYPED_TEST(ImageDecoderTest_GPU, DecodeSample_JPEG) {
  this->TestSingleFormatDecodeSample("JPEG", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeSample_JPEG_YCbCr) {
  this->TestSingleFormatDecodeSample("JPEG", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeSample_JPEG_GRAY) {
  this->TestSingleFormatDecodeSample("JPEG", DALI_GRAY);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeBatch_JPEG) {
  this->TestSingleFormatDecodeBatch("JPEG", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeBatch_JPEG_YCbCr) {
  this->TestSingleFormatDecodeBatch("JPEG", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeBatch_JPEG_GRAY) {
  this->TestSingleFormatDecodeBatch("JPEG", DALI_GRAY);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeSample_TIFF) {
  this->TestSingleFormatDecodeSample("TIFF", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeSample_TIFF_YCbCr) {
  this->TestSingleFormatDecodeSample("TIFF", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeSample_TIFF_GRAY) {
  this->TestSingleFormatDecodeSample("TIFF", DALI_GRAY);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeBatch_TIFF) {
  this->TestSingleFormatDecodeBatch("TIFF", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeBatch_TIFF_YCbCr) {
  this->TestSingleFormatDecodeBatch("TIFF", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeBatch_TIFF_GRAY) {
  this->TestSingleFormatDecodeBatch("TIFF", DALI_GRAY);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeSample_JPEG2000) {
  this->TestSingleFormatDecodeSample("JPEG2000", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeSample_JPEG2000_YCbCr) {
  this->TestSingleFormatDecodeSample("JPEG2000", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeSample_JPEG2000_GRAY) {
  this->TestSingleFormatDecodeSample("JPEG2000", DALI_GRAY);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeBatch_JPEG2000) {
  this->TestSingleFormatDecodeBatch("JPEG2000", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeBatch_JPEG2000_YCbCr) {
  this->TestSingleFormatDecodeBatch("JPEG2000", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeBatch_JPEG2000_GRAY) {
  this->TestSingleFormatDecodeBatch("JPEG2000", DALI_GRAY);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeSample_BMP) {
  this->TestSingleFormatDecodeSample("BMP", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeSample_BMP_YCbCr) {
  this->TestSingleFormatDecodeSample("BMP", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeSample_BMP_GRAY) {
  this->TestSingleFormatDecodeSample("BMP", DALI_GRAY);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeBatch_BMP) {
  this->TestSingleFormatDecodeBatch("BMP", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeBatch_BMP_YCbCr) {
  this->TestSingleFormatDecodeBatch("BMP", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeBatch_BMP_GRAY) {
  this->TestSingleFormatDecodeBatch("BMP", DALI_GRAY);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeSample_PNM) {
  this->TestSingleFormatDecodeSample("PNM", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeSample_PNM_YCbCr) {
  this->TestSingleFormatDecodeSample("PNM", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeSample_PNM_GRAY) {
  this->TestSingleFormatDecodeSample("PNM", DALI_GRAY);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeBatch_PNM) {
  this->TestSingleFormatDecodeBatch("PNM", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeBatch_PNM_YCbCr) {
  this->TestSingleFormatDecodeBatch("PNM", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeBatch_PNM_GRAY) {
  this->TestSingleFormatDecodeBatch("PNM", DALI_GRAY);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeSample_PNG) {
  this->TestSingleFormatDecodeSample("PNG", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeSample_PNG_YCbCr) {
  this->TestSingleFormatDecodeSample("PNG", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeSample_PNG_GRAY) {
  this->TestSingleFormatDecodeSample("PNG", DALI_GRAY);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeBatch_PNG) {
  this->TestSingleFormatDecodeBatch("PNG", DALI_RGB);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeBatch_PNG_YCbCr) {
  this->TestSingleFormatDecodeBatch("PNG", DALI_YCbCr);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeBatch_PNG_GRAY) {
  this->TestSingleFormatDecodeBatch("PNG", DALI_GRAY);
}

TYPED_TEST(ImageDecoderTest_GPU, DecodeBatch_NoFallback) {
  this->DisableFallback();
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
  auto out = this->Decode(make_span(srcs), this->GetParams(DALI_RGB), {}, false);
  int i = 0;
  ExpectSuccess(out.res[i]);
  AssertSimilar(out.view[i++], jpeg_samples[0].ref);
  ExpectSuccess(out.res[i]);
  AssertEqualSatNorm(out.view[i++], tiff_samples[1].ref);
  ExpectSuccess(out.res[i]);
  AssertEqualSatNorm(out.view[i++], tiff_samples[0].ref);
  ExpectSuccess(out.res[i]);
  AssertSimilar(out.view[i++], jpeg_samples[1].ref);
  ExpectSuccess(out.res[i]);
  AssertEqualSatNorm(out.view[i++], jpeg2000_samples[0].ref);
  ExpectSuccess(out.res[i]);
  AssertEqualSatNorm(out.view[i++], jpeg2000_samples[1].ref);
  ExpectSuccess(out.res[i]);
  AssertEqualSatNorm(out.view[i++], jpeg2000_samples[2].ref);
  ExpectSuccess(out.res[i]);
  AssertSimilar(out.view[i++], jpeg_samples[2].ref);
}


}  // namespace test
}  // namespace imgcodec
}  // namespace dali
