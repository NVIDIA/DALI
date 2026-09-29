// Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/imgcodec/util/convert_gpu.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include "dali/core/convert.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/kernels/imgproc/color_manipulation/color_space_conversion_impl.h"
#include "dali/operators/imgcodec/util/convert.h"
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

namespace dali {
namespace imgcodec {
namespace test {

namespace {

template <class Input, class Output>
struct ConversionTestType {
  using In = Input;
  using Out = Output;
};

using TensorTestData = std::vector<std::vector<std::vector<float>>>;

TensorShape<> get_data_shape(const TensorTestData &data) {
  return {
      static_cast<int>(data.size()),
      static_cast<int>(data[0].size()),
      static_cast<int>(data[0][0].size()),
  };
}

template <class T>
void init_test_tensor_list(kernels::TestTensorList<T> &list, const TensorTestData &data) {
  auto shape = get_data_shape(data);
  list.invalidate_cpu();
  list.invalidate_gpu();
  list.reshape({{shape}});
  auto tv = list.cpu()[0];
  for (int i = 0; i < shape[0]; i++)
    for (int j = 0; j < shape[1]; j++)
      for (int k = 0; k < shape[2]; k++)
        *tv(TensorShape<>{i, j, k}) = ConvertSatNorm<T>(data[i][j][k]);
}

TensorTestData empty_data(TensorShape<> shape) {
  return std::vector(shape[0], std::vector(shape[1], std::vector(shape[2], 0.0f)));
}

template <class T>
SampleView<GPUBackend> get_gpu_sample_view(kernels::TestTensorList<T> &list) {
  auto tv = list.gpu()[0];
  return SampleView<GPUBackend>(tv.data, tv.shape, type2id<T>::value);
}

// Helper values, to make testing data more readable
const std::vector<float> pixelA = {0.00f, 0.01f, 0.02f};
const std::vector<float> pixelB = {0.10f, 0.11f, 0.12f};
const std::vector<float> pixelC = {0.20f, 0.21f, 0.22f};
const std::vector<float> pixelD = {0.30f, 0.31f, 0.32f};
const std::vector<float> pixelE = {0.40f, 0.41f, 0.42f};
const std::vector<float> pixelF = {0.50f, 0.51f, 0.52f};

}  // namespace

template <typename ConversionType>
class ConvertGPUTest : public ::testing::Test {
 public:
  using Input = typename ConversionType::In;
  using Output = typename ConversionType::Out;

  void SetReference(const TensorTestData &data) {
    init_test_tensor_list(reference_list_, data);
    init_test_tensor_list(output_list_, empty_data(get_data_shape(data)));
  }

  void SetInput(const TensorTestData &data) {
    init_test_tensor_list(input_list_, data);
  }

  void CheckConvert(TensorLayout out_layout, DALIImageType out_format, TensorLayout in_layout,
                    DALIImageType in_format, const ROI &roi = {},
                    nvimgcodecOrientation_t orientation = {}, float multiplier = 1.0f) {
    int device_id;
    CUDA_CALL(cudaGetDevice(&device_id));
    auto out = get_gpu_sample_view(output_list_);
    auto in = get_gpu_sample_view(input_list_);
    auto stream = CUDAStreamPool::instance().Get(device_id);
    ConvertGPU(out, out_layout, out_format, in, in_layout, in_format, stream, roi, orientation,
               multiplier);
    output_list_.invalidate_cpu();
    auto tv = output_list_.cpu(stream)[0];  // here d2h copy happens
    CUDA_CALL(cudaStreamSynchronize(stream));
    Check(output_list_.cpu()[0], reference_list_.cpu()[0], EqualConvertNorm(eps_));
  }

 private:
  kernels::TestTensorList<Input> input_list_;
  kernels::TestTensorList<Output> output_list_;
  kernels::TestTensorList<Output> reference_list_;
  const float eps_ = 0.01f;
};

using ConversionTypes =
    ::testing::Types<ConversionTestType<uint8_t, int16_t>, ConversionTestType<float, uint8_t>,
                     ConversionTestType<uint16_t, float>>;

TYPED_TEST_SUITE(ConvertGPUTest, ConversionTypes);

TYPED_TEST(ConvertGPUTest, Multiply) {
  this->SetInput({
      {{0.00f, 0.01f, 0.02f}, {0.03f, 0.04f, 0.05f}},
      {{0.10f, 0.11f, 0.12f}, {0.13f, 0.14f, 0.15f}},
  });

  this->SetReference({
      {{0.00f, 0.02f, 0.04f}, {0.06f, 0.08f, 0.10f}},
      {{0.20f, 0.22f, 0.24f}, {0.26f, 0.28f, 0.30f}},
  });

  this->CheckConvert("HWC", DALI_RGB, "HWC", DALI_RGB, {}, {}, 2.0f);
}

TYPED_TEST(ConvertGPUTest, PlanarToInterleaved) {
  this->SetInput({
      {
          {0.00f, 0.01f, 0.02f, 0.03f},
          {0.10f, 0.11f, 0.12f, 0.13f},
      },
      {
          {0.20f, 0.21f, 0.22f, 0.23f},
          {0.30f, 0.31f, 0.32f, 0.33f},
      },
      {
          {0.40f, 0.41f, 0.42f, 0.43f},
          {0.50f, 0.51f, 0.52f, 0.53f},
      },
  });

  this->SetReference({
      {{0.00f, 0.20f, 0.40f}, {0.01f, 0.21f, 0.41f}, {0.02f, 0.22f, 0.42f}, {0.03f, 0.23f, 0.43f}},
      {{0.10f, 0.30f, 0.50f}, {0.11f, 0.31f, 0.51f}, {0.12f, 0.32f, 0.52f}, {0.13f, 0.33f, 0.53f}},
  });

  this->CheckConvert("HWC", DALI_RGB, "CHW", DALI_RGB);
}

TYPED_TEST(ConvertGPUTest, InterleavedToPlanar) {
  this->SetInput({
      {{0.00f, 0.20f, 0.40f}, {0.01f, 0.21f, 0.41f}, {0.02f, 0.22f, 0.42f}, {0.03f, 0.23f, 0.43f}},
      {{0.10f, 0.30f, 0.50f}, {0.11f, 0.31f, 0.51f}, {0.12f, 0.32f, 0.52f}, {0.13f, 0.33f, 0.53f}},
  });

  this->SetReference({{
                          {0.00f, 0.01f, 0.02f, 0.03f},
                          {0.10f, 0.11f, 0.12f, 0.13f},
                      },
                      {
                          {0.20f, 0.21f, 0.22f, 0.23f},
                          {0.30f, 0.31f, 0.32f, 0.33f},
                      },
                      {
                          {0.40f, 0.41f, 0.42f, 0.43f},
                          {0.50f, 0.51f, 0.52f, 0.53f},
                      }});

  this->CheckConvert("CHW", DALI_RGB, "HWC", DALI_RGB);
}

TYPED_TEST(ConvertGPUTest, TransposeWithRoi2D) {
  this->SetInput({
      {
          {0.00f, 0.01f, 0.02f, 0.03f},
          {0.10f, 0.11f, 0.12f, 0.13f},
      },
      {
          {0.20f, 0.21f, 0.22f, 0.23f},
          {0.30f, 0.31f, 0.32f, 0.33f},
      },
      {
          {0.40f, 0.41f, 0.42f, 0.43f},
          {0.50f, 0.51f, 0.52f, 0.53f},
      },
  });

  this->SetReference({
      {{0.12f, 0.32f, 0.52f}, {0.13f, 0.33f, 0.53f}},
  });

  this->CheckConvert("HWC", DALI_RGB, "CHW", DALI_RGB, {{1, 2}, {2, 4}});
}

TYPED_TEST(ConvertGPUTest, TransposeWithRoi3D) {
  this->SetInput({
      {
          {0.00f, 0.01f, 0.02f, 0.03f},
          {0.10f, 0.11f, 0.12f, 0.13f},
      },
      {
          {0.20f, 0.21f, 0.22f, 0.23f},
          {0.30f, 0.31f, 0.32f, 0.33f},
      },
      {
          {0.40f, 0.41f, 0.42f, 0.43f},
          {0.50f, 0.51f, 0.52f, 0.53f},
      },
  });

  this->SetReference({
      {{0.12f, 0.32f, 0.52f}, {0.13f, 0.33f, 0.53f}},
  });

  this->CheckConvert("HWC", DALI_RGB, "CHW", DALI_RGB, {{1, 2, 0}, {2, 4, 3}});
}

TYPED_TEST(ConvertGPUTest, RGBToYCbCr) {
  this->SetInput({
      {
          {0.1f, 0.2f, 0.3f},
      },
  });

  this->SetReference({
      {
          {0.218f, 0.558f, 0.449f},
      },
  });

  this->CheckConvert("HWC", DALI_YCbCr, "HWC", DALI_RGB);
}

TYPED_TEST(ConvertGPUTest, RGBToBGR) {
  this->SetInput({
      {
          {0.1f, 0.2f, 0.3f},
      },
  });

  this->SetReference({
      {
          {0.3f, 0.2f, 0.1f},
      },
  });

  this->CheckConvert("HWC", DALI_BGR, "HWC", DALI_RGB);
}

TYPED_TEST(ConvertGPUTest, RGBToGray) {
  this->SetInput({
      {
          {0.1f, 0.2f, 0.3f},
      },
  });

  this->SetReference({
      {
          {0.181f},
      },
  });

  this->CheckConvert("HWC", DALI_GRAY, "HWC", DALI_RGB);
}

TYPED_TEST(ConvertGPUTest, Rotation90) {
  this->SetInput({
      {pixelA, pixelB, pixelC},
      {pixelD, pixelE, pixelF},
  });

  this->SetReference({
      {pixelC, pixelF},
      {pixelB, pixelE},
      {pixelA, pixelD},
  });

  nvimgcodecOrientation_t orientation{NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION,
                                      sizeof(nvimgcodecOrientation_t),
                                      nullptr,
                                      90,
                                      false,
                                      false};
  this->CheckConvert("HWC", DALI_RGB, "HWC", DALI_RGB, {}, orientation);
}

TYPED_TEST(ConvertGPUTest, Rotation90FlipX) {
  this->SetInput({
      {pixelA, pixelB, pixelC},
      {pixelD, pixelE, pixelF},
  });

  this->SetReference({
      {pixelF, pixelC},
      {pixelE, pixelB},
      {pixelD, pixelA},
  });

  nvimgcodecOrientation_t orientation{NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION,
                                      sizeof(nvimgcodecOrientation_t),
                                      nullptr,
                                      90,
                                      true,
                                      false};
  this->CheckConvert("HWC", DALI_RGB, "HWC", DALI_RGB, {}, orientation);
}

TYPED_TEST(ConvertGPUTest, Rotation180) {
  this->SetInput({
      {pixelA, pixelB, pixelC},
      {pixelD, pixelE, pixelF},
  });

  this->SetReference({
      {pixelF, pixelE, pixelD},
      {pixelC, pixelB, pixelA},
  });

  nvimgcodecOrientation_t orientation{NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION,
                                      sizeof(nvimgcodecOrientation_t),
                                      nullptr,
                                      180,
                                      false,
                                      false};
  this->CheckConvert("HWC", DALI_RGB, "HWC", DALI_RGB, {}, orientation);
}

TYPED_TEST(ConvertGPUTest, Rotation270) {
  this->SetInput({
      {pixelA, pixelB, pixelC},
      {pixelD, pixelE, pixelF},
  });

  this->SetReference({
      {pixelD, pixelA},
      {pixelE, pixelB},
      {pixelF, pixelC},
  });

  nvimgcodecOrientation_t orientation{NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION,
                                      sizeof(nvimgcodecOrientation_t),
                                      nullptr,
                                      270,
                                      false,
                                      false};
  this->CheckConvert("HWC", DALI_RGB, "HWC", DALI_RGB, {}, orientation);
}

TYPED_TEST(ConvertGPUTest, FlipX) {
  this->SetInput({
      {pixelA, pixelB, pixelC},
      {pixelD, pixelE, pixelF},
  });

  this->SetReference({
      {pixelC, pixelB, pixelA},
      {pixelF, pixelE, pixelD},
  });

  nvimgcodecOrientation_t orientation{NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION,
                                      sizeof(nvimgcodecOrientation_t),
                                      nullptr,
                                      0,
                                      true,
                                      false};
  this->CheckConvert("HWC", DALI_RGB, "HWC", DALI_RGB, {}, orientation);
}

TYPED_TEST(ConvertGPUTest, FlipY) {
  this->SetInput({
      {pixelA, pixelB, pixelC},
      {pixelD, pixelE, pixelF},
  });

  this->SetReference({
      {pixelD, pixelE, pixelF},
      {pixelA, pixelB, pixelC},
  });

  nvimgcodecOrientation_t orientation{NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION,
                                      sizeof(nvimgcodecOrientation_t),
                                      nullptr,
                                      0,
                                      false,
                                      true};
  this->CheckConvert("HWC", DALI_RGB, "HWC", DALI_RGB, {}, orientation);
}

TYPED_TEST(ConvertGPUTest, TransposeAndRotate_PlanarToInterleaved) {
  this->SetInput({
      {
          {0.00f, 0.01f, 0.02f, 0.03f},
          {0.10f, 0.11f, 0.12f, 0.13f},
      },
      {
          {0.20f, 0.21f, 0.22f, 0.23f},
          {0.30f, 0.31f, 0.32f, 0.33f},
      },
      {
          {0.40f, 0.41f, 0.42f, 0.43f},
          {0.50f, 0.51f, 0.52f, 0.53f},
      },
  });

  this->SetReference({
      {{0.03f, 0.23f, 0.43f}, {0.13f, 0.33f, 0.53f}},
      {{0.02f, 0.22f, 0.42f}, {0.12f, 0.32f, 0.52f}},
      {{0.01f, 0.21f, 0.41f}, {0.11f, 0.31f, 0.51f}},
      {{0.00f, 0.20f, 0.40f}, {0.10f, 0.30f, 0.50f}},
  });

  nvimgcodecOrientation_t orientation{NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION,
                                      sizeof(nvimgcodecOrientation_t),
                                      nullptr,
                                      90,
                                      false,
                                      false};
  this->CheckConvert("HWC", DALI_RGB, "CHW", DALI_RGB, {}, orientation);
}

/**
 * @brief Checks that ConvertCPU and ConvertGPU agree for arbitrary (in particular: signed)
 *        input types, when converting both the data type and the color space.
 *
 * ConvertGPU first normalizes the data to the output type and then converts the color space,
 * while ConvertCPU converts directly from the input type. For signed inputs, this used to produce
 * completely different results (e.g. negative samples would contribute negative terms to the
 * color conversion matrix on the CPU).
 */
template <typename ConversionType>
class ConvertCPUvsGPUTest : public ::testing::Test {
 public:
  using Input = typename ConversionType::In;
  using Output = typename ConversionType::Out;

  static std::vector<Input> TestValues(int n) {
    std::vector<Input> values;
    if constexpr (std::is_integral_v<Input>) {
      values = {min_value<Input>(), static_cast<Input>(min_value<Input>() + 1), 0, 1,
                static_cast<Input>(max_value<Input>() / 3),
                static_cast<Input>(max_value<Input>() / 2 + 1),
                static_cast<Input>(max_value<Input>() - 1), max_value<Input>()};
      if (std::is_signed_v<Input>)
        values.insert(values.end(), {static_cast<Input>(-1),
                                     static_cast<Input>(-(max_value<Input>() / 3))});
    } else {
      values = {-2.0f, -1.0f, -0.5f, 0.0f, 0.25f, 0.5f, 1.0f, 1.5f};
    }
    // The SNORM min_value is clamped to -1 by the CPU color conversion, while the GPU path
    // normalizes with ConvertSatNorm-like scaling, which doesn't clamp it for floating point
    // output. Skip this single value for floating point outputs.
    if (std::is_floating_point_v<Output> && std::is_signed_v<Input> &&
        std::is_integral_v<Input>)
      values.erase(values.begin());
    std::mt19937_64 rng(42);
    while (static_cast<int>(values.size()) < n) {
      if constexpr (std::is_integral_v<Input>) {
        std::uniform_int_distribution<int64_t> dist(
          std::is_floating_point_v<Output> ? min_value<Input>() + 1 : min_value<Input>(),
          max_value<Input>());
        values.push_back(static_cast<Input>(dist(rng)));
      } else {
        std::uniform_real_distribution<float> dist(-2, 2);
        values.push_back(dist(rng));
      }
    }
    return values;
  }

  void Run(DALIImageType out_format, DALIImageType in_format, double int_eps) {
    int in_channels = NumberOfChannels(in_format, 3);
    int out_channels = NumberOfChannels(out_format, 3);
    // All combinations of the edge values in each channel of the first pixels, then random data
    const int H = 16, W = 128;
    auto values = TestValues(H * W);
    TensorShape<> in_shape{H, W, in_channels}, out_shape{H, W, out_channels};

    kernels::TestTensorList<Input> in_list;
    in_list.reshape(uniform_list_shape(1, in_shape));
    auto in_cpu = in_list.cpu()[0];
    int64_t npixels = H * W;
    int nedge = std::min<int>(values.size(), 10);
    for (int64_t p = 0; p < npixels; p++) {
      for (int c = 0; c < in_channels; c++) {
        int64_t idx = p;
        // enumerate the combinations of edge values in the first nedge^in_channels pixels
        int64_t combo = p;
        for (int k = 0; k < c; k++) combo /= nedge;
        if (p < std::pow(nedge, in_channels))
          idx = combo % nedge;
        else
          idx = (p * in_channels + c) % values.size();
        in_cpu.data[p * in_channels + c] = values[idx];
      }
    }

    // CPU
    std::vector<Output> cpu_out(volume(out_shape));
    ConstSampleView<CPUBackend> cpu_in_view(in_cpu.data, in_shape, type2id<Input>::value);
    SampleView<CPUBackend> cpu_out_view(cpu_out.data(), out_shape, type2id<Output>::value);
    ConvertCPU(cpu_out_view, "HWC", out_format, cpu_in_view, "HWC", in_format, {});

    // GPU
    int device_id;
    CUDA_CALL(cudaGetDevice(&device_id));
    auto stream = CUDAStreamPool::instance().Get(device_id);
    kernels::TestTensorList<Output> out_list;
    out_list.reshape(uniform_list_shape(1, out_shape));
    auto in_gpu = in_list.gpu(stream)[0];
    auto out_gpu = out_list.gpu(stream)[0];
    SampleView<GPUBackend> gpu_out_view(out_gpu.data, out_shape, type2id<Output>::value);
    ConstSampleView<GPUBackend> gpu_in_view(in_gpu.data, in_shape, type2id<Input>::value);
    ConvertGPU(gpu_out_view, "HWC", out_format, gpu_in_view, "HWC", in_format, stream, {}, {},
               1.0f);
    auto gpu_out = out_list.cpu(stream)[0];
    CUDA_CALL(cudaStreamSynchronize(stream));

    // For integral outputs, `int_eps` accounts for the intermediate rounding done by the GPU path,
    // amplified by the conversion matrix.
    double abs_eps = std::is_integral_v<Output> ? int_eps : 1e-5;
    if (in_format == DALI_YCbCr) {
      // Luma footroom and chroma offsets are powers of two (e.g. 16 and 128 for 8 bits, 4096 and
      // 32768 for 16 bits), so their normalized values differ slightly between types. The CPU
      // interprets them in the input type, while the GPU first normalizes the data to the
      // output type and then interprets them in the output type - account for that difference.
      auto norm_bias = [](auto type_tag, double bias_fraction) {
        using T = decltype(type_tag);
        if constexpr (std::is_integral_v<T>) {
          constexpr int bits = sizeof(T) * 8 - std::is_signed_v<T>;
          return std::ldexp(bias_fraction, bits) / max_value<T>();
        } else {
          return bias_fraction;
        }
      };
      double dy = std::abs(norm_bias(Input(), 1.0 / 16) - norm_bias(Output(), 1.0 / 16));
      double dc = std::abs(norm_bias(Input(), 0.5) - norm_bias(Output(), 0.5));
      double unit = std::is_integral_v<Output> ? static_cast<double>(max_value<Output>()) : 1.0;
      abs_eps += (255.0 / 219 * dy + 2.02 * dc) * unit;
    }
    for (int64_t p = 0; p < npixels; p++) {
      for (int c = 0; c < out_channels; c++) {
        int64_t i = p * out_channels + c;
        ASSERT_NEAR(static_cast<double>(cpu_out[i]), static_cast<double>(gpu_out.data[i]),
                    abs_eps)
            << "pixel " << p << " channel " << c << " input: "
            << static_cast<double>(in_cpu.data[p * in_channels]) << " "
            << (in_channels > 1 ? static_cast<double>(in_cpu.data[p * in_channels + 1]) : 0.0)
            << " "
            << (in_channels > 2 ? static_cast<double>(in_cpu.data[p * in_channels + 2]) : 0.0);
      }
    }
  }
};

using CPUvsGPUConversionTypes = ::testing::Types<
    ConversionTestType<int8_t, uint8_t>, ConversionTestType<int16_t, uint8_t>,
    ConversionTestType<int32_t, uint8_t>, ConversionTestType<uint8_t, uint8_t>,
    ConversionTestType<uint16_t, uint8_t>, ConversionTestType<uint32_t, uint8_t>,
    ConversionTestType<float, uint8_t>,
    ConversionTestType<int32_t, uint16_t>, ConversionTestType<int8_t, int16_t>,
    ConversionTestType<int16_t, int16_t>, ConversionTestType<int32_t, int16_t>,
    ConversionTestType<uint16_t, int16_t>, ConversionTestType<float, int16_t>,
    ConversionTestType<int8_t, float>, ConversionTestType<int32_t, float>,
    ConversionTestType<uint8_t, float>>;

TYPED_TEST_SUITE(ConvertCPUvsGPUTest, CPUvsGPUConversionTypes);

TYPED_TEST(ConvertCPUvsGPUTest, RGBToYCbCr) {
  this->Run(DALI_YCbCr, DALI_RGB, 1);
}

TYPED_TEST(ConvertCPUvsGPUTest, RGBToGray) {
  this->Run(DALI_GRAY, DALI_RGB, 1);
}

TYPED_TEST(ConvertCPUvsGPUTest, BGRToYCbCr) {
  this->Run(DALI_YCbCr, DALI_BGR, 1);
}

TYPED_TEST(ConvertCPUvsGPUTest, YCbCrToRGB) {
  // intermediate rounding (0.5) * (1.164 + 2.017) + rounding of both results (0.5 + 0.5)
  this->Run(DALI_RGB, DALI_YCbCr, 3);
}

TYPED_TEST(ConvertCPUvsGPUTest, YCbCrToGray) {
  this->Run(DALI_GRAY, DALI_YCbCr, 1);
}

TYPED_TEST(ConvertCPUvsGPUTest, GrayToYCbCr) {
  this->Run(DALI_YCbCr, DALI_GRAY, 1);
}

TYPED_TEST(ConvertCPUvsGPUTest, GrayToRGB) {
  this->Run(DALI_RGB, DALI_GRAY, 1);
}

TYPED_TEST(ConvertCPUvsGPUTest, RGBToRGB) {
  this->Run(DALI_RGB, DALI_RGB, 1);
}

/**
 * @brief A pixel from a signed int32 TIFF, for which CPU and GPU decoding to YCbCr used to differ.
 *
 * Normalized (SNORM) RGB: R = 0.98824, G = -0.97255 (not representable in uint8 -> 0),
 * B = 0.99608, which in ITU-R BT.601 YCbCr is Y = 105.58, Cb = 202.21, Cr = 220.54.
 * GRAY (JPEG luma) is 104.30.
 */
TEST(ConvertSignedTest, Int32RGBToYCbCrAndGray) {
  std::vector<int32_t> in = {2122218880, -2088533504, 2139061888};
  TensorShape<> in_shape{1, 1, 3}, ycbcr_shape{1, 1, 3}, gray_shape{1, 1, 1};
  ConstSampleView<CPUBackend> in_view(in.data(), in_shape, DALI_INT32);

  std::vector<uint8_t> ycbcr(3), gray(1);
  ConvertCPU(SampleView<CPUBackend>(ycbcr.data(), ycbcr_shape, DALI_UINT8), "HWC", DALI_YCbCr,
             in_view, "HWC", DALI_RGB, {});
  EXPECT_EQ(ycbcr, (std::vector<uint8_t>{106, 202, 221}));
  ConvertCPU(SampleView<CPUBackend>(gray.data(), gray_shape, DALI_UINT8), "HWC", DALI_GRAY,
             in_view, "HWC", DALI_RGB, {});
  EXPECT_EQ(gray, (std::vector<uint8_t>{104}));

  int device_id;
  CUDA_CALL(cudaGetDevice(&device_id));
  auto stream = CUDAStreamPool::instance().Get(device_id);
  kernels::TestTensorList<int32_t> in_list;
  kernels::TestTensorList<uint8_t> ycbcr_list, gray_list;
  in_list.reshape(uniform_list_shape(1, in_shape));
  ycbcr_list.reshape(uniform_list_shape(1, ycbcr_shape));
  gray_list.reshape(uniform_list_shape(1, gray_shape));
  std::copy(in.begin(), in.end(), in_list.cpu()[0].data);
  ConstSampleView<GPUBackend> gpu_in(in_list.gpu(stream)[0].data, in_shape, DALI_INT32);
  ConvertGPU(SampleView<GPUBackend>(ycbcr_list.gpu(stream)[0].data, ycbcr_shape, DALI_UINT8),
             "HWC", DALI_YCbCr, gpu_in, "HWC", DALI_RGB, stream);
  ConvertGPU(SampleView<GPUBackend>(gray_list.gpu(stream)[0].data, gray_shape, DALI_UINT8),
             "HWC", DALI_GRAY, gpu_in, "HWC", DALI_RGB, stream);
  auto gpu_ycbcr = ycbcr_list.cpu(stream)[0];
  auto gpu_gray = gray_list.cpu(stream)[0];
  CUDA_CALL(cudaStreamSynchronize(stream));
  EXPECT_EQ(std::vector<uint8_t>(gpu_ycbcr.data, gpu_ycbcr.data + 3),
            (std::vector<uint8_t>{106, 202, 221}));
  EXPECT_EQ(gpu_gray.data[0], 104);
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
