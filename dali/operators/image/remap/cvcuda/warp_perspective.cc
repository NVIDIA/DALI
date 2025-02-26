// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cvcuda/OpWarpPerspective.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include "dali/core/dev_buffer.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/operator/operator.h"

#include "dali/operators/image/remap/cvcuda/matrix_adjust.h"
#include "dali/operators/nvcvop/nvcvop.h"

namespace dali {


DALI_SCHEMA(experimental__WarpPerspective)
    .DocStr(R"doc(
Performs a perspective transform on the images.
)doc")
    .NumInput(1, 2)
    .InputDox(0, "input", "TensorList of uint8, uint16, int16 or float",
              "Input data. Must be images in HWC or CHW layout, or a sequence of those.")
    .InputDox(1, "matrix", "2D TensorList of float",
              "3x3 Perspective transform matrix for per sample homography, same device as input.")
    .NumOutput(1)
    .InputLayout(0, {"HW", "HWC", "FHWC", "CHW", "FCHW"})
    .AddOptionalArg<float>("size",
                           R"code(Output size, in pixels/points.

The channel dimension should be excluded (for example, for RGB images,
specify ``(480,640)``, not ``(480,640,3)``.
)code",
                           std::vector<float>({}), true)
    .AddOptionalArg<float>("matrix",
                           R"doc(
  3x3 Perspective transform matrix of destination to source coordinates.
  If `inverse_map` argument is set to false, the matrix is interpreted
  as a source to destination coordinates mapping.

It is equivalent to OpenCV's ``warpPerspective`` operation with the `inverse_map` argument being
analog to the ``WARP_INVERSE_MAP`` flag.

.. note::
  Instead of this argument, the operator can take a second positional input, in which
  case the matrix can be placed on the GPU.)doc",
                           std::vector<float>({}), true, true)
    .AddOptionalArg("border_mode",
                    "Border mode to be used when accessing elements outside input image.\n"
                    "Supported values are: \"constant\", \"replicate\", "
                    "\"reflect\", \"reflect_101\", \"wrap\".",
                    "constant")
    .AddOptionalArg("interp_type", "Type of interpolation used.", DALI_INTERP_LINEAR)
    .AddOptionalArg("pixel_origin", R"doc(Pixel origin. Possible values: "corner", "center".

Determines the meaning of (0, 0) coordinates - "corner" places the origin at the top-left corner of
the top-left pixel (like in OpenGL); "center" places (0, 0) in the center of
the top-left pixel (like in OpenCV).))doc",
                    "corner")
    .AddOptionalArg<float>("fill_value",
                           "Value used to fill areas that are outside the source image when the "
                           "\"constant\" border_mode is chosen.",
                           std::vector<float>({}))
    .AddOptionalArg<bool>("inverse_map",
                          "If set to true (default), the matrix is interpreted as "
                          "destination to source coordinates mapping. "
                          "Otherwise it's interpreted as source to destination "
                          "coordinates mapping.",
                          true);

bool OCVCompatArg(std::string_view arg) {
  if (arg == "corner") {
    return false;
  } else if (arg == "center") {
    return true;
  } else {
    DALI_FAIL(make_string("Invalid pixel_origin argument: ", arg));
  }
}

template <typename T>
T GetFillValue(const std::vector<float> &fill_value_arg, int channels) {
  if (fill_value_arg.size() > 1) {
    if (channels > 0) {
      if (channels != static_cast<int>(fill_value_arg.size())) {
        DALI_FAIL(make_string(
            "Number of values provided as a fill_value should match the number of channels.\n"
            "Number of channels: ",
            channels, ". Number of values provided: ", fill_value_arg.size(), "."));
      }
      assert(channels <= 4);
      T fill_value{0, 0, 0, 0};
      if constexpr (std::is_same<T, float4>::value) {
        std::memcpy(&fill_value, fill_value_arg.data(), fill_value_arg.size() * sizeof(float));
      } else {
        static_assert(std::is_same<T, cv::Scalar>::value, "Unsupported fill value type.");
        std::copy(fill_value_arg.begin(), fill_value_arg.end(), fill_value.val);
      }
      return fill_value;
    } else {
      DALI_FAIL("Only scalar fill_value can be provided when processing data in planar layout.");
    }
  } else if (fill_value_arg.size() == 1) {
    auto fv = fill_value_arg[0];
    T fill_value{fv, fv, fv, fv};
    return fill_value;
  } else {
    return T{0, 0, 0, 0};
  }
}

template <typename Backend>
void ValidateTypes(const Workspace &ws) {
  auto inp_type = ws.Input<Backend>(0).type();
  DALI_ENFORCE(inp_type == DALI_UINT8 || inp_type == DALI_INT16 || inp_type == DALI_UINT16 ||
                   inp_type == DALI_FLOAT,
               "The operator accepts the following input types: "
               "uint8, int16, uint16, float.");
  if (ws.NumInput() > 1) {
    auto mat_type = ws.Input<Backend>(1).type();
    DALI_ENFORCE(mat_type == DALI_FLOAT,
                 "Transformation matrix can be provided only as float32 values.");
  }
}


class WarpPerspective : public nvcvop::NVCVSequenceOperator<StatelessOperator> {
 public:
  explicit WarpPerspective(const OpSpec &spec)
      : nvcvop::NVCVSequenceOperator<StatelessOperator>(spec),
        border_mode_(nvcvop::GetBorderMode(spec.GetArgument<std::string>("border_mode"))),
        interp_type_(nvcvop::GetInterpolationType(spec.GetArgument<DALIInterpType>("interp_type"))),
        fill_value_arg_(spec.GetArgument<std::vector<float>>("fill_value")),
        inverse_map_(spec.GetArgument<bool>("inverse_map")),
        ocv_pixel_(OCVCompatArg(spec.GetArgument<std::string>("pixel_origin"))) {
    matrix_data_.SetContiguity(BatchContiguity::Contiguous);
  }

  bool ShouldExpandChannels(int input_idx) const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    ValidateTypes<GPUBackend>(ws);
    const auto &input = ws.Input<GPUBackend>(0);
    auto input_shape = input.shape();
    auto input_layout = input.GetLayout();
    output_desc.resize(1);

    auto output_shape = input_shape;
    int channels = (input_layout.find('C') != -1) ? input_shape[0][input_layout.find('C')] : -1;
    if (channels > 4)
      DALI_FAIL("Images with more than 4 channels are not supported.");
    fill_value_ = GetFillValue<float4>(fill_value_arg_, channels);
    if (size_arg_.HasExplicitValue()) {
      size_arg_.Acquire(spec_, ws, input_shape.size(), TensorShape<1>(2));
      for (int i = 0; i < input_shape.size(); i++) {
        auto height = std::max<int>(std::roundf(size_arg_[i].data[0]), 1);
        auto width = std::max<int>(std::roundf(size_arg_[i].data[1]), 1);
        auto out_sample_shape = (channels != -1) ? TensorShape<>({height, width, channels}) :
                                                   TensorShape<>({height, width});
        output_shape.set_tensor_shape(i, out_sample_shape);
      }
    }

    output_desc[0] = {output_shape, input.type()};
    return true;
  }

  void RunImpl(Workspace &ws) override {
    const auto &input = ws.Input<GPUBackend>(0);
    auto &output = ws.Output<GPUBackend>(0);
    output.SetLayout(input.GetLayout());

    kernels::DynamicScratchpad scratchpad(AccessOrder(ws.stream()));

    nvcv::Tensor matrix{};
    if (ws.NumInput() > 1) {
      DALI_ENFORCE(!matrix_arg_.HasExplicitValue(),
                   "Matrix input and `matrix` argument should not be provided at the same time.");
      auto &matrix_input = ws.Input<GPUBackend>(1);
      DALI_ENFORCE(matrix_input.shape() ==
                       uniform_list_shape(matrix_input.num_samples(), TensorShape<2>(3, 3)),
                   make_string("Expected a uniform list of 3x3 matrices. "
                               "Instead got data with shape: ",
                               matrix_input.shape()));

      matrix_data_.Copy(matrix_input, AccessOrder(ws.stream()));
      Tensor<GPUBackend> matrix_tensor = matrix_data_.AsTensor();
      matrix = nvcvop::AsTensor(matrix_tensor, "NW", TensorShape<2>{input.num_samples(), 9});
    } else {
      matrix = AcquireTensorArgument(ws, scratchpad, matrix_arg_, TensorShape<2>{3, 3},
                                     nvcvop::GetDataType<float>(), "W", TensorShape<1>{9});
    }
    if (!ocv_pixel_) {
      warp_perspective::adjustMatrices(matrix, ws.stream());
    }

    auto input_images = GetInputBatch(ws, 0);
    auto output_images = GetOutputBatch(ws, 0);
    if (!warp_perspective_ || input.num_samples() > op_batch_size_) {
      op_batch_size_ = std::max(op_batch_size_ * 2, input.num_samples());
      warp_perspective_.emplace(op_batch_size_);
    }
    int32_t flags = interp_type_;
    if (inverse_map_) {
      flags |= NVCV_WARP_INVERSE_MAP;
    }
    (*warp_perspective_)(ws.stream(), input_images, output_images, matrix, flags, border_mode_,
                         fill_value_);
  }

 private:
  USE_OPERATOR_MEMBERS();
  ArgValue<float, 2> matrix_arg_{"matrix", spec_};
  ArgValue<float, 1> size_arg_{"size", spec_};
  int op_batch_size_ = 0;
  NVCVBorderType border_mode_ = NVCV_BORDER_CONSTANT;
  NVCVInterpolationType interp_type_ = NVCV_INTERP_NEAREST;
  std::vector<float> fill_value_arg_{0, 0, 0, 0};
  float4 fill_value_{};
  bool inverse_map_ = false;
  bool ocv_pixel_ = true;
  std::optional<cvcuda::WarpPerspective> warp_perspective_;
  TensorList<GPUBackend> matrix_data_;
};

DALI_REGISTER_OPERATOR(experimental__WarpPerspective, WarpPerspective, GPU);


class WarpPerspectiveCPU : public SequenceOperator<CPUBackend, StatelessOperator> {
 public:
  explicit WarpPerspectiveCPU(const OpSpec &spec)
      : SequenceOperator(spec),
        border_mode_(GetBorderMode(spec.GetArgument<std::string>("border_mode"))),
        interp_type_(GetInterpolationType(spec.GetArgument<DALIInterpType>("interp_type"))),
        fill_value_arg_(spec.GetArgument<std::vector<float>>("fill_value")),
        inverse_map_(spec.GetArgument<bool>("inverse_map")),
        ocv_pixel_(OCVCompatArg(spec.GetArgument<std::string>("pixel_origin"))) {}

 private:
  cv::BorderTypes GetBorderMode(std::string_view border_mode) {
    if (border_mode == "constant") {
      return cv::BorderTypes::BORDER_CONSTANT;
    } else if (border_mode == "replicate") {
      return cv::BorderTypes::BORDER_REPLICATE;
    } else if (border_mode == "reflect") {
      return cv::BorderTypes::BORDER_REFLECT;
    } else if (border_mode == "reflect_101") {
      return cv::BorderTypes::BORDER_REFLECT_101;
    } else if (border_mode == "wrap") {
      return cv::BorderTypes::BORDER_WRAP;
    } else {
      DALI_FAIL(make_string("Invalid border_mode argument: ", border_mode));
    }
  }

  cv::InterpolationFlags GetInterpolationType(DALIInterpType interpolation_type) {
    switch (interpolation_type) {
      case DALIInterpType::DALI_INTERP_NN:
        return cv::InterpolationFlags::INTER_NEAREST;
      case DALIInterpType::DALI_INTERP_LINEAR:
        return cv::InterpolationFlags::INTER_LINEAR;
      case DALIInterpType::DALI_INTERP_CUBIC:
        return cv::InterpolationFlags::INTER_CUBIC;
      default:
        DALI_FAIL(
            make_string("Unknown interpolation type: ", static_cast<int>(interpolation_type)));
    }
  }


  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    ValidateTypes<CPUBackend>(ws);
    const auto &input = ws.Input<CPUBackend>(0);
    auto input_shape = input.shape();
    auto input_layout = input.GetLayout();
    output_desc.resize(1);

    auto output_shape = input_shape;
    const auto chIdx = input_layout.find('C');
    if (chIdx == -1 && ws.GetInputDim(0) > 2) {
      DALI_FAIL("Layout not specified and number of dims > 2, can't determine channel count.");
    } else if (chIdx != -1 && chIdx != input_layout.size() - 1) {
      DALI_FAIL("Channel dimension must be the last one.");
    }

    int channels = (chIdx != -1) ? input_shape[0][chIdx] : -1;
    if (channels > 4)
      DALI_FAIL("Images with more than 4 channels are not supported.");

    fill_value_ = GetFillValue<cv::Scalar>(fill_value_arg_, channels);
    if (size_arg_.HasExplicitValue()) {
      size_arg_.Acquire(spec_, ws, input_shape.size(), TensorShape<1>(2));
      for (int i = 0; i < input_shape.size(); i++) {
        auto height = std::max<int>(std::roundf(size_arg_[i].data[0]), 1);
        auto width = std::max<int>(std::roundf(size_arg_[i].data[1]), 1);
        auto out_sample_shape = (channels != -1) ? TensorShape<>({height, width, channels}) :
                                                   TensorShape<>({height, width});
        output_shape.set_tensor_shape(i, out_sample_shape);
      }
    }

    channels_ = std::max(1, channels);  // If channels not specified in layout (-1) then must be 1

    output_desc[0] = {output_shape, input.type()};
    return true;
  }

  bool ShouldExpandChannels(int input_idx) const override {
    return true;
  }

  /**
   * @brief Converts OpenGL perspective warp format to OpenCV format
   * @param matrix 3x3 matrix to convert inplace
   */
  void ConvertOpenGLtoOpenCVFormat(cv::Matx33f &matrix) {
    // clang-format off
    const cv::Matx33f shift = {
      1, 0, 0.5,
      0, 1, 0.5,
      0, 0, 1,
    };
    const cv::Matx33f shiftBack = {
      1, 0, -0.5,
      0, 1, -0.5,
      0, 0, 1,
    };
    // clang-format on
    matrix = shiftBack * (matrix * shift);
  }

  /**
   * @brief Convert DALI data type to OpenCV matrix type
   */
  int matTypeFromDALI(DALIDataType dtype) {
    switch (dtype) {
      case DALI_UINT8:
        return CV_8U;
      case DALI_INT16:
        return CV_16S;
      case DALI_UINT16:
        return CV_16U;
      case DALI_FLOAT:
        return CV_32F;
      default:
        DALI_FAIL("Unsupported input type");
    }
  }

  /**
   * @brief Construct a full OpenCV matrix type from DALI data type and number of channels
   */
  int fullMatTypeFromDALI(DALIDataType dtype, int channels) {
    return CV_MAKETYPE(matTypeFromDALI(dtype), channels);
  }

  void RunImpl(Workspace &ws) override {
    const auto &input = ws.Input<CPUBackend>(0);
    auto &output = ws.Output<CPUBackend>(0);
    output.SetLayout(input.GetLayout());

    const int num_samples = ws.GetInputBatchSize(0);
    std::vector<cv::Matx33f> matrices(num_samples);
    if (ws.NumInput() > 1) {
      DALI_ENFORCE(!matrix_arg_.HasExplicitValue(),
                   "Matrix input and `matrix` argument should not be provided at the same time.");
      auto &matrix_input = ws.Input<CPUBackend>(1);
      DALI_ENFORCE(matrix_input.shape() == uniform_list_shape(num_samples, TensorShape<2>(3, 3)),
                   make_string("Expected a uniform list of 3x3 matrices. "
                               "Instead got data with shape: ",
                               matrix_input.shape()));

      for (int i = 0; i < num_samples; i++) {
        std::memcpy(matrices[i].val, matrix_input.raw_tensor(i), sizeof(cv::Matx33f));
      }
    } else {
      matrix_arg_.Acquire(spec_, ws, num_samples, TensorShape<2>(3, 3));
      for (int i = 0; i < num_samples; ++i) {
        std::memcpy(matrices[i].val, matrix_arg_[i].data, sizeof(cv::Matx33f));
      }
    }
    if (!ocv_pixel_) {
      for (auto &matrix : matrices) {
        ConvertOpenGLtoOpenCVFormat(matrix);
      }
    }

    auto &tPool = ws.GetThreadPool();
    int warpFlags = interp_type_;
    if (inverse_map_) {
      warpFlags |= cv::WARP_INVERSE_MAP;
    }
    for (int i = 0; i < num_samples; ++i) {
      tPool.AddWork([&, i](int) {
        const auto inImage = input[i];
        auto outImage = output[i];
        const int dtype = fullMatTypeFromDALI(inImage.type(), channels_);

        const auto &inShape = inImage.shape();
        const cv::Mat inMat(static_cast<int>(inShape[0]), static_cast<int>(inShape[1]), dtype,
                            const_cast<void *>(inImage.raw_data()));

        const auto &outShape = outImage.shape();
        cv::Mat outMat(static_cast<int>(outShape[0]), static_cast<int>(outShape[1]), dtype,
                       outImage.raw_mutable_data());

        cv::warpPerspective(inMat, outMat, matrices[i], cv::Size(outMat.cols, outMat.rows),
                            warpFlags, border_mode_, fill_value_);
      });
    }
    tPool.RunAll();
  }

  USE_OPERATOR_MEMBERS();
  ArgValue<float, 2> matrix_arg_{"matrix", spec_};
  ArgValue<float, 1> size_arg_{"size", spec_};
  int channels_ = 1;
  cv::BorderTypes border_mode_ = cv::BorderTypes::BORDER_CONSTANT;
  cv::InterpolationFlags interp_type_ = cv::InterpolationFlags::INTER_LINEAR;
  std::vector<float> fill_value_arg_{0, 0, 0, 0};
  cv::Scalar fill_value_{};
  bool inverse_map_ = false;
  bool ocv_pixel_ = true;
};

DALI_REGISTER_OPERATOR(experimental__WarpPerspective, WarpPerspectiveCPU, CPU);

}  // namespace dali
