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
#include <optional>
#include "dali/core/dev_buffer.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/operator/operator.h"

#include "dali/operators/nvcvop/nvcvop.h"
#include "dali/operators/image/remap/cvcuda/matrix_adjust.h"

namespace dali {


DALI_SCHEMA(experimental__WarpPerspective)
    .DocStr(R"doc(
Performs a perspective transform on the images.
)doc")
    .NumInput(1, 2)
    .InputDox(0, "input", "TensorList of uint8, uint16, int16 or float",
              "Input data. Must be images in HWC or CHW layout, or a sequence of those.")
    .InputDox(1, "matrix_gpu", "1D TensorList of float",
              "Transformation matrix data. Should be used to pass the GPU data. "
              "For CPU data, the `matrix` argument should be used.")
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
  Perspective transform mapping of destination to source coordinates.
  If `inverse_map` argument is set to false, the matrix is interpreted
  as a source to destination coordinates mapping.

It is equivalent to OpenCV's ``warpPerspective`` operation with the ``inverse_map`` argument being
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
the top-left pixel (like in OpenCV).))doc", "corner")
    .AddOptionalArg<float>("fill_value",
                           "Value used to fill areas that are outside the source image when the "
                           "\"constant\" border_mode is chosen.",
                           std::vector<float>({}))
    .AddOptionalArg<bool>("inverse_map",
                          "If set to true (default), the matrix is interpreted as "
                          "destination to source coordinates mapping. "
                          "Otherwise it's interpreted as source to destination "
                          "coordinates mapping.", true);


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

  float4 GetFillValue(int channels) const {
    if (fill_value_arg_.size() > 1) {
      if (channels > 0) {
        if (channels != static_cast<int>(fill_value_arg_.size())) {
          DALI_FAIL(make_string(
              "Number of values provided as a fill_value should match the number of channels.\n"
              "Number of channels: ",
              channels, ". Number of values provided: ", fill_value_arg_.size(), "."));
        }
        assert(channels <= 4);
        float4 fill_value{0, 0, 0, 0};
        memcpy(&fill_value, fill_value_arg_.data(), channels * sizeof(float));
        return fill_value;
      } else {
        DALI_FAIL("Only scalar fill_value can be provided when processing data in planar layout.");
      }
    } else if (fill_value_arg_.size() == 1) {
      auto fv = fill_value_arg_[0];
      float4 fill_value{fv, fv, fv, fv};
      return fill_value;
    } else {
      return float4{0, 0, 0, 0};
    }
  }

  void ValidateTypes(const Workspace &ws) const {
    auto inp_type = ws.Input<GPUBackend>(0).type();
    DALI_ENFORCE(inp_type == DALI_UINT8 || inp_type == DALI_INT16 || inp_type == DALI_UINT16 ||
                 inp_type == DALI_FLOAT,
                 "The operator accepts the following input types: "
                 "uint8, int16, uint16, float.");
    if (ws.NumInput() > 1) {
      auto mat_type = ws.Input<GPUBackend>(1).type();
      DALI_ENFORCE(mat_type == DALI_FLOAT,
                   "Transformation matrix can be provided only as float32 values.");
    }
  }

  bool OCVCompatArg(const std::string &arg) {
    if (arg == "corner") {
      return false;
    } else if (arg == "center") {
      return true;
    } else {
      DALI_FAIL(make_string("Invalid pixel_origin argument: ", arg));
    }
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    ValidateTypes(ws);
    const auto &input = ws.Input<GPUBackend>(0);
    auto input_shape = input.shape();
    auto input_layout = input.GetLayout();
    output_desc.resize(1);

    auto output_shape = input_shape;
    int channels = (input_layout.find('C') != -1) ? input_shape[0][input_layout.find('C')] : -1;
    if (channels > 4)
      DALI_FAIL("Images with more than 4 channels are not supported.");
    fill_value_ = GetFillValue(channels);
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

    kernels::DynamicScratchpad scratchpad({}, AccessOrder(ws.stream()));

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

}  // namespace dali
