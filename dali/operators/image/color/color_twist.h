// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_COLOR_COLOR_TWIST_H_
#define DALI_OPERATORS_IMAGE_COLOR_COLOR_TWIST_H_

#include <memory>
#include <string>
#include <vector>
#include "dali/core/geom/mat.h"
#include "dali/core/static_switch.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/kernels/imgproc/pointwise/linear_transformation_cpu.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

#define COLOR_TWIST_SUPPORTED_TYPES (uint8_t, int16_t, int32_t, float, float16)

namespace dali {
namespace color {

/**
 * Names of arguments
 */
const std::string kHue = "hue";                 // NOLINT
const std::string kSaturation = "saturation";   // NOLINT
const std::string kValue = "value";             // NOLINT
const std::string kBrightness = "brightness";   // NOLINT
const std::string kContrast = "contrast";       // NOLINT
const std::string kOutputType = "dtype";        // NOLINT

/**
 * Color space conversion
 */
const mat3 Rgb2Yiq = {{
                              {.299f, .587f, .114f},
                              {.596f, -.274f, -.321f},
                              {.211f, -.523f, .311f}
                      }};

const mat3 Yiq2Rgb = inverse(Rgb2Yiq);

/**
 * Composes transformation matrix for hue
 */
inline mat3 hue_mat(float hue /* hue hue hue */ ) {
  const float h_rad = hue * M_PI / 180;
  mat3 ret = mat3::eye();  // rotation matrix
  // Hue change in YIQ color space is a rotation along the Y axis
  ret(1, 1) = cos(h_rad);
  ret(2, 2) = cos(h_rad);
  ret(1, 2) = sin(h_rad);
  ret(2, 1) = -sin(h_rad);
  return ret;
}


/**
 * Composes transformation matrix for saturation
 */
inline mat3 sat_mat(float saturation) {
  mat3 ret = mat3::eye();
  // In the YIQ color space, saturation change is a
  // uniform scaling in IQ dimensions
  ret(1, 1) = saturation;
  ret(2, 2) = saturation;
  return ret;
}

}  // namespace color



template <typename Backend>
class ColorTwistBase : public Operator<Backend> {
 public:
  ~ColorTwistBase() override = default;

  DISABLE_COPY_MOVE_ASSIGN(ColorTwistBase);

 protected:
  explicit ColorTwistBase(const OpSpec &spec)
      : Operator<Backend>(spec),
        output_type_arg_(spec.GetArgument<DALIDataType>(color::kOutputType)),
        output_type_(DALI_NO_TYPE) {
    if (std::is_same<Backend, GPUBackend>::value) {
      kernel_manager_.Resize(1);
    } else {
      kernel_manager_.Resize(max_batch_size_);
    }
  }

  bool CanInferOutputs() const override {
    return true;
  }

  void AcquireArguments(const workspace_t<Backend> &ws) {
    auto curr_batch_size = ws.GetInputBatchSize(0);
    if (this->spec_.ArgumentDefined(color::kHue)) {
      this->GetPerSampleArgument(hue_, color::kHue, ws, curr_batch_size);
    } else {
      hue_ = std::vector<float>(curr_batch_size, 0);
    }

    if (this->spec_.ArgumentDefined(color::kSaturation)) {
      this->GetPerSampleArgument(saturation_, color::kSaturation, ws, curr_batch_size);
    } else {
      saturation_ = std::vector<float>(curr_batch_size, 1);
    }

    if (this->spec_.ArgumentDefined(color::kValue)) {
      this->GetPerSampleArgument(value_, color::kValue, ws, curr_batch_size);
    } else {
      value_ = std::vector<float>(curr_batch_size, 1);
    }

    if (this->spec_.ArgumentDefined(color::kBrightness)) {
      this->GetPerSampleArgument(brightness_, color::kBrightness, ws, curr_batch_size);
    } else {
      brightness_ = std::vector<float>(curr_batch_size, 1);
    }

    if (this->spec_.ArgumentDefined(color::kContrast)) {
      this->GetPerSampleArgument(contrast_, color::kContrast, ws, curr_batch_size);
    } else {
      contrast_ = std::vector<float>(curr_batch_size, 1);
    }

    auto in_type = ws.template Input<Backend>(0).type();
    output_type_ = output_type_arg_ != DALI_NO_TYPE ? output_type_arg_ : in_type;

    if (in_type == DALI_FLOAT16 || in_type == DALI_FLOAT || in_type == DALI_FLOAT64) {
      half_range_ = 0.5f;
    } else {
      half_range_ = 128.f;
    }
  }

  /**
   * @brief Creates transformation matrices based on given args
   */
  void DetermineTransformation(const workspace_t<Backend> &ws) {
    using namespace color;  // NOLINT
    AcquireArguments(ws);
    assert(hue_.size() == saturation_.size() && hue_.size() == brightness_.size());
    assert(hue_.size() == contrast_.size());
    auto size = hue_.size();
    tmatrices_.resize(size);
    toffsets_.resize(size);
    for (size_t i = 0; i < size; i++) {
      tmatrices_[i] =
               mat3(brightness_[i]) * mat3(contrast_[i]) *
               Yiq2Rgb * hue_mat(hue_[i]) * sat_mat(saturation_[i]) * mat3(value_[i]) * Rgb2Yiq;
      toffsets_[i] = (half_range_ - half_range_ * contrast_[i]) * brightness_[i];
    }
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc,
                 const workspace_t<Backend> &ws) override {
    const auto &input = ws.template Input<Backend>(0);
    const auto &output = ws.template Output<Backend>(0);
    DetermineTransformation(ws);
    auto sh = input.shape();
    assert(static_cast<size_t>(sh.num_samples()) == tmatrices_.size());
    assert(ImageLayoutInfo::IsChannelLast(input.GetLayout()));
    assert(sh.sample_dim() == 3 || sh.sample_dim() == 4);
    output_desc.resize(1);
    output_desc[0] = {sh, output_type_};
    return true;
  }

  USE_OPERATOR_MEMBERS();
  float half_range_ = 0.0f;
  std::vector<float> hue_, saturation_, value_, brightness_, contrast_;
  std::vector<mat3> tmatrices_;
  std::vector<vec3> toffsets_;
  DALIDataType output_type_arg_, output_type_;
  kernels::KernelManager kernel_manager_;
};


class ColorTwistCpu : public ColorTwistBase<CPUBackend> {
 public:
  explicit ColorTwistCpu(const OpSpec &spec) : ColorTwistBase(spec) {}

  /*
   * So that compiler wouldn't complain, that
   * "overloaded virtual function `dali::Operator<dali::CPUBackend>::RunImpl`
   * is only partially overridden in class `dali::ColorTwistCpu`"
   */
  using Operator<CPUBackend>::RunImpl;

  ~ColorTwistCpu() override = default;

  DISABLE_COPY_MOVE_ASSIGN(ColorTwistCpu);

 protected:
  void RunImpl(workspace_t<CPUBackend> &ws) override;

  template <typename OutputType, typename InputType>
  void RunImplHelper(workspace_t<CPUBackend> &ws);
};


class ColorTwistGpu : public ColorTwistBase<GPUBackend> {
 public:
  explicit ColorTwistGpu(const OpSpec &spec) : ColorTwistBase(spec) {}

  ~ColorTwistGpu() override = default;

  DISABLE_COPY_MOVE_ASSIGN(ColorTwistGpu);

 protected:
  void RunImpl(workspace_t<GPUBackend> &ws) override;

  template <typename OutputType, typename InputType>
  void RunImplHelper(workspace_t<GPUBackend> &ws);
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_COLOR_COLOR_TWIST_H_
