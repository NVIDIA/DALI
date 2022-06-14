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

#ifndef DALI_OPERATORS_IMAGE_COLOR_BRIGHTNESS_CONTRAST_H_
#define DALI_OPERATORS_IMAGE_COLOR_BRIGHTNESS_CONTRAST_H_

#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "dali/core/format.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/sequence_operator.h"

#define BRIGHTNESS_CONTRAST_SUPPORTED_TYPES (uint8_t, int16_t, int32_t, float)

namespace dali {

namespace brightness_contrast {

template <typename T>
constexpr float FullRange() {
  return std::is_integral<T>::value
    ? static_cast<float>(std::numeric_limits<T>::max())
    : 1.0f;
}

template <typename T>
constexpr float HalfRange() {
  return std::is_integral<T>::value
    ? (1 << (8*sizeof(T) - std::is_signed<T>::value - 1))
    : 0.5f;
}

}  // namespace brightness_contrast

const float kDefaultBrightness = 1.f;
const float kDefaultBrightnessShift = 0;
const float kDefaultContrast = 1.f;

template <typename Backend>
class BrightnessContrastOp : public SequenceOperator<Backend> {
 public:
  ~BrightnessContrastOp() override = default;

  DISABLE_COPY_MOVE_ASSIGN(BrightnessContrastOp);

 protected:
  explicit BrightnessContrastOp(const OpSpec &spec)
      : SequenceOperator<Backend>(spec),
        output_type_(DALI_NO_TYPE),
        input_type_(DALI_NO_TYPE) {
    spec.TryGetArgument(output_type_arg_, "dtype");
  }

  bool CanInferOutputs() const override {
    return true;
  }

  // The operator needs 4 dim path for DHWC data, so use it to avoid inflating
  // the number of samples and parameters unnecessarily for FHWC when there are no
  // per-frame parameters provided.
  bool ShouldExpand(const workspace_t<Backend> &ws) override {
    return SequenceOperator<Backend>::ShouldExpand(ws) && this->HasPerFrameArgInputs(ws);
  }

  template <typename OutputType, typename InputType>
  void OpArgsToKernelArgs(float &addend, float &multiplier, float brightness,
                          float brightness_shift, float contrast,
                          float contrast_center) {
    float brightness_range = brightness_contrast::FullRange<OutputType>();
    // The formula is:
    // out = brightness_shift * brightness_range +
    //       brightness * (contrast_center + contrast * (in - contrast_center)
    //
    // It can be rearranged as:
    // out = (brightness_shift * brightness_range +
    //        brightness * (contrast_center - contrast * contrast_center)) +
    //        brightness * contrast * in
    addend = brightness_shift * brightness_range +
             brightness * (contrast_center - contrast * contrast_center);
    multiplier = brightness * contrast;
  }

  void AcquireArguments(const workspace_t<Backend> &ws) {
    auto curr_batch_size = ws.GetInputBatchSize(0);
    if (this->spec_.ArgumentDefined("brightness")) {
      this->GetPerSampleArgument(brightness_, "brightness", ws, curr_batch_size);
    } else {
      brightness_ = std::vector<float>(curr_batch_size, kDefaultBrightness);
    }

    if (this->spec_.ArgumentDefined("brightness_shift")) {
      this->GetPerSampleArgument(brightness_shift_, "brightness_shift", ws, curr_batch_size);
    } else {
      brightness_shift_ = std::vector<float>(curr_batch_size, kDefaultBrightnessShift);
    }

    if (this->spec_.ArgumentDefined("contrast")) {
      this->GetPerSampleArgument(contrast_, "contrast", ws, curr_batch_size);
    } else {
      contrast_ = std::vector<float>(curr_batch_size, kDefaultContrast);
    }

    input_type_ = ws.template Input<Backend>(0).type();
    output_type_ = output_type_arg_ != DALI_NO_TYPE ? output_type_arg_ : input_type_;
  }

  template <typename InputType>
  const vector<float> &GetContrastCenter(const workspace_t<Backend> &ws, int num_samples) {
    if (this->spec_.ArgumentDefined("contrast_center")) {
      this->GetPerSampleArgument(contrast_center_, "contrast_center", ws, num_samples);
    } else {
      // argument cannot stop being defined in a built pipeline,
      // so just fill in missing samples if needed
      contrast_center_.resize(num_samples, brightness_contrast::HalfRange<InputType>());
    }
    return contrast_center_;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    const auto &input = ws.template Input<Backend>(0);
    AcquireArguments(ws);

    auto sh = input.shape();
    assert(ImageLayoutInfo::IsChannelLast(input.GetLayout()));
    assert(sh.sample_dim() == 3 || sh.sample_dim() == 4);
    output_desc.resize(1);
    output_desc[0] = {sh, output_type_};
    return true;
  }

  USE_OPERATOR_MEMBERS();
  std::vector<float> brightness_, brightness_shift_, contrast_, contrast_center_;
  DALIDataType output_type_arg_ = DALI_NO_TYPE;
  DALIDataType output_type_ = DALI_NO_TYPE;
  DALIDataType input_type_ = DALI_NO_TYPE;
  kernels::KernelManager kernel_manager_;
};


class BrightnessContrastCpu : public BrightnessContrastOp<CPUBackend> {
 public:
  explicit BrightnessContrastCpu(const OpSpec &spec) : BrightnessContrastOp(spec) {}

  /*
   * So that compiler wouldn't complain, that
   * "overloaded virtual function `dali::Operator<dali::CPUBackend>::RunImpl` is only partially
   * overridden in class `dali::brightness_contrast::BrightnessContrast<dali::CPUBackend>`"
   */
  using SequenceOperator<CPUBackend>::RunImpl;

  ~BrightnessContrastCpu() override = default;

  DISABLE_COPY_MOVE_ASSIGN(BrightnessContrastCpu);

 protected:
  void RunImpl(workspace_t<CPUBackend> &ws) override;

  template <typename OutputType, typename InputType, int ndim>
  void RunImplHelper(workspace_t<CPUBackend> &ws);
};


class BrightnessContrastGpu : public BrightnessContrastOp<GPUBackend> {
 public:
  explicit BrightnessContrastGpu(const OpSpec &spec) : BrightnessContrastOp(spec) {}

  ~BrightnessContrastGpu() override = default;

  DISABLE_COPY_MOVE_ASSIGN(BrightnessContrastGpu);

 protected:
  void RunImpl(workspace_t<GPUBackend> &ws) override;

  template <typename OutputType, typename InputType>
  void RunImplHelper(workspace_t<GPUBackend> &ws);

  std::vector<float> addends_, multipliers_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_COLOR_BRIGHTNESS_CONTRAST_H_
