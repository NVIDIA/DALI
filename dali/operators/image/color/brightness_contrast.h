// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/static_switch.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/core/format.h"

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
class BrightnessContrastOp : public Operator<Backend> {
 public:
  ~BrightnessContrastOp() override = default;

  DISABLE_COPY_MOVE_ASSIGN(BrightnessContrastOp);

 protected:
  explicit BrightnessContrastOp(const OpSpec &spec)
      : Operator<Backend>(spec),
        output_type_arg_(spec.GetArgument<DALIDataType>("dtype")),
        output_type_(DALI_NO_TYPE),
        input_type_(DALI_NO_TYPE) {
    if (spec.HasArgument("contrast_center"))
      contrast_center_ = spec.GetArgument<float>("contrast_center");
  }

  bool CanInferOutputs() const override {
    return true;
  }

  template <typename OutputType, typename InputType>
  void OpArgsToKernelArgs(float &addend, float &multiplier,
    float brightness, float brightness_shift, float contrast) {
    float contrast_center = std::isnan(contrast_center_)
      ? brightness_contrast::HalfRange<InputType>()
      : contrast_center_;
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

    input_type_ = ws.template InputRef<Backend>(0).type().id();
    output_type_ = output_type_arg_ != DALI_NO_TYPE ? output_type_arg_ : input_type_;
  }

  void KMgrResize(int num_threads, int batch_size) {
    if (std::is_same<Backend, GPUBackend>::value) {
      kernel_manager_.Resize(1, 1);
    } else {
      kernel_manager_.Resize(num_threads, batch_size);
    }
  }

  USE_OPERATOR_MEMBERS();
  std::vector<float> brightness_, brightness_shift_, contrast_;
  DALIDataType output_type_arg_, output_type_, input_type_;
  float contrast_center_ = std::nanf("");
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
  using Operator<CPUBackend>::RunImpl;

  ~BrightnessContrastCpu() override = default;

  DISABLE_COPY_MOVE_ASSIGN(BrightnessContrastCpu);

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<CPUBackend> &ws) override;

  void RunImpl(workspace_t<CPUBackend> &ws) override;

 private:
  template <typename Kernel, typename InputType>
  TensorListShape<> CallSetup(const TensorVector<CPUBackend> &input) {
    kernels::KernelContext ctx;
    TensorListShape<> sh = input.shape();
    TensorListShape<> ret(sh.num_samples(), 3);
    assert(static_cast<size_t>(sh.num_samples()) == brightness_.size());
    for (int i = 0; i < sh.num_samples(); i++) {
      const auto tvin = view<const InputType, 3>(input[i]);
      const auto reqs = kernel_manager_.Setup<Kernel>(i, ctx, tvin, brightness_[i], contrast_[i]);
      const TensorListShape<> &out_sh = reqs.output_shapes[0];
      ret.set_tensor_shape(i, out_sh.tensor_shape(0));
    }
    return ret;
  }
};


class BrightnessContrastGpu : public BrightnessContrastOp<GPUBackend> {
 public:
  explicit BrightnessContrastGpu(const OpSpec &spec) : BrightnessContrastOp(spec) {}

  ~BrightnessContrastGpu() override = default;

  DISABLE_COPY_MOVE_ASSIGN(BrightnessContrastGpu);

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<GPUBackend> &ws) override;

  void RunImpl(workspace_t<GPUBackend> &ws) override;

 private:
  template <typename Kernel, typename InputType>
  const TensorListShape<> &CallSetup(const DeviceWorkspace &ws, const TensorList<GPUBackend> &tl) {
    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();
    const auto tvin = view<const InputType, 3>(tl);
    const auto &reqs = kernel_manager_.Setup<Kernel>(0, ctx, tvin, brightness_, contrast_);
    return reqs.output_shapes[0];
  }
  std::vector<float> addends_, multipliers_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_COLOR_BRIGHTNESS_CONTRAST_H_
