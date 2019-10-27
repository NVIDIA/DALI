// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_COLOR_BRIGHTNESS_CONTRAST_H_
#define DALI_OPERATORS_COLOR_BRIGHTNESS_CONTRAST_H_

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

const std::string kBrightness = "brightness_delta";  // NOLINT
const std::string kContrast = "contrast_delta";      // NOLINT
const std::string kOutputType = "output_type";       // NOLINT

}  // namespace brightness_contrast


template <typename Backend>
class BrightnessContrastOp : public Operator<Backend> {
 public:
  ~BrightnessContrastOp() override = default;

  DISABLE_COPY_MOVE_ASSIGN(BrightnessContrastOp);

 protected:
  explicit BrightnessContrastOp(const OpSpec &spec) :
          Operator<Backend>(spec),
          output_type_(spec.GetArgument<DALIDataType>(brightness_contrast::kOutputType)) {
    if (std::is_same<Backend, GPUBackend>::value) {
      kernel_manager_.Resize(1, 1);
    } else {
      kernel_manager_.Resize(num_threads_, batch_size_);
    }
  }


  bool CanInferOutputs() const override {
    return true;
  }


  void AcquireArguments(const ArgumentWorkspace &ws) {
    this->GetPerSampleArgument(brightness_, brightness_contrast::kBrightness, ws);
    this->GetPerSampleArgument(contrast_, brightness_contrast::kContrast, ws);
  }


  USE_OPERATOR_MEMBERS();
  std::vector<float> brightness_, contrast_;
  DALIDataType output_type_;
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
  TensorListShape<> CallSetup(const TensorVector<CPUBackend> &input, int instance_idx) {
    kernels::KernelContext ctx;
    TensorListShape<> sh = input.shape();
    TensorListShape<> ret(sh.num_samples(), 3);
    assert(static_cast<size_t>(sh.num_samples()) == brightness_.size());
    for (int i = 0; i < sh.num_samples(); i++) {
      const auto tvin = view<const InputType, 3>(input[i]);
      const auto reqs = kernel_manager_.Setup<Kernel>(instance_idx, ctx, tvin, brightness_[i],
                                                      contrast_[i]);
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
  TensorListShape<> CallSetup(const TensorList<GPUBackend> &tl, int instance_idx) {
    kernels::KernelContext ctx;
    const auto tvin = view<const InputType, 3>(tl);
    const auto reqs = kernel_manager_.Setup<Kernel>(instance_idx, ctx, tvin, brightness_,
                                                    contrast_);
    return reqs.output_shapes[0];
  }
};

}  // namespace dali

#endif  // DALI_OPERATORS_COLOR_BRIGHTNESS_CONTRAST_H_
