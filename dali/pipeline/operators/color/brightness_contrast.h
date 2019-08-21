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

#ifndef DALI_PIPELINE_OPERATORS_COLOR_BRIGHTNESS_CONTRAST_H_
#define DALI_PIPELINE_OPERATORS_COLOR_BRIGHTNESS_CONTRAST_H_

#include <vector>
#include <memory>
#include <string>
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/kernels/imgproc/color_manipulation/brightness_contrast.h"
#include "dali/kernels/imgproc/color_manipulation/brightness_contrast_gpu.h"
#include "dali/kernels/kernel_manager.h"

namespace dali {
namespace brightness_contrast {

namespace detail {

const std::string kBrightness = "brightness_delta";  // NOLINT
const std::string kContrast = "contrast_delta";      // NOLINT
const std::string kOutputType = "output_type";      // NOLINT


template <class Backend, class Out, class In, size_t ndims>
struct Kernel {
  using type = kernels::BrightnessContrastCpu<Out, In, ndims>;
};


template <class Out, class In, size_t ndims>
struct Kernel<GPUBackend, Out, In, ndims> {
  using type = kernels::brightness_contrast::BrightnessContrastGpu<Out, In, ndims>;
};


template <class Backend>
struct ArgumentType {
  static_assert(
          std::is_same<Backend, CPUBackend>::value || std::is_same<Backend, GPUBackend>::value,
          "Unsupported Backend");
  using type = float;
};


template <>
struct ArgumentType<GPUBackend> {
  using type = std::vector<float>;
};


/**
 * Select proper type for argument (for either sample processing or batch processing cases)
 */
template <class Backend>
using argument_t = typename ArgumentType<Backend>::type;


/**
 * Chooses proper kernel (CPU or GPU) for given template parameters
 */
template <class Backend, class OutputType, class InputType, size_t ndims>
using BrightnessContrastKernel = typename Kernel<Backend, OutputType, InputType, ndims>::type;


/**
 * Assign argument, whether it is a single value (for sample-wise processing)
 * or vector of values (for batch processing, where every argument is defined per-sample)
 */
template <class Backend>
void assign_argument_value(const OpSpec &, const std::string &, argument_t<Backend> &) {
  DALI_FAIL("Unsupported Backend. You may want to write your own specialization.");
}


template <>
void assign_argument_value<CPUBackend>(const OpSpec &spec, const std::string &arg_name,
                                       argument_t<CPUBackend> &arg) {
  std::vector<float> tmp;
  GetSingleOrRepeatedArg(spec, tmp, arg_name);
  arg = tmp[0];
}


template <>
void assign_argument_value<GPUBackend>(const OpSpec &spec, const std::string &arg_name,
                                       argument_t<GPUBackend> &arg) {
  GetSingleOrRepeatedArg(spec, arg, arg_name);
}

}  // namespace detail


template <class Backend>
class BrightnessContrast : public Operator<Backend> {
 public:
  explicit BrightnessContrast(const OpSpec &spec) :
          Operator<Backend>(spec),
          output_type_(spec.GetArgument<DALIDataType>(detail::kOutputType)) {
    detail::assign_argument_value<Backend>(spec, detail::kBrightness, brightness_);
    detail::assign_argument_value<Backend>(spec, detail::kContrast, contrast_);
    if (std::is_same<Backend, GPUBackend>::value) {
      kernel_manager_.Resize(1, 1);
    } else {
      kernel_manager_.Resize(num_threads_, batch_size_);
    }
  }


  ~BrightnessContrast() = default;


  DISABLE_COPY_MOVE_ASSIGN(BrightnessContrast);


 protected:
  bool CanInferOutputs() const override {
    return true;
  }


  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                 const ::dali::workspace_t<Backend> &ws) override {
    const auto &input = ws.template InputRef<Backend>(0);
    const auto &output = ws.template OutputRef<Backend>(0);
    output_desc.resize(1);

    // @autoformat:off
    DALI_TYPE_SWITCH(output_type_, OutputType,
        {
          TypeInfo type;
          type.SetType<OutputType>(output_type_);
          output_desc[0] = {input.shape(), type};
        }
    )
    // @autoformat:on
    return true;
  }


  void RunImpl(Workspace<Backend> &ws) override {
    const auto &input = ws.template Input<Backend>(0);
    auto &output = ws.template Output<Backend>(0);

    // @autoformat:off
      DALI_TYPE_SWITCH(input.type().id(), InputType,
          DALI_TYPE_SWITCH(output_type_, OutputType,
              {
                  using TheKernel =
                          detail::BrightnessContrastKernel<Backend, OutputType, InputType, 3>;
                  auto tvin = view<const InputType, 3>(input);
                  auto tvout = view<OutputType, 3>(output);
                  kernel_manager_.Initialize<TheKernel>();
                  kernels::KernelContext ctx;
                  kernel_manager_.Setup<TheKernel>(ws.data_idx(),
                          ctx, tvin, brightness_, contrast_);
                  kernel_manager_.Run<TheKernel>(ws.thread_idx(), ws.data_idx(),
                          ctx, tvout, tvin, brightness_, contrast_);
              }
           )
        )
    // @autoformat:on
  }


 private:
  USE_OPERATOR_MEMBERS();
  detail::argument_t<Backend> brightness_, contrast_;
  DALIDataType output_type_;
  kernels::KernelManager kernel_manager_;
};

}  // namespace brightness_contrast
}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_COLOR_BRIGHTNESS_CONTRAST_H_
