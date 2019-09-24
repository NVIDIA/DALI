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

#ifndef DALI_PIPELINE_OPERasdfasdfBRIGHTNESS_CONTRAST_H_
#define DALI_PIPELINE_OPERasdfasdfBRIGHTNESS_CONTRAST_H_

#include <vector>
#include <memory>
#include <string>
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/kernels/kernel_manager.h"

namespace dali {
namespace hsv {


const std::string kHue = "hue_delta";                // NOLINT
const std::string kSaturation = "saturation_delta";  // NOLINT
const std::string kValue = "value_delta";            // NOLINT
const std::string kOutputType = "output_type";       // NOLINT


//template <typename Backend, typename Out, typename In, size_t ndims>
//struct Kernel {
//  using type = kernels::BrightnessContrastCpu<Out, In, ndims>;
//};


//template <typename Out, typename In, size_t ndims>
//struct Kernel<GPUBackend, Out, In, ndims> {
//  using type = kernels::brightness_contrast::BrightnessContrastGpu<Out, In, ndims>;
//};


//template <typename Backend>
//struct ArgumentType {
//  static_assert(
//          std::is_same<Backend, CPUBackend>::value || std::is_same<Backend, GPUBackend>::value,
//          "Unsupported Backend");
//  using type = float;
//};


//template <>
//struct ArgumentType<GPUBackend> {
//  using type = std::vector<float>;
//};


///**
// * Select proper type for argument (for either sample processing or batch processing cases)
// */
//template <typename Backend>
//using argument_t = typename ArgumentType<Backend>::type;


///**
// * Chooses proper kernel (CPU or GPU) for given template parameters
// */
//template <typename Backend, typename OutputType, typename InputType, int ndims>
//using BrightnessContrastKernel = typename Kernel<Backend, OutputType, InputType, ndims>::type;


///**
// * Assign argument, whether it is a single value (for sample-wise processing)
// * or vector of values (for batch processing, where every argument is defined per-sample)
// */
//template <typename Backend>
//void assign_argument_value(const OpSpec &, const std::string &, argument_t<Backend> &) {
//  DALI_FAIL("Unsupported Backend. You may want to write your own specialization.");
//}


//template <>
//void assign_argument_value<CPUBackend>(const OpSpec &spec, const std::string &arg_name,
//                                       argument_t<CPUBackend> &arg) {
//  std::vector<float> tmp;
//  GetSingleOrRepeatedArg(spec, tmp, arg_name);
//  arg = tmp[0];
//}


//template <>
//void assign_argument_value<GPUBackend>(const OpSpec &spec, const std::string &arg_name,
//                                       argument_t<GPUBackend> &arg) {
//  GetSingleOrRepeatedArg(spec, arg, arg_name);
//}

}  // namespace hsv


template <typename Backend>
class Hsv : public Operator<Backend> {
 public:
  explicit Hsv(const OpSpec &spec) :
          Operator<Backend>(spec),
          output_type_(spec.GetArgument<DALIDataType>(detail::kOutputType)) {
//    detail::assign_argument_value<Backend>(spec, detail::kBrightness, brightness_);
//    detail::assign_argument_value<Backend>(spec, detail::kContrast, contrast_);
//    if (std::is_same<Backend, GPUBackend>::value) {
//      kernel_manager_.Resize(1, 1);
//    } else {
//      kernel_manager_.Resize(num_threads_, batch_size_);
//    }
  }


  ~Hsv() = default;


  DISABLE_COPY_MOVE_ASSIGN(Hsv);


 protected:
  bool CanInferOutputs() const override {
    return true;
  }


  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                 const ::dali::workspace_t<Backend> &ws) override {
//    const auto &input = ws.template InputRef<Backend>(0);
//    const auto &output = ws.template OutputRef<Backend>(0);
//    output_desc.resize(1);
//
//    TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
//            {
//              TypeInfo type;
//              type.SetType<OutputType>(output_type_);
//              output_desc[0] = {input.shape(), type};
//            }
//    ), DALI_FAIL("Unsupported image type"))  // NOLINT
//    return true;
  }

  /**
   * So that compiler wouldn't complain, that
   * "overloaded virtual function `dali::Operator<dali::CPUBackend>::RunImpl` is only partially
   * overridden in class `dali::brightness_contrast::BrightnessContrast<dali::CPUBackend>`"
   */
  using Operator<Backend>::RunImpl;


  void RunImpl(Workspace<Backend> &ws) {
//    const auto &input = ws.template Input<Backend>(0);
//    auto &output = ws.template Output<Backend>(0);
//
//    TYPE_SWITCH(input.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float), (
//            TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
//                    {
//                      using BrightnessContrastKernel =
//                      detail::BrightnessContrastKernel<Backend, OutputType, InputType, 3>;
//                      auto tvin = view<const InputType, 3>(input);
//                      auto tvout = view<OutputType, 3>(output);
//                      kernel_manager_.Initialize<BrightnessContrastKernel>();
//                      kernels::KernelContext ctx;
//                      kernel_manager_.Setup<BrightnessContrastKernel>(ws.data_idx(),
//                                                                      ctx, tvin, brightness_, contrast_);
//                      kernel_manager_.Run<BrightnessContrastKernel>(ws.thread_idx(), ws.data_idx(),
//                                                                    ctx, tvout, tvin, brightness_, contrast_);
//                    }
//            ), DALI_FAIL("Unsupported output type"))  // NOLINT
//    ), DALI_FAIL("Unsupported input type"))  // NOLINT
  }


 private:
  USE_OPERATOR_MEMBERS();
//  detail::argument_t<Backend> brightness_, contrast_;
  DALIDataType output_type_;
  kernels::KernelManager kernel_manager_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_COLOR_BRIGHTNESS_CONTRAST_H_
