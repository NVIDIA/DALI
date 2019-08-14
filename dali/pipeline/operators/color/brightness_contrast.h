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

#ifndef DALI_PIPESADFASDFASDFFADFF_
#define DALI_PIPESADFASDFASDFFADFF_

#include <dali/pipeline/data/views.h>
#include "dali/pipeline/operators/operator.h"
#include "dali/kernels/imgproc/color_manipulation/brightness_contrast.h"

namespace dali {
namespace brightness_contrast {


namespace detail {

const std::string kBrightness = "brightness_delta";  // NOLINT
const std::string kContrast = "contrast_delta";      // NOLINT
const std::string kOutputType = "output_type";      // NOLINT


template <class Backend, class Out, class In, size_t ndims>
struct Kernel {
  using type = kernels::BrightnessContrastCPU<Out, In, ndims>;
};


template <class Out, class In, size_t ndims>
struct BrightnessContrastGpuKernelStub;


template <class Out, class In, size_t ndims>
struct Kernel<GPUBackend, Out, In, ndims> {
  using type = BrightnessContrastGpuKernelStub<Out, In, ndims>;
};


/**
 * Chooses proper kernel for given template parameters
 */
template <class Backend, class OutputType, class InputType, size_t ndims>
using BrightnessContrastKernel = typename Kernel<Backend, OutputType, InputType, ndims>::type;

}  // namespace detail


template <class Backend>
class BrightnessContrast : public Operator<Backend> {

 public:
  explicit BrightnessContrast(const OpSpec &spec);

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
    TypeInfo t;
    t.SetType<float>(output_type_);
    output_desc[0] = {input.shape(), t};
//    output_desc[0] = {input.shape(), output.type()};
    return true;
  }


  void RunImpl(Workspace<Backend> *ws) {
    const auto &input = ws->template Input<Backend>(0);
    auto &output = ws->template Output<Backend>(0);

    auto tvin = view<const float, 3>(input);
    auto tvout = view<float, 3>(output);

    detail::BrightnessContrastKernel<Backend, float, float, 3> kernel;
    kernels::KernelContext ctx;
    auto reqs = kernel.Setup(ctx, tvin, brightness_, contrast_);
    kernel.Run(ctx, tvout, tvin, brightness_, contrast_);


//    auto out = output.template mutable_data<float>();
//    for (int i=0;i<volume(output.shape());i++){
//      out[i]=1.f;
//    }

    cout << "ASDASDASDASDASDASDASDASDASDASDASDASD\n";
  }


 private:
  //TODO const
  float brightness_, contrast_;
  DALIDataType output_type_;

};


}
}

#endif
