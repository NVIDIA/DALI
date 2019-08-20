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
#include <dali/pipeline/operators/common.h>
#include "dali/pipeline/operators/operator.h"
#include "dali/kernels/imgproc/color_manipulation/brightness_contrast.h"
#include "dali/kernels/imgproc/color_manipulation/brightness_contrast_gpu.h"
#include "dali/kernels/type_tag.h"
#include "dali/core/static_switch.h"

namespace dali {
namespace brightness_contrast {


namespace detail {

const std::string kBrightness = "brightness_delta";  // NOLINT
const std::string kContrast = "contrast_delta";      // NOLINT
const std::string kOutputType = "output_type";      // NOLINT

template<class Backend>
struct Backend2Storage {
    static_assert(std::is_same<Backend, CPUBackend>::value || std::is_same<Backend, GPUBackend>::value,
                  "Unsupported Backend");
    using type = kernels::StorageCPU;
};

template<>
struct Backend2Storage<GPUBackend> {
    using type = kernels::StorageGPU;
};

template<class Backend>
using backend_to_storage_t = typename Backend2Storage<Backend>::type;


template<class Backend, class Out, class In, size_t ndims>
struct Kernel {
    using type = kernels::BrightnessContrastCpu<Out, In, ndims>;
};


template<class Out, class In, size_t ndims>
struct BrightnessContrastGpuKernelStub;


template<class Out, class In, size_t ndims>
struct Kernel<GPUBackend, Out, In, ndims> {
  using type = kernels::brightness_contrast::BrightnessContrastGpu<Out, In, ndims>;
//    using type = BrightnessContrastGpuKernelStub<Out, In, ndims>;
};

template<class Backend>
struct ArgumentType {
    static_assert(std::is_same<Backend, CPUBackend>::value || std::is_same<Backend, GPUBackend>::value,
                  "Unsupported Backend");
    using type = float;
};

template<>
struct ArgumentType<GPUBackend> {
    using type = std::vector<float>;
};

/**
 * Select proper type for argument (for either sample processing or batch processing cases)
 */
template<class Backend>
using argument_t = typename ArgumentType<Backend>::type;


/**
 * Chooses proper kernel for given template parameters
 */
template<class Backend, class OutputType, class InputType, size_t ndims>
using BrightnessContrastKernel = typename Kernel<Backend, OutputType, InputType, ndims>::type;


/**
 * Assign argument, whether it is a single value (for sample-wise processing)
 * or vector of values (for batch processing, where every argument is defined per-sample)
 */
template<class Backend>
void assign_argument_value(const OpSpec &, const std::string &, argument_t<Backend> &) {
    DALI_FAIL("Unsupported Backend. You may want to write your own specialization.");
}


template<>
void assign_argument_value<CPUBackend>(const OpSpec &spec, const std::string &arg_name, argument_t<CPUBackend> &arg) {
    std::vector<float> tmp;
    GetSingleOrRepeatedArg(spec, tmp, arg_name);
    arg = tmp[0];
}


template<>
void assign_argument_value<GPUBackend>(const OpSpec &spec, const std::string &arg_name, argument_t<GPUBackend> &arg) {
    GetSingleOrRepeatedArg(spec, arg, arg_name);
}


}  // namespace detail


template<class Backend>
class BrightnessContrast : public Operator<Backend> {

public:
    explicit BrightnessContrast(const OpSpec &spec) :
            Operator<Backend>(spec),
            output_type_(spec.GetArgument<DALIDataType>(detail::kOutputType)) {
        detail::assign_argument_value<Backend>(spec, detail::kBrightness, brightness_);
        detail::assign_argument_value<Backend>(spec, detail::kContrast, contrast_);
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

//    using OutputType=uint8_t;
        DALI_TYPE_SWITCH(output_type_, OutputType,
                         {
                             TypeInfo type;
                             type.SetType<OutputType>(output_type_);
                             output_desc[0] = {input.shape(), type};
                         }
        )
        return true;
    }


    void RunImpl(Workspace<Backend> &ws) override{
        const auto &input = ws.template Input<Backend>(0);
        auto &output = ws.template Output<Backend>(0);

// Convenient alias for DALI_TYPE_SWITCH
#define TS(...) DALI_TYPE_SWITCH (__VA_ARGS__)
        TS(input.type().id(), InputType,
           TS(output_type_, OutputType,
              {
                       auto tvin = view<const InputType, 3>(input);
                       auto tvout = view<OutputType, 3>(output);
                       detail::BrightnessContrastKernel<Backend, OutputType, InputType, 3> kernel;
                       kernels::KernelContext ctx;
                       auto reqs = kernel.Setup(ctx, tvin, brightness_, contrast_);
                       kernel.Run(ctx, tvout, tvin, brightness_, contrast_);
              }
           )
        )
#undef TS
    }


private:
    detail::argument_t<Backend> brightness_, contrast_;
    DALIDataType output_type_;

};

//template<>
//void BrightnessContrast<CPUBackend>::RunImpl(Workspace<CPUBackend>&ws) {
//    const auto &input = ws.template Input<CPUBackend>(0);
//    auto &output = ws.template Output<CPUBackend>(0);
//
//// Convenient alias for DALI_TYPE_SWITCH
//#define TS(...) DALI_TYPE_SWITCH (__VA_ARGS__)
//    TS(input.type().id(), InputType,
//       TS(output_type_, OutputType,
//          {
//                       auto tvin = view<const InputType, 3>(input);
//                      auto tvout = view<OutputType, 3>(output);
//                       detail::BrightnessContrastKernel<CPUBackend, OutputType, InputType, 3> kernel;
//                       kernels::KernelContext ctx;
//
//                       auto reqs = kernel.Setup(ctx, tvin, brightness_, contrast_);
//
//                       kernels::TensorView<kernels::StorageCPU, OutputType, 3> tvout(out.data(), out_shape.template to_static<3>());
//
//                       kernel.Run(ctx, tvout, tvin, brightness_, contrast_);
//               }
//       )
//    )
//#undef TS
//}
//
//template<>
//void BrightnessContrast<GPUBackend>::RunImpl(Workspace<GPUBackend>&ws) {
////    const auto &input = ws.template Input<Backend>(0);
////    auto &output = ws.template Output<Backend>(0);
////
////// Convenient alias for DALI_TYPE_SWITCH
////#define TS(...) DALI_TYPE_SWITCH (__VA_ARGS__)
////    TS(input.type().id(), InputType,
////       TS(output_type_, OutputType,
////          {
////                       auto tvin = view<const InputType, 3>(input);
//////                      auto tvout = view<OutputType, 3>(output);
////                       detail::BrightnessContrastKernel<Backend, OutputType, InputType, 3> kernel;
////                       kernels::KernelContext ctx;
//////                      if (ws.has_stream()) ctx.gpu.stream = ws.stream();
////
////                       auto reqs = kernel.Setup(ctx, tvin, brightness_, contrast_);
////                       auto out_shape = reqs.output_shapes[0][0];
////                       std::vector<OutputType> out;
////                       out.resize(dali::volume(out_shape));
////                       kernels::TensorView<detail::backend_to_storage_t<Backend>, OutputType, 3> tvout(out.data(),   out_shape.template to_static<3>());
////
////                       kernel.Run(ctx, tvout, tvin, brightness_, contrast_);
////               }
////       )
////    )
////#undef TS
//}


}
}
#endif
