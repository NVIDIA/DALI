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

#ifndef DUPAJASIU
#define DUPAJASIU

#include <vector>
#include <memory>
#include <string>
#include <dali/pipeline/operators/common.h>
#include "dali/core/geom/mat.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"


namespace dali {
namespace hsv {


const std::string kHue = "hue_delta";                // NOLINT
const std::string kSaturation = "saturation_delta";  // NOLINT
const std::string kValue = "value_delta";            // NOLINT
const std::string kOutputType = "output_type";       // NOLINT

const mat3 Rgb2Yiq = {{
                              {.299f, .587f, .114f},
                              {.596f, -.274f, -.321f},
                              {.211f, -.523f, .311f}
                      }};

// Inversion of Rgb2Yiq, but pre-calculated, cause mat<> doesn't do inversion.
const mat3 Yiq2Rgb = {{
                              {1, .956f, .621f},
                              {1, -.272f, -.647f},
                              {1, -1.107f, 1.705f}
                      }};


mat3 compose_hue(float hue) {
    const float pi = 3.14159265359f;
    const auto h_rad = hue * pi / 180;
    mat3 ret = eye<3, 3>();
    ret(1, 1) = cos(h_rad);
    ret(2, 2) = cos(h_rad);
    ret(1, 2) = -sin(h_rad);
    ret(2, 1) = sin(h_rad);
    return ret;
}


mat3 compose_saturation(float saturation) {
    auto ret = eye<3, 3>();
    ret(1, 1) = saturation;
    ret(2, 2) = saturation;
    return ret;
}


mat3 compose_value(float value) {
    return eye<3, 3>() * value;
}


mat3 transformation_matrix(float hue, float saturation, float value) {
    return Rgb2Yiq * compose_hue(hue) * compose_saturation(saturation) * compose_value(value) *
           Yiq2Rgb;
}


template<typename Backend, typename Out, typename In, size_t ndims>
struct Kernel {
    using type = ;
//  using type = kernels::BrightnessContrastCpu<Out, In, ndims>;
};


template<typename Out, typename In, size_t ndims>
struct Kernel<GPUBackend, Out, In, ndims> {
    using type = int;
};


template<typename Backend>
struct ArgumentType {
    static_assert(
            std::is_same<Backend, CPUBackend>::value || std::is_same<Backend, GPUBackend>::value,
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
template<typename Backend>
using argument_t = typename ArgumentType<Backend>::type;


/**
 * Chooses proper kernel (CPU or GPU) for given template parameters
 */
template<typename Backend, typename OutputType, typename InputType, int ndims>
using KernelSelector = typename hsv::Kernel<Backend, OutputType, InputType, ndims>::type;


/**
 * Assign argument, whether it is a single value (for sample-wise processing)
 * or vector of values (for batch processing, where every argument is defined per-sample)
 */
template<typename Backend>
void assign_argument_value(const OpSpec &, const std::string &, argument_t<Backend> &) {
    DALI_FAIL("Unsupported Backend. You may want to write your own specialization.");
}


template<>
void assign_argument_value<CPUBackend>(const OpSpec &spec, const std::string &arg_name,
                                       argument_t<CPUBackend> &arg) {
    std::vector<float> tmp;
    GetSingleOrRepeatedArg(spec, tmp, arg_name);
    arg = tmp[0];
}


template<>
void assign_argument_value<GPUBackend>(const OpSpec &spec, const std::string &arg_name,
                                       argument_t<GPUBackend> &arg) {
    GetSingleOrRepeatedArg(spec, arg, arg_name);
}

}  // namespace hsv


template<typename Backend>
class Hsv : public Operator<Backend> {
public:
    explicit Hsv(const OpSpec &spec) :
            Operator<Backend>(spec),
            output_type_(spec.GetArgument<DALIDataType>(hsv::kOutputType)) {
        hsv::assign_argument_value<Backend>(spec, hsv::kHue, hue_);
        hsv::assign_argument_value<Backend>(spec, hsv::kSaturation, saturation_);
        hsv::assign_argument_value<Backend>(spec, hsv::kValue, value_);
        if (std::is_same<Backend, GPUBackend>::value) {
            kernel_manager_.Resize(1, 1);
        } else {
            kernel_manager_.Resize(num_threads_, batch_size_);
        }
    }


    ~Hsv() override = default;


    DISABLE_COPY_MOVE_ASSIGN(Hsv);


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
        TYPE_SWITCH(input.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float), (
                TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
                        {
                            using TheKernel = hsv::KernelSelector<Backend, OutputType, InputType, 3>;
                            auto tvin = view<const InputType, 3>(input);
                            kernel_manager_.Initialize<TheKernel>();
                            kernels::KernelContext ctx;
                            const auto reqs = kernel_manager_.Setup<TheKernel>(ws.data_idx(), ctx, tvin, hue_, saturation_, value_);
                            TypeInfo type;
                            type.SetType<OutputType>(output_type_);
                            output_desc[0] = {reqs.output_shapes[0], type};
                        }
                ), DALI_FAIL("Unsupported output type"))  // NOLINT
        ), DALI_FAIL("Unsupported input type"))  // NOLINT
        // @autoformat:on
    }

    /*
     * So that compiler wouldn't complain, that
     * "overloaded virtual function `dali::Operator<dali::CPUBackend>::RunImpl` is only partially
     * overridden in class `dali::brightness_contrast::BrightnessContrast<dali::CPUBackend>`"
     */
    using Operator<Backend>::RunImpl;


    void RunImpl(Workspace<Backend> &ws) {
        const auto &input = ws.template Input<Backend>(0);
        auto &output = ws.template Output<Backend>(0);

        // @autoformat:off
        TYPE_SWITCH(input.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float), (
                TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
                        {
                            using TheKernel = hsv::KernelSelector<Backend, OutputType, InputType, 3>;
                            auto tvin = view<const InputType, 3>(input);
                            auto tvout = view<OutputType, 3>(output);
                            kernels::KernelContext ctx;
                            kernel_manager_.Run<TheKernel>(ws.thread_idx(),ws.data_idx(), ctx, tvout,tvin, hue_, saturation_,value_);
                        }
                ), DALI_FAIL("Unsupported output type"))  // NOLINT
        ), DALI_FAIL("Unsupported input type"))  // NOLINT
        // @autoformat:on

    }


private:
    USE_OPERATOR_MEMBERS();
    hsv::argument_t<Backend> hue_, saturation_, value_;
    DALIDataType output_type_;
    kernels::KernelManager kernel_manager_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_COLOR_BRIGHTNESS_CONTRAST_H_
