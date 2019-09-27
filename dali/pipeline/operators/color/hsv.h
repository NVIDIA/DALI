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
//#include  "dali/kernels/imgproc/pointwise/linear_transformation_cpu.h"
#include <dali/kernels/tensor_shape_print.h>


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


inline mat3 compose_hue(float hue) {
  const float pi = 3.14159265359f;
  const auto h_rad = hue * pi / 180;
  mat3 ret = eye<3, 3>();
  ret(1, 1) = cos(h_rad);
  ret(2, 2) = cos(h_rad);
  ret(1, 2) = -sin(h_rad);
  ret(2, 1) = sin(h_rad);
  return ret;
}


inline mat3 compose_saturation(float saturation) {
  auto ret = eye<3, 3>();
  ret(1, 1) = saturation;
  ret(2, 2) = saturation;
  return ret;
}


inline mat3 compose_value(float value) {
  return eye<3, 3>() * value;
}




//template <typename Backend, typename Out, typename In, int channels_out, int channels_in, int ndims>
//struct Kernel {
//  using type= kernels::LinearTransformationCpu<Out, In, channels_out, channels_in, ndims>;
//};

template <typename Backend, typename Out, typename In, int channels_out, int channels_in, int ndims>
struct Kernel;

//template<typename Backend, typename Out, typename In, int channels_out, int channels_in, int ndims>
//struct Kernel<GPUBackend, Out, In, channels_out, channels_in, ndims> {
//    using type = int;
//};


template <typename Backend, typename T>
struct ArgumentType {
  static_assert(
          std::is_same<Backend, CPUBackend>::value || std::is_same<Backend, GPUBackend>::value,
          "Unsupported Backend");
  using type = T;
};


template <typename T>
struct ArgumentType<GPUBackend, T> {
  using type = std::vector<T>;
};


/**
 * Select proper type for argument (for either sample processing or batch processing cases)
 */
template <typename Backend, typename T=float>
using argument_t = typename ArgumentType<Backend, T>::type;


/**
 * Chooses proper kernel (CPU or GPU) for given template parameters
 */
template <typename Backend, typename Out, typename In, int channels_out, int channels_in, int ndims>
using KernelSelector = typename hsv::Kernel<Backend, Out, In, channels_out, channels_in, ndims>::type;


/**
 * Assign argument, whether it is a single value (for sample-wise processing)
 * or vector of values (for batch processing, where every argument is defined per-sample)
 */
template <typename Backend>
void assign_argument_value(const OpSpec &, const std::string &, argument_t<Backend> &) {
  DALI_FAIL("Unsupported Backend. You may want to write your own specialization.");
}


template <>
inline void assign_argument_value<CPUBackend>(const OpSpec &spec, const std::string &arg_name,
                                              argument_t<CPUBackend> &arg) {
  std::vector<float> tmp;
  GetSingleOrRepeatedArg(spec, tmp, arg_name);
  arg = tmp[0];
}


template <>
inline void assign_argument_value<GPUBackend>(const OpSpec &spec, const std::string &arg_name,
                                              argument_t<GPUBackend> &arg) {
  GetSingleOrRepeatedArg(spec, arg, arg_name);
}


template <typename Backend>
inline argument_t<Backend, mat3>
transformation_matrix(const argument_t<Backend> &hue, const argument_t<Backend> &saturation,
                      const argument_t<Backend> &value) {
  DALI_FAIL("Unsupported backend.")
}


template <>
inline argument_t<CPUBackend, mat3>
transformation_matrix<CPUBackend>(const argument_t<CPUBackend> &hue, const argument_t<CPUBackend> &saturation,
                                  const argument_t<CPUBackend> &value) {
  return Rgb2Yiq * compose_hue(hue) * compose_saturation(saturation) * compose_value(value) *
         Yiq2Rgb;
}


template <>
inline argument_t<GPUBackend, mat3>
transformation_matrix<GPUBackend>(const argument_t<GPUBackend> &hue, const argument_t<GPUBackend> &saturation,
                                  const argument_t<GPUBackend> &value) {
  std::vector<mat3> ret;
  DALI_ENFORCE(hue.size() == saturation.size());
  DALI_ENFORCE(saturation.size() == value.size());
  for (size_t i = 0; i < hue.size(); i++) {
    ret.emplace_back(Rgb2Yiq * compose_hue(hue[i]) * compose_saturation(saturation[i]) *
                     compose_value(value[i]) * Yiq2Rgb);
  }
  return ret;
}

}  // namespace hsv


template <typename Backend>
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
                 const workspace_t<Backend> &ws) override {
    const auto &input = ws.template InputRef<Backend>(0);
    const auto &output = ws.template OutputRef<Backend>(0);
    output_desc.resize(1);
    // @autoformat:off
    TYPE_SWITCH(input.type().id(), type2id, InputType, (uint8_t/*, int16_t, int32_t, float, float16*/), (
        TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t/*, int16_t, int32_t, float, float16*/), (
            {
                using TheKernel = hsv::KernelSelector<Backend, OutputType, InputType, 3, 3, 2>;
                kernel_manager_.Initialize<TheKernel>();
                auto shapes = CallSetup<TheKernel, InputType>(input, ws.data_idx());
                TypeInfo type;
                type.SetType<OutputType>(output_type_);
                output_desc[0] = {shapes, type};
            }
        ), DALI_FAIL("Unsupported output type"))  // NOLINT
    ), DALI_FAIL("Unsupported input type"))  // NOLINT
    // @autoformat:on
    return true;
  }

  /*
   * So that compiler wouldn't complain, that
   * "overloaded virtual function `dali::Operator<dali::CPUBackend>::RunImpl` is only partially
   * overridden in class `dali::brightness_contrast::BrightnessContrast<dali::CPUBackend>`"
   */
  using Operator<Backend>::RunImpl;


  void RunImpl(Workspace<Backend> &ws) override {
    const auto &input = ws.template Input<Backend>(0);
    auto &output = ws.template Output<Backend>(0);
    // @autoformat:off
    TYPE_SWITCH(input.type().id(), type2id, InputType, (uint8_t/*, int16_t, int32_t, float, float16*/), (
        TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t/*, int16_t, int32_t, float, float16*/), (
            {
                using TheKernel = hsv::KernelSelector<Backend, OutputType, InputType, 3, 3, 2>;
                kernels::KernelContext ctx;
                auto sh = input.shape();
                auto tvin = view<const InputType, 3>(input);
                auto tvout = view<OutputType, 3>(output);
                kernel_manager_.Run<TheKernel>(ws.thread_idx(),ws.data_idx(), ctx, tvout, tvin,
                        hsv::transformation_matrix(hue_, saturation_, value_));
//                                kernel_manager_.Run<TheKernel>(ws.thread_idx(),ws.data_idx(), ctx, tvout, tvin);
            }
        ), DALI_FAIL("Unsupported output type"))  // NOLINT
    ), DALI_FAIL("Unsupported input type"))  // NOLINT
    // @autoformat:on

  }


 private:
  template <typename Kernel, typename InputType>
  kernels::TensorListShape<> CallSetup(const TensorVector<Backend> &input, int instance_idx) {
    kernels::KernelContext ctx;
    kernels::TensorListShape<> sh = input.shape();
    std::vector<kernels::TensorShape<>> shapes;
    shapes.resize(sh.num_samples());
    for (size_t i = 0; i < shapes.size(); i++) {
      const auto tvin = view<const InputType, 3>(input[i]);
//      const auto reqs = kernel_manager_.Setup<Kernel>(instance_idx, ctx, tvin, hsv::transformation_matrix(hue_, saturation_, value_));
      const auto reqs = kernel_manager_.Setup<Kernel>(instance_idx, ctx, tvin);
      kernels::TensorListShape<> out_sh = reqs.output_shapes[0];
      shapes[i] = out_sh.tensor_shape(0);
    }
    return {shapes};
  }


  template <typename Kernel, typename InputType>
  kernels::TensorListShape<> CallSetup(const TensorList<Backend> &tl, int instance_idx) {
    kernels::KernelContext ctx;
    kernels::TensorListView<kernels::StorageGPU, const InputType, 3> tvin = view<const InputType, 3>(tl);
    const auto reqs = kernel_manager_.Setup<Kernel>(instance_idx, ctx, tvin, make_cspan(hsv::transformation_matrix(hue_, saturation_,value_)));
//    const auto reqs = kernel_manager_.Setup<Kernel>(instance_idx, ctx, tvin);
    return reqs.output_shapes[0];
  }


  USE_OPERATOR_MEMBERS();
  hsv::argument_t<Backend> hue_, saturation_, value_;
  hsv::argument_t<Backend> zero_vec_;
  DALIDataType output_type_;
  kernels::KernelManager kernel_manager_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_COLOR_BRIGHTNESS_CONTRAST_H_
