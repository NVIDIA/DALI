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

#ifndef DALI_PIPELINE_OPERATORS_COLOR_HSV_H_
#define DALI_PIPELINE_OPERATORS_COLOR_HSV_H_

#include <vector>
#include <memory>
#include <string>
#include "dali/core/static_switch.h"
#include "dali/core/geom/mat.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/tensor_shape_print.h"
#include "dali/kernels/imgproc/pointwise/linear_transformation_cpu.h"


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
  mat3 ret = mat3::eye();
  ret(1, 1) = cos(h_rad);
  ret(2, 2) = cos(h_rad);
  ret(1, 2) = -sin(h_rad);
  ret(2, 1) = sin(h_rad);
  return ret;
}


inline mat3 compose_saturation(float saturation) {
  mat3 ret = mat3::eye();
  ret(1, 1) = saturation;
  ret(2, 2) = saturation;
  return ret;
}


inline mat3 compose_value(float value) {
  return mat3::eye() * value;
}


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
template <typename Backend, typename T = float>
using argument_t = typename ArgumentType<Backend, T>::type;


template <typename Backend>
void
assign_argument_value(const OpSpec &spec, const std::string &arg_name, argument_t<Backend> &arg) {
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
struct WorkspaceInputType {
  using type = TensorVector<CPUBackend>;
};

template <>
struct WorkspaceInputType<GPUBackend> {
  using type = TensorList<GPUBackend>;
};

template <typename Backend>
using workspace_input_t = typename WorkspaceInputType<Backend>::type;


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

  USE_OPERATOR_MEMBERS();
  hsv::argument_t<Backend> hue_, saturation_, value_;
  DALIDataType output_type_;
  kernels::KernelManager kernel_manager_;
};

class HsvCpu : public Hsv<CPUBackend> {
 public:
  explicit HsvCpu(const OpSpec &spec) : Hsv(spec) {}


  ~HsvCpu() override = default;


  DISABLE_COPY_MOVE_ASSIGN(HsvCpu);


 protected:
  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                 const workspace_t<CPUBackend> &ws) override;


  void RunImpl(Workspace<CPUBackend> &ws) override;


 private:
  static mat3 transformation_matrix(float hue, float saturation, float value) {
    using namespace hsv;  // NOLINT
    return Rgb2Yiq * compose_hue(hue) * compose_saturation(saturation) * compose_value(value) *
           Yiq2Rgb;
  }


  template <typename Kernel, typename InputType>
  kernels::TensorListShape<> CallSetup(const TensorVector<CPUBackend> &input, int instance_idx) {
    kernels::KernelContext ctx;
    kernels::TensorListShape<> sh = input.shape();
    std::vector<kernels::TensorShape<>> shapes;
    shapes.resize(sh.num_samples());
    for (size_t i = 0; i < shapes.size(); i++) {
      const auto tvin = view<const InputType, 3>(input[i]);
      const auto reqs = kernel_manager_.Setup<Kernel>(instance_idx, ctx, tvin,
                                                      transformation_matrix(hue_, saturation_,
                                                                            value_));
      kernels::TensorListShape<> out_sh = reqs.output_shapes[0];
      shapes[i] = out_sh.tensor_shape(0);
    }
    return {shapes};
  }
};


class HsvGpu : public Hsv<GPUBackend> {
 public:
  explicit HsvGpu(const OpSpec &spec) : Hsv(spec) {}


  ~HsvGpu() override = default;


  DISABLE_COPY_MOVE_ASSIGN(HsvGpu);


 protected:
  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                 const workspace_t<GPUBackend> &ws) override;


  void RunImpl(Workspace<GPUBackend> &ws) override;


 private:
  std::vector<mat3>
  transformation_matrix(const std::vector<float> &hue, const std::vector<float> &saturation,
                        const std::vector<float> &value) {
    using namespace hsv;  // NOLINT
    std::vector<mat3> ret;
    DALI_ENFORCE(hue.size() == saturation.size());
    DALI_ENFORCE(saturation.size() == value.size());
    for (size_t i = 0; i < hue.size(); i++) {
      ret.emplace_back(Rgb2Yiq * compose_hue(hue[i]) * compose_saturation(saturation[i]) *
                       compose_value(value[i]) * Yiq2Rgb);
    }
    return ret;
  }


  template <typename Kernel, typename InputType>
  kernels::TensorListShape<> CallSetup(const TensorList<GPUBackend> &tl, int instance_idx) {
    kernels::KernelContext ctx;
    auto tvin = view<const InputType, 3>(tl);
    auto tmat = transformation_matrix(hue_, saturation_, value_);
    const auto reqs = kernel_manager_.Setup<Kernel>(instance_idx, ctx, tvin, make_cspan(tmat));
    return reqs.output_shapes[0];
  }
};


}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_COLOR_HSV_H_
