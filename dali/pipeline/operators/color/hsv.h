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

/**
 * Names of arguments
 */
extern const std::string kHue;
extern const std::string kSaturation;
extern const std::string kValue;
extern const std::string kOutputType;

/**
 * Color space conversion
 */
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


inline mat3 compose_hue(float hue /* hue hue hue */ ) {
  const auto h_rad = hue * M_PI / 180;
  mat3 ret = mat3::eye();  // rotation matrix
  // Hue change in YIQ color space is a rotation along the Y axis
  ret(1, 1) = cos(h_rad);
  ret(2, 2) = cos(h_rad);
  ret(1, 2) = -sin(h_rad);
  ret(2, 1) = sin(h_rad);
  return ret;
}

inline mat3 compose_saturation(float saturation) {
  mat3 ret = mat3::eye();
  // In the YIQ color space, saturation change is a
  // homothetic transformation in IQ dimensions
  ret(1, 1) = saturation;
  ret(2, 2) = saturation;
  return ret;
}


inline mat3 compose_value(float value) {
  // In the YIQ color space, value change is a
  // homothetic transformation across all dimensions
  return mat3::diag(value);
}

}  // namespace hsv



template <typename Backend>
class Hsv : public Operator<Backend> {
 public:
  ~Hsv() override = default;

  DISABLE_COPY_MOVE_ASSIGN(Hsv);

 protected:
  explicit Hsv(const OpSpec &spec) :
          Operator<Backend>(spec),
          output_type_(spec.GetArgument<DALIDataType>(hsv::kOutputType)) {
    GetSingleOrRepeatedArg(spec, hue_, hsv::kHue, batch_size_);
    GetSingleOrRepeatedArg(spec, saturation_, hsv::kSaturation, batch_size_);
    GetSingleOrRepeatedArg(spec, value_, hsv::kValue, batch_size_);
    tmatrices_ = determine_transformation(hue_, saturation_, value_);
    if (std::is_same<Backend, GPUBackend>::value) {
      kernel_manager_.Resize(1, 1);
    } else {
      kernel_manager_.Resize(num_threads_, batch_size_);
    }
    assert(hue_.size() == static_cast<size_t>(batch_size_));
    assert(saturation_.size() == static_cast<size_t>(batch_size_));
    assert(value_.size() == static_cast<size_t>(batch_size_));
  }


  bool CanInferOutputs() const override {
    return true;
  }


  /**
   * @brief Creates transformation matrices based on given args
   */
  std::vector<mat3>
  determine_transformation(const std::vector<float> &hue, const std::vector<float> &saturation,
                           const std::vector<float> &value) const {
    using namespace hsv;  // NOLINT
    std::vector<mat3> ret;

    auto size = hue.size();
    ret.resize(size);
    for (size_t i = 0; i < size; i++) {
      ret[i] = Yiq2Rgb * compose_hue(hue[i]) * compose_saturation(saturation[i]) *
               compose_value(value[i]) * Rgb2Yiq;
    }
    return ret;
  }


  USE_OPERATOR_MEMBERS();
  std::vector<float> hue_, saturation_, value_;
  std::vector<mat3> tmatrices_;
  DALIDataType output_type_;
  kernels::KernelManager kernel_manager_;
};



class HsvCpu : public Hsv<CPUBackend> {
 public:
  explicit HsvCpu(const OpSpec &spec) : Hsv(spec) {}

  /*
   * So that compiler wouldn't complain, that
   * "overloaded virtual function `dali::Operator<dali::CPUBackend>::RunImpl`
   * is only partially overridden in class `dali::HsvCpu`"
   */
  using Operator<CPUBackend>::RunImpl;

  ~HsvCpu() override = default;

  DISABLE_COPY_MOVE_ASSIGN(HsvCpu);

 protected:
  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                 const workspace_t<CPUBackend> &ws) override;

  void RunImpl(workspace_t<CPUBackend> &ws) override;

 private:
  template <typename Kernel, typename InputType>
  kernels::TensorListShape<> CallSetup(const TensorVector<CPUBackend> &input, int instance_idx) {
    kernels::KernelContext ctx;
    kernels::TensorListShape<> sh = input.shape();
    kernels::TensorListShape<> ret(sh.num_samples(), 3);
    assert(static_cast<size_t>(sh.num_samples()) == tmatrices_.size());
    for (int i = 0; i < sh.num_samples(); i++) {
      const auto tvin = view<const InputType, 3>(input[i]);
      const auto reqs = kernel_manager_.Setup<Kernel>(instance_idx, ctx, tvin, tmatrices_[i]);
      const kernels::TensorListShape<> &out_sh = reqs.output_shapes[0];
      ret.set_tensor_shape(i, out_sh.tensor_shape(0));
    }
    return ret;
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

  void RunImpl(workspace_t<GPUBackend> &ws) override;

 private:
  template <typename Kernel, typename InputType>
  kernels::TensorListShape<> CallSetup(const TensorList<GPUBackend> &tl, int instance_idx) {
    kernels::KernelContext ctx;
    const auto tvin = view<const InputType, 3>(tl);
    const auto reqs = kernel_manager_.Setup<Kernel>(instance_idx, ctx, tvin,
                                                    make_cspan(tmatrices_));
    return reqs.output_shapes[0];
  }
};


}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_COLOR_HSV_H_
