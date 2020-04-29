// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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


#ifndef DALI_OPERATORS_IMAGE_COLOR_COLOR_TWIST_H_
#define DALI_OPERATORS_IMAGE_COLOR_COLOR_TWIST_H_

#include <vector>
#include <memory>
#include <cmath>
#include "dali/pipeline/operator/operator.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/core/geom/mat.h"

namespace dali {

class ColorAugment {
 public:
  static const int nDim = 4;
  using mat_t = mat<nDim, nDim, float>;

  virtual void operator() (mat_t &m) {
    m = matrix_ * m;
  }

  virtual void Prepare(Index i, const OpSpec &spec, const ArgumentWorkspace * ws) = 0;

  virtual ~ColorAugment() = default;

 protected:
  mat_t matrix_ = mat_t::eye();
};

class Brightness : public ColorAugment {
 public:
  void Prepare(Index i, const OpSpec &spec, const ArgumentWorkspace * ws) override {
    auto brightness = spec.GetArgument<float>("brightness", ws, i);
    matrix_ = mat_t(brightness);
    matrix_(nDim-1, nDim-1) = 1.;
  }
};

class Contrast : public ColorAugment {
 public:
  void Prepare(Index i, const OpSpec &spec, const ArgumentWorkspace * ws) override {
    auto contrast = spec.GetArgument<float>("contrast", ws, i);
    matrix_ = mat_t(contrast);
    matrix_(nDim-1, nDim-1) = 1.;
    for (int r = 0; r < nDim-1; ++r) matrix_(r, nDim-1) = (1 - contrast) * 128.f;
  }
};

class Hue : public ColorAugment {
 public:
  void Prepare(Index i, const OpSpec &spec, const ArgumentWorkspace * ws) override {
    auto hue = spec.GetArgument<float>("hue", ws, i);
    const float U = cos(hue * M_PI / 180.0);
    const float V = sin(hue * M_PI / 180.0);
    const mat_t const_mat = mat_t({{.299, .587, .114, 0.0},
                                   {.299, .587, .114, 0.0},
                                   {.299, .587, .114, 0.0},
                                   {.0,   .0,   .0,   1.0}});

    const mat_t U_mat = mat_t({{ .701, -.587, -.114, 0.0},
                               {-.299,  .413, -.114, 0.0},
                               {-.300, -.588, .886,  0.0},
                               { .0,    .0,   .0,    0.0}});

    const mat_t V_mat = mat_t({{ .168,   .330, -.497, 0.0},
                                 {-.328,   .035,  .292, 0.0},
                                 {1.25,  -1.05,  -.203, 0.0},
                                 {  .0,    .0,    .0,   0.0}});

    matrix_ = const_mat + U_mat * U + V_mat * V;
  }
};

class Saturation : public ColorAugment {
 public:
  void Prepare(Index i, const OpSpec &spec, const ArgumentWorkspace * ws) override {
    auto saturation = spec.GetArgument<float>("saturation", ws, i);
    const mat_t const_mat = mat_t({{.299, .587, .114, 0.0},
                                   {.299, .587, .114, 0.0},
                                   {.299, .587, .114, 0.0},
                                   {.0,   .0,   .0,   1.0}});

    const mat_t U_mat = mat_t({{ .701, -.587, -.114, 0.0},
                               {-.299,  .413, -.114, 0.0},
                               {-.300, -.588, .886,  0.0},
                               { .0,    .0,   .0,    0.0}});

    matrix_ = const_mat + U_mat * saturation;
  }
};

template <typename Backend>
class ColorTwistBase : public Operator<Backend> {
 public:
  static const int nDim = 4;

  inline explicit ColorTwistBase(const OpSpec &spec) : Operator<Backend>(spec),
                      C_(IsColor(spec.GetArgument<DALIImageType>("image_type")) ? 3 : 1) {
    DALI_ENFORCE(C_ == 3, "Color transformation is implemented only for RGB images");
  }

  ~ColorTwistBase() override = default;

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override;

  bool CanInferOutputs() const override;

  void RunImpl(Workspace<Backend> &ws) override;

  std::vector<std::unique_ptr<ColorAugment>> augments_;
  const int C_;

  USE_OPERATOR_MEMBERS();

 private:
  std::vector<mat<3, 3, float>> mats{};
  std::vector<vec<3, float>> vecs{};
  kernels::KernelManager kernel_manager_;
};



template <typename Backend>
class BrightnessAdjust : public ColorTwistBase<Backend> {
 public:
  inline explicit BrightnessAdjust(const OpSpec &spec) : ColorTwistBase<Backend>(spec) {
    this->augments_.emplace_back(std::make_unique<Brightness>());
  }

  ~BrightnessAdjust() override = default;
};

template <typename Backend>
class ContrastAdjust : public ColorTwistBase<Backend> {
 public:
  inline explicit ContrastAdjust(const OpSpec &spec) : ColorTwistBase<Backend>(spec) {
    this->augments_.emplace_back(std::make_unique<Contrast>());
  }

  ~ContrastAdjust() override = default;
};

template<typename Backend>
class HueAdjust : public ColorTwistBase<Backend> {
 public:
  inline explicit HueAdjust(const OpSpec &spec) : ColorTwistBase<Backend>(spec) {
    this->augments_.emplace_back(std::make_unique<Hue>());
  }

  ~HueAdjust() override = default;
};

template<typename Backend>
class SaturationAdjust : public ColorTwistBase<Backend> {
 public:
  inline explicit SaturationAdjust(const OpSpec &spec) : ColorTwistBase<Backend>(spec) {
    this->augments_.emplace_back(std::make_unique<Saturation>());
  }

  ~SaturationAdjust() override = default;
};

template<typename Backend>
class ColorTwistAdjust : public ColorTwistBase<Backend> {
 public:
  inline explicit ColorTwistAdjust(const OpSpec &spec) : ColorTwistBase<Backend>(spec) {
    this->augments_.emplace_back(std::make_unique<Hue>());
    this->augments_.emplace_back(std::make_unique<Saturation>());
    this->augments_.emplace_back(std::make_unique<Contrast>());
    this->augments_.emplace_back(std::make_unique<Brightness>());
  }

  ~ColorTwistAdjust() override = default;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_COLOR_COLOR_TWIST_H_
