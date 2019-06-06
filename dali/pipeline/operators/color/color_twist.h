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


#ifndef DALI_PIPELINE_OPERATORS_COLOR_COLOR_TWIST_H_
#define DALI_PIPELINE_OPERATORS_COLOR_COLOR_TWIST_H_

#include <vector>
#include <memory>
#include <cmath>
#include "dali/pipeline/operators/operator.h"

namespace dali {

class ColorAugment {
 public:
  static const int nDim = 4;

  virtual void operator() (float * matrix) = 0;
  virtual void Prepare(Index i, const OpSpec &spec, const ArgumentWorkspace * ws) = 0;

  virtual ~ColorAugment() = default;
};

class Brightness : public ColorAugment {
 public:
  void operator() (float * matrix) override {
    for (int i = 0; i < nDim - 1; ++i) {
      for (int j = 0; j < nDim; ++j) {
        matrix[i * nDim + j] *= brightness_;
      }
    }
  }

  void Prepare(Index i, const OpSpec &spec, const ArgumentWorkspace * ws) override {
    brightness_ = spec.GetArgument<float>("brightness", ws, i);
  }

 private:
  float brightness_;
};

class Contrast : public ColorAugment {
 public:
  void operator() (float * matrix) override {
    for (int i = 0; i < nDim - 1; ++i) {
      for (int j = 0; j < nDim - 1; ++j) {
        matrix[i * nDim + j] *= contrast_;
      }
      matrix[i * nDim + nDim - 1] = matrix[i * nDim + nDim - 1] * contrast_ +
                                    (1 - contrast_) * 128.f;
    }
  }

  void Prepare(Index i, const OpSpec &spec, const ArgumentWorkspace * ws) override {
    contrast_ = spec.GetArgument<float>("contrast", ws, i);
  }

 private:
  float contrast_;
};

class Hue : public ColorAugment {
 public:
  void operator() (float * matrix) override {
    float temp[nDim*nDim];  // NOLINT(*)
    for (int i = 0; i < nDim * nDim; ++i) {
        temp[i] = matrix[i];
    }
    const float U = cos(hue_ * M_PI / 180.0);
    const float V = sin(hue_ * M_PI / 180.0);

    // Single matrix transform for both hue and saturation change. Matrix taken
    // from https://beesbuzz.biz/code/hsv_color_transforms.php. Derived by
    // transforming first to HSV, then do the modification, and transfom back to RGB.

    const float const_mat[] = {.299, .587, .114, 0.0,
                               .299, .587, .114, 0.0,
                               .299, .587, .114, 0.0,
                               .0,   .0,   .0,   1.0};

    const float U_mat[] = { .701, -.587, -.114, 0.0,
                           -.299,  .413, -.114, 0.0,
                           -.300, -.588, .886,  0.0,
                            .0,    .0,   .0,    0.0};

    const float V_mat[] = { .168,   .330, -.497, 0.0,
                           -.328,   .035,  .292, 0.0,
                           1.25, -1.05,  -.203, 0.0,
                            .0,    .0,    .0,   0.0};
    // The last row stays the same so we update only nDim - 1 rows
    for (int i = 0 ; i < nDim - 1; ++i) {
      for (int j = 0; j < nDim; ++j) {
        float sum = 0;
        for (int k = 0; k < nDim; ++k) {
          sum += temp[k * nDim + j] * (const_mat[i * nDim + k] +
                                       U_mat[i * nDim + k] * U +
                                       V_mat[i * nDim + k] * V);
        }
        matrix[i * nDim + j] = sum;
      }
    }
  }

  void Prepare(Index i, const OpSpec &spec, const ArgumentWorkspace * ws) override {
    hue_ = spec.GetArgument<float>("hue", ws, i);
  }

 private:
  float hue_;
};

class Saturation : public ColorAugment {
 public:
  void operator() (float * matrix) override {
    float temp[nDim*nDim];  // NOLINT(*)
    for (int i = 0; i < nDim * nDim; ++i) {
        temp[i] = matrix[i];
    }

    // Single matrix transform for both hue and saturation change. Matrix taken
    // from https://beesbuzz.biz/code/hsv_color_transforms.php. Derived by
    // transforming first to HSV, then do the modification, and transfom back to RGB.

    const float const_mat[] = {.299, .587, .114, 0.0,
                               .299, .587, .114, 0.0,
                               .299, .587, .114, 0.0,
                               .0,   .0,   .0,   1.0};

    const float U_mat[] = { .701, -.587, -.114, 0.0,
                           -.299,  .413, -.114, 0.0,
                           -.300, -.588, .886,  0.0,
                            .0,    .0,   .0,    0.0};

    // The last row stays the same so we update only nDim - 1 rows
    for (int i = 0 ; i < nDim - 1; ++i) {
      for (int j = 0; j < nDim; ++j) {
        float sum = 0;
        for (int k = 0; k < nDim; ++k) {
          sum += temp[k * nDim + j] * (const_mat[i * nDim + k] +
                                       U_mat[i * nDim + k] * saturation_);
        }
        matrix[i * nDim + j] = sum;
      }
    }
  }

  void Prepare(Index i, const OpSpec &spec, const ArgumentWorkspace * ws) override {
    saturation_ = spec.GetArgument<float>("saturation", ws, i);
  }

 private:
  float saturation_;
};

template <typename Backend>
class ColorTwistBase : public Operator<Backend> {
 public:
  static const int nDim = 4;

  inline explicit ColorTwistBase(const OpSpec &spec) : Operator<Backend>(spec),
                      C_(IsColor(spec.GetArgument<DALIImageType>("image_type")) ? 3 : 1) {
    DALI_ENFORCE(C_ == 3, "Color transformation is implemented only for RGB images");
  }

  ~ColorTwistBase() override {
    for (auto * a : augments_) {
      delete a;
    }
  }

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;

  std::vector<ColorAugment*> augments_;
  const int C_;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;

 private:
  void IdentityMatrix(float * matrix) {
    for (int i = 0; i < nDim; ++i) {
      for (int j = 0; j < nDim; ++j) {
        if (i == j) {
          matrix[i * nDim + j] = 1;
        } else {
          matrix[i * nDim + j] = 0;
        }
      }
    }
  }
};

template <typename Backend>
class BrightnessAdjust : public ColorTwistBase<Backend> {
 public:
  inline explicit BrightnessAdjust(const OpSpec &spec) : ColorTwistBase<Backend>(spec) {
    this->augments_.push_back(new Brightness());
  }

  ~BrightnessAdjust() override = default;
};

template <typename Backend>
class ContrastAdjust : public ColorTwistBase<Backend> {
 public:
  inline explicit ContrastAdjust(const OpSpec &spec) : ColorTwistBase<Backend>(spec) {
    this->augments_.push_back(new Contrast());
  }

  ~ContrastAdjust() override = default;
};

template<typename Backend>
class HueAdjust : public ColorTwistBase<Backend> {
 public:
  inline explicit HueAdjust(const OpSpec &spec) : ColorTwistBase<Backend>(spec) {
    this->augments_.push_back(new Hue());
  }

  ~HueAdjust() override = default;
};

template<typename Backend>
class SaturationAdjust : public ColorTwistBase<Backend> {
 public:
  inline explicit SaturationAdjust(const OpSpec &spec) : ColorTwistBase<Backend>(spec) {
    this->augments_.push_back(new Saturation());
  }

  ~SaturationAdjust() override = default;
};

template<typename Backend>
class ColorTwistAdjust : public ColorTwistBase<Backend> {
 public:
  inline explicit ColorTwistAdjust(const OpSpec &spec) : ColorTwistBase<Backend>(spec) {
    this->augments_.push_back(new Hue());
    this->augments_.push_back(new Saturation());
    this->augments_.push_back(new Contrast());
    this->augments_.push_back(new Brightness());
  }

  ~ColorTwistAdjust() override = default;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_COLOR_COLOR_TWIST_H_
