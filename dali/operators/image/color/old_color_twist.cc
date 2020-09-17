// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <cmath>
#include <memory>
#include <vector>
#include "dali/pipeline/operator/operator.h"
#include "dali/image/transform.h"
#include "dali/util/npp.h"

namespace dali {
namespace old {

class ColorAugment {
 public:
  static const int nDim = 4;

  virtual void operator()(float *matrix) = 0;
  virtual void Prepare(Index i, const OpSpec &spec, const ArgumentWorkspace *ws) = 0;

  virtual ~ColorAugment() = default;
};

class Brightness : public ColorAugment {
 public:
  void operator()(float *matrix) override {
    for (int i = 0; i < nDim - 1; ++i) {
      for (int j = 0; j < nDim; ++j) {
        matrix[i * nDim + j] *= brightness_;
      }
    }
  }

  void Prepare(Index i, const OpSpec &spec, const ArgumentWorkspace *ws) override {
    brightness_ = spec.GetArgument<float>("brightness", ws, i);
  }

 private:
  float brightness_;
};

class Contrast : public ColorAugment {
 public:
  void operator()(float *matrix) override {
    for (int i = 0; i < nDim - 1; ++i) {
      for (int j = 0; j < nDim - 1; ++j) {
        matrix[i * nDim + j] *= contrast_;
      }
      matrix[i * nDim + nDim - 1] =
          matrix[i * nDim + nDim - 1] * contrast_ + (1 - contrast_) * 128.f;
    }
  }

  void Prepare(Index i, const OpSpec &spec, const ArgumentWorkspace *ws) override {
    contrast_ = spec.GetArgument<float>("contrast", ws, i);
  }

 private:
  float contrast_;
};

class Hue : public ColorAugment {
 public:
  void operator()(float *matrix) override {
    float temp[nDim * nDim];  // NOLINT(*)
    for (int i = 0; i < nDim * nDim; ++i) {
      temp[i] = matrix[i];
    }
    const float U = cos(hue_ * M_PI / 180.0);
    const float V = sin(hue_ * M_PI / 180.0);

    // Single matrix transform for both hue and saturation change. Matrix taken
    // from https://beesbuzz.biz/code/hsv_color_transforms.php. Derived by
    // transforming first to HSV, then do the modification, and transfom back to RGB.

    const float const_mat[] = {.299, .587, .114, 0.0, .299, .587, .114, 0.0,
                               .299, .587, .114, 0.0, .0,   .0,   .0,   1.0};

    const float U_mat[] = {.701,  -.587, -.114, 0.0, -.299, .413, -.114, 0.0,
                           -.300, -.588, .886,  0.0, .0,    .0,   .0,    0.0};

    const float V_mat[] = {.168, .330,  -.497, 0.0, -.328, .035, .292, 0.0,
                           1.25, -1.05, -.203, 0.0, .0,    .0,   .0,   0.0};
    // The last row stays the same so we update only nDim - 1 rows
    for (int i = 0; i < nDim - 1; ++i) {
      for (int j = 0; j < nDim; ++j) {
        float sum = 0;
        for (int k = 0; k < nDim; ++k) {
          sum += temp[k * nDim + j] *
                 (const_mat[i * nDim + k] + U_mat[i * nDim + k] * U + V_mat[i * nDim + k] * V);
        }
        matrix[i * nDim + j] = sum;
      }
    }
  }

  void Prepare(Index i, const OpSpec &spec, const ArgumentWorkspace *ws) override {
    hue_ = spec.GetArgument<float>("hue", ws, i);
  }

 private:
  float hue_;
};

class Saturation : public ColorAugment {
 public:
  void operator()(float *matrix) override {
    float temp[nDim * nDim];  // NOLINT(*)
    for (int i = 0; i < nDim * nDim; ++i) {
      temp[i] = matrix[i];
    }

    // Single matrix transform for both hue and saturation change. Matrix taken
    // from https://beesbuzz.biz/code/hsv_color_transforms.php. Derived by
    // transforming first to HSV, then do the modification, and transfom back to RGB.

    const float const_mat[] = {.299, .587, .114, 0.0, .299, .587, .114, 0.0,
                               .299, .587, .114, 0.0, .0,   .0,   .0,   1.0};

    const float U_mat[] = {.701,  -.587, -.114, 0.0, -.299, .413, -.114, 0.0,
                           -.300, -.588, .886,  0.0, .0,    .0,   .0,    0.0};

    // The last row stays the same so we update only nDim - 1 rows
    for (int i = 0; i < nDim - 1; ++i) {
      for (int j = 0; j < nDim; ++j) {
        float sum = 0;
        for (int k = 0; k < nDim; ++k) {
          sum += temp[k * nDim + j] * (const_mat[i * nDim + k] + U_mat[i * nDim + k] * saturation_);
        }
        matrix[i * nDim + j] = sum;
      }
    }
  }

  void Prepare(Index i, const OpSpec &spec, const ArgumentWorkspace *ws) override {
    saturation_ = spec.GetArgument<float>("saturation", ws, i);
  }

 private:
  float saturation_;
};
}  // namespace old

template <typename Backend>
class OldColorTwistBase : public Operator<Backend> {
 public:
  static const int nDim = 4;

  inline explicit OldColorTwistBase(const OpSpec &spec)
      : Operator<Backend>(spec),
        C_(IsColor(spec.GetArgument<DALIImageType>("image_type")) ? 3 : 1) {
    DALI_ENFORCE(C_ == 3, "Color transformation is implemented only for RGB images");
  }

  ~OldColorTwistBase() override {
    for (auto *a : augments_) {
      delete a;
    }
  }

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    return false;
  }

  void RunImpl(Workspace<Backend> &ws) override;

  std::vector<old::ColorAugment *> augments_;
  const int C_;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;

 private:
  void IdentityMatrix(float *matrix) {
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
class OldColorTwistAdjust : public OldColorTwistBase<Backend> {
 public:
  inline explicit OldColorTwistAdjust(const OpSpec &spec) : OldColorTwistBase<Backend>(spec) {
    this->augments_.push_back(new old::Hue());
    this->augments_.push_back(new old::Saturation());
    this->augments_.push_back(new old::Contrast());
    this->augments_.push_back(new old::Brightness());
  }

  ~OldColorTwistAdjust() override = default;
};

typedef NppStatus (*colorTwistFunc)(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32f aTwist[3][4]);

template<>
void OldColorTwistBase<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  DALI_ENFORCE(IsType<uint8_t>(input.type()), "Color augmentations accept only uint8 tensors");
  auto &output = ws.Output<GPUBackend>(0);
  output.ResizeLike(input);
  output.SetLayout(input.GetLayout());

  cudaStream_t old_stream = nppGetStream();
  nppSetStream(ws.stream());

  for (size_t i = 0; i < input.ntensor(); ++i) {
    if (!augments_.empty()) {
      float matrix[nDim][nDim];
      float *m = reinterpret_cast<float *>(matrix);
      IdentityMatrix(m);
      for (size_t j = 0; j < augments_.size(); ++j) {
        augments_[j]->Prepare(i, spec_, &ws);
        (*augments_[j])(m);
      }
      NppiSize size;
      size.height = input.tensor_shape(i)[0];
      size.width = input.tensor_shape(i)[1];
      const int nStep = C_ * size.width;  // W * C
      colorTwistFunc twist_func = C_ == 3 ? nppiColorTwist32f_8u_C3R : nppiColorTwist32f_8u_C1R;
      DALI_CHECK_NPP(twist_func(input.tensor<uint8_t>(i), nStep, output.mutable_tensor<uint8_t>(i),
                                nStep, size, matrix));
    } else {
      CUDA_CALL(cudaMemcpyAsync(output.raw_mutable_tensor(i), input.raw_tensor(i),
                                volume(input.tensor_shape(i)), cudaMemcpyDefault, ws.stream()));
    }
  }
  nppSetStream(old_stream);
}

template <>
void OldColorTwistBase<CPUBackend>::RunImpl(SampleWorkspace &ws) {
  const auto &input = ws.Input<CPUBackend>(0);
  auto &output = ws.Output<CPUBackend>(0);
  const auto &input_shape = input.shape();

  CheckParam(input, "Color augmentation");

  const auto H = input_shape[0];
  const auto W = input_shape[1];
  const auto C = input_shape[2];

  output.ResizeLike(input);
  output.SetLayout(input.GetLayout());

  auto pImgInp = input.template data<uint8>();
  auto pImgOut = output.template mutable_data<uint8>();

  if (!augments_.empty()) {
    float matrix[nDim][nDim];
    float *m = reinterpret_cast<float *>(matrix);
    IdentityMatrix(m);
    for (size_t j = 0; j < augments_.size(); ++j) {
      augments_[j]->Prepare(ws.data_idx(), spec_, &ws);
      (*augments_[j])(m);
    }

    MakeColorTransformation(pImgInp, H, W, C, m, pImgOut);
  } else {
    memcpy(pImgOut, pImgInp, H * W * C);
  }
}

DALI_SCHEMA(OldColorTwist)
    .DocStr(R"code(A combination of hue, saturation, contrast, and brightness.

.. note::
    This is an old implementation which uses NPP.)code")
    .Deprecate("ColorTwist")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("hue", R"code(Hue change, in degrees.)code", 0.f, true)
    .AddOptionalArg("saturation",
                    R"code(Saturation change factor.

Values must be non-negative.

Example values:

- `0` – Completely desaturated image.
- `1` - No change to image's saturation.
)code", 1.f, true)
    .AddOptionalArg("contrast",
                    R"code(Contrast change factor.

Values must be non-negative.

Example values:

* `0` - Uniform grey image.
* `1` - No change.
* `2` - Increase brightness twice.
)code", 1.f, true)
    .AddOptionalArg("brightness",
                    R"code(Brightness change factor.

Values must be non-negative.

Example values:

* `0` - Black image.
* `1` - No change.
* `2` - Increase brightness twice.
)code", 1.f, true)
    .AddParent("ColorTransformBase")
    .InputLayout(0, "HWC");

DALI_REGISTER_OPERATOR(OldColorTwist, OldColorTwistAdjust<GPUBackend>, GPU);
DALI_REGISTER_OPERATOR(OldColorTwist, OldColorTwistAdjust<CPUBackend>, CPU);

}  // namespace dali
