// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifndef NDLL_PIPELINE_OPERATORS_WARPAFFINE_H_
#define NDLL_PIPELINE_OPERATORS_WARPAFFINE_H_

#include <ctgmath>
#include <vector>
#include "ndll/pipeline/operator.h"
#include "ndll/pipeline/operators/displacement_filter.h"

namespace ndll {

class WarpAffineAugment {
 public:
  static const int size = 6;
  explicit WarpAffineAugment(const OpSpec& spec)
    : use_image_center(spec.GetArgument<bool>("use_image_center")) {
    std::vector<float> tmp = spec.GetRepeatedArgument<float>("matrix");
    NDLL_ENFORCE(tmp.size() == size, "Warp affine matrix needs to have 6 elements");
    for (int i = 0; i < size; ++i) {
      matrix[i] = tmp[i];
    }
  }

  template <typename T>
  DISPLACEMENT_IMPL
  Point<T> operator()(int h, int w, int c, int H, int W, int C) {
    Point<T> p;
    T hp = h;
    T wp = w;
    if (use_image_center) {
      hp -= H/2.0f;
      wp -= W/2.0f;
    }
    T newX = matrix[0] * wp + matrix[1] * hp + matrix[2];
    T newY = matrix[3] * wp + matrix[4] * hp + matrix[5];
    if (use_image_center) {
      newX += W/2.0f;
      newY += H/2.0f;
    }

    p.x = newX >= 0 && newX < W ? newX : -1;
    p.y = newY >= 0 && newY < H ? newY : -1;
    return p;
  }

  void Cleanup() {}

 private:
  float matrix[6];
  bool use_image_center;
};

template <typename Backend>
class WarpAffine : public DisplacementFilter<Backend, WarpAffineAugment> {
 public:
    inline explicit WarpAffine(const OpSpec &spec)
      : DisplacementFilter<Backend, WarpAffineAugment>(spec) {}

    virtual ~WarpAffine() = default;
};

#ifndef M_PI
const float M_PI =  3.14159265358979323846;
#endif

template <typename Backend>
class Rotate : public Operator<Backend> {
 public:
  inline explicit Rotate(const OpSpec &spec)
    : Operator<Backend>(spec),
      angle_(spec.GetArgument<float>("angle")) {
    std::vector<float> matrix(6);
    float angle_rad = angle_ * M_PI / 180.0;
    matrix[0] = cos(angle_rad);
    matrix[1] = sin(angle_rad);
    matrix[2] = 0.0;
    matrix[3] = -sin(angle_rad);
    matrix[4] = cos(angle_rad);
    matrix[5] = 0.0;
    OpSpec tmp = spec;
    tmp.set_name("WarpAffine");
    tmp.AddArg("matrix", matrix);
    tmp.AddArg("use_image_center", true);
    augmentor_ = new WarpAffine<Backend>(tmp);
  }

  void RunImpl(Workspace<Backend> *ws, int idx = 0) override {
    augmentor_->RunImpl(ws, idx);
  }

  void SetupSharedSampleParams(Workspace<Backend> *ws) {
    augmentor_->SetupSharedSampleParams(ws);
  }

 private:
  WarpAffine<Backend> * augmentor_;
  float angle_;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_WARPAFFINE_H_
