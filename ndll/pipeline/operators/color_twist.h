// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifndef NDLL_PIPELINE_OPERATORS_COLOR_TWIST_H_
#define NDLL_PIPELINE_OPERATORS_COLOR_TWIST_H_

#include <vector>
#include "ndll/pipeline/operator.h"

namespace ndll {

typedef NppStatus (*colorTwistFunc)(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32f aTwist[3][4]);

template <typename Backend>
class ColorTwist : public Operator<Backend> {
 public:
  inline explicit ColorTwist(const OpSpec &spec) : Operator<Backend>(spec),
                      color_(IsColor(spec.GetArgument<NDLLImageType>("image_type"))),
                      C_(color_ ? 3 : 1) {
    twistFunc_ = colorImgs() ? nppiColorTwist32f_8u_C3R : nppiColorTwist32f_8u_C1R;

    // Resize per-image data
    input_ptrs_.resize(batch_size_);
    output_ptrs_.resize(batch_size_);
    sizes_.resize(batch_size_);
  }

  virtual ~ColorTwist() = default;

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;

  virtual bool reshapeBatch() const               { return true; }
  virtual bool twistMatr(float pMatr[][4]) const  { return false; }
  inline int getC() const                         { return C_; }
  inline bool colorImgs() const                   { return color_; }

 private:
  const bool color_;
  const int C_;
  colorTwistFunc twistFunc_;
  vector<const uint8 *> input_ptrs_;
  vector<uint8 *> output_ptrs_;
  vector<NDLLSize> sizes_;

  USE_OPERATOR_MEMBERS();
};

template <typename Backend>
class ColorIntensity : public ColorTwist<Backend> {
 public:
  inline explicit ColorIntensity(const OpSpec &spec) : ColorTwist<Backend>(spec) {
    InitColorTwistParam(spec);
  }

  virtual ~ColorIntensity() = default;

 protected:
  virtual bool twistMatr(float pMatr[][4]) const {
    bool brightness_matrix(const float *pScale, float pMatr[][4]);
    return brightness_matrix(scale(), pMatr);
  }

  void InitColorTwistParam(const OpSpec &spec) {
    if (ColorTwist<Backend>::colorImgs())
      scale_ = spec.GetArgument<vector<float>>("RGB_level");
    else
      scale_ = {spec.GetArgument<float>("GRAY_level"), 0., 0.};
  }

  inline const float *scale() const { return scale_.data(); }

 private:
  vector<float> scale_;
};

template <typename Backend>
class ColorOffset : public ColorIntensity<Backend> {
 public:
  inline explicit ColorOffset(const OpSpec &spec) : ColorIntensity<Backend>(spec) {}
  virtual ~ColorOffset() = default;

 protected:
  virtual bool twistMatr(float pMatr[][4]) const {
    bool color_offset_matrix(const float *pScale, float pMatr[][4]);
    return color_offset_matrix(ColorIntensity<Backend>::scale(), pMatr);
  }
};

template<typename Backend>
class HueSaturation : public ColorTwist<Backend> {
 public:
  inline explicit HueSaturation(const OpSpec &spec) : ColorTwist<Backend>(spec),
                      hue_(spec.GetArgument<float>("hue")),
                      saturation_(spec.GetArgument<float>("saturation")) {}

  virtual ~HueSaturation() = default;

 protected:
  virtual bool twistMatr(float pMatr[][4]) const {
    bool hue_saturation_matrix(float hue, float saturation, float pMatr[][4]);
    return hue_saturation_matrix(hue_, saturation_, pMatr);
  }

 private:
  const float hue_;
  const float saturation_;
};

template<typename Backend>
class ColorContrast : public ColorTwist<Backend> {
 public:
  inline explicit ColorContrast(const OpSpec &spec) : ColorTwist<Backend>(spec),
                       slop_(spec.GetArgument<float>("slope")),
                       bias_(spec.GetArgument<float>("bias")) {}

  virtual ~ColorContrast() = default;

 protected:
  virtual bool twistMatr(float pMatr[][4]) const {
    bool contrast_matrix(float slop, float bias, float pMatr[][4]);
    return contrast_matrix(slop_, bias_, pMatr);
  }

 private:
  const float slop_;
  const float bias_;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_COLOR_TWIST_H_
