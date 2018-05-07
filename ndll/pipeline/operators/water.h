// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifndef NDLL_PIPELINE_OPERATORS_WATER_H_
#define NDLL_PIPELINE_OPERATORS_WATER_H_

#include <ctgmath>
#include <vector>
#include <string>

#include "ndll/pipeline/operators/displacement_filter.h"

namespace ndll {

class WaterAugment {
 public:
  class WaveDescr {
   public:
    WaveDescr(const OpSpec &spec, const char *direction)
      : ampl(spec.GetArgument<float>(string("ampl") + direction)),
        freq(spec.GetArgument<float>(string("freq") + direction)),
        phase(spec.GetArgument<float>(string("phase") + direction)) {}

    float ampl;
    float freq;
    float phase;
  };

  explicit WaterAugment(const OpSpec& spec)
    : x_desc_(spec, "_x"),
      y_desc_(spec, "_y") {}

  void Cleanup() {}

  template <typename T>
  DISPLACEMENT_IMPL
  Point<T> operator()(int h, int w, int c, int H, int W, int C) {
    const WaveDescr &wX = x_desc_;
    const WaveDescr &wY = y_desc_;

    const T newX = w + wX.ampl * sinf(wX.freq * h + wX.phase);
    const T newY = h + wY.ampl * cosf(wY.freq * w + wY.phase);

    Point<T> p;
    p.x = newX >= 0 && newX < W ? newX : -1;
    p.y = newY >= 0 && newY < H ? newY : -1;

    return p;
  }

 private:
  WaveDescr x_desc_, y_desc_;
};

template <typename Backend>
class Water : public DisplacementFilter<Backend, WaterAugment> {
 public:
  explicit Water(const OpSpec& spec)
    : DisplacementFilter<Backend, WaterAugment>(spec) {}

  virtual ~Water() = default;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_WATER_H_
