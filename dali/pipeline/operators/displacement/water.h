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


#ifndef DALI_PIPELINE_OPERATORS_DISPLACEMENT_WATER_H_
#define DALI_PIPELINE_OPERATORS_DISPLACEMENT_WATER_H_

#include <ctgmath>
#include <vector>
#include <string>

#include "dali/pipeline/operators/displacement/displacement_filter.h"

namespace dali {

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

    return CreatePointLimited(newX, newY, W, H);
  }

 private:
  WaveDescr x_desc_, y_desc_;
};

template <typename Backend>
class Water : public DisplacementFilter<Backend, WaterAugment> {
 public:
  explicit Water(const OpSpec& spec)
    : DisplacementFilter<Backend, WaterAugment>(spec) {}

  ~Water() override = default;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DISPLACEMENT_WATER_H_
