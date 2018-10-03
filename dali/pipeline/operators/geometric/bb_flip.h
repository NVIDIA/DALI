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

#ifndef DALI_PIPELINE_OPERATORS_GEOMETRIC_BB_FLIP_H_
#define DALI_PIPELINE_OPERATORS_GEOMETRIC_BB_FLIP_H_

#include <dali/pipeline/operators/operator.h>
#include <dali/pipeline/operators/common.h>

namespace dali {

class BbFlip : public Operator<CPUBackend> {
 public:
  explicit BbFlip(const OpSpec &spec);

  virtual ~BbFlip() = default;
  DISABLE_COPY_MOVE_ASSIGN(BbFlip);

 protected:
  void RunImpl(SampleWorkspace *ws, const int idx) override;

 private:
  const int kBbTypeSize = 4;  /// Bounding box is always vector of 4 floats

  /**
   * Bounding box can be represented in two ways:
   * 1. Upper-left corner, width, height (`wh_type`)
   *    (x1, y1,  w,  h)
   * 2. Upper-left and Lower-right corners (`two-point type`)
   *    (x1, y1, x2, y2)
   *
   * Both of them have coordinates in image coordinate system
   * (i.e. 0.0-1.0)
   *
   * If `coordinates_type_wh_` is true, then we deal with 1st type. Otherwise, the 2nd one.
   */
  const bool coordinates_type_wh_;

  /**
   * If true, flip is performed along vertical (x) axis
   */
  const bool flip_type_vertical_;

  /**
   * If true, Operator if turned on
   */
  const bool on_off_switch_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_GEOMETRIC_BB_FLIP_H_
