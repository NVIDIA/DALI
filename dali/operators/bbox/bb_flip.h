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

#ifndef DALI_OPERATORS_BBOX_BB_FLIP_H_
#define DALI_OPERATORS_BBOX_BB_FLIP_H_

#include <string>
#include <vector>

#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/arg_helper.h"

namespace dali {

template <typename Backend>
class BbFlip : public Operator<Backend> {
 public:
  explicit BbFlip(const OpSpec &spec) :
      Operator<Backend>(spec),
      ltrb_(spec.GetArgument<bool>("ltrb")),
      horz_("horizontal", spec),
      vert_("vertical", spec) {}

  ~BbFlip() override = default;
  DISABLE_COPY_MOVE_ASSIGN(BbFlip);

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_descs, const workspace_t<Backend> &ws) override {
    const auto &input = ws.template InputRef<Backend>(0);
    DALI_ENFORCE(input.type().id() == DALI_FLOAT, "Bounding box in wrong format");
    auto nsamples = input.shape().size();
    horz_.Acquire(spec_, ws, nsamples, TensorShape<0>{});
    vert_.Acquire(spec_, ws, nsamples, TensorShape<0>{});
    output_descs.resize(1);  // only one output
    output_descs[0].type = input.type();
    output_descs[0].shape = input.shape();
    return true;
  }

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
   * If `coordinates_type_ltrb_` is true, then we deal with 2nd type. Otherwise,
   * the 1st one.
   */
  const bool ltrb_;

  ArgValue<int> horz_;
  ArgValue<int> vert_;

  using Operator<Backend>::spec_;
};

class BbFlipCPU : public BbFlip<CPUBackend> {
 public:
  explicit BbFlipCPU(const OpSpec &spec) : BbFlip<CPUBackend>(spec) {}

 protected:
  void RunImpl(workspace_t<CPUBackend> &ws) override;
  using BbFlip<CPUBackend>::RunImpl;

 private:
  using BbFlip<CPUBackend>::horz_;
  using BbFlip<CPUBackend>::vert_;
  using BbFlip<CPUBackend>::ltrb_;
};


}  // namespace dali

#endif  // DALI_OPERATORS_BBOX_BB_FLIP_H_
