// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_SSD_RANDOM_CROP_H_
#define DALI_OPERATORS_SSD_RANDOM_CROP_H_

#include <cfloat>
#include <vector>
#include <random>
#include <memory>
#include <utility>

#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/util/batch_rng.h"
#include "dali/operators/random/rng_base_cpu.h"

namespace dali {

template <typename Backend>
class SSDRandomCrop : public rng::OperatorWithRng<Backend> {
 public:
  explicit inline SSDRandomCrop(const OpSpec &spec) :
    rng::OperatorWithRng<Backend>(spec),
    num_attempts_(spec.GetArgument<int>("num_attempts")),
    int_dis_(0, 6),        // sample option
    float_dis_(0.3, 1.) {  // w, h generation
    // setup all possible sample types
    sample_options_.push_back(SampleOption{false, -1.f});
    sample_options_.push_back(SampleOption{false, 0.1});
    sample_options_.push_back(SampleOption{false, 0.3});
    sample_options_.push_back(SampleOption{false, 0.5});
    sample_options_.push_back(SampleOption{false, 0.7});
    sample_options_.push_back(SampleOption{false, 0.9});
    sample_options_.push_back(SampleOption{true, 0});
  }

  inline ~SSDRandomCrop() override = default;

  DISABLE_COPY_MOVE_ASSIGN(SSDRandomCrop);

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    return false;
  }

  void RunImpl(legacy_workspace_t<Backend> &ws) override;
  void SetupSharedSampleParams(legacy_workspace_t<Backend> &ws) override;

 private:
  struct CropInfo {
    int x, y;
    int w, h;
  };

  struct SampleOption {
    bool no_crop_ = false;
    float min_iou_ = FLT_MAX;

    SampleOption(bool no_crop, float min_iou) :
      no_crop_(no_crop), min_iou_(min_iou) {}

    bool no_crop() const {
      return no_crop_;
    }
    float min_iou() const {
      return min_iou_;
    }
  };

  std::vector<SampleOption> sample_options_;

  int num_attempts_;

  // RNG stuff
  std::uniform_int_distribution<> int_dis_;
  std::uniform_real_distribution<float> float_dis_;
};

}  // namespace dali


#endif  // DALI_OPERATORS_SSD_RANDOM_CROP_H_
