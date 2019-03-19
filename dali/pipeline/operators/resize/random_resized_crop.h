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

#ifndef DALI_PIPELINE_OPERATORS_RESIZE_RANDOM_RESIZED_CROP_H_
#define DALI_PIPELINE_OPERATORS_RESIZE_RANDOM_RESIZED_CROP_H_

#include <vector>
#include <random>
#include <memory>
#include <utility>

#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/operators/resize/resize_base.h"
#include "dali/util/random_crop_generator.h"
#include "dali/kernels/imgproc/resample/params.h"

namespace dali {

template <typename Backend>
class RandomResizedCrop : public Operator<Backend>
                        , protected ResizeBase {
 public:
  explicit inline RandomResizedCrop(const OpSpec &spec)
      : Operator<Backend>(spec)
      , ResizeBase(spec)
      , num_attempts_(spec.GetArgument<int>("num_attempts"))
      , interp_type_(spec.GetArgument<DALIInterpType>("interp_type")) {
    GetSingleOrRepeatedArg(spec, &size_, "size", 2);
    GetSingleOrRepeatedArg(spec, &aspect_ratio_range_, "random_aspect_ratio", 2);
    GetSingleOrRepeatedArg(spec, &area_range_, "random_area", 2);
    DALI_ENFORCE(aspect_ratio_range_[0] <= aspect_ratio_range_[1],
        "Provided empty range");
    DALI_ENFORCE(area_range_[0] <= area_range_[1],
        "Provided empty range");
    InitParams(spec);
    BackendInit();
  }

  inline ~RandomResizedCrop() override = default;

  DISABLE_COPY_MOVE_ASSIGN(RandomResizedCrop);

  USE_OPERATOR_MEMBERS();

 protected:
  void RunImpl(Workspace<Backend> * ws, const int idx) override;
  void SetupSharedSampleParams(Workspace<Backend> *ws) override;

 private:
  void BackendInit();

  struct Params {
    void Initialize(
        int num_gens,
        int64_t seed,
        const std::pair<float, float> &aspect_ratio_range,
        const std::pair<float, float> &area_range,
        int num_attempts) {
      std::seed_seq seq{seed};
      std::vector<int> seeds(num_gens);
      seq.generate(seeds.begin(), seeds.end());

      crop_gens.resize(num_gens);
      for (int i = 0; i < num_gens; i++) {
        crop_gens[i] = RandomCropGenerator(aspect_ratio_range, area_range, seeds[i], num_attempts);
      }

      crops.resize(num_gens);
    }

    std::vector<RandomCropGenerator> crop_gens;
    std::vector<CropWindow> crops;
  };


  void CalcResamplingParams() {
    const int n = params_.crops.size();
    resample_params_.resize(n);
    for (int i = 0; i < n; i++)
      resample_params_[i] = CalcResamplingParams(i);
  }

  kernels::ResamplingParams2D CalcResamplingParams(int index) const {
    auto &wnd = params_.crops[index];
    auto params = shared_params_;
    params[0].roi = kernels::ResamplingParams::ROI(wnd.y, wnd.y+wnd.h);
    params[1].roi = kernels::ResamplingParams::ROI(wnd.x, wnd.x+wnd.w);
    return params;
  }

  void InitParams(const OpSpec &spec) {
    auto seed = spec.GetArgument<int64_t>("seed");
    params_.Initialize(
        batch_size_, seed,
        { aspect_ratio_range_[0], aspect_ratio_range_[1] },
        { area_range_[0], area_range_[1] },
        num_attempts_);

    shared_params_[0].output_size = size_[0];
    shared_params_[1].output_size = size_[1];
    shared_params_[0].min_filter = shared_params_[1].min_filter = min_filter_;
    shared_params_[0].mag_filter = shared_params_[1].mag_filter = mag_filter_;
  }

  Params params_;
  int num_attempts_;

  std::vector<int> size_;
  DALIInterpType interp_type_;
  kernels::ResamplingParams2D shared_params_;

  std::vector<float> aspect_ratio_range_;
  std::vector<float> area_range_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_RESIZE_RANDOM_RESIZED_CROP_H_
