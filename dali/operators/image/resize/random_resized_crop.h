// Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_RESIZE_RANDOM_RESIZED_CROP_H_
#define DALI_OPERATORS_IMAGE_RESIZE_RANDOM_RESIZED_CROP_H_

#include <vector>
#include <random>
#include <memory>
#include <utility>

#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/common.h"
#include "dali/operators/image/resize/resize_base.h"
#include "dali/operators/image/resize/resize_attr.h"
#include "dali/operators/image/crop/random_crop_attr.h"
#include "dali/kernels/imgproc/resample/params.h"

namespace dali {

template <typename Backend>
class RandomResizedCrop : public Operator<Backend>
                        , protected ResizeBase<Backend> {
 public:
  explicit inline RandomResizedCrop(const OpSpec &spec)
      : Operator<Backend>(spec), ResizeBase<Backend>(spec), crop_attr_(spec) {
    GetSingleOrRepeatedArg(spec, size_, "size", 2);
    BackendInit();
  }

  inline ~RandomResizedCrop() override = default;

  DISABLE_COPY_MOVE_ASSIGN(RandomResizedCrop);

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;

  bool CanInferOutputs() const override { return true; }

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    auto curr_batch_size = ws.GetInputBatchSize(0);
    InitParams(curr_batch_size);
    auto &input = ws.template InputRef<Backend>(0);
    const auto &in_shape = input.shape();
    DALIDataType in_type = input.type().id();

    auto layout = input.GetLayout();

    int N = in_shape.num_samples();

    resample_params_.resize(N);
    resampling_attr_.PrepareFilterParams(spec_, ws, N);

    auto out_type = resampling_attr_.GetOutputType(in_type);

    int width_idx  = layout.find('W');
    int height_idx = layout.find('H');
    assert(width_idx >= 0 && "Width dimension not found");
    assert(height_idx >= 0 && "Height dimension not found");
    assert(width_idx == height_idx + 1 && "Width must immediately follow height dim.");

    for (int sample_idx = 0; sample_idx < N; sample_idx++) {
      auto sample_shape = in_shape.tensor_shape_span(sample_idx);
      int H = sample_shape[height_idx];
      int W = sample_shape[width_idx];
      crops_[sample_idx] = crop_attr_.GetCropWindowGenerator(sample_idx)({H, W}, "HW");
      resample_params_[sample_idx] = CalcResamplingParams(sample_idx);
    }
    resampling_attr_.ApplyFilterParams(make_span(resample_params_));

    output_desc.resize(1);
    this->SetupResize(output_desc[0].shape, out_type, in_shape, in_type,
                      make_cspan(resample_params_), height_idx);
    output_desc[0].type = TypeTable::GetTypeInfo(out_type);
    return true;
  }

  void RunImpl(workspace_t<Backend> &ws) override;

 private:
  void BackendInit();

  void CalcResamplingParams() {
    const int n = crops_.size();
    resample_params_.resize(n);
    for (int i = 0; i < n; i++)
      resample_params_[i] = CalcResamplingParams(i);
  }

  kernels::ResamplingParams2D CalcResamplingParams(int index) const {
    auto &wnd = crops_[index];
    auto params = shared_params_;
    for (int d = 0; d < 2; d++) {
      params[d].roi = kernels::ResamplingParams::ROI(wnd.anchor[d], wnd.anchor[d] + wnd.shape[d]);
    }
    return params;
  }

  void InitParams(int curr_batch_size) {
    crops_.resize(curr_batch_size);
    shared_params_[0].output_size = size_[0];
    shared_params_[1].output_size = size_[1];
  }

  ResamplingFilterAttr resampling_attr_;
  RandomCropAttr crop_attr_;

  std::vector<int> size_;
  kernels::ResamplingParams2D shared_params_;
  std::vector<kernels::ResamplingParams2D> resample_params_;
  std::vector<CropWindow> crops_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_RESIZE_RANDOM_RESIZED_CROP_H_
