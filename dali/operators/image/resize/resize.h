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

#ifndef DALI_OPERATORS_IMAGE_RESIZE_RESIZE_H_
#define DALI_OPERATORS_IMAGE_RESIZE_RESIZE_H_

#include <random>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/pipeline/operator/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/operators/image/resize/resize_crop_mirror.h"
#include "dali/operators/image/resize/resize_base.h"
#include "dali/operators/image/resize/resize_attr.h"
#include "dali/kernels/context.h"
#include "dali/kernels/scratch.h"
#include "dali/kernels/imgproc/resample/params.h"

namespace dali {
namespace detail {
  kernels::ResamplingParams2D GetResamplingParams(
    const TransformMeta &meta, kernels::FilterDesc min_filter, kernels::FilterDesc mag_filter);
}  // namespace detail

template <typename Backend>
class Resize : public Operator<Backend>
             , protected ResizeBase<Backend> {
 public:
  explicit Resize(const OpSpec &spec);

 protected:
  int NumSpatialDims() const { return resize_attr_.spatial_ndim_; }
  int FirstSpatialDim() const { return resize_attr_.first_spatial_dim_; }

  bool CanInferOutputs() const override { return true; }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override;

  void RunImpl(workspace_t<Backend> &ws) override;

  void SaveAttrs(const TensorListView<StorageCPU, int, 1> &shape_data,
                 const TensorListShape<> &orig_shape) const {
    int N = orig_shape.num_samples();
    int D = NumSpatialDims();
    assert(shape_data.sample_dim() == 1);
    for (int i = 0; i < N; i++) {
      auto sample_shape = orig_shape.tensor_shape_span(i);
      assert(static_cast<int>(shape_data.shape[i][0]) == D);
      int *out_shape = shape_data.data[i];
      for (int d = 0; d < D; d++) {
        out_shape[d] = sample_shape[FirstSpatialDim() + d];
      }
    }
  }

  void PrepareParams(const ArgumentWorkspace &ws, const TensorListShape<> &input_shape,
                     const TensorLayout &layout) {
    resize_attr_.PrepareResizeParams(spec_, ws, input_shape, layout);
    assert(NumSpatialDims() >= 1 && NumSpatialDims() <= 3);
    assert(FirstSpatialDim() >= 0);
    int N = input_shape.num_samples();
    resample_params_.resize(N * NumSpatialDims());
    resampling_attr_.PrepareFilterParams(spec_, ws, N);
    resampling_attr_.GetResamplingParams(make_span(resample_params_),
                                         make_cspan(resize_attr_.params_));
  }

  void InitializeBackend();

  USE_OPERATOR_MEMBERS();
  std::vector<kernels::ResamplingParams> resample_params_;
  TensorList<CPUBackend> attr_staging_;
  using Operator<Backend>::RunImpl;
  bool save_attrs_ = false;

  ResizeAttr resize_attr_;
  ResamplingFilterAttr resampling_attr_;
};

template <typename Backend>
bool Resize<Backend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                const workspace_t<Backend> &ws) {
  output_desc.resize(save_attrs_ ? 2 : 1);
  auto &input = ws.template InputRef<Backend>(0);

  const auto &in_shape = input.shape();
  auto in_type = input.type().id();
  auto in_layout = input.GetLayout();
  int N = in_shape.num_samples();

  PrepareParams(ws, in_shape, in_layout);

  auto out_type = resampling_attr_.GetOutputType(in_type);

  output_desc[0].type = TypeTable::GetTypeInfo(out_type);
  this->SetupResize(output_desc[0].shape, out_type, in_shape, in_type,
                    make_cspan(this->resample_params_), NumSpatialDims(), FirstSpatialDim());

  if (save_attrs_) {
    output_desc[1].shape = uniform_list_shape(N, TensorShape<1>({ NumSpatialDims() }));
    output_desc[1].type = TypeTable::GetTypeInfo(DALI_INT32);
  }
  return true;
}


}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_RESIZE_RESIZE_H_
