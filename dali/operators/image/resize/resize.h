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
             , protected ResizeAttr
             , protected ResizeBase<Backend> {
 public:
  explicit Resize(const OpSpec &spec);

 protected:
  bool CanInferOutputs() const override { return true; }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override;

  void RunImpl(workspace_t<Backend> &ws) override;

  void SaveAttrs(const TensorListView<StorageCPU, int, 1> &shape_data,
                 const TensorListShape<> &orig_shape) const {
    int N = orig_shape.num_samples();
    for (int i = 0; i < N; i++) {
      auto sample_shape = orig_shape.tensor_shape_span(i);
      int *out_shape = shape_data.data[i];
      for (int d = 0; d < spatial_ndim_; d++) {
        out_shape[d] = sample_shape[first_spatial_dim_ + d];
      }
    }
  }

  USE_OPERATOR_MEMBERS();
  std::vector<kernels::ResamplingParams2D> resample_params_;
  TensorList<CPUBackend> attr_staging_;
  using Operator<Backend>::RunImpl;
  bool save_attrs_ = false;
  DALIDataType out_type_ = DALI_NO_TYPE;

  int spatial_ndim_ = 2;
  int first_spatial_dim_ = 0;
};

template <typename Backend>
bool Resize<Backend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                const workspace_t<Backend> &ws) {
  output_desc.resize(save_attrs_ ? 2 : 1);
  auto &input = ws.template InputRef<Backend>(0);
  if (!spec_.TryGetArgument(out_type_, "dtype")) {
    out_type_ = input.type().id();
  }

  output_desc[0].type = TypeTable::GetTypeInfo(out_type_);

  int spatial_ndim = 2, first_spatial_dim = 0;
  ResizeAttr::ParseLayout(spatial_ndim, first_spatial_dim, input.GetLayout());

  DALI_ENFORCE(ws.NumOutput() == 1 || ws.NumOutput() == 2,
    "Resize and produce 1 or 2 outputs - if there are two outputs, the 2nd one receives the "
    "original size of the images.");

  const auto &in_shape = input.shape();
  auto in_type = input.type().id();
  this->SetupResize(output_desc[0].shape, out_type_, in_shape, in_type, make_cspan(this->params_),
                    spatial_ndim, first_spatial_dim);

  if (save_attrs_) {
    ImageLayoutInfo::NumSpatialDims(input.GetLayout());
    output_desc[1].type = TypeTable::GetTypeInfo(DALI_INT32);
  }
  return true;
}


}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_RESIZE_RESIZE_H_
