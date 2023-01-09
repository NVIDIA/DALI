// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_GENERIC_RESIZE_TENSOR_RESIZE_H_
#define DALI_OPERATORS_GENERIC_RESIZE_TENSOR_RESIZE_H_

#include <cassert>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "dali/operators/image/resize/resize_base.h"
#include "dali/operators/image/resize/tensor_resize_attr.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {
namespace tensor_resize {


template <typename Backend>
class TensorResize : public Operator<Backend>
                   , protected ResizeBase<Backend> {
 public:
  explicit TensorResize(const OpSpec &spec);

 protected:
  int NumSpatialDims() const { return resize_attr_.spatial_ndim_; }
  int FirstSpatialDim() const { return resize_attr_.first_spatial_dim_; }

  bool CanInferOutputs() const override { return true; }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;

  void RunImpl(Workspace &ws) override;

  void PrepareParams(const ArgumentWorkspace &ws, const TensorListShape<> &input_shape) {
    int nsamples = input_shape.num_samples();
    resize_attr_.PrepareResizeParams(spec_, ws, input_shape);

    if (NumSpatialDims() < 2) {
      // PrepareResizeParams should expand to at least
      throw std::logic_error("1D resizing is not supported");
    }

    if (NumSpatialDims() < 1) {
      throw std::invalid_argument(
          "TensorResize expects at least one spatial dimension to be resized");
    }

    if (NumSpatialDims() > 3) {
      throw std::invalid_argument(make_string(
          "TensorResize can not resize more than 3 consecutive spatial dimensions. "
          "The input may contain extra leading and trailing dimensions which are not resized.\n\n"
          "Example 1: Input shape (2, 2, 4, 3, 2), Output shape (2, 2, 5, 5, 2). Only the 3rd and "
          "4th dimensions are resized (2 spatial dimensions)\n\n"
          "Example 2: Input shape (2, 2, 4, 3, 2), Output shape (2, 3, 4, 5, 2). Only the 2nd and "
          "4th dimensions are resized, but we can only consider a block of consecutive dimensions "
          "so we include the 3rd dimension as well (3 spatial dimensions).\n\nGot: ",
          resize_attr_.spatial_ndim_, " spatial dimensions starting at dim ",
          resize_attr_.first_spatial_dim_));
    }

    assert(NumSpatialDims() >= 2 && NumSpatialDims() <= 3);
    assert(FirstSpatialDim() >= 0);
    resample_params_.resize(nsamples * NumSpatialDims());
    resampling_attr_.PrepareFilterParams(spec_, ws, nsamples);
    resampling_attr_.GetResamplingParams(make_span(resample_params_),
                                         make_cspan(resize_attr_.params_));
  }

  void InitializeBackend();

  USE_OPERATOR_MEMBERS();
  std::vector<kernels::ResamplingParams> resample_params_;
  using Operator<Backend>::RunImpl;

  TensorResizeAttr resize_attr_;
  ResamplingFilterAttr resampling_attr_;
};

template <typename Backend>
TensorResize<Backend>::TensorResize(const OpSpec &spec)
    : Operator<Backend>(spec), ResizeBase<Backend>(spec), resize_attr_(spec) {
  InitializeBackend();
}

template <typename Backend>
bool TensorResize<Backend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                      const Workspace &ws) {
  output_desc.resize(1);
  auto &input = ws.Input<Backend>(0);
  auto in_type = input.type();

  PrepareParams(ws, input.shape());
  auto expanded_in_shape = resize_attr_.ExpandedInputShape();
  int leading_dummy_ndim = resize_attr_.LeadingDummyDims();
  int N = expanded_in_shape.num_samples();
  auto out_type = resampling_attr_.GetOutputType(in_type);

  output_desc[0].type = out_type;
  this->SetupResize(output_desc[0].shape, out_type, expanded_in_shape, in_type,
                    make_cspan(this->resample_params_), NumSpatialDims(), FirstSpatialDim());

  if (leading_dummy_ndim > 0) {
    using shape_blocks_t = SmallVector<std::pair<int, int>, 6>;
    auto groups_dim = shape_blocks_t{{0, leading_dummy_ndim + 1}};
    output_desc[0].shape = collapse_dims(output_desc[0].shape, make_cspan(groups_dim));
  }
  return true;
}


template <typename Backend>
void TensorResize<Backend>::RunImpl(Workspace &ws) {
  const auto &input = ws.Input<Backend>(0);
  auto &output = ws.Output<Backend>(0);
  this->RunResize(ws, output, input);
  output.SetLayout(input.GetLayout());
}

}  // namespace tensor_resize
}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_RESIZE_TENSOR_RESIZE_H_
