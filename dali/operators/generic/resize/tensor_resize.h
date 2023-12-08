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
#include "dali/core/expand_dims.h"
#include "dali/operators/image/resize/resize_base.h"
#include "dali/operators/image/resize/tensor_resize_attr.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {
namespace tensor_resize {


template <typename Backend>
class TensorResize : public StatelessOperator<Backend>
                   , protected ResizeBase<Backend> {
 public:
  explicit TensorResize(const OpSpec &spec);

 protected:
  int NumSpatialDims() const { return spatial_ndim_; }
  int FirstSpatialDim() const { return first_spatial_dim_; }

  bool CanInferOutputs() const override { return true; }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;

  void RunImpl(Workspace &ws) override;

  void TrimSpatialDims(const TensorListShape<> &input_shape, int min_ndim) {
    int nsamples = input_shape.num_samples();
    auto unchanged_dim = [&](int d) {
      for (int i = 0; i < nsamples; i++) {
        int64_t extent = input_shape.tensor_shape_span(i)[d];
        if (static_cast<int64_t>(resize_params_[i].dst_size[d]) != extent ||
            static_cast<int64_t>(resize_params_[i].src_lo[d]) != 0 ||
            static_cast<int64_t>(resize_params_[i].src_hi[d]) != extent) {
          return false;
        }
      }
      return true;
    };

    // at least min_ndim should remain
    int ndim_to_trim = std::max(0, spatial_ndim_ - min_ndim);
    int new_first_spatial_dim = first_spatial_dim_;
    int new_end_spatial_dim = first_spatial_dim_ + spatial_ndim_;

    for (int d = first_spatial_dim_; d < first_spatial_dim_ + spatial_ndim_; d++) {
      if (ndim_to_trim == 0 || !unchanged_dim(d))
        break;
      new_first_spatial_dim++;
      ndim_to_trim--;
    }
    for (int d = new_end_spatial_dim - 1; d > new_first_spatial_dim; d--) {
      if (ndim_to_trim == 0 || !unchanged_dim(d))
        break;
      new_end_spatial_dim--;
      ndim_to_trim--;
    }

    int new_spatial_ndim = new_end_spatial_dim - new_first_spatial_dim;
    if (first_spatial_dim_ == new_first_spatial_dim && spatial_ndim_ == new_spatial_ndim)
      return;

    for (int s = 0; s < nsamples; s++) {
      auto &p = resize_params_[s];
      for (int d = 0; d < new_spatial_ndim ; d++) {
        int orig_d =  new_first_spatial_dim - first_spatial_dim_ + d;
        p.dst_size[d] = p.dst_size[orig_d];
        p.src_hi[d] = p.src_hi[orig_d];
        p.src_lo[d] = p.src_lo[orig_d];
      }
      p.dst_size.resize(new_spatial_ndim);
      p.src_hi.resize(new_spatial_ndim);
      p.src_lo.resize(new_spatial_ndim);
    }
    first_spatial_dim_ = new_first_spatial_dim;
    spatial_ndim_ = new_spatial_ndim;
  }

  void PrepareParams(const ArgumentWorkspace &ws, const TensorListShape<> &input_shape,
                     const TensorLayout &layout) {
    resize_attr_.PrepareResizeParams(spec_, ws, input_shape, layout);

    // 1st: Expand number of dimensions to at least 2D
    int min_spatial_ndim = 2;  // ResizeBase requires it.
    int orig_spatial_ndim = resize_attr_.NumSpatialDims();
    add_leading_spatial_ndim_ = std::max(0, min_spatial_ndim - orig_spatial_ndim);
    first_spatial_dim_ = 0;
    spatial_ndim_ = orig_spatial_ndim + add_leading_spatial_ndim_;
    expand_dims(expanded_input_shape_, input_shape, first_spatial_dim_, spatial_ndim_);
    // should use expanded_input_shape_ from now on

    auto params_view = resize_attr_.Params();
    resize_params_.clear();
    resize_params_.reserve(params_view.size());
    for (auto p : params_view) {
      for (int i = 0; i < add_leading_spatial_ndim_; i++) {
        p.dst_size.insert(p.dst_size.begin(), 1);
        p.src_lo.insert(p.src_lo.begin(), 0);
        p.src_hi.insert(p.src_hi.begin(), 1);
      }
      resize_params_.emplace_back(std::move(p));
    }

    // 2nd: Remove leading and trailing dimensions that are not resized,
    // while keeping at least 2D.
    TrimSpatialDims(expanded_input_shape_, min_spatial_ndim);

    if (NumSpatialDims() > 3) {
      throw std::invalid_argument(make_string(
          "TensorResize can not resize more than 3 consecutive spatial dimensions. "
          "The input may contain extra leading and trailing dimensions which are not resized.\n\n"
          "Example 1: Input shape (2, 2, 4, 3, 2), Output shape (2, 2, 5, 5, 2). Only the 3rd and "
          "4th dimensions are resized (2 spatial dimensions)\n\n"
          "Example 2: Input shape (2, 2, 4, 3, 2), Output shape (2, 3, 4, 5, 2). Only the 2nd and "
          "4th dimensions are resized, but we can only consider a block of consecutive dimensions "
          "so we include the 3rd dimension as well (3 spatial dimensions).\n\nGot: ",
          NumSpatialDims(), " spatial dimensions starting at dim ",
          FirstSpatialDim()));
    }
    assert(NumSpatialDims() >= 2 && NumSpatialDims() <= 3);
    assert(FirstSpatialDim() >= 0);
    int nsamples = expanded_input_shape_.num_samples();
    resample_params_.resize(nsamples * NumSpatialDims());
    resampling_attr_.PrepareFilterParams(spec_, ws, nsamples);
    resampling_attr_.GetResamplingParams(make_span(resample_params_),
                                         make_cspan(resize_params_));
  }

  void InitializeBackend();

  USE_OPERATOR_MEMBERS();
  std::vector<kernels::ResamplingParams> resample_params_;
  using Operator<Backend>::RunImpl;

  TensorResizeAttr resize_attr_;
  ResamplingFilterAttr resampling_attr_;

  TensorListShape<> expanded_input_shape_;  // expanded if needed
  int spatial_ndim_ = -1;
  int first_spatial_dim_ = -1;
  int add_leading_spatial_ndim_ = 0;
  std::vector<ResizeParams> resize_params_;
};

template <typename Backend>
TensorResize<Backend>::TensorResize(const OpSpec &spec)
    : StatelessOperator<Backend>(spec), ResizeBase<Backend>(spec), resize_attr_(spec) {
  InitializeBackend();
}

template <typename Backend>
bool TensorResize<Backend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                      const Workspace &ws) {
  output_desc.resize(1);
  auto &input = ws.Input<Backend>(0);
  auto in_type = input.type();
  PrepareParams(ws, input.shape(), input.GetLayout());
  int N = expanded_input_shape_.num_samples();
  auto out_type = resampling_attr_.GetOutputType(in_type);

  output_desc[0].type = out_type;
  this->SetupResize(output_desc[0].shape, out_type, expanded_input_shape_, in_type,
                    make_cspan(this->resample_params_), NumSpatialDims(), FirstSpatialDim());

  if (add_leading_spatial_ndim_ > 0) {
    using shape_blocks_t = SmallVector<std::pair<int, int>, 6>;
    auto groups_dim = shape_blocks_t{{0, add_leading_spatial_ndim_ + 1}};
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
