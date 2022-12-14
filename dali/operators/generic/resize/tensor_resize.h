// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

    auto unchanged_dim = [&](int d) {
      for (int i = 0; i < nsamples; i++) {
        int64_t extent = input_shape.tensor_shape_span(i)[d];
        if (static_cast<int64_t>(resize_attr_.params_[i].dst_size[d]) != extent ||
            static_cast<int64_t>(resize_attr_.params_[i].src_lo[d]) != 0 ||
            static_cast<int64_t>(resize_attr_.params_[i].src_hi[d]) != extent) {
          return false;
        }
      }
      return true;
    };

    for (int d = 0; d < resize_attr_.ndim_; d++) {
      if (!unchanged_dim(d))
        break;
      resize_attr_.first_spatial_dim_++;
      resize_attr_.spatial_ndim_--;
    }
    for (int d = resize_attr_.ndim_ - 1; d >= 0; d--) {
      if (!unchanged_dim(d))
        break;
      resize_attr_.spatial_ndim_--;
    }

    int spatial_ndim = resize_attr_.spatial_ndim_;
    for (int s = 0; s < nsamples; s++) {
      for (int d = 0; d < spatial_ndim ; d++) {
        int orig_d = resize_attr_.first_spatial_dim_ + d;
        resize_attr_.params_[s].dst_size[d] = resize_attr_.params_[s].dst_size[orig_d];
        resize_attr_.params_[s].src_hi[d] = resize_attr_.params_[s].src_hi[orig_d];
        resize_attr_.params_[s].src_lo[d] = resize_attr_.params_[s].src_lo[orig_d];
      }
      resize_attr_.params_[s].dst_size.resize(spatial_ndim);
      resize_attr_.params_[s].src_hi.resize(spatial_ndim);
      resize_attr_.params_[s].src_lo.resize(spatial_ndim);
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

    assert(NumSpatialDims() >= 1 && NumSpatialDims() <= 3);
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
  resample_params_.resize(num_threads_);
  InitializeBackend();
}

template <typename Backend>
bool TensorResize<Backend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                      const Workspace &ws) {
  output_desc.resize(1);
  auto &input = ws.Input<Backend>(0);

  const auto &in_shape = input.shape();
  auto in_type = input.type();
  int N = in_shape.num_samples();

  PrepareParams(ws, in_shape);

  auto out_type = resampling_attr_.GetOutputType(in_type);

  output_desc[0].type = out_type;
  this->SetupResize(output_desc[0].shape, out_type, in_shape, in_type,
                    make_cspan(this->resample_params_), NumSpatialDims(), FirstSpatialDim());
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
