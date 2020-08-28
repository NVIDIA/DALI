// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_GENERIC_SLICE_SLICE_BASE_H_
#define DALI_OPERATORS_GENERIC_SLICE_SLICE_BASE_H_

#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include "dali/core/any.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/slice/slice_kernel_utils.h"
#include "dali/operators/generic/slice/out_of_bounds_policy.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/util/operator_impl_utils.h"
#include "dali/util/crop_window.h"

#define SLICE_TYPES (uint8_t, uint16_t, uint32_t, uint64_t, \
                     int8_t,  int16_t,  int32_t,  int64_t, \
                     float16, float, double)
#define SLICE_DIMS (1, 2, 3, 4)

namespace dali {

template <typename Backend>
class SliceBase : public Operator<Backend> {
 public:
  explicit inline SliceBase(const OpSpec &spec)
      : Operator<Backend>(spec),
        output_type_(spec.GetArgument<DALIDataType>("dtype")),
        out_of_bounds_policy_(GetOutOfBoundsPolicy(spec)) {
    if (out_of_bounds_policy_ == OutOfBoundsPolicy::Pad) {
      fill_values_ = spec.GetRepeatedArgument<float>("fill_values");
    }
  }

  template <typename OutputType, int Dims>
  void FillArgs(std::vector<kernels::SliceArgs<OutputType, Dims>>& slice_args,
                const workspace_t<Backend> &ws) {
    this->ProcessCroppingAttrs(ws);
    const auto &input = ws.template InputRef<Backend>(0);
    auto in_shape = input.shape();
    int nsamples = in_shape.num_samples();
    int ndim = in_shape.sample_dim();
    auto in_layout = input.GetLayout();
    if (in_layout.empty())
      in_layout = GetDefaultLayout(ndim);
    slice_args.clear();
    slice_args.reserve(nsamples);
    for (int i = 0; i < nsamples; i++) {
      auto crop_win_gen = this->GetCropWindowGenerator(i);
      assert(crop_win_gen);
      CropWindow win = crop_win_gen(in_shape[i], in_layout);
      ApplySliceBoundsPolicy(out_of_bounds_policy_, in_shape[i], win.anchor, win.shape);
      slice_args.emplace_back(
          ToSliceArgs<OutputType, Dims>(win, in_layout, make_cspan(fill_values_)));
    }
  }


 protected:
   /**
   * @brief Implementation specific (Crop, Slice, ...)
   */
  virtual void ProcessCroppingAttrs(const workspace_t<Backend> &ws) = 0;
  virtual const CropWindowGenerator& GetCropWindowGenerator(std::size_t data_idx) const = 0;

  bool CanInferOutputs() const override {
    return true;
  }
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override;
  void RunImpl(workspace_t<Backend> &ws) override;

  std::vector<float> fill_values_;
  DALIDataType input_type_ = DALI_NO_TYPE;
  DALIDataType output_type_ = DALI_NO_TYPE;
  int ndim_ = -1;
  OutOfBoundsPolicy out_of_bounds_policy_ = OutOfBoundsPolicy::Error;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;

 private:
  template <typename OutputType, int Dims>
  kernels::SliceArgs<OutputType, Dims> ToSliceArgs(const CropWindow &win,
                                                   const TensorLayout &layout,
                                                   span<const float> fill_values) {
    kernels::SliceArgs<OutputType, Dims> args;
    auto channel_dim = layout.find('C');
    args.anchor = win.anchor.to_static<Dims>();
    args.shape = win.shape.to_static<Dims>();
    args.channel_dim = -1;
    if (!fill_values_.empty()) {
      args.fill_values.clear();
      for (auto val : fill_values_)
        args.fill_values.push_back(static_cast<OutputType>(val));
      if (fill_values_.size() > 1) {
        DALI_ENFORCE((channel_dim >= 0 && channel_dim < Dims),
                      "Multi-channel fill_values was provided but channel dimension could not be "
                      "found in layout");
        args.channel_dim = channel_dim;
      }
    }
    return args;
  }

  inline TensorLayout GetDefaultLayout(int ndims) {
    switch (ndims) {
      case 2:
        return "HW";
      case 3:
        return "HWC";
      case 4:
        return "DHWC";
      default:
        return "";
    }
  }

  std::unique_ptr<OpImplBase<Backend>> impl_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_SLICE_SLICE_BASE_H_
