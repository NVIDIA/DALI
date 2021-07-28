// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_GENERIC_SLICE_SUBSCRIPT_H_
#define DALI_OPERATORS_GENERIC_SLICE_SUBSCRIPT_H_

#include <string>
#include <utility>
#include <vector>

#include "dali/core/any.h"
#include "dali/core/convert.h"
#include "dali/core/static_switch.h"
#include "dali/core/util.h"
#include "dali/core/math_util.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/slice/slice_kernel_utils.h"

namespace dali {

constexpr const int kMaxSubscripts = 32;

#define INTEGER_TYPES int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t

enum class ArgSource {
  None = 0,
  Value = 1,
  Input = 2,
};

struct SubscriptArg {
  void Init(const OpSpec &spec, const std::string &name) {
    this->name = name;
    src = spec.HasTensorArgument(name)
              ? ArgSource::Input
              : spec.HasArgument(name)
                  ? ArgSource::Value
                  : ArgSource::None;
    if (src == ArgSource::Value)
      scalar_value = spec.GetArgument<int64_t>(name);
  }

  bool IsDefined() const {
    return src != ArgSource::None;
  }
  string name;
  ArgSource src = ArgSource::None;
  vector<int64_t> values;
  int64_t scalar_value = 0;

  void Load(const ArgumentWorkspace &ws, int batch_size) {
    if (src == ArgSource::Input) {
      const TensorVector<CPUBackend> &inp = ws.ArgumentInput(name);
      const auto &shape = inp.shape();

      int n = batch_size;
      DALI_ENFORCE(shape.num_samples() == n, make_string(
          "Unexpected number of samples in argument `", name, "`. Got: ",
          shape.num_samples(), ", expected ", n, "."));

      DALI_ENFORCE(shape.sample_dim() == 0,
        make_string("Array indices must be scalar (0D). Got ", shape.sample_dim(), "D tensor."));

      values.resize(n);
      DALIDataType type_id = inp.type().id();
      TYPE_SWITCH(type_id, type2id, T, (INTEGER_TYPES), (
        for (int i = 0; i < n; i++) {
          // TODO(michalz): Add tensor<T> and mutable_tensor<T> to TensorVector?
          values[i] = ConvertSat<int64_t>(static_cast<const T*>(inp.raw_tensor(i))[0]);
        }
      ), DALI_FAIL(make_string("Array index must be of integral type. Got: ", type_id)));  // NOLINT
    } else if (src == ArgSource::Value) {
      values.resize(batch_size, scalar_value);
    }
  }
};

struct SubscriptInfo {
  void Init(const OpSpec &spec, int i) {
    at.Init(spec, make_string("at_", i));
    lo.Init(spec, make_string("lo_", i));
    hi.Init(spec, make_string("hi_", i));
    step.Init(spec, make_string("step_", i));

    if (at.IsDefined()) {
      if (lo.IsDefined() || hi.IsDefined()) {
        DALI_FAIL(make_string("The subscriptg for dimension ", i,
                              " is specified both as an index and as a range."));
      }
      if (step.IsDefined()) {
        DALI_FAIL(make_string("The subscriptg for dimension ", i,
                              " is specified as an index - it cannot have a step."));
      }
    }
  }

  bool IsDefined() const {
    return at.IsDefined() || lo.IsDefined() || hi.IsDefined() || step.IsDefined();
  }

  bool IsIndex() const {
    return at.IsDefined();
  }

  bool IsRange() const {
    return !IsIndex() && IsDefined();
  }

  SubscriptArg at, lo, hi, step;

  void Load(const ArgumentWorkspace &ws, int batch_size) {
    at.Load(ws, batch_size);
    lo.Load(ws, batch_size);
    hi.Load(ws, batch_size);
    step.Load(ws, batch_size);
  }
};


template <typename Backend>
class TensorSubscript : public Operator<Backend> {
 public:
  explicit TensorSubscript(const OpSpec &spec) : Operator<Backend>(spec) {
    InitArgs();
  }

  USE_OPERATOR_MEMBERS();

  void InitArgs() {
    subscripts_.resize(kMaxSubscripts);
    int num_subscripts = 0;
    for (int i = 0; i < kMaxSubscripts; i++) {
      SubscriptInfo &s = subscripts_[i];
      s.Init(spec_, i);


      if (s.IsDefined()) {
        num_subscripts = i + 1;
      }
    }
    subscripts_.resize(num_subscripts);
  }

  bool SetupImpl(vector<OutputDesc> &outputs, const workspace_t<Backend> &ws) override {
    const auto &input = ws.template InputRef<Backend>(0);
    DALI_ENFORCE(input.sample_dim() > 0, "Cannot apply an index to a scalar.");
    const auto &input_shape = input.shape();
    outputs.resize(1);
    outputs[0].type = input.type();
    GetRanges(ws, input_shape);
    ProcessShapes(outputs[0].shape, input_shape);
    return true;
  }

  void GetRanges(const workspace_t<Backend> &ws, const TensorListShape<> &in_shape) {
    int nsub = subscripts_.size();
    int ndim = in_shape.sample_dim();
    DALI_ENFORCE(ndim >= nsub, "Too many indices.");
    int nsamples = in_shape.num_samples();
    lo_.resize(nsamples, ndim);
    hi_ = in_shape;
    shape_ = in_shape;

    for (int d = 0; d < nsub; d++) {
      SubscriptInfo &s = subscripts_[d];
      s.Load(ws, nsamples);
      GetRange(s, d, in_shape);
    }
  }

  void GetRange(SubscriptInfo &s, int d, const TensorListShape<> &in_shape) {
    int nsamples = in_shape.num_samples();

    for (int64_t step : s.step.values) {
      DALI_ENFORCE(step == 1, "Strided indexing is not implemted");
    }

    if (s.IsIndex()) {
      for (int i = 0; i < nsamples; i++) {
        int64_t in_extent = in_shape.tensor_shape_span(i)[d];
        int64_t at = s.at.values[i];
        int64_t idx = at < 0 ? in_extent + at : at;
        DALI_ENFORCE(idx >= 0 && idx < in_extent, make_string("Tensor index out of range!\n"
          "Index ", idx, " at axis ", d, " is out of range for a tensor of shape ", in_shape[i],
          ". Detected while processing sample #", idx));
        lo_.tensor_shape_span(i)[d] = idx;
        hi_.tensor_shape_span(i)[d] = idx + 1;
        shape_.tensor_shape_span(i)[d] = 1;
      }
    }
    if (s.IsRange()) {
      for (int i = 0; i < nsamples; i++) {
        int64_t in_extent = in_shape.tensor_shape_span(i)[d];
        int64_t lo = s.lo.IsDefined() ? s.lo.values[i] : 0;
        int64_t hi = s.hi.IsDefined() ? s.hi.values[i] : in_extent;
        if (lo < 0) lo += in_extent;
        if (hi < 0) hi += in_extent;
        lo = clamp(lo, 0_i64, in_extent);
        hi = clamp(hi, 0_i64, in_extent);
        lo_.tensor_shape_span(i)[d] = lo;
        hi_.tensor_shape_span(i)[d] = hi;
        shape_.tensor_shape_span(i)[d] = abs(hi - lo);
      }
    }
  }

  void ProcessShapes(TensorListShape<> &out_shape, const TensorListShape<> &in_shape) {
    int in_dims = in_shape.sample_dim();
    int nsub = subscripts_.size();

    out_dim_map_.clear();
    dim_groups_.clear();

    int group_start = 0;
    int d = 0;
    for (; d < nsub; d++) {
      if (subscripts_[d].IsDefined()) {
        if (d != group_start)
          dim_groups_.push_back({ group_start, d - group_start });
        group_start = d + 1;
      }

      if (!subscripts_[d].IsIndex()) {  // indices are not present in the output
        out_dim_map_.push_back(d);
      }
    }
    for (; d < in_dims; d++) {
      out_dim_map_.push_back(d);
    }

    if (in_dims != group_start)
      dim_groups_.push_back({ group_start, in_dims - group_start });

    collapse_dims(simplified_in_shape_, in_shape, dim_groups_);
    collapse_dims(simplified_out_shape_, shape_, dim_groups_);
    collapse_dims(simplified_anchor_, lo_, dim_groups_);

    out_shape.resize(in_shape.num_samples(), out_dim_map_.size());
    for (int i = 0; i < out_shape.num_samples(); i++) {
      auto out_sample = out_shape.tensor_shape_span(i);
      auto sh = shape_.tensor_shape_span(i);
      for (int d = 0; d < out_shape.sample_dim(); d++) {
        out_sample[d] = sh[out_dim_map_[d]];
      }
    }
  }

  TensorLayout GetOutputLayout(const TensorLayout &input_layout) const {
    if (input_layout.empty())
      return {};
    TensorLayout out_layout;
    out_layout.resize(out_dim_map_.size());
    for (int i = 0; i < out_layout.ndim(); i++)
      out_layout[i] = input_layout[out_dim_map_[i]];
    return out_layout;
  }

  bool CanInferOutputs() const override { return true; }

  using Operator<Backend>::RunImpl;
  void RunImpl(workspace_t<Backend> &ws) override {
    const auto &input = ws.template InputRef<Backend>(0);
    auto &output = ws.template OutputRef<Backend>(0);
    output.SetLayout(GetOutputLayout(input.GetLayout()));
    VALUE_SWITCH(simplified_in_shape_.sample_dim(), ndim,
      (1,  2,  3,  4,  5,  6/*,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16*/),
      (VALUE_SWITCH(input.type().size(), element_size, (1, 2, 4, 8),
        (RunTyped<ndim, element_size>(ws);),
        (DALI_FAIL(make_string("Unsupported input type: ", input.type().id()));))),
      (DALI_FAIL("Subscript too complex.\n"
        "The subscript operator supports up to 32 total and up to  16 non-collapsible dimensions.\n"
        "Adjacent dimensions from which no index or slice is taken can be collapsed.");)
    );  // NOLINT
  }

 private:
  template <int ndim, int element_size>
  void RunTyped(workspace_t<Backend> &ws);

  vector<SubscriptInfo> subscripts_;

  // Ranges and output shapes in input space - that is, not including
  // the dimensions which are removed by indexing or ones collapsed as a result of simplification.
  TensorListShape<> lo_, hi_, shape_;

  // Grouping of indices, used for simplification
  SmallVector<std::pair<int, int>, 6> dim_groups_;
  // Input shape where adjacent dimensions are collapsed where there's no indexing done
  TensorListShape<> simplified_in_shape_;
  // Output anchor & shape simplified in the same way as input shape
  TensorListShape<> simplified_anchor_, simplified_out_shape_;

  // Mapping from output to input indices
  SmallVector<int, 6> out_dim_map_;

  kernels::KernelManager kmgr_;
  any ctx_;
};


}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_SLICE_SUBSCRIPT_H_
