// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_GENERIC_CONSTANT_VALUE_H_
#define DALI_OPERATORS_GENERIC_CONSTANT_VALUE_H_

#include <vector>
#include "dali/core/static_switch.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/core/float16.h"

#define DALI_CONSTANT_VALUE_TYPES                                                           \
  uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float, float16, \
      double, bool

namespace dali {

template <typename Backend>
class ConstantValue : public StatelessOperator<Backend> {
 public:
  explicit ConstantValue(const OpSpec &spec, bool has_fill_value = false,
                         bool is_shape_like = false)
      : StatelessOperator<Backend>(spec),
        has_fill_value_(has_fill_value),
        is_shape_like_(is_shape_like),
        has_shape_(spec.ArgumentDefined("shape")),
        has_dtype_(spec.ArgumentDefined("dtype")) {
    dtype_ = has_dtype_ ? spec.GetArgument<DALIDataType>("dtype") : DALI_INT32;
  }

  int GetBatchSize(const Workspace &ws) const {
    if (is_shape_like_)
      return ws.GetInputBatchSize(shape_like_input_idx_);
    else
      return ws.GetRequestedBatchSize(0);
  }


  bool CanBroadcastShapes(span<int64_t> shape1, span<int64_t> shape2) {
    size_t len1 = shape1.size();
    size_t len2 = shape2.size();
    size_t max_len = std::max(len1, len2);
    for (size_t i = 0; i < max_len; ++i) {
      // Get the dimensions from each shape, defaulting to 1 if out of bounds
      int dim1 = (i < len1) ? shape1[len1 - 1 - i] : 1;
      int dim2 = (i < len2) ? shape2[len2 - 1 - i] : 1;
      // Check if the dimensions are compatible
      if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
        return false;
      }
    }
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    int nsamples = GetBatchSize(ws);
    output_desc.resize(1);
    auto &dtype = output_desc[0].type;
    auto &shape = output_desc[0].shape;
    dtype = is_shape_like_ && !has_dtype_ ? ws.GetInputDataType(shape_like_input_idx_) : dtype_;

    if (is_shape_like_) {
      shape = ws.GetInputShape(shape_like_input_idx_);
    } else if (has_shape_) {
      GetShapeArgument(shape, spec_, "shape", ws, nsamples);
    } else {
      shape = uniform_list_shape(nsamples, TensorShape<0>{});
    }

    if (has_fill_value_) {
      auto& fill_value = ws.Input<Backend>(value_input_idx_);
      auto fill_value_shape = fill_value.shape();
      auto fill_value_dtype = fill_value.type();
      int new_ndim = shape.sample_dim() + fill_value_shape.sample_dim();
      for (int i = 0; i < nsamples; i++) {
        auto orig_shape = shape.tensor_shape_span(i);
        auto fill_value_sh = fill_value_shape.tensor_shape_span(i);
        if (!CanBroadcastShapes(orig_shape, fill_value_sh)) {
          DALI_FAIL(make_string("Shapes ", shape.tensor_shape(i), " and ",
                                fill_value_shape.tensor_shape(i), " can't be broadcast."));
        }
      }
      if (!has_dtype_ && !is_shape_like_) {
        dtype = fill_value_dtype;
      }
    }
    return true;
  }

  void SetConstValue(int value) {
    has_const_value_ = true;
    const_value_ = value;
  }

  void RunImpl(Workspace &ws) override;

 protected:
  using Operator<Backend>::spec_;
  using Operator<Backend>::max_batch_size_;
  bool has_fill_value_;
  bool is_shape_like_;
  bool has_shape_, has_dtype_;
  DALIDataType dtype_;

  bool has_const_value_ = false;
  int const_value_ = 0;

  int shape_like_input_idx_ = is_shape_like_ ? 0 : -1;
  int value_input_idx_ = is_shape_like_ ? 1 : 0;
};

template <typename Backend>
class Full : public ConstantValue<Backend> {
 public:
  explicit Full(const OpSpec &spec): ConstantValue<Backend>(spec, true, false) {
  }
};

template <typename Backend>
class FullLike : public ConstantValue<Backend> {
 public:
  explicit FullLike(const OpSpec &spec): ConstantValue<Backend>(spec, true, true) {
  }
};

template <typename Backend>
class Zeros : public ConstantValue<Backend> {
 public:
  explicit Zeros(const OpSpec &spec): ConstantValue<Backend>(spec, false, false) {
    ConstantValue<Backend>::SetConstValue(0);
  }
};

template <typename Backend>
class ZerosLike : public ConstantValue<Backend> {
 public:
  explicit ZerosLike(const OpSpec &spec): ConstantValue<Backend>(spec, false, true) {
    ConstantValue<Backend>::SetConstValue(0);
  }
};

template <typename Backend>
class Ones : public ConstantValue<Backend> {
 public:
  explicit Ones(const OpSpec &spec): ConstantValue<Backend>(spec, false, false) {
    ConstantValue<Backend>::SetConstValue(1);
  }
};

template <typename Backend>
class OnesLike : public ConstantValue<Backend> {
 public:
  explicit OnesLike(const OpSpec &spec): ConstantValue<Backend>(spec, false, true) {
    ConstantValue<Backend>::SetConstValue(1);
  }
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_CONSTANT_VALUE_H_
