// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_SEQUENCE_SEQUENCE_REARRANGE_H_
#define DALI_OPERATORS_SEQUENCE_SEQUENCE_REARRANGE_H_

#include <tuple>
#include <vector>

#include "dali/core/format.h"
#include "dali/core/tensor_layout.h"
#include "dali/core/tensor_view.h"
#include "dali/kernels/common/scatter_gather.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

inline int64_t GetSeqLength(const TensorShape<>& seq_shape) {
  return seq_shape[0];
}

void ValidateSeqRearrange(const TensorShape<>& in_sample_shape,
                          const TensorView<StorageCPU, const int, 1>& new_order, int sample_idx);

TensorShape<> GetSeqRearrangedShape(const TensorShape<>& in_sample_shape,
                                    const TensorView<StorageCPU, const int, 1>& new_order);

struct copy_desc {
  const void* from;
  void* to;
  size_t size;
};

copy_desc GetCopyDesc(char* output_sample, const char* input_sample, int out_elem_idx,
                      int in_elem_idx, int64_t element_sizeof);

template <typename Backend>
class SequenceRearrange : public Operator<Backend> {
 public:
  inline explicit SequenceRearrange(const OpSpec& spec)
      : Operator<Backend>(spec), scatter_gather_(kMaxSizePerBlock) {
    if (spec.HasArgument("new_order")) {
      new_order_ = spec.GetRepeatedArgument<int>("new_order");
      DALI_ENFORCE(!new_order_.empty(), "Empty result sequences are not allowed.");
      single_order_ = true;
    }
  }

  DISABLE_COPY_MOVE_ASSIGN(SequenceRearrange);

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc>& output_desc, const workspace_t<Backend>& ws) override {
    const auto& input = ws.template InputRef<Backend>(0);
    const auto& in_shape = input.shape();  // temporary in some cases
    DALI_ENFORCE(in_shape.sample_dim() > 1, "Sequence elements must have at least 1 dim");
    output_desc.resize(1);
    output_desc[0].type = input.type();
    output_desc[0].shape = TensorListShape<>(in_shape.num_samples(), in_shape.sample_dim());
    auto curr_batch_size = ws.GetInputBatchSize(0);
    if (single_order_) {
      TensorView<StorageCPU, const int, 1> new_order(new_order_.data(),
                                                     TensorShape<1>(new_order_.size()));
      for (int i = 0; i < curr_batch_size; i++) {
        ValidateSeqRearrange(in_shape[i], new_order, i);
        output_desc[0].shape.set_tensor_shape(i, GetSeqRearrangedShape(in_shape[i], new_order));
      }
    } else {
      const auto& new_orders = ws.ArgumentInput("new_order");
      for (int i = 0; i < curr_batch_size; i++) {
        auto new_order = view<const int, 1>(new_orders[i]);
        ValidateSeqRearrange(in_shape[i], new_order, i);
        output_desc[0].shape.set_tensor_shape(i, GetSeqRearrangedShape(in_shape[i], new_order));
      }
    }

    auto layout = input.GetLayout();
    DALI_ENFORCE(layout.empty() || layout.find('F') == 0,
                 make_string("Expected sequence as the input, where outermost dimension represents "
                             "frames dimension `F`, got data with layout = \"",
                             layout, "\"."));

    return true;
  }

  void RunImpl(workspace_t<Backend>& ws) override;

 private:
  USE_OPERATOR_MEMBERS();
  bool single_order_ = false;
  std::vector<int> new_order_;
  kernels::ScatterGatherGPU scatter_gather_;
  static constexpr size_t kMaxSizePerBlock = 1 << 18;  // 256 kB per block
};

}  // namespace dali

#endif  // DALI_OPERATORS_SEQUENCE_SEQUENCE_REARRANGE_H_
