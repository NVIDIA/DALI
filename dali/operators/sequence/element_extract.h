// Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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


#ifndef DALI_OPERATORS_SEQUENCE_ELEMENT_EXTRACT_H_
#define DALI_OPERATORS_SEQUENCE_ELEMENT_EXTRACT_H_

#include <vector>
#include "dali/core/common.h"
#include "dali/core/format.h"
#include "dali/core/tensor_layout.h"
#include "dali/core/tensor_shape.h"
#include "dali/kernels/common/scatter_gather.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

namespace detail {
  static void CheckInputShape(const span<const int64_t>& tensor_shape,
                              const std::vector<int>& element_map,
                              const TensorLayout& input_layout) {
    if (!input_layout.empty()) {
      DALI_ENFORCE(
          VideoLayoutInfo::IsSequence(input_layout),
          make_string("Input layout must describe a sequence - it must start with 'F', got '",
                      input_layout, "' instead."));
    }

    DALI_ENFORCE(tensor_shape.size() > 1,
                 "Input must have at least two dimenstions - outermost for sequence and at least "
                 "one for data elements.");
    auto N_input = tensor_shape[0];

    for (auto elem : element_map)
        DALI_ENFORCE(elem < N_input,
            "index " + std::to_string(elem) + " out of bounds");
  }

  static TensorListShape<> GetOutputShape(const TensorListShape<> &input_shape,
                                          const std::vector<int> &element_map,
                                          const TensorLayout& input_layout) {
    for (int i = 0; i < input_shape.num_samples(); ++i) {
      auto shape = input_shape.tensor_shape_span(i);
      CheckInputShape(shape, element_map, input_layout);
    }
    return input_shape.last(input_shape.sample_dim() - 1);
  }

}  // namespace detail

template <typename Backend>
class ElementExtract : public Operator<Backend> {
 public:
  inline explicit ElementExtract(const OpSpec &spec)
    : Operator<Backend>(spec), scatter_gather_(kMaxSizePerBlock) {
    element_map_ = spec.GetRepeatedArgument<int>("element_map");

    DALI_ENFORCE(!element_map_.empty(),
        "No 'element_map' indexes provided");

    for (auto elem : element_map_) {
        DALI_ENFORCE(elem >= 0,
            "index " + std::to_string(elem) + " out of bounds.");
    }
  }

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    const auto &input = ws.template InputRef<Backend>(0);
    output_desc.resize(element_map_.size());
    auto output_shape = detail::GetOutputShape(input.shape(), element_map_, input.GetLayout());
    for (auto &desc : output_desc) {
      desc.shape = output_shape;
      desc.type = input.type();
    }
    return true;
  }

  void RunImpl(workspace_t<Backend> &ws) override;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;

 private:
  std::vector<int> element_map_;
  kernels::ScatterGatherGPU scatter_gather_;
  static constexpr size_t kMaxSizePerBlock = 1 << 18;  // 256 kB per block
};

}  // namespace dali

#endif  // DALI_OPERATORS_SEQUENCE_ELEMENT_EXTRACT_H_
