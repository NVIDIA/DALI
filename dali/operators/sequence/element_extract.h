// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <type_traits>
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
                              const std::vector<int>& element_map) {
    assert(tensor_shape.size() > 0);
    auto N_input = tensor_shape[0];

    for (auto elem : element_map)
      DALI_ENFORCE(
          elem < N_input,
          make_string(
              "Index `", elem,
              "` from `element_map` is out of bounds for sample with sequence length equal `",
              N_input, "`."));
  }

  static TensorListShape<> GetOutputShape(const TensorListShape<> &input_shape,
                                          const std::vector<int> &element_map,
                                          const TensorLayout& input_layout) {
    if (!input_layout.empty()) {
      DALI_ENFORCE(
          VideoLayoutInfo::IsSequence(input_layout),
          make_string("Input layout must describe a sequence - it must start with 'F', got '",
                      input_layout, "' instead."));
    }

    DALI_ENFORCE(input_shape.sample_dim() > 1,
                 "Input must have at least two dimensions - outermost for sequence and at least "
                 "one for data elements.");

    for (int i = 0; i < input_shape.num_samples(); ++i) {
      auto shape = input_shape.tensor_shape_span(i);
      CheckInputShape(shape, element_map);
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

    DALI_ENFORCE(!element_map_.empty(), "No `element_map` indicies provided");

    for (auto elem : element_map_) {
      DALI_ENFORCE(
          elem >= 0,
          make_string("Negative indices in `element_map` are not allowed, found: ", elem, "."));
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

  void RunImpl(workspace_t<Backend> &ws) override {
    auto &input = ws.template InputRef<Backend>(0);
    auto element_layout = VideoLayoutInfo::GetFrameLayout(input.GetLayout());
    int elements_per_sample = element_map_.size();
    auto data_type = input.type();
    for (int k = 0; k < elements_per_sample; k++) {
      int element = element_map_[k];
      auto &output = ws.template OutputRef<Backend>(k);
      for (unsigned int i = 0; i < input.ntensor(); i++) {
        auto tensor_shape = input.tensor_shape(i);
        auto element_size = volume(tensor_shape.begin() + 1, tensor_shape.end());
        auto input_offset_bytes = element * element_size * data_type.size();
        scatter_gather_.AddCopy(
            output.raw_mutable_tensor(i),
            static_cast<const uint8_t *>(input.raw_tensor(i)) + input_offset_bytes,
            element_size * data_type.size());
      }
      output.SetLayout(element_layout);
    }
    RunCopies(ws);
  }

  void RunCopies(workspace_t<Backend> &ws);

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;

 private:
  std::vector<int> element_map_;

  std::conditional_t<
      std::is_same<Backend, CPUBackend>::value,
      kernels::ScatterGatherCPU,
      kernels::ScatterGatherGPU> scatter_gather_;
  // 256 kB per block for GPU
  static constexpr size_t kMaxSizePerBlock =
      std::is_same<Backend, CPUBackend>::value ? kernels::ScatterGatherCPU::kAnyBlockSize : 1 << 18;
};

}  // namespace dali

#endif  // DALI_OPERATORS_SEQUENCE_ELEMENT_EXTRACT_H_
