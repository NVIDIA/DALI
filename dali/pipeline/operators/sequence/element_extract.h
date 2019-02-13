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


#ifndef DALI_PIPELINE_OPERATORS_SEQUENCE_ELEMENT_EXTRACT_H_
#define DALI_PIPELINE_OPERATORS_SEQUENCE_ELEMENT_EXTRACT_H_

#include <vector>
#include "dali/common.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

namespace detail {
  static void CheckInputShape(const Dims& tensor_shape,
                              const std::vector<int>& element_map) {
    DALI_ENFORCE(tensor_shape.size() > 0);
    auto N_input = tensor_shape[0];

    int N_output = element_map.size();
    DALI_ENFORCE(N_input >= N_output,
        "Requested more elements than available");

    for (auto elem : element_map)
        DALI_ENFORCE(elem < N_input,
            "index " + std::to_string(elem) + " out of bounds");
  }
}  // namespace detail

template <typename Backend>
class ElementExtract : public Operator<Backend> {
 public:
  inline explicit ElementExtract(const OpSpec &spec)
    : Operator<Backend>(spec) {
    element_map_ = spec.GetRepeatedArgument<int>("element_map");

    DALI_ENFORCE(!element_map_.empty(),
        "No 'element_map' indexes provided");
  }

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;

  USE_OPERATOR_MEMBERS();

 private:
  std::vector<int> element_map_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_SEQUENCE_ELEMENT_EXTRACT_H_
