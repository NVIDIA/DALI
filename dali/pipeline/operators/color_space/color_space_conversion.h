// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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


#ifndef DALI_PIPELINE_OPERATORS_COLOR_SPACE_COLOR_SPACE_CONVERSION_H_
#define DALI_PIPELINE_OPERATORS_COLOR_SPACE_COLOR_SPACE_CONVERSION_H_

#include "dali/pipeline/operators/operator.h"

namespace dali {

template <typename Backend>
class ColorSpaceConversion : public Operator<Backend> {
 public:
  inline explicit ColorSpaceConversion(const OpSpec &spec)
    : Operator<Backend>(spec)
    , input_type_(spec.GetArgument<DALIImageType>("image_type"))
    , output_type_(spec.GetArgument<DALIImageType>("output_type")) {
  }

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;

  const DALIImageType input_type_;
  const DALIImageType output_type_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_COLOR_SPACE_COLOR_SPACE_CONVERSION_H_
