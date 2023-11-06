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

#ifndef DALI_OPERATORS_IMGCODEC_PEEK_IMAGE_SHAPE_H_
#define DALI_OPERATORS_IMGCODEC_PEEK_IMAGE_SHAPE_H_

#include <vector>
#include "dali/core/backend_tags.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/core/static_switch.h"
#include "dali/imgcodec/parsers/tiff.h"
#include "dali/imgcodec/image_source.h"

namespace dali {
namespace imgcodec {

class ImgcodecPeekImageShape : public StatelessOperator<CPUBackend> {
 public:
  ImgcodecPeekImageShape(const ImgcodecPeekImageShape &) = delete;

  explicit ImgcodecPeekImageShape(const OpSpec &spec);

  bool CanInferOutputs() const override;

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;

  void RunImpl(Workspace &ws) override;

  USE_OPERATOR_MEMBERS();
  using Operator<CPUBackend>::RunImpl;

 private:
  DALIDataType output_type_ = DALI_INT64;
  bool use_orientation_;
  DALIImageType image_type_;
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_OPERATORS_IMGCODEC_PEEK_IMAGE_SHAPE_H_
