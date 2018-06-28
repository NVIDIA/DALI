// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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


#ifndef DALI_PIPELINE_OPERATORS_CROP_CROP_H_
#define DALI_PIPELINE_OPERATORS_CROP_CROP_H_

#include <vector>
#include <utility>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/fused/crop_cast_permute.h"

namespace dali {

template <typename Backend>
class Crop : public CropCastPermute<Backend> {
 public:
  explicit inline Crop(const OpSpec &spec) :
    CropCastPermute<Backend>(Crop::PrepareOpSpec(spec)),
    image_type_(spec.GetArgument<DALIImageType>("image_type")),
    color_(IsColor(image_type_)),
    C_(color_ ? 3 : 1) {
  }

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;

  void SetupSharedSampleParams(Workspace<Backend> *ws) override;

  static OpSpec PrepareOpSpec(const OpSpec& spec) {
    OpSpec newSpec = spec;
    newSpec.AddArg<DALIDataType>("output_dtype", DALI_NO_TYPE);
    newSpec.AddArg<DALITensorLayout>("output_layout", DALI_SAME);
    return newSpec;
  }

  int crop_h_;
  int crop_w_;

  // Input/output channel meta-data
  DALIImageType image_type_;
  bool color_;
  int C_;

  // Output data layout
  DALITensorLayout output_layout_;



  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_CROP_H_

