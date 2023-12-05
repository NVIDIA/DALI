// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_CROP_BBOX_CROP_H_
#define DALI_OPERATORS_IMAGE_CROP_BBOX_CROP_H_

#include <memory>
#include <vector>
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/operators/random/rng_base_cpu.h"
#include "dali/pipeline/util/operator_impl_utils.h"

namespace dali {

template <typename Backend>
class RandomBBoxCrop : public rng::OperatorWithRng<Backend> {
 public:
  explicit inline RandomBBoxCrop(const OpSpec &spec);
  ~RandomBBoxCrop() override;

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;
  void RunImpl(Workspace &ws) override;

 private:
  std::unique_ptr<OpImplBase<Backend>> impl_;
  int impl_ndim_ = -1;
  using Operator<Backend>::spec_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CROP_BBOX_CROP_H_
