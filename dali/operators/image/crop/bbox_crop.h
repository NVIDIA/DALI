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
#include <string>
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/operator_impl_utils.h"

namespace dali {

template <typename Backend>
class RandomBBoxCropImplBase : public OpImplBase<Backend> {
 public:
  virtual void SaveStateImpl(OpCheckpoint &cpt, AccessOrder order) = 0;

  virtual void RestoreStateImpl(const OpCheckpoint &cpt) = 0;

  virtual std::string SerializeCheckpointImpl(const OpCheckpoint &cpt) const = 0;

  virtual void DeserializeCheckpointImpl(OpCheckpoint &cpt, const std::string &data) const = 0;
};

template <typename Backend>
class RandomBBoxCrop : public Operator<Backend> {
 public:
  explicit inline RandomBBoxCrop(const OpSpec &spec);
  ~RandomBBoxCrop() override;

  void SaveState(OpCheckpoint &cpt, AccessOrder order) override {
    impl_->SaveStateImpl(cpt, order);
  }

  void RestoreState(const OpCheckpoint &cpt) override {
    impl_->RestoreStateImpl(cpt);
  }

  std::string SerializeCheckpoint(const OpCheckpoint &cpt) const override {
    return impl_->SerializeCheckpointImpl(cpt);
  }

  void DeserializeCheckpoint(OpCheckpoint &cpt, const std::string &data) const override {
    impl_->DeserializeCheckpointImpl(cpt, data);
  }

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;
  void RunImpl(Workspace &ws) override;

 private:
  std::unique_ptr<RandomBBoxCropImplBase<Backend>> impl_;
  int impl_ndim_ = -1;
  using Operator<Backend>::spec_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CROP_BBOX_CROP_H_
