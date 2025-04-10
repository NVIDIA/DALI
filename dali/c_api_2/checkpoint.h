// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_C_API_2_CHECKPOINT_H_
#define DALI_C_API_2_CHECKPOINT_H_

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "dali/dali.h"
#include "dali/pipeline/operator/checkpointing/checkpoint.h"

// A dummy base that the handle points to
struct _DALICheckpoint {
 protected:
  _DALICheckpoint() = default;
  ~_DALICheckpoint() = default;
};


namespace dali::c_api {

class PipelineWrapper;

class CheckpointWrapper : public _DALICheckpoint {
 public:
  explicit CheckpointWrapper(Checkpoint &&cpt)
  : cpt_(std::move(cpt)) {}

  const std::string &Serialized() const & {
    return serialized_.value();
  }

  void Serialize(const PipelineWrapper &pipeline);

  daliCheckpointExternalData_t ExternalData() const {
    daliCheckpointExternalData_t ext;
    ext.iterator_data.data = cpt_.external_ctx_cpt_.pipeline_data.data();
    ext.iterator_data.size = cpt_.external_ctx_cpt_.pipeline_data.size();
    ext.pipeline_data.data = cpt_.external_ctx_cpt_.pipeline_data.data();
    ext.pipeline_data.size = cpt_.external_ctx_cpt_.pipeline_data.size();
    return ext;
  }

  Checkpoint *Unwrap() & {
    return &cpt_;
  }

  const Checkpoint *Unwrap() const & {
    return &cpt_;
  }

 private:
  Checkpoint cpt_;
  std::optional<std::string> serialized_;
};

}  // namespace dali::c_api

#endif  // DALI_C_API_2_CHECKPOINT_H_
