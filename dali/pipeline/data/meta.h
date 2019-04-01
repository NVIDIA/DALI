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

#ifndef DALI_PIPELINE_DATA_META_H_
#define DALI_PIPELINE_DATA_META_H_

#include <string>
#include "dali/pipeline/data/types.h"

namespace dali {

class DALIMeta {
 public:
  DALIMeta() {
  }

  explicit DALIMeta(DALITensorLayout layout) : layout_(layout) {
  }

  inline DALITensorLayout GetLayout() const {
    return layout_;
  }

  inline void SetLayout(DALITensorLayout layout) {
    layout_ = layout;
  }

  inline std::string GetSourceInfo() const {
    return source_info_;
  }

  inline void SetSourceInfo(std::string source_info) {
    source_info_ = source_info;
  }

  inline void SetSkipSample(bool skip_sample) {
    skip_sample_ = skip_sample;
  }

  inline bool ShouldSkipSample() const {
    return skip_sample_;
  }

 private:
  DALITensorLayout layout_;
  std::string source_info_;
  bool skip_sample_ = false;
};

}  // namespace dali


#endif  // DALI_PIPELINE_DATA_META_H_
