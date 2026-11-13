// Copyright (c) 2017-2018, 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  DALIMeta() = default;

  explicit DALIMeta(const TensorLayout &layout) : layout_(layout) {
  }

  inline const TensorLayout &GetLayout() const {
    return layout_;
  }

  inline void SetLayout(const TensorLayout &layout) {
    layout_ = layout;
  }

  inline const std::string &GetSourceInfo() const {
    return source_info_;
  }

  inline void SetSourceInfo(const std::string &source_info) {
    source_info_ = source_info;
  }

  /**
   * @brief Key identifying the data source in the decoded image cache.
   *
   * The image cache is shared by every pipeline that runs on the same device, so the key must
   * identify the data source unambiguously (e.g. the full path of a file), not only its name
   * as reported to the user via the source info. If no dedicated key was set, the source info
   * is used.
   */
  inline const std::string &GetCacheKey() const {
    return cache_key_.empty() ? source_info_ : cache_key_;
  }

  inline void SetCacheKey(const std::string &cache_key) {
    cache_key_ = cache_key;
  }

  inline void SetSkipSample(bool skip_sample) {
    skip_sample_ = skip_sample;
  }

  inline bool ShouldSkipSample() const {
    return skip_sample_;
  }

 private:
  TensorLayout layout_;
  std::string source_info_;
  std::string cache_key_;
  bool skip_sample_ = false;
};

}  // namespace dali


#endif  // DALI_PIPELINE_DATA_META_H_
