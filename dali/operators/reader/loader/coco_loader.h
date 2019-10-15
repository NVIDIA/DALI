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

#ifndef DALI_OPERATORS_READER_LOADER_COCO_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_COCO_LOADER_H_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "dali/operators/reader/loader/file_loader.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"

namespace dali {
namespace detail {
template<size_t N>
inline int safe_strcmp(const char *str1, const char (&str2)[N]) {
    return strncmp(str1, str2, N-1);
}

}  // namespace detail

using ImageIdPairs = std::vector<std::pair<std::string, int>>;
class CocoLoader : public FileLoader {
 public:
  explicit inline CocoLoader(
    const OpSpec& spec,
    std::vector<int> &offsets,
    std::vector<float> &boxes,
    std::vector<int> &labels,
    std::vector<int> &counts,
    bool save_img_ids,
    std::vector<int> &original_ids,
    bool shuffle_after_epoch = false) :
      FileLoader(spec, std::vector<std::pair<string, int>>(), shuffle_after_epoch),
      spec_(spec),
      parse_meta_files_(spec.HasArgument("meta_files_path")),
      offsets_(offsets),
      boxes_(boxes),
      labels_(labels),
      counts_(counts),
      save_img_ids_(save_img_ids),
      original_ids_(original_ids) {}

 protected:
  void PrepareMetadataImpl() override {
    if (parse_meta_files_) {
      ParseMetafiles();
    } else {
       ParseJsonAnnotations();
    }

    DALI_ENFORCE(Size() > 0, "No files found.");
    if (shuffle_) {
      // seeded with hardcoded value to get
      // the same sequence on every shard
      std::mt19937 g(524287);
      std::shuffle(image_label_pairs_.begin(), image_label_pairs_.end(), g);
    }
    Reset(true);
  }

  void ParseMetafiles();

  void ParseJsonAnnotations();

  void DumpMetaFiles(std::string path, const ImageIdPairs &image_id_pairs);

 private:
  const OpSpec &spec_;
  bool parse_meta_files_;

  std::vector<int> &offsets_;
  std::vector<float> &boxes_;
  std::vector<int> &labels_;
  std::vector<int> &counts_;

  bool save_img_ids_;
  std::vector<int> &original_ids_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_COCO_LOADER_H_
