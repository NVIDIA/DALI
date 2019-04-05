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

#ifndef DALI_PIPELINE_OPERATORS_READER_LOADER_COCO_LOADER_H_
#define DALI_PIPELINE_OPERATORS_READER_LOADER_COCO_LOADER_H_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "dali/pipeline/operators/reader/loader/file_loader.h"
#include "dali/pipeline/operators/reader/parser/coco_parser.h"
#include "dali/common.h"
#include "dali/error_handling.h"

namespace dali {

namespace detail {

void ParseAnnotationFilesHelper(
  std::vector<std::string> &annotations_filename,
  AnnotationMap &annotations_multimap,
  std::vector<std::pair<std::string, int>> &image_id_pairs,
  bool ltrb, bool ratio, float size_threshold, bool skip_empty);

}

class CocoLoader : public FileLoader {
 public:
  explicit inline CocoLoader(
    const OpSpec& spec,
    AnnotationMap &annotations_multimap,
    bool shuffle_after_epoch = false) :
      FileLoader(spec, std::vector<std::pair<string, int>>(), shuffle_after_epoch),
      annotations_multimap_(annotations_multimap),
      annotations_filename_(spec.GetRepeatedArgument<std::string>("annotations_file")),
      ltrb_(spec.GetArgument<bool>("ltrb")),
      ratio_(spec.GetArgument<bool>("ratio")),
      size_threshold_(spec.GetArgument<float>("size_threshold")),
      skip_empty_(spec.GetArgument<bool>("skip_empty")) {}

 protected:
  void PrepareMetadataImpl() override {
    detail::ParseAnnotationFilesHelper(
      annotations_filename_,
      annotations_multimap_,
      image_label_pairs_,
      ltrb_,
      ratio_,
      size_threshold_,
      skip_empty_);

    DALI_ENFORCE(Size() > 0, "No files found.");
    if (shuffle_) {
      // seeded with hardcoded value to get
      // the same sequence on every shard
      std::mt19937 g(524287);
      std::shuffle(image_label_pairs_.begin(), image_label_pairs_.end(), g);
    }
    Reset(true);
  }

  AnnotationMap &annotations_multimap_;
  std::vector<std::string> annotations_filename_;
  bool ltrb_;
  bool ratio_;
  float size_threshold_;
  bool skip_empty_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_LOADER_COCO_LOADER_H_
