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

#ifndef DALI_PIPELINE_OPERATORS_READER_COCO_READER_OP_H_
#define DALI_PIPELINE_OPERATORS_READER_COCO_READER_OP_H_

#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <istream>
#include <memory>

#include "dali/pipeline/operators/reader/reader_op.h"
#include "dali/pipeline/operators/reader/loader/file_loader.h"
#include "dali/pipeline/operators/reader/loader/coco_loader.h"
#include "dali/pipeline/operators/reader/parser/coco_parser.h"

namespace dali {

class COCOReader : public DataReader<CPUBackend, ImageLabelWrapper> {
 public:
  explicit COCOReader(const OpSpec& spec)
  : DataReader<CPUBackend, ImageLabelWrapper>(spec) {
    bool shuffle_after_epoch = spec.GetArgument<bool>("shuffle_after_epoch");
    bool stick_to_shard = spec.GetArgument<bool>("stick_to_shard");

    if (shuffle_after_epoch || stick_to_shard)
      DALI_ENFORCE(
        !shuffle_after_epoch || !stick_to_shard,
        "shuffle_after_epoch and stick_to_shard cannot be both true");

    if (spec.HasArgument("file_list"))
      loader_ = InitLoader<FileLoader>(
        spec,
        std::vector<std::pair<string, int>>(),
        shuffle_after_epoch);
    else
      loader_ = InitLoader<CocoLoader>(
        spec,
        annotations_multimap_,
        shuffle_after_epoch);
    parser_.reset(new COCOParser(spec, annotations_multimap_));
  }

  void RunImpl(SampleWorkspace* ws, const int i) override {
    parser_->Parse(GetSample(ws->data_idx()), ws);
  }

 protected:
  AnnotationMap annotations_multimap_;

  USE_READER_OPERATOR_MEMBERS(CPUBackend, ImageLabelWrapper);
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_COCO_READER_OP_H_
