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
#include "dali/pipeline/operators/reader/parser/coco_parser.h"

namespace dali {

void ParseAnnotationFilesHelper(std::vector<std::string> &annotations_filename,
                                AnnotationMap &annotations_multimap,
                                std::vector<std::pair<std::string, int>> &image_id_pairs,
                                bool ltrb, bool ratio);

class COCOReader : public DataReader<CPUBackend, ImageLabelWrapper> {
 public:
  explicit COCOReader(const OpSpec& spec)
  : DataReader<CPUBackend, ImageLabelWrapper>(spec),
    annotations_filename_(spec.GetRepeatedArgument<std::string>("annotations_file")),
    ltrb_(spec.GetArgument<bool>("ltrb")),
    ratio_(spec.GetArgument<bool>("ratio")),
    save_img_ids_(spec.GetArgument<bool>("save_img_ids")) {
    ParseAnnotationFiles();
    loader_.reset(new FileLoader(spec, image_id_pairs_));
    parser_.reset(new COCOParser(spec, annotations_multimap_, save_img_ids_));
  }

  void RunImpl(SampleWorkspace* ws, const int i) override {
    const int idx = ws->data_idx();

    auto* image_label = prefetched_batch_[idx];

    parser_->Parse(*image_label, ws);

    return;
  }

 protected:
  void ParseAnnotationFiles() {
    ParseAnnotationFilesHelper(annotations_filename_, annotations_multimap_,
                                image_id_pairs_, ltrb_, ratio_);
  }

  std::vector<std::string> annotations_filename_;
  AnnotationMap annotations_multimap_;
  std::vector<std::pair<std::string, int>> image_id_pairs_;
  bool ltrb_;
  bool ratio_;
  bool save_img_ids_;
  USE_READER_OPERATOR_MEMBERS(CPUBackend, ImageLabelWrapper);
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_COCO_READER_OP_H_
