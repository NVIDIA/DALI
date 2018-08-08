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

#include "dali/pipeline/operators/reader/reader_op.h"
#include "dali/pipeline/operators/reader/loader/file_loader.h"
#include "dali/pipeline/operators/reader/parser/coco_parser.h"
#include "dali/util/json.h"

namespace dali {

class COCOReader : public DataReader<CPUBackend> {
 public:
  explicit COCOReader(const OpSpec& spec)
  : DataReader<CPUBackend>(spec),
    annotations_filename_(spec.GetArgument<std::string>("annotations_file")),
    ltrb_(spec.GetArgument<bool>("ltrb")),
    ratio_(spec.GetArgument<bool>("ratio")) {
    ParseAnnotationFiles();
    loader_.reset(new FileLoader(spec, image_id_pairs_));
    parser_.reset(new COCOParser(spec, annotations_multimap_));
  }

  DEFAULT_READER_DESTRUCTOR(COCOReader, CPUBackend);

  void RunImpl(SampleWorkspace* ws, const int i) override {
    const int idx = ws->data_idx();

    auto* raw_data = prefetched_batch_[idx];

    parser_->Parse(raw_data->data<uint8_t>(), raw_data->size(), ws);

    return;
  }

 protected:
  void ParseAnnotationFiles() {
    // Loading raw json into the RAM
    std::ifstream f(annotations_filename_);
    DALI_ENFORCE(f, "Could not open JSON annotations file");
    std::string raw_json;
    f.seekg(0, std::ios::end);
    raw_json.reserve(f.tellg());
    f.seekg(0, std::ios::beg);
    raw_json.assign((std::istreambuf_iterator<char>(f)),
                    std::istreambuf_iterator<char>());

    // Parsing the JSON into image vector and annotation multimap
    std::stringstream ss;
    ss << raw_json;
    auto j = nlohmann::json::parse(ss);

    // mapping each image_id to its WH dimension
    std::unordered_map<int, std::pair<int, int> > image_id_to_wh;

    // Parse images
    auto images = j.find("images");
    DALI_ENFORCE(images != j.end(), "`images` not found in JSON annotions file");
    for (auto& im : *images) {
      auto id_it = im.find("id");
      DALI_ENFORCE(id_it != im.end(), "`id` not found in JSON annotions file");
      auto file_name_it = im.find("file_name");
      DALI_ENFORCE(file_name_it != im.end(), "`file_name` not found in JSON annotions file");
      auto width_it = im.find("width");
      DALI_ENFORCE(width_it != im.end(), "`width` not found in JSON annotions file");
      auto height_it = im.find("height");
      DALI_ENFORCE(height_it != im.end(), "`height` not found in JSON annotions file");

      int id = id_it.value().get<int>();
      std::string file_name = file_name_it.value().get<std::string>();

      image_id_pairs_.push_back(std::make_pair(file_name, id));

      int width = width_it.value().get<int>();
      int height = height_it.value().get<int>();
      image_id_to_wh.insert(std::make_pair(id, std::make_pair(width, height)));
    }

    // Parse annotations
    auto annotations = j.find("annotations");
    DALI_ENFORCE(annotations != j.end(), "`annotations` not found in JSON annotions file");
    int annotation_size = (*annotations).size();

    for (auto& an : *annotations) {
      auto image_id_it = an.find("image_id");
      DALI_ENFORCE(image_id_it != an.end(), "`image_id` not found in JSON annotions file");
      auto category_id_it = an.find("category_id");
      DALI_ENFORCE(category_id_it != an.end(), "`category_id` not found in JSON annotions file");
      auto bbox_it = an.find("bbox");
      DALI_ENFORCE(bbox_it != an.end(), "`bbox` not found in JSON annotions file");

      int image_id = image_id_it.value().get<int>();
      int category_id = category_id_it.value().get<int>();
      std::vector<float> bbox = bbox_it.value().get<std::vector<float>>();

      if (ltrb_) {
        bbox[2] += bbox[0];
        bbox[3] += bbox[1];
      }

      if (ratio_) {
        auto wh_it = image_id_to_wh.find(image_id);
        DALI_ENFORCE(wh_it != image_id_to_wh.end(),
            "annotation has an invalid image_id: " + std::to_string(image_id));
        auto wh = wh_it->second;
        bbox[0] /= static_cast<float>(wh.first);
        bbox[1] /= static_cast<float>(wh.second);
        bbox[2] /= static_cast<float>(wh.first);
        bbox[3] /= static_cast<float>(wh.second);
      }

      annotations_multimap_.insert(
          std::make_pair(image_id,
            Annotation(bbox[0], bbox[1], bbox[2], bbox[3], category_id)));
    }

    f.close();
  }

  std::string annotations_filename_;
  AnnotationMap annotations_multimap_;
  std::vector<std::pair<std::string, int>> image_id_pairs_;
  bool ltrb_;
  bool ratio_;
  USE_READER_OPERATOR_MEMBERS(CPUBackend);
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_COCO_READER_OP_H_
