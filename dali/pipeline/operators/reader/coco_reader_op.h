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

template <typename I>
nlohmann::json::const_iterator find_in_json(const I &im, std::string field) {
  auto it = im.find(field);
  DALI_ENFORCE(it != im.end(), "`" + field + "` not found in JSON annotions file");
  return it;
}

template <typename T, typename I>
T get_from_json(const I &im, std::string field) {
  auto it = im.find(field);
  DALI_ENFORCE(it != im.end(), "`" + field + "` not found in JSON annotions file");
  return it.value();
}

namespace dali {

class COCOReader : public DataReader<CPUBackend, ImageLabelWrapper> {
 public:
  explicit COCOReader(const OpSpec& spec)
  : DataReader<CPUBackend, ImageLabelWrapper>(spec),
    annotations_filename_(spec.GetRepeatedArgument<std::string>("annotations_file")),
    ltrb_(spec.GetArgument<bool>("ltrb")),
    ratio_(spec.GetArgument<bool>("ratio")) {
    ParseAnnotationFiles();
    loader_.reset(new FileLoader(spec, image_id_pairs_));
    parser_.reset(new COCOParser(spec, annotations_multimap_));
  }

  void RunImpl(SampleWorkspace* ws, const int i) override {
    const int idx = ws->data_idx();

    auto* image_label = prefetched_batch_[idx];

    parser_->Parse(*image_label, ws);

    return;
  }

 protected:
  void ParseAnnotationFiles() {
    for (auto& file_name : annotations_filename_) {
      // Loading raw json into the RAM
      std::ifstream f(file_name);
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
      auto images = find_in_json(j, "images");
      for (auto& im : *images) {
        auto id = get_from_json<int>(im, "id");
        auto image_file_name = get_from_json<std::string>(im, "file_name");
        auto width = get_from_json<int>(im, "width");
        auto height = get_from_json<int>(im, "height");

        image_id_pairs_.push_back(std::make_pair(image_file_name, id));
        image_id_to_wh.insert(std::make_pair(id, std::make_pair(width, height)));
      }

      // Parse annotations
      auto annotations = find_in_json(j, "annotations");
      int annotation_size = (*annotations).size();

      // Change categories IDs to be in range [1, 80]
      std::vector<int> deleted_categories{ 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91 };
      std::map<int, int> new_category_ids;
      int current_id = 1;
      int vector_id = 0;
      for (int i = 1; i <= 90; i++) {
        if (i == deleted_categories[vector_id]) {
          vector_id++;
        } else {
          new_category_ids.insert(std::make_pair(i, current_id));
          current_id++;
        }
      }

      for (auto& an : *annotations) {
        auto image_id = get_from_json<int>(an, "image_id");
        auto category_id = get_from_json<int>(an, "category_id");
        auto bbox = get_from_json<std::array<float, 4>>(an, "bbox");

        if (bbox[2] < 0.1 || bbox[3] < 0.1) {
          continue;
        }

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
              Annotation(bbox[0], bbox[1], bbox[2], bbox[3], new_category_ids[category_id])));
      }

      f.close();
    }
  }

  std::vector<std::string> annotations_filename_;
  AnnotationMap annotations_multimap_;
  std::vector<std::pair<std::string, int>> image_id_pairs_;
  bool ltrb_;
  bool ratio_;
  USE_READER_OPERATOR_MEMBERS(CPUBackend, ImageLabelWrapper);
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_COCO_READER_OP_H_
