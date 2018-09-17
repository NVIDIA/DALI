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

#define FIND_IN_JSON(im, it, field)             \
      auto it = im.find(#field);               \
      DALI_ENFORCE(it != im.end(), "`" #field "` not found in JSON annotions file");

#define GET_FROM_JSON(im, field, type)          \
      ({auto it_##field = im.find(#field);     \
      DALI_ENFORCE(it_##field != im.end(), "`" #field "` not found in JSON annotions file"); \
      it_##field.value().get<type>();})

namespace dali {

class COCOReader : public DataReader<CPUBackend> {
 public:
  explicit COCOReader(const OpSpec& spec)
  : DataReader<CPUBackend>(spec),
    annotations_filename_(spec.GetRepeatedArgument<std::string>("annotations_file")),
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
      FIND_IN_JSON(j, images, images);
      for (auto& im : *images) {
        int id = GET_FROM_JSON(im, id, int);
        std::string image_file_name = GET_FROM_JSON(im, file_name, std::string);
        int width = GET_FROM_JSON(im, width, int);
        int height = GET_FROM_JSON(im, height, int);

        image_id_pairs_.push_back(std::make_pair(image_file_name, id));
        image_id_to_wh.insert(std::make_pair(id, std::make_pair(width, height)));
      }

      // Change categories IDs to be in range [1, 80]
      std::vector<int> deleted_categories{ 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91 };
      std::map<int, int> new_categories_ids;
      int current_id = 1;
      int vector_id = 0;
      for (int i = 1; i <= 90; i++) {
        if (i == deleted_categories[vector_id]) {
          vector_id++;
        } else {
          new_categories_ids.insert(std::make_pair(i, current_id));
          current_id++;
        }
      }

      // Parse annotations
      FIND_IN_JSON(j, annotations, annotations);
      int annotation_size = (*annotations).size();

      for (auto& an : *annotations) {
        int image_id = GET_FROM_JSON(an, image_id, int);
        int category_id = GET_FROM_JSON(an, category_id, int);
        std::vector<float> bbox = GET_FROM_JSON(an, bbox, std::vector<float>);

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
              Annotation(bbox[0], bbox[1], bbox[2], bbox[3], new_categories_ids[category_id])));
      }

      f.close();
    }
  }

  std::vector<std::string> annotations_filename_;
  AnnotationMap annotations_multimap_;
  std::vector<std::pair<std::string, int>> image_id_pairs_;
  bool ltrb_;
  bool ratio_;
  USE_READER_OPERATOR_MEMBERS(CPUBackend);
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_COCO_READER_OP_H_
