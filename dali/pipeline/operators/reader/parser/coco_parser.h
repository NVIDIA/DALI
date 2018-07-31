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

#ifndef DALI_PIPELINE_OPERATORS_READER_PARSER_COCO_PARSER_H_
#define DALI_PIPELINE_OPERATORS_READER_PARSER_COCO_PARSER_H_

#include <string>

#include "dali/pipeline/operators/reader/parser/parser.h"

namespace dali {

struct Annotation {
  Annotation(float x, float y, float w, float h, int category_id)
      : x(x), y(y), w(w), h(h), category_id(category_id){}
  float x;
  float y;
  float w;
  float h;
  int category_id;

  friend std::ostream& operator<<(std::ostream& os, Annotation& an);
};

std::ostream& operator<<(std::ostream& os, Annotation& an) {
  os << "Annotation(category_id=" << an.category_id
  << ",bbox = [" << an.x << "," << an.y
  << "," << an.w << "," << an.h << "])";
  return os;
}


class COCOParser: public Parser {
 public:
  explicit COCOParser(const OpSpec& spec, std::multimap<int, Annotation>& annotations_multimap)
   : Parser(spec),
     annotations_multimap_(annotations_multimap) {}
  void Parse(const uint8_t* data, const size_t size, SampleWorkspace* ws) override {
    Index image_size = size - sizeof(int);
    auto *image_output = ws->Output<CPUBackend>(0);
    auto *bbox_output = ws->Output<CPUBackend>(1);
    int image_id =
         *reinterpret_cast<const int*>(data + image_size);

    auto range = annotations_multimap_.equal_range(image_id);
    auto n_bboxes = std::distance(range.first, range.second);
    DALI_ENFORCE(n_bboxes > 0, "Annotations not found for image by image_id " + std::to_string(image_id));

    image_output->Resize({image_size});
    image_output->mutable_data<uint8_t>();
    bbox_output->Resize({n_bboxes});
    bbox_output->mutable_data<float>();

    for (auto it = range.first; it != range.second; ++it) {
      Annotation& an = it->second;
    }
  }

  std::multimap<int, Annotation>& annotations_multimap_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_PARSER_COCO_PARSER_H_
