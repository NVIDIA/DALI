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

#include <map>
#include <string>
#include <array>

#include "dali/pipeline/operators/reader/parser/parser.h"
#include "dali/pipeline/operators/reader/loader/file_loader.h"

namespace dali {

struct Annotation {
  /*
   * Currently, only bboxes and labels are supported,
   * while key points and segmentation masks need to be handled
   * somehow in the future
   */
  Annotation(float x, float y, float w, float h, int category_id)
    : bbox({x, y, w, h}),
      category_id(category_id) {}

  // bbox contains 5 values: x, y, W, H, category_id
  std::array<float, 4> bbox;
  int category_id;
  friend std::ostream& operator<<(std::ostream& os, Annotation& an);
};

std::ostream& operator<<(std::ostream& os, Annotation& an) {
  std::array<float, 4>& bbox = an.bbox;
  os << "Annotation(category_id=" << an.category_id
  << ",bbox = [" << bbox[0] << "," << bbox[1]
  << "," << bbox[2] << "," << bbox[3] << "])";
  return os;
}

using AnnotationMap = std::multimap<int, Annotation>;

class COCOParser: public Parser<ImageLabelWrapper> {
 public:
  explicit COCOParser(const OpSpec& spec, const AnnotationMap& annotations_multimap,
                      const bool& save_img_ids)
    : Parser<ImageLabelWrapper>(spec),
    annotations_multimap_(annotations_multimap),
    save_img_ids_(save_img_ids) {}

  void Parse(const ImageLabelWrapper& image_label, SampleWorkspace* ws) override {
    Index image_size = image_label.image.size();
    auto *image_output = ws->Output<CPUBackend>(0);
    auto *bbox_output = ws->Output<CPUBackend>(1);
    auto *label_output = ws->Output<CPUBackend>(2);
    int image_id = image_label.label;

    auto range = annotations_multimap_.equal_range(image_id);
    auto n_bboxes = std::distance(range.first, range.second);

    image_output->Resize({image_size});
    image_output->mutable_data<uint8_t>();
    bbox_output->Resize({n_bboxes, 4});
    bbox_output->mutable_data<float>();
    label_output->Resize({n_bboxes, 1});
    label_output->mutable_data<int>();


    if (save_img_ids_) {
      auto *image_id_output = ws->Output<CPUBackend>(3);
      image_id_output->Resize({1});
      image_id_output->mutable_data<int>();
      std::memcpy(image_id_output->raw_mutable_data(),
                &image_label.label,
                sizeof(int));
    }

    std::memcpy(image_output->raw_mutable_data(),
                image_label.image.raw_data(),
                image_size);

    int stride = 0;
    for (auto it = range.first; it != range.second; ++it) {
      const Annotation& an = it->second;
      int i = std::distance(range.first, it);
      std::memcpy(
          bbox_output->mutable_data<float>() + an.bbox.size() * i,
          an.bbox.data(),
          an.bbox.size() * sizeof (float));
      label_output->mutable_data<int>()[i] = an.category_id;
    }
  }

  const AnnotationMap& annotations_multimap_;
  const bool save_img_ids_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_PARSER_COCO_PARSER_H_
