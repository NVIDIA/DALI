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

namespace dali {
using ImageIdPairs = std::vector<std::pair<std::string, int>>;

class COCOReader : public DataReader<CPUBackend, ImageLabelWrapper> {
 public:
  explicit COCOReader(const OpSpec& spec):
    DataReader<CPUBackend, ImageLabelWrapper>(spec),
    save_img_ids_(spec.GetArgument<bool>("save_img_ids")) {
    ValidateOptions(spec);
    bool shuffle_after_epoch = spec.GetArgument<bool>("shuffle_after_epoch");
    loader_ = InitLoader<FileLoader>(
      spec,
      spec.HasArgument("meta_files_path") ? ParseMetafiles(spec) : ParseJsonAnnotations(spec),
      shuffle_after_epoch);
  }

  void RunImpl(SampleWorkspace* ws, const int i) override {
    const ImageLabelWrapper& image_label = GetSample(ws->data_idx());

    Index image_size = image_label.image.size();
    auto &image_output = ws->Output<CPUBackend>(0);
    int image_id = image_label.label;

    image_output.Resize({image_size});
    image_output.mutable_data<uint8_t>();
    std::memcpy(
      image_output.raw_mutable_data(),
      image_label.image.raw_data(),
      image_size);
    image_output.SetSourceInfo(image_label.image.GetSourceInfo());

    auto &boxes_output = ws->Output<CPUBackend>(1);
    boxes_output.Resize({counts_[image_id], 4});
    auto boxes_out_data = boxes_output.mutable_data<float>();
    memcpy(
      boxes_out_data,
      boxes_.data() + 4 * offsets_[image_id],
      counts_[image_id] * 4 * sizeof(float));

    auto &labels_output = ws->Output<CPUBackend>(2);
    labels_output.Resize({counts_[image_id], 1});
    auto labels_out_data = labels_output.mutable_data<int>();
    memcpy(
      labels_out_data,
      labels_.data() + offsets_[image_id],
      counts_[image_id] * sizeof(int));

    // if (save_img_ids_) {
    //   auto &id_output = ws->Output<CPUBackend>(3);
    //   id_output.Resize({1});
    //   auto id_out_data = id_output.mutable_data<int>();
    //   memcpy(
    //     id_out_data,
    //     original_ids_.data() + image_id,
    //     sizeof(int));
    // }
  }

 protected:
  USE_READER_OPERATOR_MEMBERS(CPUBackend, ImageLabelWrapper);

 private:
  std::vector<int> offsets_;
  std::vector<float> boxes_;
  std::vector<int> labels_;
  std::vector<int> counts_;

  bool save_img_ids_;
  std::vector<int> original_ids_;

  ImageIdPairs ParseMetafiles(const OpSpec& spec);

  ImageIdPairs ParseJsonAnnotations(const OpSpec& spec);

  void DumpMetaFiles(std::string path, const ImageIdPairs &image_id_pairs);

  void ValidateOptions(const OpSpec &spec);
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_COCO_READER_OP_H_
