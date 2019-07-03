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

#ifndef DALI_PIPELINE_OPERATORS_READER_TFRECORD_READER_OP_H_
#define DALI_PIPELINE_OPERATORS_READER_TFRECORD_READER_OP_H_

#ifdef DALI_BUILD_PROTO3

#include <fstream>
#include <vector>
#include <iterator>
#include <iostream>
#include <string>

#include "dali/pipeline/operators/reader/reader_op.h"
#include "dali/pipeline/operators/reader/loader/indexed_file_loader.h"
#include "dali/pipeline/operators/reader/parser/tfrecord_parser.h"

template<typename T>
void load_vector_from_file(std::vector<T> &output, std::string path) {
  std::ifstream file(path);
  if (file) {
    T val;
    while (file >> val)
      output.push_back(val);
  } else {
    DALI_FAIL("TFRecord meta file error.");
  }
}

namespace dali {
class TFRecordReader : public DataReader<CPUBackend, Tensor<CPUBackend>> {
 public:
  explicit TFRecordReader(const OpSpec& spec)
  : DataReader<CPUBackend, Tensor<CPUBackend>>(spec) {
    loader_ = InitLoader<IndexedFileLoader>(spec);
    parser_.reset(new TFRecordParser(spec));
    DALI_ENFORCE(!skip_cached_images_,
      "TFRecordReader doesn't support `skip_cached_images` option");

      auto meta_files_path = spec.GetArgument<string>("meta_files_path");
      parse_meta_files_ = meta_files_path != std::string();

      if (parse_meta_files_) {
        load_vector_from_file(
          offsets_,
          meta_files_path + "offsets.txt");

        load_vector_from_file(
          boxes_,
          meta_files_path + "boxes.txt");

        load_vector_from_file(
          labels_,
          meta_files_path + "labels.txt");

        load_vector_from_file(
          counts_,
          meta_files_path + "counts.txt");
      }
  }

  void RunImpl(SampleWorkspace* ws, const int i) override {
    const auto& tensor = GetSample(ws->data_idx());
    parser_->Parse(tensor, ws);

    if (parse_meta_files_) {
      const auto idx = ws->Output<CPUBackend>(1).data<int64>()[0];

      auto &boxes_output = ws->Output<CPUBackend>(2);
      boxes_output.Resize({counts_[idx], 4});
      auto boxes_out_data = boxes_output.mutable_data<float>();
      memcpy(
        boxes_out_data,
        boxes_.data() + 4 * offsets_[idx],
        counts_[idx] * 4 * sizeof(float));

      auto &labels_output = ws->Output<CPUBackend>(3);
      labels_output.Resize({counts_[idx], 1});
      auto labels_out_data = labels_output.mutable_data<int>();
      memcpy(
        labels_out_data,
        labels_.data() + offsets_[idx],
        counts_[idx] * sizeof(int));
    }
  }

 protected:
  USE_READER_OPERATOR_MEMBERS(CPUBackend, Tensor<CPUBackend>);

 private:
  std::vector<int> offsets_;
  std::vector<float> boxes_;
  std::vector<int> labels_;
  std::vector<int> counts_;
  bool parse_meta_files_;
};

}  // namespace dali

#endif  // DALI_BUILD_PROTO3
#endif  // DALI_PIPELINE_OPERATORS_READER_TFRECORD_READER_OP_H_
