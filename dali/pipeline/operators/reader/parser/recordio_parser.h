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

#ifndef DALI_PIPELINE_OPERATORS_READER_PARSER_RECORDIO_PARSER_H_
#define DALI_PIPELINE_OPERATORS_READER_PARSER_RECORDIO_PARSER_H_

#include <string>
#include <vector>

#include "dali/pipeline/operators/reader/parser/parser.h"

namespace dali {

struct ImageRecordIOHeader {
  uint32_t flag;
  float label;
  uint64_t image_id[2];
};

class RecordIOParser : public Parser<Tensor<CPUBackend>> {
 public:
  explicit RecordIOParser(const OpSpec& spec) :
    Parser<Tensor<CPUBackend>>(spec) {
  }

  void Parse(const Tensor<CPUBackend>& data, SampleWorkspace* ws) override {
    auto& image = ws->Output<CPUBackend>(0);
    auto& label = ws->Output<CPUBackend>(1);
    ReadSingleImageRecordIO(image, label, data.data<uint8_t>());
    image.SetSourceInfo(data.GetSourceInfo());
  }

 private:
  inline uint32_t DecodeFlag(uint32_t rec) {
    return (rec >> 29U) & 7U;
  }

  inline uint32_t DecodeLength(uint32_t rec) {
    return rec & ((1U << 29U) - 1U);
  }

  template <typename T>
  void ReadSingle(const uint8_t** in, T* out) {
    memcpy(out, *in, sizeof(T));
    *in += sizeof(T);
  }

  inline void ReadSingleImageRecordIO(Tensor<CPUBackend>& o_image,
                Tensor<CPUBackend>& o_label,
                const uint8_t* input) {
    uint32_t magic;
    const uint32_t kMagic = 0xced7230a;
    ReadSingle<uint32_t>(&input, &magic);
    DALI_ENFORCE(magic == kMagic, "Invalid RecordIO: wrong magic number");

    uint32_t length_flag;
    ReadSingle(&input, &length_flag);
    uint32_t cflag = DecodeFlag(length_flag);
    uint32_t clength = DecodeLength(length_flag);
    ImageRecordIOHeader hdr;
    ReadSingle(&input, &hdr);

    if (hdr.flag == 0) {
      o_label.Resize({1});
      o_label.mutable_data<float>()[0] = hdr.label;
    } else {
      o_label.Resize({hdr.flag});
      o_label.mutable_data<float>();
    }

    int64_t data_size = clength - sizeof(ImageRecordIOHeader);
    int64_t label_size = hdr.flag * sizeof(float);
    int64_t image_size = data_size - label_size;
    if (cflag == 0) {
      o_image.Resize({image_size});
      uint8_t* data = o_image.mutable_data<uint8_t>();
      memcpy(data, input + label_size, image_size);
      if (hdr.flag > 0) {
        float * label = o_label.mutable_data<float>();
        memcpy(label, input, label_size);
      }
    } else {
      std::vector<uint8_t> temp_vec(data_size);
      memcpy(&temp_vec[0], input, data_size);
      input += data_size;

      while (true) {
        size_t pad = clength - (((clength + 3U) >> 2U) << 2U);
        input += pad;

        if (cflag != 3) {
          size_t s = temp_vec.size();
          temp_vec.resize(static_cast<int64_t>(s + sizeof(kMagic)));
          memcpy(&temp_vec[s], &kMagic, sizeof(kMagic));
        } else {
          break;
        }
        ReadSingle(&input, &magic);
        ReadSingle(&input, &length_flag);
        cflag = DecodeFlag(length_flag);
        clength = DecodeLength(length_flag);
        size_t s = temp_vec.size();
        temp_vec.resize(static_cast<int64_t>(s + clength));
        memcpy(&temp_vec[s], input, clength);
        input += clength;
      }
      o_image.Resize({static_cast<Index>(temp_vec.size() - label_size)});
      uint8_t* data = o_image.mutable_data<uint8_t>();
      memcpy(data, (&temp_vec[0]) + label_size, temp_vec.size() - label_size);
      if (hdr.flag > 0) {
        float * label = o_label.mutable_data<float>();
        memcpy(label, &temp_vec[0], label_size);
      }
    }
  }
};

};  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_PARSER_RECORDIO_PARSER_H_
