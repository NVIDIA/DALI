// Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_PARSER_RECORDIO_PARSER_H_
#define DALI_OPERATORS_READER_PARSER_RECORDIO_PARSER_H_

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "dali/operators/reader/parser/parser.h"

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

  void Parse(const Tensor<CPUBackend>& tensor, SampleWorkspace* ws) override {
    auto& image = ws->Output<CPUBackend>(0);
    auto& label = ws->Output<CPUBackend>(1);
    ReadSingleImageRecordIO(image, label, tensor.data<uint8_t>(), tensor.nbytes(),
                            tensor.GetSourceInfo());
    image.SetSourceInfo(tensor.GetSourceInfo());
  }

 private:
  inline uint32_t DecodeFlag(uint32_t rec) {
    return (rec >> 29U) & 7U;
  }

  inline uint32_t DecodeLength(uint32_t rec) {
    return rec & ((1U << 29U) - 1U);
  }

  inline size_t RemainingBytes(const uint8_t* in, const uint8_t* end) {
    return static_cast<size_t>(end - in);
  }

  inline void CheckAvailable(const uint8_t* in, const uint8_t* end, size_t size,
                             const string& source_info, const string& context) {
    size_t available = RemainingBytes(in, end);
    if (size > available) {
      throw std::runtime_error(
        make_string("Invalid RecordIO file: ", source_info, " (", context, " requires ", size,
                    " bytes, but only ", available, " bytes are available)."));
    }
  }

  inline size_t PaddedLength(uint32_t length) {
    return (static_cast<size_t>(length) + 3U) & ~static_cast<size_t>(3U);
  }

  inline void CopyBytes(void* out, const void* in, size_t size) {
    if (size > 0) {
      std::memcpy(out, in, size);
    }
  }

  template <typename T>
  void ReadSingle(const uint8_t** in, const uint8_t* end, T* out, const string& source_info,
                  const string& context) {
    CheckAvailable(*in, end, sizeof(T), source_info, context);
    std::memcpy(out, *in, sizeof(T));
    *in += sizeof(T);
  }

  inline void ReadSingleImageRecordIO(Tensor<CPUBackend>& o_image,
                Tensor<CPUBackend>& o_label,
                const uint8_t* input,
                size_t record_size,
                const string& source_info) {
    const uint8_t* end = input + record_size;
    uint32_t magic;
    const uint32_t kMagic = 0xced7230a;
    ReadSingle<uint32_t>(&input, end, &magic, source_info, "magic number");
    DALI_ENFORCE(magic == kMagic, "Invalid RecordIO: wrong magic number");

    uint32_t length_flag;
    ReadSingle(&input, end, &length_flag, source_info, "length flag");
    uint32_t cflag = DecodeFlag(length_flag);
    uint32_t clength = DecodeLength(length_flag);
    CheckAvailable(input, end, clength, source_info, "record payload");
    ImageRecordIOHeader hdr;
    ReadSingle(&input, end, &hdr, source_info, "record header");

    if (clength < sizeof(ImageRecordIOHeader)) {
      throw std::runtime_error(
        make_string("Invalid RecordIO file: ", source_info, " (record payload length: ", clength,
                    " bytes, minimum is ", sizeof(ImageRecordIOHeader), " bytes)."));
    }

    if (hdr.flag == 0) {
      o_label.Resize({1}, DALI_FLOAT);
      o_label.mutable_data<float>()[0] = hdr.label;
    } else {
      o_label.Resize({hdr.flag}, DALI_FLOAT);
    }

    size_t data_size = clength - sizeof(ImageRecordIOHeader);
    size_t label_size = static_cast<size_t>(hdr.flag) * sizeof(float);
    if (label_size > data_size) {
      throw std::runtime_error(
        make_string("Invalid RecordIO file: ", source_info, " (label size: ", label_size,
                    " bytes, available data: ", data_size, " bytes)."));
    }
    size_t image_size = data_size - label_size;
    CheckAvailable(input, end, data_size, source_info, "record data");
    if (cflag == 0) {
      o_image.Resize({static_cast<Index>(image_size)}, DALI_UINT8);
      uint8_t* data = o_image.mutable_data<uint8_t>();
      CopyBytes(data, input + label_size, image_size);
      if (hdr.flag > 0) {
        float * label = o_label.mutable_data<float>();
        CopyBytes(label, input, label_size);
      }
    } else {
      std::vector<uint8_t> temp_vec(data_size);
      CopyBytes(temp_vec.data(), input, data_size);
      input += data_size;

      while (true) {
        size_t pad = PaddedLength(clength) - clength;
        CheckAvailable(input, end, pad, source_info, "record padding");
        input += pad;

        if (cflag != 3) {
          size_t s = temp_vec.size();
          temp_vec.resize(s + sizeof(kMagic));
          std::memcpy(&temp_vec[s], &kMagic, sizeof(kMagic));
        } else {
          break;
        }
        ReadSingle(&input, end, &magic, source_info, "segment magic number");
        DALI_ENFORCE(magic == kMagic, "Invalid RecordIO: wrong magic number");
        ReadSingle(&input, end, &length_flag, source_info, "segment length flag");
        cflag = DecodeFlag(length_flag);
        clength = DecodeLength(length_flag);
        CheckAvailable(input, end, clength, source_info, "segment payload");
        size_t s = temp_vec.size();
        temp_vec.resize(s + clength);
        if (clength > 0) {
          std::memcpy(temp_vec.data() + s, input, clength);
        }
        input += clength;
      }
      if (label_size > temp_vec.size()) {
        throw std::runtime_error(
          make_string("Invalid RecordIO file: ", source_info, " (label size: ", label_size,
                      " bytes, decoded data: ", temp_vec.size(), " bytes)."));
      }
      size_t decoded_image_size = temp_vec.size() - label_size;
      o_image.Resize({static_cast<Index>(decoded_image_size)}, DALI_UINT8);
      uint8_t* data = o_image.mutable_data<uint8_t>();
      if (decoded_image_size > 0) {
        std::memcpy(data, temp_vec.data() + label_size, decoded_image_size);
      }
      if (hdr.flag > 0) {
        float * label = o_label.mutable_data<float>();
        CopyBytes(label, temp_vec.data(), label_size);
      }
    }
  }
};

};  // namespace dali

#endif  // DALI_OPERATORS_READER_PARSER_RECORDIO_PARSER_H_
