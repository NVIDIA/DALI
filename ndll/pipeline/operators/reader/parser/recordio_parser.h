// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_READER_PARSER_RECORDIO_PARSER_H_
#define NDLL_PIPELINE_OPERATORS_READER_PARSER_RECORDIO_PARSER_H_

#include <string>

#include "ndll/pipeline/operators/reader/parser/parser.h"

namespace ndll {

struct ImageRecordIOHeader {
  uint32_t flag;
  float label;
  uint64_t image_id[2];
};

class RecordIOParser : public Parser {
 public:
  explicit RecordIOParser(const OpSpec& spec) :
    Parser(spec) {
  }

  void Parse(const uint8_t* data, const size_t size, SampleWorkspace* ws) override {
    auto* image = ws->Output<CPUBackend>(0);
    auto* label = ws->Output<CPUBackend>(1);
    ReadSingleImageRecordIO(image, label, data);
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

  inline void ReadSingleImageRecordIO(Tensor<CPUBackend>* o_image,
                Tensor<CPUBackend> *o_label,
                const uint8_t* input) {
    uint32_t magic;
    const uint32_t kMagic = 0xced7230a;
    ReadSingle<uint32_t>(&input, &magic);
    NDLL_ENFORCE(magic == kMagic, "Invalid RecordIO: wrong magic number");

    uint32_t length_flag;
    ReadSingle(&input, &length_flag);
    uint32_t cflag = DecodeFlag(length_flag);
    uint32_t clength = DecodeLength(length_flag);
    ImageRecordIOHeader hdr;
    ReadSingle(&input, &hdr);

    o_label->Resize({1});
    o_label->mutable_data<float>()[0] = hdr.label;

    int64_t data_size = clength - sizeof(ImageRecordIOHeader);
    o_image->Resize({data_size});

    uint8_t* data = o_image->mutable_data<uint8_t>();
    memcpy(data, input, data_size);
    input += data_size;

    if (cflag != 0) {
      while (true) {
        size_t pad = clength - (((clength + 3U) >> 2U) << 2U);
        input += pad;

        if (cflag != 3) {
          size_t s = o_image->nbytes();
          o_image->Resize({static_cast<int64_t>(s + sizeof(kMagic))});
          memcpy(data + s, &kMagic, sizeof(kMagic));
        } else {
          break;
        }
        ReadSingle(&input, &magic);
        ReadSingle(&input, &length_flag);
        cflag = DecodeFlag(length_flag);
        clength = DecodeLength(length_flag);
        size_t s = o_image->nbytes();
        o_image->Resize({static_cast<int64_t>(s + clength)});
        memcpy(data + s, input, clength);
        input += clength;
      }
    }
  }
};

};  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_READER_PARSER_RECORDIO_PARSER_H_
