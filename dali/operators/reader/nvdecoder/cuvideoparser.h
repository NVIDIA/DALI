// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_NVDECODER_CUVIDEOPARSER_H_
#define DALI_OPERATORS_READER_NVDECODER_CUVIDEOPARSER_H_

#include <algorithm>
#include <cstring>

#include "dali/operators/reader/nvdecoder/dynlink_nvcuvid.h"
#include "dali/core/error_handling.h"
#include "dali/core/cuda_utils.h"


namespace dali {

enum class Codec {
  H264,
  HEVC,
  MPEG4,
  MJPEG,
  VP8,
  VP9
};

class CUVideoParser {
 public:
  CUVideoParser()
    : parser_{0}, parser_info_{}, parser_extinfo_{}, initialized_{false}
  {
  }


  template <typename Decoder>
  void init(Codec codec, Decoder* decoder, int decode_surfaces,
            uint8_t* extradata, int extradata_size) {
    switch (codec) {
      case Codec::H264:
        parser_info_.CodecType = cudaVideoCodec_H264;
        break;
      case Codec::HEVC:
        parser_info_.CodecType = cudaVideoCodec_HEVC;
        parser_info_.ulMaxNumDecodeSurfaces = 20;
        break;
      case Codec::MJPEG:
        parser_info_.CodecType = cudaVideoCodec_JPEG;
        parser_info_.ulMaxNumDecodeSurfaces = 20;
        break;
      case Codec::MPEG4:
        parser_info_.CodecType = cudaVideoCodec_MPEG4;
        parser_info_.ulMaxNumDecodeSurfaces = 20;
        break;
      case Codec::VP9:
        parser_info_.CodecType = cudaVideoCodec_VP9;
        parser_info_.ulMaxNumDecodeSurfaces = 20;
        break;
      case Codec::VP8:
        parser_info_.CodecType = cudaVideoCodec_VP8;
        parser_info_.ulMaxNumDecodeSurfaces = 20;
        break;
      default:
        DALI_FAIL("Invalid codec: must be H.264, HEVC, MPEG4 or VP9");
        return;
    }
    memset(&parser_extinfo_, 0, sizeof(parser_extinfo_));
    parser_info_.ulMaxNumDecodeSurfaces = decode_surfaces;
    parser_info_.pUserData = decoder;

    /* Called before decoding frames and/or whenever there is a fmt change */
    parser_info_.pfnSequenceCallback = Decoder::handle_sequence;

    /* Called when a picture is ready to be decoded (decode order) */
    parser_info_.pfnDecodePicture = Decoder::handle_decode;

    /* Called whenever a picture is ready to be displayed (display order) */
    parser_info_.pfnDisplayPicture = Decoder::handle_display;

    parser_info_.pExtVideoInfo = &parser_extinfo_;
    if (extradata_size > 0) {
      auto hdr_size = std::min(sizeof(parser_extinfo_.raw_seqhdr_data),
                                static_cast<std::size_t>(extradata_size));
      parser_extinfo_.format.seqhdr_data_length = hdr_size;
      memcpy(parser_extinfo_.raw_seqhdr_data, extradata, hdr_size);
    }
    NVCUVID_CALL(cuvidCreateVideoParser(&parser_, &parser_info_));
    initialized_ = true;
  }

  explicit CUVideoParser(CUvideoparser parser)
      : parser_{parser}, initialized_{true}
  {
  }

  ~CUVideoParser() {
    if (initialized_) {
      NVCUVID_CALL(cuvidDestroyVideoParser(parser_));
    }
  }

  explicit CUVideoParser(CUVideoParser&& other)
      : parser_{other.parser_}, initialized_{other.initialized_}
  {
    other.parser_ = 0;
    other.initialized_ = false;
  }

  CUVideoParser& operator=(CUVideoParser&& other) {
    if (initialized_) {
      NVCUVID_CALL(cuvidDestroyVideoParser(parser_));
    }
    parser_ = other.parser_;
    parser_info_ = other.parser_info_;
    parser_extinfo_ = other.parser_extinfo_;
    initialized_ = other.initialized_;
    other.parser_ = 0;
    other.initialized_ = false;
    return *this;
  }

  bool initialized() const {
    return initialized_;
  }

  operator CUvideoparser() const {
    return parser_;
  }

 private:
  CUvideoparser parser_;
  CUVIDPARSERPARAMS parser_info_;
  CUVIDEOFORMATEX parser_extinfo_;

  bool initialized_;
};


}  // namespace dali

#endif  // DALI_OPERATORS_READER_NVDECODER_CUVIDEOPARSER_H_
