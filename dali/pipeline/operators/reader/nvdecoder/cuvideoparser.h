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

#ifndef DALI_PIPELINE_OPERATORS_READER_NVDECODER_CUVIDEOPARSER_H_
#define DALI_PIPELINE_OPERATORS_READER_NVDECODER_CUVIDEOPARSER_H_

#include <cstring>

#include "dali/pipeline/operators/reader/nvdecoder/nvcuvid.h"
#include "dali/error_handling.h"


namespace dali {

enum class Codec {
    H264,
    HEVC
};

class CUVideoParser {
  public:
    CUVideoParser() : parser_{0}, initialized_{false} {}

    CUVideoParser(Codec codec, NvDecoder* decoder, int decode_surfaces)
        : CUVideoParser{codec, decoder, decode_surfaces, nullptr, 0} {}

    CUVideoParser(Codec codec, NvDecoder* decoder, int decode_surfaces,
                  uint8_t* extradata, int extradata_size)
        : parser_{0}, parser_info_{}, parser_extinfo_{}, initialized_{false}
    {
        init_params(codec, decoder, decode_surfaces, extradata, extradata_size);

        CUDA_CALL(cuvidCreateVideoParser(&parser_, &parser_info_));
        initialized_ = true;
    }


    void init_params(Codec codec, NvDecoder* decoder, int decode_surfaces,
                     uint8_t* extradata, int extradata_size) {
        switch (codec) {
            case Codec::H264:
                parser_info_.CodecType = cudaVideoCodec_H264;
                break;
            case Codec::HEVC:
                parser_info_.CodecType = cudaVideoCodec_HEVC;
                // this can probably be better
                parser_info_.ulMaxNumDecodeSurfaces = 20;
                break;
            default:
                std::cerr << "Invalid codec\n";
                return;
        }
        parser_info_.ulMaxNumDecodeSurfaces = decode_surfaces;
        parser_info_.pUserData = decoder;

        /* Called before decoding frames and/or whenever there is a fmt change */
        parser_info_.pfnSequenceCallback = NvDecoder::handle_sequence;

        /* Called when a picture is ready to be decoded (decode order) */
        parser_info_.pfnDecodePicture = NvDecoder::handle_decode;

        /* Called whenever a picture is ready to be displayed (display order) */
        parser_info_.pfnDisplayPicture = NvDecoder::handle_display;

        parser_info_.pExtVideoInfo = &parser_extinfo_;
        if (extradata_size > 0) {
            auto hdr_size = std::min(sizeof(parser_extinfo_.raw_seqhdr_data),
                                     static_cast<std::size_t>(extradata_size));
            parser_extinfo_.format.seqhdr_data_length = hdr_size;
            memcpy(parser_extinfo_.raw_seqhdr_data, extradata, hdr_size);
        }
    }

    CUVideoParser(CUvideoparser parser)
        : parser_{parser}, initialized_{true}
    {
    }

    ~CUVideoParser() {
        if (initialized_) {
            CUDA_CALL(cuvidDestroyVideoParser(parser_));
        }
    }

    CUVideoParser(CUVideoParser&& other)
        : parser_{other.parser_}, initialized_{other.initialized_}
    {
        other.parser_ = 0;
        other.initialized_ = false;
    }

    CUVideoParser& operator=(CUVideoParser&& other) {
        if (initialized_) {
            CUDA_CALL(cuvidDestroyVideoParser(parser_));
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

#endif  // DALI_PIPELINE_OPERATORS_READER_NVDECODER_CUVIDEOPARSER_H_
