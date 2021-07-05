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

#include "dali/operators/reader/nvdecoder/cuvideodecoder.h"
#include "dali/operators/reader/nvdecoder/cuvideoparser.h"
#include "dali/operators/reader/nvdecoder/nvdecoder.h"
#include "dali/core/error_handling.h"

namespace dali {

namespace {

const char* GetVideoCodecString(cudaVideoCodec eCodec) {
    static struct {
        cudaVideoCodec eCodec;
        const char *name;
    } aCodecName[] = {
        { cudaVideoCodec_MPEG1,     "MPEG-1"       },
        { cudaVideoCodec_MPEG2,     "MPEG-2"       },
        { cudaVideoCodec_MPEG4,     "MPEG-4 (ASP)" },
        { cudaVideoCodec_VC1,       "VC-1/WMV"     },
        { cudaVideoCodec_H264,      "AVC/H.264"    },
        { cudaVideoCodec_JPEG,      "M-JPEG"       },
        { cudaVideoCodec_H264_SVC,  "H.264/SVC"    },
        { cudaVideoCodec_H264_MVC,  "H.264/MVC"    },
        { cudaVideoCodec_HEVC,      "H.265/HEVC"   },
        { cudaVideoCodec_VP8,       "VP8"          },
        { cudaVideoCodec_VP9,       "VP9"          },
        { cudaVideoCodec_NumCodecs, "Invalid"      },
        { cudaVideoCodec_YUV420,    "YUV  4:2:0"   },
        { cudaVideoCodec_YV12,      "YV12 4:2:0"   },
        { cudaVideoCodec_NV12,      "NV12 4:2:0"   },
        { cudaVideoCodec_YUYV,      "YUYV 4:2:2"   },
        { cudaVideoCodec_UYVY,      "UYVY 4:2:2"   },
    };

    if (eCodec >= 0 && eCodec <= cudaVideoCodec_NumCodecs) {
        return aCodecName[eCodec].name;
    }
    for (size_t i = static_cast<size_t>(cudaVideoCodec_NumCodecs) + 1;
         i < sizeof(aCodecName) / sizeof(aCodecName[0]);
         i++) {
        if (eCodec == aCodecName[i].eCodec) {
          return aCodecName[eCodec].name;
        }
    }
    return "Unknown";
}

const char* GetVideoChromaFormatString(cudaVideoChromaFormat eChromaFormat) {
    static struct {
        cudaVideoChromaFormat eChromaFormat;
        const char *name;
    } aChromaFormatName[] = {
        { cudaVideoChromaFormat_Monochrome, "YUV 400 (Monochrome)" },
        { cudaVideoChromaFormat_420,        "YUV 420"              },
        { cudaVideoChromaFormat_422,        "YUV 422"              },
        { cudaVideoChromaFormat_444,        "YUV 444"              },
    };

    if (static_cast<size_t>(eChromaFormat)
           < sizeof(aChromaFormatName) / sizeof(aChromaFormatName[0])) {
        return aChromaFormatName[eChromaFormat].name;
    }
    return "Unknown";
}

}  // namespace

CUVideoDecoder::CUVideoDecoder(int max_height, int max_width, int additional_decode_surfaces)
                              : decoder_{0}, decoder_info_{}, caps_{},
                                max_height_{max_height}, max_width_{max_width},
                                additional_decode_surfaces_{additional_decode_surfaces} {
}

CUVideoDecoder::CUVideoDecoder() : CUVideoDecoder(0, 0, 0) {
}

CUVideoDecoder::CUVideoDecoder(CUvideodecoder decoder)
    : CUVideoDecoder(0, 0, 0) {
  decoder_ = decoder;
}

CUVideoDecoder::~CUVideoDecoder() {
  if (decoder_) {
    NVCUVID_CALL(cuvidDestroyDecoder(decoder_));
  }
}

CUVideoDecoder::CUVideoDecoder(CUVideoDecoder&& other)
    : decoder_{other.decoder_}, decoder_info_{other.decoder_info_},
      caps_{other.caps_}, max_height_{other.max_height_}, max_width_{other.max_width_},
      additional_decode_surfaces_{other.additional_decode_surfaces_} {
    other.decoder_ = 0;
    other.max_height_ = 0;
    other.max_width_ = 0;
}

CUVideoDecoder& CUVideoDecoder::operator=(CUVideoDecoder&& other) {
    if (decoder_) {
        NVCUVID_CALL(cuvidDestroyDecoder(decoder_));
    }
    decoder_ = other.decoder_;
    max_height_ = other.max_height_;
    max_width_ = other.max_width_;
    other.decoder_ = 0;
    other.max_height_ = 0;
    other.max_width_ = 0;
    return *this;
}

void CUVideoDecoder::reconfigure(unsigned int height, unsigned int width) {
    DALI_ENFORCE(NVCUVID_API_EXISTS(cuvidReconfigureDecoder),
                 "cuvidReconfigureDecoder API is not available.");

    CUVIDRECONFIGUREDECODERINFO reconfigParams = { 0 };

    DALI_ENFORCE(decoder_, "Trying to reconfigure uninitialized decoder");

    DALI_ENFORCE(width >= caps_.nMinWidth && height >= caps_.nMinHeight,
                 make_string("Video is too small in at least one dimension. Provided size is ",
                 width, "x", height, " while the decoder requires at least ", caps_.nMinWidth, "x",
                 caps_.nMinHeight));

    DALI_ENFORCE(width <= caps_.nMaxWidth && height <= caps_.nMaxHeight,
                 make_string("Video is too large in at least one dimension. Provided size is ",
                 width, "x", height, " while the decoder supports at most ", caps_.nMaxWidth, "x",
                 caps_.nMaxHeight));

    DALI_ENFORCE(width * height / 256 <= caps_.nMaxMBCount,
                 make_string("Video is too large (too many macroblocks). Provided video has ",
                 width * height / 256, " blocks calculated as ", width, "*", height, "/256, ",
                 " while the decoder supports up to", caps_.nMaxMBCount, " macroblocks"));

    reconfigParams.display_area.bottom = decoder_info_.display_area.bottom = height;
    reconfigParams.display_area.top = 0;
    reconfigParams.display_area.left = 0;
    reconfigParams.display_area.right = decoder_info_.display_area.right = width;

    decoder_info_.ulTargetWidth = decoder_info_.ulWidth = width;
    reconfigParams.ulTargetWidth = reconfigParams.ulWidth = width;

    decoder_info_.ulTargetHeight = decoder_info_.ulHeight = height;
    reconfigParams.ulTargetHeight = reconfigParams.ulHeight = height;

    reconfigParams.ulNumDecodeSurfaces = decoder_info_.ulNumDecodeSurfaces;


    NVCUVID_CALL(cuvidReconfigureDecoder(decoder_, &reconfigParams));
}

int CUVideoDecoder::initialize(CUVIDEOFORMAT* format) {
    if (decoder_) {
        if ((format->codec != decoder_info_.CodecType) ||
            (format->chroma_format != decoder_info_.ChromaFormat)) {
            DALI_FAIL("Encountered a dynamic video format change.");
        }
        if ((format->coded_width != decoder_info_.ulWidth) ||
            (format->coded_height != decoder_info_.ulHeight)) {
            if (NVCUVID_API_EXISTS(cuvidReconfigureDecoder)) {
              LOG_LINE << "reconfigure decoder";
              CUVideoDecoder::reconfigure(format->coded_height, format->coded_width);
            } else {
             DALI_FAIL("Encountered a dynamic video resolution change. Install Nvidia driver"
                       " version >=396 (x86) or >=415 (Power PC)");
            }
        }
        return 1;
    }

    LOG_LINE << "Hardware Decoder Input Information" << std::endl
        << "\tVideo codec     : " << GetVideoCodecString(format->codec) << std::endl
        << "\tFrame rate      : " << format->frame_rate.numerator
        << "/" << format->frame_rate.denominator
        << " = " << 1.0 * format->frame_rate.numerator / format->frame_rate.denominator
        << " fps" << std::endl
        << "\tSequence format : " << (format->progressive_sequence ? "Progressive" : "Interlaced")
         << std::endl
        << "\tCoded frame size: [" << format->coded_width << ", " << format->coded_height << "]"
        << std::endl
        << "\tDisplay area    : [" << format->display_area.left << ", "
        << format->display_area.top << ", "
        << format->display_area.right << ", " << format->display_area.bottom << "]" << std::endl
        << "\tChroma format   : " << GetVideoChromaFormatString(format->chroma_format) << std::endl
        << "\tBit depth       : " << format->bit_depth_luma_minus8 + 8 << std::endl;

    auto caps = CUVIDDECODECAPS{};
    caps.eCodecType = format->codec;
    caps.eChromaFormat = format->chroma_format;
    caps.nBitDepthMinus8 = format->bit_depth_luma_minus8;
    NVCUVID_CALL(cuvidGetDecoderCaps(&caps));
    if (!caps.bIsSupported) {
        std::stringstream ss;
        ss << "Unsupported Codec " << GetVideoCodecString(format->codec)
            << " with chroma format "
            << GetVideoChromaFormatString(format->chroma_format);
        DALI_WARN(ss.str());
        throw unsupported_exception("Decoder hardware does not support this video codec"
                                    " and/or chroma format");
    }

    caps_ = caps;
    LOG_LINE << "NVDEC Capabilities" << std::endl
        << "\tMax width : " << caps.nMaxWidth << std::endl
        << "\tMax height : " << caps.nMaxHeight << std::endl
        << "\tMax MB count : " << caps.nMaxMBCount << std::endl
        << "\tMin width : " << caps.nMinWidth << std::endl
        << "\tMin height :" << caps.nMinHeight << std::endl;
    if (format->coded_width < caps.nMinWidth ||
        format->coded_height < caps.nMinHeight) {
        DALI_FAIL("Video is too small in at least one dimension.");
    }
    if (format->coded_width > caps.nMaxWidth ||
        format->coded_height > caps.nMaxHeight) {
        DALI_FAIL("Video is too large in at least one dimension.");
    }
    if (format->coded_width * format->coded_height / 256 > caps.nMaxMBCount) {
        DALI_FAIL("Video is too large (too many macroblocks).");
    }

    decoder_info_.CodecType = format->codec;
    decoder_info_.ulWidth = format->coded_width;
    decoder_info_.ulHeight = format->coded_height;
    if (format->min_num_decode_surfaces == 0)
      decoder_info_.ulNumDecodeSurfaces = 20;
    else
      decoder_info_.ulNumDecodeSurfaces = format->min_num_decode_surfaces
                                          + additional_decode_surfaces_;
    decoder_info_.ChromaFormat = format->chroma_format;
    decoder_info_.OutputFormat = cudaVideoSurfaceFormat_NV12;
    decoder_info_.bitDepthMinus8 = format->bit_depth_luma_minus8;
    decoder_info_.DeinterlaceMode = cudaVideoDeinterlaceMode_Adaptive;
    decoder_info_.ulTargetWidth = format->display_area.right - format->display_area.left;
    decoder_info_.ulTargetHeight = format->display_area.bottom - format->display_area.top;
    decoder_info_.ulMaxWidth = static_cast<unsigned long>(max_width_);  // NOLINT
    decoder_info_.ulMaxHeight = static_cast<unsigned long>(max_height_);  // NOLINT

    auto& area = decoder_info_.display_area;
    area.left   = format->display_area.left;
    area.right  = format->display_area.right;
    area.top    = format->display_area.top;
    area.bottom = format->display_area.bottom;
    LOG_LINE << "\tUsing full size : [" << area.left << ", " << area.top
            << "], [" << area.right << ", " << area.bottom << "]" << std::endl;
    decoder_info_.ulNumOutputSurfaces = 2;
    decoder_info_.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    decoder_info_.vidLock = nullptr;

    NVCUVID_CALL(cuvidCreateDecoder(&decoder_, &decoder_info_));
    return decoder_info_.ulNumDecodeSurfaces;
}

bool CUVideoDecoder::initialized() const {
    return decoder_;
}

CUVideoDecoder::operator CUvideodecoder() const {
    return decoder_;
}

uint16_t CUVideoDecoder::width() const {
    return static_cast<uint16_t>(decoder_info_.ulTargetWidth);
}

uint16_t CUVideoDecoder::height() const {
    return static_cast<uint16_t>(decoder_info_.ulTargetHeight);
}

}  // namespace dali
