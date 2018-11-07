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

#include "dali/pipeline/operators/reader/nvdecoder/cuvideodecoder.h"
#include "dali/pipeline/operators/reader/nvdecoder/cuvideoparser.h"
#include "dali/pipeline/operators/reader/nvdecoder/nvdecoder.h"
#include "dali/error_handling.h"

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

    if (static_cast<size_t>(eChromaFormat) >= 0
        && static_cast<size_t>(eChromaFormat)
           < sizeof(aChromaFormatName) / sizeof(aChromaFormatName[0])) {
        return aChromaFormatName[eChromaFormat].name;
    }
    return "Unknown";
}

}  // namespace

CUVideoDecoder::CUVideoDecoder() : decoder_{0},
                                   decoder_info_{}, initialized_{false} {
}

CUVideoDecoder::CUVideoDecoder(CUvideodecoder decoder)
    : decoder_{decoder}, decoder_info_{}, initialized_{true} {
}

CUVideoDecoder::~CUVideoDecoder() {
    if (initialized_) {
        CUDA_CALL(cuvidDestroyDecoder(decoder_));
    }
}

CUVideoDecoder::CUVideoDecoder(CUVideoDecoder&& other)
    : decoder_{other.decoder_}, initialized_{other.initialized_} {
    other.decoder_ = 0;
    other.initialized_ = false;
}

CUVideoDecoder& CUVideoDecoder::operator=(CUVideoDecoder&& other) {
    if (initialized_) {
        CUDA_CALL(cuvidDestroyDecoder(decoder_));
    }
    decoder_ = other.decoder_;
    initialized_ = other.initialized_;
    other.decoder_ = 0;
    other.initialized_ = false;
    return *this;
}

int CUVideoDecoder::initialize(CUVIDEOFORMAT* format) {
    if (initialized_) {
        if ((format->codec != decoder_info_.CodecType) ||
            (format->coded_width != decoder_info_.ulWidth) ||
            (format->coded_height != decoder_info_.ulHeight) ||
            (format->chroma_format != decoder_info_.ChromaFormat)) {
            DALI_FAIL("Encountered a dynamic video format change.");
            return 0;
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
    CUDA_CALL(cuvidGetDecoderCaps(&caps));
    if (!caps.bIsSupported) {
        std::stringstream ss;
        ss << "Unsupported Codec " << GetVideoCodecString(format->codec)
            << " with chroma format "
            << GetVideoChromaFormatString(format->chroma_format);
        throw std::runtime_error(ss.str());
    }
    LOG_LINE << "NVDEC Capabilities" << std::endl
        << "\tMax width : " << caps.nMaxWidth << std::endl
        << "\tMax height : " << caps.nMaxHeight << std::endl
        << "\tMax MB count : " << caps.nMaxMBCount << std::endl
        << "\tMin width : " << caps.nMinWidth << std::endl
        << "\tMin height :" << caps.nMinHeight << std::endl;
    if (format->coded_width < caps.nMinWidth ||
        format->coded_height < caps.nMinHeight) {
        throw std::runtime_error("Video is too small in at least one dimension.");
    }
    if (format->coded_width > caps.nMaxWidth ||
        format->coded_height > caps.nMaxHeight) {
        throw std::runtime_error("Video is too large in at least one dimension.");
    }
    if (format->coded_width * format->coded_height / 256 > caps.nMaxMBCount) {
        throw std::runtime_error("Video is too large (too many macroblocks).");
    }

    decoder_info_.CodecType = format->codec;
    decoder_info_.ulWidth = format->coded_width;
    decoder_info_.ulHeight = format->coded_height;
    decoder_info_.ulNumDecodeSurfaces = 20;
    decoder_info_.ChromaFormat = format->chroma_format;
    decoder_info_.OutputFormat = cudaVideoSurfaceFormat_NV12;
    decoder_info_.bitDepthMinus8 = format->bit_depth_luma_minus8;
    decoder_info_.DeinterlaceMode = cudaVideoDeinterlaceMode_Adaptive;
    decoder_info_.ulTargetWidth = format->display_area.right - format->display_area.left;
    decoder_info_.ulTargetHeight = format->display_area.bottom - format->display_area.top;

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

    CUDA_CALL(cuvidCreateDecoder(&decoder_, &decoder_info_));
    initialized_ = true;
    return 1;
}

bool CUVideoDecoder::initialized() const {
    return initialized_;
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
