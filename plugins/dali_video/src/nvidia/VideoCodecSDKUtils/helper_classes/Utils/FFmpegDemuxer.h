/*
 * This copyright notice applies to this file only
 *
 * SPDX-FileCopyrightText: Copyright (c) 2010-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#pragma once

extern "C" {
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavcodec/avcodec.h>
/* Explicitly include bsf.h when building against FFmpeg 4.3 (libavcodec 58.45.100) or later for backward compatibility */
#if LIBAVCODEC_VERSION_INT >= 3824484
#include <libavcodec/bsf.h>
#endif
}
#ifndef DEMUX_ONLY
#include "cuviddec.h"
#include "nvcuvid.h"
#endif
#include "NvCodecUtils.h"
using namespace std;
//---------------------------------------------------------------------------
//! \file FFmpegDemuxer.h 
//! \brief Provides functionality for stream demuxing
//!
//! This header file is used by Decode/Transcode apps to demux input video clips before decoding frames from it. 
//---------------------------------------------------------------------------


enum SeekMode {
    /* Seek for exact frame number.
     * Suited for standalone demuxer seek. */
    EXACT_FRAME = 0,
    /* Seek for previous key frame in past.
     * Suitable for seek & decode.  */
     PREV_KEY_FRAME = 1,

     SEEK_MODE_NUM_ELEMS
};

enum SeekCriteria {
    /* Seek frame by number.
     */
    BY_NUMBER = 0,
    /* Seek frame by timestamp.
     */
     BY_TIMESTAMP = 1,

     SEEK_CRITERIA_NUM_ELEMS
};

struct SeekContext {
    /* Will be set to false for default ctor, true otherwise;
     */
    bool use_seek;

    /* Frame we want to get. Set by user.
     * Shall be set to frame timestamp in case seek is done by time.
     */
    uint64_t seek_frame;

    /* Mode in which we seek. */
    SeekMode mode;

    /* Criteria by which we seek. */
    SeekCriteria crit;

    /* PTS of frame found after seek. */
    int64_t out_frame_pts;

    /* Duration of frame found after seek. */
    int64_t out_frame_duration;

    /* Number of frames that were decoded during seek. */
    uint64_t num_frames_decoded;

    SeekContext()
        : use_seek(false), seek_frame(0), mode(PREV_KEY_FRAME), crit(BY_NUMBER),
        out_frame_pts(0), out_frame_duration(0), num_frames_decoded(0U)
    {
    }

    SeekContext(uint64_t frame_id)
        : use_seek(true), seek_frame(frame_id), mode(PREV_KEY_FRAME),
        crit(BY_NUMBER), out_frame_pts(0), out_frame_duration(0),
        num_frames_decoded(0U)
    {
    }


    SeekContext& operator=(const SeekContext& other)
    {
        use_seek = other.use_seek;
        seek_frame = other.seek_frame;
        mode = other.mode;
        crit = other.crit;
        out_frame_pts = other.out_frame_pts;
        out_frame_duration = other.out_frame_duration;
        num_frames_decoded = other.num_frames_decoded;
        return *this;
    }
};

/**
* @brief libavformat wrapper class. Retrieves the elementary encoded stream from the container format.
*/
class FFmpegDemuxer {
private:
    AVFormatContext *fmtc = NULL;
    AVIOContext *avioc = NULL;
    AVPacket* pkt = NULL; /*!< AVPacket stores compressed data typically exported by demuxers and then passed as input to decoders */
    AVPacket* pktFiltered = NULL;
    AVBSFContext *bsfc = NULL;

    int iVideoStream;
    bool bMp4H264, bMp4HEVC, bMp4MPEG4, is_seekable;
    AVCodecID eVideoCodec;
    AVPixelFormat eChromaFormat;
    int nWidth, nHeight, nBitDepth, nBPP, nChromaHeight;
    double timeBase = 0.0;
    int64_t userTimeScale = 0; 
    double framerate;
    double avg_framerate;
    AVColorSpace color_space;
    AVColorRange color_range;

    uint8_t *pDataWithHeader = NULL;

    unsigned int frameCount = 0;

public:
    class DataProvider {
    public:
        virtual ~DataProvider() {}
        virtual int GetData(uint8_t *pBuf, int nBuf) = 0;
    };

private:
    /**
    *   @brief  Private constructor to initialize libavformat resources.
    *   @param  fmtc - Pointer to AVFormatContext allocated inside avformat_open_input()
    */
    FFmpegDemuxer(AVFormatContext *fmtc, int64_t timeScale = 1000 /*Hz*/) : fmtc(fmtc) {
        if (!fmtc) {
            LOG(ERROR) << "No AVFormatContext provided.";
            return;
        }

        // Allocate the AVPackets and initialize to default values
        pkt = av_packet_alloc();
        pktFiltered = av_packet_alloc();
        if (!pkt || !pktFiltered) {
            LOG(ERROR) << "AVPacket allocation failed";
            return;
        }

        LOG(INFO) << "Media format: " << fmtc->iformat->long_name << " (" << fmtc->iformat->name << ")";

        ck(avformat_find_stream_info(fmtc, NULL));
        iVideoStream = av_find_best_stream(fmtc, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
        if (iVideoStream < 0) {
            LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__ << " " << "Could not find stream in input file";
            av_packet_free(&pkt);
            av_packet_free(&pktFiltered);
            return;
        }

        //fmtc->streams[iVideoStream]->need_parsing = AVSTREAM_PARSE_NONE;
        eVideoCodec = fmtc->streams[iVideoStream]->codecpar->codec_id;
        nWidth = fmtc->streams[iVideoStream]->codecpar->width;
        nHeight = fmtc->streams[iVideoStream]->codecpar->height;
        eChromaFormat = (AVPixelFormat)fmtc->streams[iVideoStream]->codecpar->format;
        AVRational rTimeBase = fmtc->streams[iVideoStream]->time_base;
        timeBase = av_q2d(rTimeBase);
        userTimeScale = timeScale;
        framerate = (double)fmtc->streams[iVideoStream]->r_frame_rate.num /
            (double)fmtc->streams[iVideoStream]->r_frame_rate.den;
        avg_framerate = (double)fmtc->streams[iVideoStream]->avg_frame_rate.num /
            (double)fmtc->streams[iVideoStream]->avg_frame_rate.den;
        // Set bit depth, chroma height, bits per pixel based on eChromaFormat of input
        eChromaFormat = (AVPixelFormat)fmtc->streams[iVideoStream]->codecpar->format;
        color_space = fmtc->streams[iVideoStream]->codecpar->color_space;
        color_range = fmtc->streams[iVideoStream]->codecpar->color_range;
        switch (eChromaFormat)
        {
        case AV_PIX_FMT_YUV420P10LE:
        case AV_PIX_FMT_GRAY10LE:   // monochrome is treated as 420 with chroma filled with 0x0
            nBitDepth = 10;
            nChromaHeight = (nHeight + 1) >> 1;
            nBPP = 2;
            break;
        case AV_PIX_FMT_YUV420P12LE:
            nBitDepth = 12;
            nChromaHeight = (nHeight + 1) >> 1;
            nBPP = 2;
            break;
        case AV_PIX_FMT_YUV444P10LE:
            nBitDepth = 10;
            nChromaHeight = nHeight << 1;
            nBPP = 2;
            break;
        case AV_PIX_FMT_YUV444P12LE:
            nBitDepth = 12;
            nChromaHeight = nHeight << 1;
            nBPP = 2;
            break;
        case AV_PIX_FMT_YUV444P:
            nBitDepth = 8;
            nChromaHeight = nHeight << 1;
            nBPP = 1;
            break;
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_YUVJ420P:
        case AV_PIX_FMT_YUVJ422P:   // jpeg decoder output is subsampled to NV12 for 422/444 so treat it as 420
        case AV_PIX_FMT_YUVJ444P:   // jpeg decoder output is subsampled to NV12 for 422/444 so treat it as 420
        case AV_PIX_FMT_GRAY8:      // monochrome is treated as 420 with chroma filled with 0x0
            nBitDepth = 8;
            nChromaHeight = (nHeight + 1) >> 1;
            nBPP = 1;
            break;
        default:
            LOG(WARNING) << "ChromaFormat not recognized. Assuming 420";
            eChromaFormat = AV_PIX_FMT_YUV420P;
            nBitDepth = 8;
            nChromaHeight = (nHeight + 1) >> 1;
            nBPP = 1;
        }

        bMp4H264 = eVideoCodec == AV_CODEC_ID_H264 && (
                !strcmp(fmtc->iformat->long_name, "QuickTime / MOV") 
                || !strcmp(fmtc->iformat->long_name, "FLV (Flash Video)") 
                || !strcmp(fmtc->iformat->long_name, "Matroska / WebM")
            );
        bMp4HEVC = eVideoCodec == AV_CODEC_ID_HEVC && (
                !strcmp(fmtc->iformat->long_name, "QuickTime / MOV")
                || !strcmp(fmtc->iformat->long_name, "FLV (Flash Video)")
                || !strcmp(fmtc->iformat->long_name, "Matroska / WebM")
            );

        bMp4MPEG4 = eVideoCodec == AV_CODEC_ID_MPEG4 && (
                !strcmp(fmtc->iformat->long_name, "QuickTime / MOV")
                || !strcmp(fmtc->iformat->long_name, "FLV (Flash Video)")
                || !strcmp(fmtc->iformat->long_name, "Matroska / WebM")
            );

        // Initialize bitstream filter and its required resources
        if (bMp4H264) {
            const AVBitStreamFilter *bsf = av_bsf_get_by_name("h264_mp4toannexb");
            if (!bsf) {
                LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__ << " " << "av_bsf_get_by_name() failed";
                av_packet_free(&pkt);
                av_packet_free(&pktFiltered);
                return;
            }
            ck(av_bsf_alloc(bsf, &bsfc));
            avcodec_parameters_copy(bsfc->par_in, fmtc->streams[iVideoStream]->codecpar);
            ck(av_bsf_init(bsfc));
        }
        if (bMp4HEVC) {
            const AVBitStreamFilter *bsf = av_bsf_get_by_name("hevc_mp4toannexb");
            if (!bsf) {
                LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__ << " " << "av_bsf_get_by_name() failed";
                av_packet_free(&pkt);
                av_packet_free(&pktFiltered);
                return;
            }
            ck(av_bsf_alloc(bsf, &bsfc));
            avcodec_parameters_copy(bsfc->par_in, fmtc->streams[iVideoStream]->codecpar);
            ck(av_bsf_init(bsfc));
        }

        /* Some inputs doesn't allow seek functionality.
        * Check this ahead of time. */
        is_seekable = fmtc->iformat->read_seek || fmtc->iformat->read_seek2;
    }

    AVFormatContext *CreateFormatContext(DataProvider *pDataProvider) {

        AVFormatContext *ctx = NULL;
        if (!(ctx = avformat_alloc_context())) {
            LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__;
            return NULL;
        }

        uint8_t *avioc_buffer = NULL;
        int avioc_buffer_size = 8 * 1024 * 1024;
        avioc_buffer = (uint8_t *)av_malloc(avioc_buffer_size);
        if (!avioc_buffer) {
            LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__;
            return NULL;
        }
        avioc = avio_alloc_context(avioc_buffer, avioc_buffer_size,
            0, pDataProvider, &ReadPacket, NULL, NULL);
        if (!avioc) {
            LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__;
            return NULL;
        }
        ctx->pb = avioc;

        ck(avformat_open_input(&ctx, NULL, NULL, NULL));
        return ctx;
    }

    /**
    *   @brief  Allocate and return AVFormatContext*.
    *   @param  szFilePath - Filepath pointing to input stream.
    *   @return Pointer to AVFormatContext
    */
     AVFormatContext *CreateFormatContext(const char *szFilePath) {
        avformat_network_init();

        AVFormatContext *ctx = NULL;
        ck(avformat_open_input(&ctx, szFilePath, NULL, NULL));
        return ctx;
    }

public:
    FFmpegDemuxer(const char *szFilePath, int64_t timescale = 1000 /*Hz*/) : FFmpegDemuxer(CreateFormatContext(szFilePath), timescale) {}
    FFmpegDemuxer(DataProvider *pDataProvider) : FFmpegDemuxer(CreateFormatContext(pDataProvider)) {avioc = fmtc->pb;}
    ~FFmpegDemuxer() {

        if (!fmtc) {
            return;
        }

        if (pkt) {
            av_packet_free(&pkt);
        }
        if (pktFiltered) {
            av_packet_free(&pktFiltered);
        }

        if (bsfc) {
            av_bsf_free(&bsfc);
        }

        avformat_close_input(&fmtc);

        if (avioc) {
            av_freep(&avioc->buffer);
            av_freep(&avioc);
        }

        if (pDataWithHeader) {
            av_free(pDataWithHeader);
        }
    }
    AVCodecID GetVideoCodec() {
        return eVideoCodec;
    }
    AVPixelFormat GetChromaFormat() {
        return eChromaFormat;
    }
    int GetWidth() {
        return nWidth;
    }
    int GetHeight() {
        return nHeight;
    }
    int GetBitDepth() {
        return nBitDepth;
    }
    int GetFrameSize() {
        return nWidth * (nHeight + nChromaHeight) * nBPP;
    }

    double GetFrameRate()
    {
        return framerate;
    }

    AVPixelFormat GetPixelFormat() const { return eChromaFormat; }

    AVColorSpace GetColorSpace() const { return color_space; }

    AVColorRange GetColorRange() const { return color_range; }

    bool IsVFR() const { 
        return framerate != avg_framerate; 
    }
    int64_t TsFromTime(double ts_sec)
    {
        /* Internal timestamp representation is integer, so multiply to AV_TIME_BASE
         * and switch to fixed point precision arithmetics; */
        auto const ts_tbu = llround(ts_sec * AV_TIME_BASE);

        // Rescale the timestamp to value represented in stream base units;
        AVRational factor;
        factor.num = 1;
        factor.den = AV_TIME_BASE;
        return av_rescale_q(ts_tbu, factor, fmtc->streams[iVideoStream]->time_base);
    }

    int64_t TsFromFrameNumber(int64_t frame_num)
    {
        auto const ts_sec = (double)frame_num / framerate;
        return TsFromTime(ts_sec);
    }
    bool Demux(uint8_t **ppVideo, int *pnVideoBytes, int64_t *pts = NULL) {
       
        NVTX_SCOPED_RANGE("demux")
        if (!fmtc) {
            return false;
        }

        *pnVideoBytes = 0;

        if (pkt->data) {
            av_packet_unref(pkt);
        }

        int e = 0;
        while ((e = av_read_frame(fmtc, pkt)) >= 0 && pkt->stream_index != iVideoStream) {
            av_packet_unref(pkt);
        }
        if (e < 0) {
            return false;
        }

        if (bMp4H264 || bMp4HEVC) {
            if (pktFiltered->data) {
                av_packet_unref(pktFiltered);
            }
            ck(av_bsf_send_packet(bsfc, pkt));
            ck(av_bsf_receive_packet(bsfc, pktFiltered));
            *ppVideo = pktFiltered->data;
            *pnVideoBytes = pktFiltered->size;
            if (pts)
                *pts = (int64_t) (pktFiltered->pts * userTimeScale * timeBase);
        } else {

            if (bMp4MPEG4 && (frameCount == 0)) {

                int extraDataSize = fmtc->streams[iVideoStream]->codecpar->extradata_size;

                if (extraDataSize > 0) {

                    // extradata contains start codes 00 00 01. Subtract its size
                    pDataWithHeader = (uint8_t *)av_malloc(extraDataSize + pkt->size - 3*sizeof(uint8_t));

                    if (!pDataWithHeader) {
                        LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__;
                        return false;
                    }

                    memcpy(pDataWithHeader, fmtc->streams[iVideoStream]->codecpar->extradata, extraDataSize);
                    memcpy(pDataWithHeader+extraDataSize, pkt->data+3, pkt->size - 3*sizeof(uint8_t));

                    *ppVideo = pDataWithHeader;
                    *pnVideoBytes = extraDataSize + pkt->size - 3*sizeof(uint8_t);
                }

            } else {
                *ppVideo = pkt->data;
                *pnVideoBytes = pkt->size;
            }

            if (pts)
                *pts = (int64_t)(pkt->pts * userTimeScale * timeBase);
        }

        frameCount++;

        return true;
    }

    bool Seek(SeekContext& seekCtx, uint8_t** ppVideo, int* pnVideoBytes)
    {
        
        /* !!! IMPORTANT !!!
         * Across this function packet decode timestamp (DTS) values are used to
         * compare given timestamp against. This is done for reason. DTS values shall
         * monotonically increase during the course of decoding unlike PTS velues
         * which may be affected by frame reordering due to B frames presence.
         */

        if (!is_seekable) {
            cerr << "Seek isn't supported for this input." << endl;
            return false;
        }

        if (IsVFR() && (BY_NUMBER == seekCtx.crit)) {
            cerr << "Can't seek by frame number in VFR sequences. Seek by timestamp "
                "instead."
                << endl;
            return false;
        }

        // Seek for single frame;
        auto seek_frame = [&](SeekContext const& seek_ctx, int flags) {
            bool seek_backward = true;
            int64_t timestamp = 0;
            int ret = 0;

            switch (seek_ctx.crit) {
            case BY_NUMBER:
                timestamp = TsFromFrameNumber(seek_ctx.seek_frame);
                ret = av_seek_frame(fmtc, iVideoStream, timestamp,
                    seek_backward ? AVSEEK_FLAG_BACKWARD | flags : flags);
                break;
            case BY_TIMESTAMP:
                timestamp = TsFromTime(seek_ctx.seek_frame);
                ret = av_seek_frame(fmtc, iVideoStream, timestamp,
                    seek_backward ? AVSEEK_FLAG_BACKWARD | flags : flags);
                break;
            default:
                throw runtime_error("Invalid seek mode");
            }

            if (ret < 0) {
                throw runtime_error("Error seeking for frame: ");
            }
        };

        // Check if frame satisfies seek conditions;
        auto is_seek_done = [&](PacketData& pkt_data, SeekContext const& seek_ctx) {
            int64_t target_ts = 0;

            switch (seek_ctx.crit) {
            case BY_NUMBER:
                target_ts = TsFromFrameNumber(seek_ctx.seek_frame);
                break;
            case BY_TIMESTAMP:
                target_ts = TsFromTime(seek_ctx.seek_frame);
                break;
            default:
                throw runtime_error("Invalid seek criteria");
                break;
            }

            if (pkt_data.dts == target_ts) {
                return 0;
            }
            else if (pkt_data.dts > target_ts) {
                return 1;
            }
            else {
                return -1;
            };
        };

        /* This will seek for exact frame number;
         * Note that decoder may not be able to decode such frame; */
        auto seek_for_exact_frame = [&](PacketData& pkt_data, SeekContext& seek_ctx) {
            // Repetititive seek until seek condition is satisfied;
            SeekContext tmp_ctx(seek_ctx.seek_frame);
            seek_frame(tmp_ctx, AVSEEK_FLAG_ANY);

            int condition = 0;
            do {
                if (!Demux(ppVideo, pnVideoBytes)) {
                    break;
                }
                condition = is_seek_done(pkt_data, seek_ctx);

                // We've gone too far and need to seek backwards;
                if (condition > 0) {
                    tmp_ctx.seek_frame--;
                    seek_frame(tmp_ctx, AVSEEK_FLAG_ANY);
                }
                // Need to read more frames until we reach requested number;
                else if (condition < 0) {
                    continue;
                }
            } while (0 != condition);

            seek_ctx.out_frame_pts = pkt_data.pts;
            seek_ctx.out_frame_duration = pkt_data.duration;
        };

        // Seek for closest key frame in the past;
        auto seek_for_prev_key_frame = [&](PacketData& pkt_data,
            SeekContext& seek_ctx) {
                seek_frame(seek_ctx, AVSEEK_FLAG_BACKWARD);

                Demux(ppVideo, pnVideoBytes);
                seek_ctx.out_frame_pts = pkt_data.pts;
                seek_ctx.out_frame_duration = pkt_data.duration;
        };

        PacketData pktData;
        pktData.bsl_data = size_t(*ppVideo);
        pktData.bsl = *pnVideoBytes;

        switch (seekCtx.mode) {
        case EXACT_FRAME:
            seek_for_exact_frame(pktData, seekCtx);
            break;
        case PREV_KEY_FRAME:
            seek_for_prev_key_frame(pktData, seekCtx);
            break;
        default:
            throw runtime_error("Unsupported seek mode");
            break;
        }

        return true;
    }

    static int ReadPacket(void *opaque, uint8_t *pBuf, int nBuf) {
        return ((DataProvider *)opaque)->GetData(pBuf, nBuf);
    }

};

#ifndef DEMUX_ONLY
inline cudaVideoCodec FFmpeg2NvCodecId(AVCodecID id) {
    switch (id) {
    case AV_CODEC_ID_MPEG1VIDEO : return cudaVideoCodec_MPEG1;
    case AV_CODEC_ID_MPEG2VIDEO : return cudaVideoCodec_MPEG2;
    case AV_CODEC_ID_MPEG4      : return cudaVideoCodec_MPEG4;
    case AV_CODEC_ID_WMV3       :
    case AV_CODEC_ID_VC1        : return cudaVideoCodec_VC1;
    case AV_CODEC_ID_H264       : return cudaVideoCodec_H264;
    case AV_CODEC_ID_HEVC       : return cudaVideoCodec_HEVC;
    case AV_CODEC_ID_VP8        : return cudaVideoCodec_VP8;
    case AV_CODEC_ID_VP9        : return cudaVideoCodec_VP9;
    case AV_CODEC_ID_MJPEG      : return cudaVideoCodec_JPEG;
    case AV_CODEC_ID_AV1        : return cudaVideoCodec_AV1;
    default                     : return cudaVideoCodec_NumCodecs;
    }
}
#endif


