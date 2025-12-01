// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_NVDECODER_NVDECODER_H_
#define DALI_OPERATORS_READER_NVDECODER_NVDECODER_H_

extern "C" {
#include <libavformat/avformat.h>
}
#include <libavcodec/avcodec.h>

#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "dali/core/dynlink_cuda.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/operators/video/legacy/reader/nvdecoder/sequencewrapper.h"
#include "dali/operators/video/legacy/reader/nvdecoder/cuvideoparser.h"
#include "dali/operators/video/legacy/reader/nvdecoder/cuvideodecoder.h"
#include "dali/operators/video/dynlink_nvcuvid/dynlink_nvcuvid.h"
#include "dali/util/thread_safe_queue.h"
#if NVML_ENABLED
#include "dali/util/nvml.h"
#endif


struct AVPacket;
#if HAVE_AVSTREAM_CODECPAR
struct AVCodecParameters;
using CodecParameters = AVCodecParameters;
#else
struct AVCodecContext;
using CodecParameters = AVCodecContext;
#endif

#define ALIGN32(value) ((((value) + 31) >> 5) << 5)

namespace dali {

#define NVDECODER_SUPPORTED_TYPES (float, uint8_t)

struct FrameReq {
  std::string filename;
  int frame;
  int count;
  int stride;
  AVRational frame_base;
  bool full_range;
};

enum class VidReqStatus {
  REQ_READY = 0,
  REQ_IN_PROGRESS,
  REQ_NOT_STARTED,
  REQ_ERROR,
};

class NvDecoder {
 public:
  NvDecoder(int device_id,
            const CodecParameters* codecpar,
            DALIImageType image_type,
            DALIDataType dtype,
            bool normalized,
            int max_height,
            int max_width,
            int additional_decode_surfaces);

  // Some of the members are non-movable or non-copyable so the constructors below still end up
  // implicitly deleted, thus marking them explicitly deleted as this class in managed through
  // unique_ptr.
  // The culprits are: CUStream (non-copyable), ThreadSafeQueue (std::mutex) and const members.
  NvDecoder(const NvDecoder&) = delete;
  NvDecoder(NvDecoder&&) = delete;
  NvDecoder& operator=(const NvDecoder&) = delete;
  NvDecoder& operator=(NvDecoder&&) = delete;
  ~NvDecoder();

  bool initialized() const;

  // NVDEC callbacks
  static int handle_sequence(void* user_data, CUVIDEOFORMAT* format);
  static int handle_decode(void* user_data, CUVIDPICPARAMS* pic_params);
  static int handle_display(void* user_data, CUVIDPARSERDISPINFO* disp_info);

  VidReqStatus decode_packet(AVPacket* pkt, int64_t start_time, AVRational stream_base,
                             const CodecParameters*);

  void push_req(FrameReq req);

  void receive_frames(SequenceWrapper& batch);

  void finish();

 private:
  VidReqStatus decode_av_packet(AVPacket* pkt, int64_t start_time, AVRational stream_base);

  void record_sequence_event_(SequenceWrapper& sequence);

  // implem functions called in the callback
  int handle_sequence_(CUVIDEOFORMAT* format);
  int handle_decode_(CUVIDPICPARAMS* pic_params);
  int handle_display_(CUVIDPARSERDISPINFO* disp_info);

  class MappedFrame {
   public:
    MappedFrame() = delete;
    MappedFrame(CUVIDPARSERDISPINFO* disp_info, CUvideodecoder decoder,
                CUstream stream);
    ~MappedFrame();
    MappedFrame(const MappedFrame&) = delete;
    MappedFrame& operator=(const MappedFrame&) = delete;
    MappedFrame(MappedFrame&& other);
    MappedFrame& operator=(MappedFrame&&) = delete;

    uint8_t* get_ptr() const;
    unsigned int get_pitch() const;

    CUVIDPARSERDISPINFO* disp_info;
   private:
    bool valid_;
    CUvideodecoder decoder_;
    CUdeviceptr ptr_;
    unsigned int pitch_;
    CUVIDPROCPARAMS params_;
  };

  const int device_id_;
  CUDAStreamLease stream_;

  bool rgb_;
  DALIDataType dtype_;
  bool normalized_;

  CUdevice device_;
  CUVideoParser parser_;
  CUVideoDecoder decoder_;

  AVRational nv_time_base_ = {1, 10000000};

  std::vector<uint8_t> frame_in_use_;
  std::vector<bool> frame_full_range_;
  ThreadSafeQueue<FrameReq> recv_queue_;
  ThreadSafeQueue<CUVIDPARSERDISPINFO*> frame_queue_;
  FrameReq current_recv_;
  VidReqStatus req_ready_;

  volatile bool stop_;
  std::exception_ptr captured_exception_;

  std::thread thread_convert_;
#if NVML_ENABLED
  nvml::NvmlInstance nvml_handle_;
#endif
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_NVDECODER_NVDECODER_H_
