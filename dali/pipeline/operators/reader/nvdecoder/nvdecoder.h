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

#ifndef DALI_PIPELINE_OPERATORS_READER_NVDECODER_NVDECODER_H_
#define DALI_PIPELINE_OPERATORS_READER_NVDECODER_NVDECODER_H_

extern "C" {
#include <libavformat/avformat.h>
}
#include <libavcodec/avcodec.h>


#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dali/util/cucontext.h"
#include "dali/util/dynlink_cuda.h"
#include "dali/pipeline/operators/reader/nvdecoder/sequencewrapper.h"
#include "dali/pipeline/operators/reader/nvdecoder/cuvideoparser.h"
#include "dali/pipeline/operators/reader/nvdecoder/cuvideodecoder.h"
#include "dali/pipeline/operators/reader/nvdecoder/dynlink_nvcuvid.h"
#include "dali/util/thread_safe_queue.h"
#include "dali/util/custream.h"

struct AVPacket;
#if HAVE_AVSTREAM_CODECPAR
struct AVCodecParameters;
using CodecParameters = AVCodecParameters;
#else
struct AVCodecContext;
using CodecParameters = AVCodecContext;
#endif

namespace dali {

struct FrameReq {
  std::string filename;
  int frame;
  int count;
};

enum ScaleMethod {
    /**
     * The value for the nearest neighbor is used, no interpolation
     */
    ScaleMethod_Nearest,

    /**
     * Simple bilinear interpolation of four nearest neighbors
     */
    ScaleMethod_Linear
};

class NvDecoder {
 public:
  NvDecoder(int device_id,
            const CodecParameters* codecpar,
            AVRational time_base,
            DALIImageType image_type,
            DALIDataType dtype,
            bool normalized);

  NvDecoder(const NvDecoder&) = default;
  NvDecoder(NvDecoder&&) = default;
  NvDecoder& operator=(const NvDecoder&) = default;
  NvDecoder& operator=(NvDecoder&&) = default;
  ~NvDecoder();

  bool initialized() const;

  // NVDEC callbacks
  static int handle_sequence(void* user_data, CUVIDEOFORMAT* format);
  static int handle_decode(void* user_data, CUVIDPICPARAMS* pic_params);
  static int handle_display(void* user_data, CUVIDPARSERDISPINFO* disp_info);

  int decode_packet(AVPacket* pkt);

  void push_req(FrameReq req);

  void receive_frames(SequenceWrapper& batch);

  void finish();

 private:
  int decode_av_packet(AVPacket* pkt);

  void record_sequence_event_(SequenceWrapper& sequence);

  // implem functions called in the callback
  int handle_sequence_(CUVIDEOFORMAT* format);
  int handle_decode_(CUVIDPICPARAMS* pic_params);
  int handle_display_(CUVIDPARSERDISPINFO* disp_info);

  class MappedFrame {
   public:
    MappedFrame();
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

  class TextureObject {
   public:
    TextureObject();
    TextureObject(const cudaResourceDesc* pResDesc,
                  const cudaTextureDesc* pTexDesc,
                  const cudaResourceViewDesc* pResViewDesc);
    ~TextureObject();
    TextureObject(TextureObject&& other);
    TextureObject& operator=(TextureObject&& other);
    TextureObject(const TextureObject&) = delete;
    TextureObject& operator=(const TextureObject&) = delete;
    operator cudaTextureObject_t() const;
   private:
    bool valid_;
    cudaTextureObject_t object_;
  };

  struct TextureObjects {
    TextureObject luma;
    TextureObject chroma;
  };

  const TextureObjects& get_textures(uint8_t* input, unsigned int input_pitch,
                                     uint16_t input_width, uint16_t input_height,
                                     ScaleMethod scale_method);
  void convert_frame(const MappedFrame& frame, SequenceWrapper& sequence,
                     int index);


  const int device_id_;
  CUStream stream_;
  const CodecParameters* codecpar_;

  bool rgb_;
  DALIDataType dtype_;
  bool normalized_;

  CUdevice device_;
  CUContext context_;
  CUVideoParser parser_;
  CUVideoDecoder decoder_;

  AVRational time_base_;
  AVRational nv_time_base_ = {1, 10000000};
  AVRational frame_base_;

  std::vector<uint8_t> frame_in_use_;
  ThreadSafeQueue<FrameReq> recv_queue_;
  ThreadSafeQueue<CUVIDPARSERDISPINFO*> frame_queue_;
  FrameReq current_recv_;

  using TexID = std::tuple<uint8_t*, ScaleMethod>;
  struct tex_hash {
      std::hash<uint8_t*> ptr_hash;
      std::hash<int> scale_hash;
      std::size_t operator () (const TexID& tex) const {
          return ptr_hash(std::get<0>(tex))
                  ^ scale_hash(std::get<1>(tex));
      }
  };

  std::unordered_map<TexID, TextureObjects, tex_hash> textures_;

  volatile bool done_;

  std::thread thread_convert_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_NVDECODER_NVDECODER_H_
