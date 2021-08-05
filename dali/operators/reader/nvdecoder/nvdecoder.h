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
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dali/core/dynlink_cuda.h"
#include "dali/core/cuda_stream.h"
#include "dali/operators/reader/nvdecoder/sequencewrapper.h"
#include "dali/operators/reader/nvdecoder/cuvideoparser.h"
#include "dali/operators/reader/nvdecoder/cuvideodecoder.h"
#include "dali/operators/reader/nvdecoder/dynlink_nvcuvid.h"
#include "dali/util/thread_safe_queue.h"

struct AVPacket;
#if HAVE_AVSTREAM_CODECPAR
struct AVCodecParameters;
using CodecParameters = AVCodecParameters;
#else
struct AVCodecContext;
using CodecParameters = AVCodecContext;
#endif

#define ALIGN16(value) ((((value) + 15) >> 4) << 4)

namespace dali {

#define NVDECODER_SUPPORTED_TYPES (float, uint8_t)

struct FrameReq {
  std::string filename;
  int frame;
  int count;
  int stride;
  AVRational frame_base;
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

  int decode_packet(AVPacket* pkt, int64_t start_time, AVRational stream_base,
                    const CodecParameters*);

  void push_req(FrameReq req);

  void receive_frames(SequenceWrapper& batch);

  void finish();

 private:
  int decode_av_packet(AVPacket* pkt, int64_t start_time, AVRational stream_base);

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
    cudaTextureObject_t object_ = 0;
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
  CUDAStream stream_;

  bool rgb_;
  DALIDataType dtype_;
  bool normalized_;

  CUdevice device_;
  CUVideoParser parser_;
  CUVideoDecoder decoder_;

  AVRational nv_time_base_ = {1, 10000000};

  std::vector<uint8_t> frame_in_use_;
  ThreadSafeQueue<FrameReq> recv_queue_;
  ThreadSafeQueue<CUVIDPARSERDISPINFO*> frame_queue_;
  FrameReq current_recv_;

  using TexID = std::tuple<uint8_t*, ScaleMethod, uint16_t, uint16_t, unsigned int>;

  struct tex_hash {
      // hash_combine taken from
      // http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3876.pdf
      template <typename T>
      inline void hash_combine(size_t& seed, const T& value) const {
        std::hash<T> hasher;
        seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      }

      std::size_t operator () (const TexID& tex) const {
        size_t seed = 0;
        hash_combine(seed, std::get<0>(tex));
        hash_combine(seed, std::get<1>(tex));
        hash_combine(seed, std::get<2>(tex));
        hash_combine(seed, std::get<3>(tex));
        hash_combine(seed, std::get<4>(tex));

        return seed;
      }
  };

  std::unordered_map<TexID, TextureObjects, tex_hash> textures_;

  volatile bool stop_;
  std::exception_ptr captured_exception_;

  std::thread thread_convert_;
};

}  // namespace dali

namespace std {
template<>
struct hash<dali::ScaleMethod> {
 public:
  std::size_t operator()(dali::ScaleMethod const& s) const noexcept {
  return std::hash<int>()(s);
  }
};

}  // namespace std


#endif  // DALI_OPERATORS_READER_NVDECODER_NVDECODER_H_
