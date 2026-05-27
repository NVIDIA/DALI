// Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nvimgcodec.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>
#include "dali/operators.h"
#include "dali/operators/image/distortion/jpeg_compression_distortion_op.h"
#include "dali/operators/imgcodec/util/nvimagecodec_types.h"

#if not(WITH_DYNAMIC_NVIMGCODEC_ENABLED)
nvimgcodecStatus_t get_libjpeg_turbo_extension_desc(nvimgcodecExtensionDesc_t *ext_desc);
#endif

namespace dali {

DALI_SCHEMA(JpegCompressionDistortion)
    .DocStr(R"code(Introduces JPEG compression artifacts to RGB images.

JPEG is a lossy compression format which exploits characteristics of natural
images and human visual system to achieve high compression ratios. The information
loss originates from sampling the color information at a lower spatial resolution
than the brightness and from representing high frequency components of the image
with a lower effective bit depth. The conversion to frequency domain and quantization
is applied independently to 8x8 pixel blocks, which introduces additional artifacts
at block boundaries.

This operation produces images by subjecting the input to a transformation that
mimics JPEG compression with given `quality` factor followed by decompression.
)code")
    .NumInput(1)
    .InputLayout({"HWC", "FHWC"})
    .NumOutput(1)
    .AddOptionalArg(
        "quality",
        R"code(JPEG compression quality from 1 (lowest quality) to 100 (highest quality).

Any values outside the range 1-100 will be clamped.)code",
        50, true)
    .AllowSequences();

namespace {

using imgcodec::NvImageCodecCodeStream;
using imgcodec::NvImageCodecDecoder;
using imgcodec::NvImageCodecEncoder;
using imgcodec::NvImageCodecFuture;
using imgcodec::NvImageCodecImage;
using imgcodec::NvImageCodecInstance;

inline nvimgcodecImageInfo_t MakeRgbU8ImageInfo(uint8_t* buffer, int64_t width, int64_t height) {
  nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO,
                             sizeof(nvimgcodecImageInfo_t), nullptr};
  info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
  info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
  info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
  info.orientation = {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION,
                      sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};
  info.num_planes = 1;
  info.plane_info[0].height = static_cast<uint32_t>(height);
  info.plane_info[0].width = static_cast<uint32_t>(width);
  info.plane_info[0].num_channels = 3;
  info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
  info.plane_info[0].precision = 8;
  info.plane_info[0].row_stride = static_cast<uint32_t>(width) * 3u;
  info.buffer = buffer;
  info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
  return info;
}

inline nvimgcodecImageInfo_t MakeJpegOutputStreamInfo(int64_t width, int64_t height) {
  nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO,
                             sizeof(nvimgcodecImageInfo_t), nullptr};
  info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
  info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
  info.chroma_subsampling = NVIMGCODEC_SAMPLING_420;
  info.orientation = {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION,
                      sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};
  info.num_planes = 1;
  info.plane_info[0].height = static_cast<uint32_t>(height);
  info.plane_info[0].width = static_cast<uint32_t>(width);
  info.plane_info[0].num_channels = 3;
  info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
  info.plane_info[0].precision = 8;
  info.plane_info[0].row_stride = static_cast<uint32_t>(width) * 3u;
  std::snprintf(info.codec_name, NVIMGCODEC_MAX_CODEC_NAME_SIZE, "%s", "jpeg");
  return info;
}

unsigned char* ResizeVectorBufferCb(void* ctx, size_t req_size) {
  auto* v = static_cast<std::vector<uint8_t>*>(ctx);
  v->resize(req_size);
  return v->data();
}

// Drain a future, fetch per-frame processing statuses, and convert any failure
// (status-count mismatch or non-success entry) into a DALI exception. `frame_id`
// maps a status index to the frame id printed in error messages.
template <typename FrameIdFn>
void WaitAndCheck(const NvImageCodecFuture& future, size_t expected_count,
                  const char* op_name, FrameIdFn&& frame_id) {
  CHECK_NVIMGCODEC(nvimgcodecFutureWaitForAll(future));
  std::vector<nvimgcodecProcessingStatus_t> statuses(expected_count);
  size_t status_size = statuses.size();
  CHECK_NVIMGCODEC(nvimgcodecFutureGetProcessingStatus(future, statuses.data(), &status_size));
  DALI_ENFORCE(status_size == expected_count,
               make_string("nvimgcodec ", op_name, " returned ", status_size,
                           " statuses for ", expected_count, " frames"));
  for (size_t k = 0; k < statuses.size(); k++) {
    DALI_ENFORCE(statuses[k] == NVIMGCODEC_PROCESSING_STATUS_SUCCESS,
                 make_string("nvimgcodec ", op_name, " failed for frame ", frame_id(k),
                             " (status=", static_cast<int>(statuses[k]), ")"));
  }
}

}  // namespace

// Per-sample work is dispatched to nvimgcodec's internal executor rather than
// the workspace thread pool — encode/decode are issued as batched submits on
// the calling thread, so `ws.GetThreadPool()` is intentionally unused here.
class JpegCompressionDistortionCPU : public JpegCompressionDistortion<CPUBackend> {
 public:
  explicit JpegCompressionDistortionCPU(const OpSpec& spec)
      : JpegCompressionDistortion(spec) {}
  using Operator<CPUBackend>::RunImpl;

  ~JpegCompressionDistortionCPU() noexcept override {
#if not(WITH_DYNAMIC_NVIMGCODEC_ENABLED)
    // Tear down codecs before destroying extensions so encoder/decoder do not
    // outlive the extension that backs them.
    decoder_.reset();
    encoder_pool_.clear();
    for (auto& extension : extensions_) {
      nvimgcodecExtensionDestroy(extension);
    }
#endif
  }

 protected:
  void RunImpl(Workspace& ws) override;

 private:
  void EnsureCodecs();
  void EncodePass(size_t n_frames);
  void DecodePass(size_t n_frames);

  struct FrameDesc {
    const uint8_t* in_ptr;
    uint8_t* out_ptr;
    int64_t width;
    int64_t height;
    int quality;
  };

  // Per-bucket encode state. The encoder takes raw pointers into `enc_params`
  // and references into `in_imgs` / `out_streams` via the handle arrays it
  // is given, so every member must remain at a stable address until the
  // corresponding `future` is drained.
  struct EncodeBucket {
    std::vector<size_t> idxs;
    std::vector<imgcodec::NvImageCodecImage> in_imgs;
    std::vector<imgcodec::NvImageCodecCodeStream> out_streams;
    nvimgcodecEncodeParams_t enc_params{};
    NvImageCodecFuture future;
  };

  // Lazily initialized; constructed on first RunImpl invocation.
  bool codecs_ready_ = false;

  // Backend descriptor and execution params must outlive Encoder/Decoder
  // creation (nvimgcodec may retain a pointer to them past the Create call).
  // They are declared before the codec/extension/instance members so that
  // reverse-declaration destruction order tears down codecs (and any
  // implicit references they hold to these structs) first.
  nvimgcodecBackend_t cpu_backend_{NVIMGCODEC_STRUCTURE_TYPE_BACKEND,
                                   sizeof(nvimgcodecBackend_t), nullptr,
                                   NVIMGCODEC_BACKEND_KIND_CPU_ONLY,
                                   {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS,
                                    sizeof(nvimgcodecBackendParams_t), nullptr,
                                    1.0f, NVIMGCODEC_LOAD_HINT_POLICY_FIXED}};
  nvimgcodecExecutionParams_t exec_params_{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS,
                                           sizeof(nvimgcodecExecutionParams_t), nullptr};

  imgcodec::NvImageCodecInstance instance_;
  // Pool of encoders reused across RunImpl invocations. nvimgcodec stashes
  // the encode-params pointer in an encoder member that worker threads read
  // asynchronously, so two in-flight submits cannot share an encoder. The
  // pool grows on demand to match the peak number of concurrent buckets ever
  // submitted in a single RunImpl; each iteration's buckets are assigned to
  // pool[0..buckets-1] regardless of their quality value — the quality is
  // passed per-submit through enc_params, so encoders are quality-agnostic
  // across drained iterations.
  std::vector<imgcodec::NvImageCodecEncoder> encoder_pool_;
  imgcodec::NvImageCodecDecoder decoder_;
#if not(WITH_DYNAMIC_NVIMGCODEC_ENABLED)
  std::vector<nvimgcodecExtensionDesc_t> extensions_descs_;
  std::vector<nvimgcodecExtension_t> extensions_;
#endif

  // Scratch state reused across RunImpl calls; .clear()-ed at the top of
  // each invocation so capacity is amortised across pipeline iterations.
  std::vector<std::vector<uint8_t>> encoded_buffers_;
  std::vector<FrameDesc> frames_;
  // (quality, frame indices) groups. Linear find/insert is fine: the
  // distinct-quality count K is small (≤100), and per-frame group lookup is
  // O(K) — for the dominant single-quality case (K=1) it's one comparison.
  std::vector<std::pair<int, std::vector<size_t>>> by_quality_;
  std::vector<EncodeBucket> buckets_;

  // Per-call handle arrays for the nvimgcodec submits. Resized to the current
  // bucket / batch size on each entry; the underlying handles outlive each
  // submit via `buckets_` (encode) and the dec_* RAII vectors below (decode).
  std::vector<nvimgcodecImage_t> enc_in_img_handles_;
  std::vector<nvimgcodecCodeStream_t> enc_out_stream_handles_;
  std::vector<imgcodec::NvImageCodecCodeStream> dec_code_streams_;
  std::vector<imgcodec::NvImageCodecImage> dec_out_imgs_;
  std::vector<nvimgcodecCodeStream_t> dec_code_stream_handles_;
  std::vector<nvimgcodecImage_t> dec_out_img_handles_;
};

void JpegCompressionDistortionCPU::EnsureCodecs() {
  if (codecs_ready_)
    return;

  EnforceMinimumNvimgcodecVersion();

  nvimgcodecInstanceCreateInfo_t instance_create_info{
      NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      sizeof(nvimgcodecInstanceCreateInfo_t), nullptr};

  const char* log_lvl_env = std::getenv("DALI_NVIMGCODEC_LOG_LEVEL");
  int log_lvl = log_lvl_env ? std::clamp(std::atoi(log_lvl_env), 1, 5) : 2;

  instance_create_info.load_extension_modules =
      static_cast<int>(WITH_DYNAMIC_NVIMGCODEC_ENABLED);
  instance_create_info.load_builtin_modules = 1;
  instance_create_info.extension_modules_path = nullptr;
  instance_create_info.create_debug_messenger = 1;
  instance_create_info.message_severity = imgcodec::verbosity_to_severity(log_lvl);
  instance_create_info.message_category = NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL;
  instance_ = imgcodec::NvImageCodecInstance::Create(&instance_create_info);

  // Statically linked nvimgcodec doesn't pick up extensions automatically;
  // explicitly register the libjpeg-turbo extension that backs JPEG encode/decode.
#if not(WITH_DYNAMIC_NVIMGCODEC_ENABLED)
  // Drop any entries from a previous failed EnsureCodecs() attempt: extensions
  // were registered against the prior instance_, which the move-assign above
  // just destroyed, so their handles are stale. The old instance's teardown
  // cascades to its extensions; we discard the stale handles here so the
  // destructor doesn't later call ExtensionDestroy on them.
  extensions_.clear();
  extensions_descs_.clear();

  extensions_descs_.push_back(
      {NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), nullptr});
  extensions_.emplace_back();
  get_libjpeg_turbo_extension_desc(&extensions_descs_.back());
  CHECK_NVIMGCODEC(
      nvimgcodecExtensionCreate(instance_, &extensions_.back(), &extensions_descs_.back()));
#endif

  exec_params_.device_id = NVIMGCODEC_DEVICE_CPU_ONLY;
  exec_params_.backends = &cpu_backend_;
  exec_params_.num_backends = 1;
  exec_params_.pre_init = 1;
  exec_params_.skip_pre_sync = 1;

  decoder_ = imgcodec::NvImageCodecDecoder::Create(instance_, &exec_params_, "");

  codecs_ready_ = true;
}

void JpegCompressionDistortionCPU::RunImpl(Workspace& ws) {
  EnsureCodecs();

  const auto& input = ws.Input<CPUBackend>(0);
  auto& output = ws.Output<CPUBackend>(0);
  auto layout = input.GetLayout();
  output.SetLayout(layout);
  const auto& in_shape = input.shape();
  const int nsamples = input.num_samples();
  const auto in_view = view<const uint8_t>(input);
  const auto out_view = view<uint8_t>(output);

  const int w_dim = layout.find('W');
  const int h_dim = layout.find('H');
  const int f_dim = layout.find('F');
  assert(w_dim >= 0 && h_dim >= 0 && layout.find('C') >= 0);

  frames_.clear();
  for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
    auto shape = in_shape.tensor_shape_span(sample_idx);
    int ndim = shape.size();

    int64_t nframes = volume(shape.begin(), shape.begin() + f_dim + 1);
    int64_t frame_size = volume(shape.begin() + f_dim + 1, shape.begin() + ndim);
    int64_t width = shape[w_dim];
    int64_t height = shape[h_dim];

    int q = std::clamp(quality_arg_[sample_idx].data[0], 1, 100);
    for (int64_t elem = 0; elem < nframes; elem++) {
      frames_.push_back(FrameDesc{
          in_view[sample_idx].data + elem * frame_size,
          out_view[sample_idx].data + elem * frame_size,
          width, height, q});
    }
  }

  const size_t N = frames_.size();
  // Resize without clearing first so each inner vector retains its previous
  // capacity — the resize callback below reuses it on the next encode.
  encoded_buffers_.resize(N);

  EncodePass(N);
  // Empty inputs (no samples / all-zero-frame sequences) skip the decode call
  // entirely — nvimgcodec's behaviour on a zero-length batch is unspecified.
  if (N == 0) return;
  DecodePass(N);
}

void JpegCompressionDistortionCPU::EncodePass(size_t n_frames) {
  // Frames are grouped by quality (one bucket per distinct quality), and each
  // bucket is dispatched on its own encoder taken from encoder_pool_. The
  // submits are pipelined across buckets and the futures are drained together
  // at the end — independent encoders have independent curr_params_ members
  // inside nvimgcodec, so they cannot stomp on each other. The single-quality
  // common case collapses to one batched submit on a single encoder.
  by_quality_.clear();
  for (size_t i = 0; i < n_frames; i++) {
    int q = frames_[i].quality;
    auto it = std::find_if(by_quality_.begin(), by_quality_.end(),
                           [q](const auto& p) { return p.first == q; });
    if (it == by_quality_.end()) {
      by_quality_.emplace_back(q, std::vector<size_t>{i});
    } else {
      it->second.push_back(i);
    }
  }

  // Constant across all buckets; referenced via enc_params.struct_next, so it
  // must outlive any in-flight encode.
  nvimgcodecJpegEncodeParams_t jpeg_params{NVIMGCODEC_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS,
                                           sizeof(nvimgcodecJpegEncodeParams_t), nullptr,
                                           /*optimized_huffman=*/0};

  // Recycle the bucket vector across calls: .clear() destroys any handles
  // (futures, images, code streams) carried over from the previous RunImpl
  // while keeping the outer vector's capacity, then .reserve() guarantees no
  // reallocation invalidates &bucket.enc_params during the submit loop.
  buckets_.clear();
  buckets_.reserve(by_quality_.size());

  // Grow the encoder pool to cover all buckets in this iteration. Encoders
  // survive across RunImpl invocations; the pool only grows when an iteration
  // needs more concurrent submits than any previous one, so steady-state size
  // tracks the peak distinct-qualities-per-iter, not the lifetime cumulative.
  while (encoder_pool_.size() < by_quality_.size()) {
    encoder_pool_.push_back(
        imgcodec::NvImageCodecEncoder::Create(instance_, &exec_params_, ""));
  }

  size_t pool_idx = 0;
  for (auto& kv : by_quality_) {
    auto& bucket = buckets_.emplace_back();
    bucket.idxs = std::move(kv.second);
    const int batch = static_cast<int>(bucket.idxs.size());
    bucket.in_imgs.resize(batch);
    bucket.out_streams.resize(batch);

    enc_in_img_handles_.resize(batch);
    enc_out_stream_handles_.resize(batch);
    for (int k = 0; k < batch; k++) {
      const auto& fd = frames_[bucket.idxs[k]];
      // nvimgcodecImageInfo_t.buffer is void* (no const qualifier in the C API);
      // the encoder only reads from this buffer.
      auto in_info = MakeRgbU8ImageInfo(const_cast<uint8_t*>(fd.in_ptr), fd.width, fd.height);
      bucket.in_imgs[k] = imgcodec::NvImageCodecImage::Create(instance_, &in_info);
      enc_in_img_handles_[k] = bucket.in_imgs[k];

      auto out_info = MakeJpegOutputStreamInfo(fd.width, fd.height);
      bucket.out_streams[k] = imgcodec::NvImageCodecCodeStream::ToHostMem(
          instance_, &encoded_buffers_[bucket.idxs[k]], &ResizeVectorBufferCb, &out_info);
      enc_out_stream_handles_[k] = bucket.out_streams[k];
    }

    bucket.enc_params = nvimgcodecEncodeParams_t{NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS,
                                                 sizeof(nvimgcodecEncodeParams_t), &jpeg_params,
                                                 NVIMGCODEC_QUALITY_TYPE_QUALITY,
                                                 static_cast<float>(kv.first)};

    nvimgcodecFuture_t future_handle = nullptr;
    auto enc_status = nvimgcodecEncoderEncode(encoder_pool_[pool_idx++],
                                              enc_in_img_handles_.data(),
                                              enc_out_stream_handles_.data(), batch,
                                              &bucket.enc_params, &future_handle);
    bucket.future = NvImageCodecFuture(future_handle);
    CHECK_NVIMGCODEC(enc_status);
  }

  // Drain all encode futures and check per-frame statuses.
  for (const auto& bucket : buckets_) {
    WaitAndCheck(bucket.future, bucket.idxs.size(), "encode",
                 [&](size_t k) { return bucket.idxs[k]; });
  }
}

void JpegCompressionDistortionCPU::DecodePass(size_t n_frames) {
  dec_code_streams_.resize(n_frames);
  dec_out_imgs_.resize(n_frames);
  dec_code_stream_handles_.resize(n_frames);
  dec_out_img_handles_.resize(n_frames);

  for (size_t i = 0; i < n_frames; i++) {
    dec_code_streams_[i] = imgcodec::NvImageCodecCodeStream::FromHostMem(
        instance_, encoded_buffers_[i].data(), encoded_buffers_[i].size());
    dec_code_stream_handles_[i] = dec_code_streams_[i];

    auto out_info = MakeRgbU8ImageInfo(frames_[i].out_ptr, frames_[i].width, frames_[i].height);
    dec_out_imgs_[i] = imgcodec::NvImageCodecImage::Create(instance_, &out_info);
    dec_out_img_handles_[i] = dec_out_imgs_[i];
  }

  nvimgcodecDecodeParams_t dec_params{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS,
                                      sizeof(nvimgcodecDecodeParams_t), nullptr,
                                      /*apply_exif_orientation=*/0};

  nvimgcodecFuture_t future_handle = nullptr;
  auto dec_status = nvimgcodecDecoderDecode(
      decoder_, dec_code_stream_handles_.data(), dec_out_img_handles_.data(),
      static_cast<int>(n_frames), &dec_params, &future_handle);
  NvImageCodecFuture future(future_handle);
  CHECK_NVIMGCODEC(dec_status);
  WaitAndCheck(future, n_frames, "decode", [](size_t k) { return k; });
}

DALI_REGISTER_OPERATOR(JpegCompressionDistortion, JpegCompressionDistortionCPU, CPU);

}  // namespace dali
