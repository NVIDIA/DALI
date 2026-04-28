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
#include <map>
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

}  // namespace

class JpegCompressionDistortionCPU : public JpegCompressionDistortion<CPUBackend> {
 public:
  explicit JpegCompressionDistortionCPU(const OpSpec& spec)
      : JpegCompressionDistortion(spec) {}
  using Operator<CPUBackend>::RunImpl;

 protected:
  void RunImpl(Workspace& ws) override;

 private:
  void EnsureCodecs();

  // Lazily initialized; constructed on first RunImpl invocation.
  bool codecs_ready_ = false;
  imgcodec::NvImageCodecInstance instance_;
  imgcodec::NvImageCodecEncoder encoder_;
  imgcodec::NvImageCodecDecoder decoder_;
#if not(WITH_DYNAMIC_NVIMGCODEC_ENABLED)
  std::vector<nvimgcodecExtensionDesc_t> extensions_descs_;
  std::vector<nvimgcodecExtension_t> extensions_;
#endif

  // Reused across RunImpl calls.
  std::vector<std::vector<uint8_t>> encoded_buffers_;
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
#if not(WITH_DYNAMIC_NVIMGCODEC_ENABLED) && LIBJPEG_TURBO_ENABLED
  extensions_descs_.push_back(
      {NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), nullptr});
  extensions_.emplace_back();
  get_libjpeg_turbo_extension_desc(&extensions_descs_.back());
  CHECK_NVIMGCODEC(
      nvimgcodecExtensionCreate(instance_, &extensions_.back(), &extensions_descs_.back()));
#endif

  nvimgcodecBackend_t cpu_backend{NVIMGCODEC_STRUCTURE_TYPE_BACKEND,
                                  sizeof(nvimgcodecBackend_t), nullptr,
                                  NVIMGCODEC_BACKEND_KIND_CPU_ONLY,
                                  {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS,
                                   sizeof(nvimgcodecBackendParams_t), nullptr,
                                   1.0f, NVIMGCODEC_LOAD_HINT_POLICY_FIXED}};
  nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS,
                                          sizeof(nvimgcodecExecutionParams_t), nullptr};
  exec_params.device_id = NVIMGCODEC_DEVICE_CPU_ONLY;
  exec_params.backends = &cpu_backend;
  exec_params.num_backends = 1;
  exec_params.pre_init = 1;
  exec_params.skip_pre_sync = 1;

  encoder_ = imgcodec::NvImageCodecEncoder::Create(instance_, &exec_params, "");
  decoder_ = imgcodec::NvImageCodecDecoder::Create(instance_, &exec_params, "");

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

  struct FrameDesc {
    const uint8_t* in_ptr;
    uint8_t* out_ptr;
    int64_t width;
    int64_t height;
    int quality;
  };
  const int w_dim = layout.find('W');
  const int h_dim = layout.find('H');
  const int c_dim = layout.find('C');
  const int f_dim = layout.find('F');
  assert(w_dim >= 0 && h_dim >= 0 && c_dim >= 0);

  int64_t total_frames = 0;
  for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
    auto shape = in_shape.tensor_shape_span(sample_idx);
    total_frames += volume(shape.begin(), shape.begin() + f_dim + 1);
  }

  std::vector<FrameDesc> frames;
  frames.reserve(total_frames);

  for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
    auto shape = in_shape.tensor_shape_span(sample_idx);
    int ndim = shape.size();

    int64_t nframes = volume(shape.begin(), shape.begin() + f_dim + 1);
    int64_t frame_size = volume(shape.begin() + f_dim + 1, shape.begin() + ndim);
    int64_t width = shape[w_dim];
    int64_t height = shape[h_dim];

    int q = std::clamp(quality_arg_[sample_idx].data[0], 1, 100);
    for (int64_t elem = 0; elem < nframes; elem++) {
      frames.push_back(FrameDesc{
          in_view[sample_idx].data + elem * frame_size,
          out_view[sample_idx].data + elem * frame_size,
          width, height, q});
    }
  }

  const size_t N = frames.size();
  // Resize without clearing first so each inner vector retains its previous
  // capacity — the resize callback below reuses it on the next encode.
  encoded_buffers_.resize(N);

  // ---- Encode pass: bucket by quality, one batched submit per bucket ----
  std::map<int, std::vector<size_t>> by_quality;
  for (size_t i = 0; i < N; i++)
    by_quality[frames[i].quality].push_back(i);

  for (const auto& kv : by_quality) {
    const int q = kv.first;
    const auto& idxs = kv.second;
    const int batch = static_cast<int>(idxs.size());

    std::vector<imgcodec::NvImageCodecImage> in_imgs(batch);
    std::vector<imgcodec::NvImageCodecCodeStream> out_streams(batch);
    std::vector<nvimgcodecImage_t> in_img_handles(batch);
    std::vector<nvimgcodecCodeStream_t> out_stream_handles(batch);

    for (int k = 0; k < batch; k++) {
      const auto& fd = frames[idxs[k]];
      auto in_info = MakeRgbU8ImageInfo(const_cast<uint8_t*>(fd.in_ptr), fd.width, fd.height);
      in_imgs[k] = imgcodec::NvImageCodecImage::Create(instance_, &in_info);
      in_img_handles[k] = in_imgs[k];

      auto out_info = MakeJpegOutputStreamInfo(fd.width, fd.height);
      out_streams[k] = imgcodec::NvImageCodecCodeStream::ToHostMem(
          instance_, &encoded_buffers_[idxs[k]], &ResizeVectorBufferCb, &out_info);
      out_stream_handles[k] = out_streams[k];
    }

    nvimgcodecJpegEncodeParams_t jpeg_params{NVIMGCODEC_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS,
                                             sizeof(nvimgcodecJpegEncodeParams_t), nullptr,
                                             /*optimized_huffman=*/0};
    nvimgcodecEncodeParams_t enc_params{NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS,
                                        sizeof(nvimgcodecEncodeParams_t), &jpeg_params,
                                        NVIMGCODEC_QUALITY_TYPE_QUALITY, static_cast<float>(q)};

    nvimgcodecFuture_t future_handle = nullptr;
    auto enc_status = nvimgcodecEncoderEncode(encoder_, in_img_handles.data(),
                                              out_stream_handles.data(), batch,
                                              &enc_params, &future_handle);
    NvImageCodecFuture future(future_handle);
    CHECK_NVIMGCODEC(enc_status);
    CHECK_NVIMGCODEC(nvimgcodecFutureWaitForAll(future));
    std::vector<nvimgcodecProcessingStatus_t> statuses(batch);
    size_t status_size = statuses.size();
    CHECK_NVIMGCODEC(nvimgcodecFutureGetProcessingStatus(future, statuses.data(), &status_size));
    DALI_ENFORCE(status_size == static_cast<size_t>(batch),
                 make_string("nvimgcodec encode returned ", status_size,
                             " statuses for ", batch, " frames"));
    for (size_t k = 0; k < statuses.size(); k++) {
      DALI_ENFORCE(statuses[k] == NVIMGCODEC_PROCESSING_STATUS_SUCCESS,
                   make_string("nvimgcodec encode failed for frame ", idxs[k],
                               " (status=", static_cast<int>(statuses[k]), ")"));
    }
  }

  // ---- Decode pass: one batched submit for the whole batch ----
  std::vector<imgcodec::NvImageCodecCodeStream> code_streams(N);
  std::vector<imgcodec::NvImageCodecImage> out_imgs(N);
  std::vector<nvimgcodecCodeStream_t> code_stream_handles(N);
  std::vector<nvimgcodecImage_t> out_img_handles(N);

  for (size_t i = 0; i < N; i++) {
    code_streams[i] = imgcodec::NvImageCodecCodeStream::FromHostMem(
        instance_, encoded_buffers_[i].data(), encoded_buffers_[i].size());
    code_stream_handles[i] = code_streams[i];

    auto out_info = MakeRgbU8ImageInfo(frames[i].out_ptr, frames[i].width, frames[i].height);
    out_imgs[i] = imgcodec::NvImageCodecImage::Create(instance_, &out_info);
    out_img_handles[i] = out_imgs[i];
  }

  nvimgcodecDecodeParams_t dec_params{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS,
                                      sizeof(nvimgcodecDecodeParams_t), nullptr,
                                      /*apply_exif_orientation=*/0};

  nvimgcodecFuture_t future_handle = nullptr;
  auto dec_status = nvimgcodecDecoderDecode(decoder_, code_stream_handles.data(),
                                            out_img_handles.data(), static_cast<int>(N),
                                            &dec_params, &future_handle);
  NvImageCodecFuture future(future_handle);
  CHECK_NVIMGCODEC(dec_status);
  CHECK_NVIMGCODEC(nvimgcodecFutureWaitForAll(future));
  std::vector<nvimgcodecProcessingStatus_t> statuses(N);
  size_t status_size = statuses.size();
  CHECK_NVIMGCODEC(nvimgcodecFutureGetProcessingStatus(future, statuses.data(), &status_size));
  DALI_ENFORCE(status_size == N,
               make_string("nvimgcodec decode returned ", status_size,
                           " statuses for ", N, " frames"));
  for (size_t i = 0; i < statuses.size(); i++) {
    DALI_ENFORCE(statuses[i] == NVIMGCODEC_PROCESSING_STATUS_SUCCESS,
                 make_string("nvimgcodec decode failed for frame ", i,
                             " (status=", static_cast<int>(statuses[i]), ")"));
  }
}

DALI_REGISTER_OPERATOR(JpegCompressionDistortion, JpegCompressionDistortionCPU, CPU);

}  // namespace dali
