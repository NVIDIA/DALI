// Copyright (c) 2018-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>
#include <memory>
#include "dali/operators/reader/parser/sequence_parser.h"

namespace dali {

nvimgcodecSampleFormat_t image_type_to_sample_format(DALIImageType img_type) {
  switch (img_type) {
    case DALI_ANY_DATA:
      return NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED;
    case DALI_GRAY:
      return NVIMGCODEC_SAMPLEFORMAT_P_Y;
    case DALI_RGB:
      return NVIMGCODEC_SAMPLEFORMAT_I_RGB;
    case DALI_BGR:
      return NVIMGCODEC_SAMPLEFORMAT_I_BGR;
    case DALI_YCbCr:
    default:
      return NVIMGCODEC_SAMPLEFORMAT_UNSUPPORTED;
  }
}

void SequenceParser::Parse(const TensorSequence& data, SampleWorkspace* ws) {
  auto& decoder = GetDecoder(ws->thread_idx());
  if (!decoder)
    decoder = imgcodec::NvImageCodecDecoder::Create(instance_, &exec_params_, {});
  Index seq_length = data.tensors.size();
  TensorShape<4> seq_shape;
  seq_shape[0] = seq_length;
  int64_t& nchannels = seq_shape[3];
  auto& sequence = ws->Output<CPUBackend>(0);

  nvimgcodecDecodeParams_t decode_params = {NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS,
                                            sizeof(nvimgcodecDecodeParams_t), nullptr};
  decode_params.apply_exif_orientation = 1;
  decode_params.enable_roi = 0;

  for (int i = 0; i < static_cast<int>(data.tensors.size()); i++) {
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t),
                               nullptr};

    auto file_name = data.tensors[i].GetSourceInfo();
    const uint8_t* data_ptr = data.tensors[i].data<uint8_t>();
    size_t data_size = data.tensors[i].size();
    auto encoded_stream =
        imgcodec::NvImageCodecCodeStream::FromHostMem(instance_, data_ptr, data_size);
    CHECK_NVIMGCODEC(nvimgcodecCodeStreamGetImageInfo(encoded_stream, &info));

    int64_t new_nchannels = image_type_ == DALI_GRAY                           ? 1 :
                            image_type_ == DALI_RGB || image_type_ == DALI_BGR ? 3 :
                            info.num_planes > 1                                ? info.num_planes :
                                                  info.plane_info[0].num_channels;
    if (i == 0) {
      seq_shape[1] = info.plane_info[0].height;
      seq_shape[2] = info.plane_info[0].width;
      nchannels = new_nchannels;
      sequence.SetLayout("FHWC");
      sequence.Resize(seq_shape, DALI_UINT8);
    } else {
      DALI_ENFORCE(
          info.plane_info[0].height == seq_shape[1] && info.plane_info[0].width == seq_shape[2] &&
              new_nchannels == nchannels,
          make_string("Expected all frames to have same shape. (", info.plane_info[0].height, ", ",
                      info.plane_info[0].width, ", ", info.plane_info[0].num_channels, ") != ( ",
                      seq_shape[1], ", ", seq_shape[2], ", ", seq_shape[3], ")"));
    }

    auto view_i = sequence.SubspaceTensor(i);
    info.buffer_size = volume(view_i.shape());
    info.buffer = view_i.raw_mutable_data();

    // Decode to format
    info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
    info.sample_format = image_type_to_sample_format(image_type_);
    DALI_ENFORCE(info.sample_format != NVIMGCODEC_SAMPLEFORMAT_UNSUPPORTED,
      "Format not supported. Only RGB, BGR, GRAY or ANY_DATA are supported on this operator");

    info.cuda_stream = nullptr;
    info.region.ndim = 0;
    info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
    info.num_planes = 1;
    info.plane_info[0].height = seq_shape[1];
    info.plane_info[0].width = seq_shape[2];
    info.plane_info[0].num_channels = seq_shape[3];
    info.plane_info[0].row_stride = seq_shape[2] * seq_shape[3];
    info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
    info.plane_info[0].precision = 0;
    auto image = imgcodec::NvImageCodecImage::Create(instance_, &info);

    nvimgcodecFuture_t decode_future;
    auto enc_stream = encoded_stream.get();
    auto img_handle = image.get();
    CHECK_NVIMGCODEC(nvimgcodecDecoderDecode(decoder, &enc_stream, &img_handle, 1, &decode_params,
                                             &decode_future));
    size_t status_size;
    nvimgcodecProcessingStatus_t decode_status;
    nvimgcodecFutureGetProcessingStatus(decode_future, &decode_status, &status_size);
    nvimgcodecFutureDestroy(decode_future);
    DALI_ENFORCE(decode_status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS,
                 make_string("Failed to decode the ", i, "-th frame (", file_name,
                             ") with error code ", decode_status));
  }
}

}  // namespace dali
