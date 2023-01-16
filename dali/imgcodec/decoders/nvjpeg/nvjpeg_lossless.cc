// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <map>
#include <string>
#include <utility>
#include "dali/core/device_guard.h"
#include "dali/imgcodec/decoders/nvjpeg/nvjpeg_helper.h"
#include "dali/imgcodec/decoders/nvjpeg/nvjpeg_lossless.h"
#include "dali/imgcodec/registry.h"
#include "dali/imgcodec/parsers/jpeg.h"

namespace dali {
namespace imgcodec {

NvJpegLosslessDecoderInstance::NvJpegLosslessDecoderInstance(
    int device_id, const std::map<std::string, any> &params)
    : BatchedApiDecoderImpl(device_id, params),
      stream_(CUDAStreamPool::instance().Get(device_id)),
      event_(CUDAEvent::Create(device_id)) {
  DeviceGuard dg(device_id_);
  CUDA_CALL(nvjpegCreateEx(NVJPEG_BACKEND_LOSSLESS_JPEG, NULL, NULL, 0, &nvjpeg_handle_));
  CUDA_CALL(nvjpegJpegStreamCreate(nvjpeg_handle_, &jpeg_stream_));
  CUDA_CALL(nvjpegJpegStateCreate(nvjpeg_handle_, &state_));
}

NvJpegLosslessDecoderInstance::~NvJpegLosslessDecoderInstance() {
  DeviceGuard dg(device_id_);
  CUDA_CALL(cudaStreamSynchronize(stream_));
  CUDA_CALL(nvjpegJpegStreamDestroy(jpeg_stream_));
  CUDA_CALL(nvjpegJpegStateDestroy(state_));
  CUDA_CALL(nvjpegDestroy(nvjpeg_handle_));
}

bool NvJpegLosslessDecoderInstance::CanDecode(DecodeContext ctx, ImageSource *in, DecodeParams opts, const ROI &roi) {
  if (opts.dtype != DALI_UINT16 ||
      opts.format != DALI_ANY_DATA ||
      opts.planar == true ||
      opts.use_orientation == true) {
    return false;
  }

  try {
    CUDA_CALL(nvjpegJpegStreamParseHeader(nvjpeg_handle_, in->RawData<unsigned char>(), in->Size(), jpeg_stream_));

    int is_supported = 0;  // 0 means yes
    CUDA_CALL(nvjpegDecodeBatchedSupported(nvjpeg_handle_, jpeg_stream_, &is_supported));
    if (is_supported != 0)
      return false;

    unsigned int precision;
    CUDA_CALL(nvjpegJpegStreamGetSamplePrecision(jpeg_stream_, &precision));
    return precision > 8;  // this backend only supports U16
  } catch(...) {
    return false;
  }
}

DecodeResult NvJpegLosslessDecoderInstance::DecodeImplBatch(cudaStream_t stream,
                                                            span<SampleView<GPUBackend>> out,
                                                            cspan<ImageSource *> in,
                                                            DecodeParams opts, cspan<ROI> rois) {
  if (opts.dtype != DALI_UINT16)
    throw std::invalid_argument("Only uint16 is supported.");
  if (opts.format != DALI_ANY_DATA)
    throw std::invalid_argument("Only ANY_DATA is supported.");

  int nsamples = in.size();
  assert(out.size() == nsamples);
  std::vector<DecodeResult> ret(nsamples, {false, nullptr});
  encoded_.clear();
  encoded_.reserve(nsamples);
  encoded_len_.clear();
  encoded_len_.reserve(nsamples);
  output_imgs_.clear();
  output_imgs_.reserve(nsamples);

  for (auto* sample : in) {
    assert(sample->Kind() == InputKind::HostMemory);
    auto* data_ptr = sample->RawData<unsigned char>();
    auto data_size = sample->Size();
    encoded_.push_back(data_ptr);
    encoded_len_.push_back(data_size);
  }
  for (auto& out_sample : out) {
    output_imgs_.emplace_back();
    auto &o =  output_imgs_.back();
    auto sh = out_sample.shape();
    o.channel[0] = static_cast<uint8_t*>(out_sample.raw_mutable_data());
    o.pitch[0] = sh[1] * sh[2] * sizeof(uint16_t);
    std::cout << "Pitch " << sh[1] << " x " << sh[2] << " x " << sizeof(uint16_t) << "\n";
    std::cout << "Out shape :" << sh.num_elements() * sizeof(uint16_t) << "\n";
  }

  CUDA_CALL(nvjpegDecodeBatchedInitialize(nvjpeg_handle_, state_, nsamples, 1, NVJPEG_OUTPUT_UNCHANGEDI_U16));
  CUDA_CALL(nvjpegDecodeBatched(nvjpeg_handle_, state_, encoded_.data(), encoded_len_.data(), output_imgs_.data(), stream));
  return {true, nullptr};
}

REGISTER_DECODER("JPEG", NvJpegLosslessDecoderFactory, CUDADecoderPriority - 1);

}  // namespace imgcodec
}  // namespace dali
