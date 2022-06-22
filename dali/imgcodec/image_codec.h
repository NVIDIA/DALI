// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_IMGCODEC_IMAGE_CODEC_H_
#define DALI_IMGCODEC_IMAGE_CODEC_H_

#include <memory>
#include <stdexcept>
#include <vector>
#include "dali/core/span.h"
#include "dali/core/tensor_shape.h"
#include "dali/imgcodec/image_format.h"
#include "dali/pipeline/data/sample_view.h"

namespace dali {
namespace imgcodec {

struct DecodeParams {
  bool use_roi;
  struct {
    TensorShape<> begin, end;
  } roi;
  DALIDataType dtype;
  DALIImageType format;
  bool planar;  // not required initially
};

struct DecodeResult {
  bool success;
  std::exception_ptr exception;
};

struct ImageCodecProperties {
  // the codec natively supports region-of-interest decoding
  bool roi_support = false;
  InputKind supported_input_kinds;

  // if true and the codec fails to decode an image,
  // an attempt will be made to use other compatible codecs
  bool fallback = true;
};

class ImageCodecInstance {
 public:
  virtual ~ImageCodecInstance() = default;
  virtual bool CanDecode(EncodedImage *in, DecodeParams opts);
  virtual std::vector<bool> CanDecode(span<EncodedImage *> in, span<DecodeParams> opts);
  virtual DecodeResult Decode(SampleView<CPUBackend> out, EncodedImage *in, DecodeParams opts);
  virtual std::vector<DecodeResult> Decode(span<SampleView<CPUBackend>> out,
                                           span<EncodedImage *> in, span<DecodeParams> opts);
  virtual DecodeResult Decode(SampleView<GPUBackend> out, EncodedImage *in, DecodeParams opts);
  virtual std::vector<DecodeResult> Decode(span<SampleView<GPUBackend>> out,
                                           span<EncodedImage *> in, span<DecodeParams> opts);

  /* the `key` is deliberately opaque and it can be a codec-specific enum */
  virtual void SetParam(const char *key, any value) = 0;
  virtual const any &GetParam(const char *key) const = 0;
};

class ImageCodec {
 public:
  virtual ImageCodecProperties GetProperties() const;
  virtual bool IsSupported(int device_id) const;
  virtual std::shared_ptr<ImageCodecInstance> Create(int device_id, ThreadPool &tp);
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_IMAGE_CODEC_H_
