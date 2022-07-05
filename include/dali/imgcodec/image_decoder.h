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

#ifndef DALI_IMGCODEC_IMAGE_DECODER_H_
#define DALI_IMGCODEC_IMAGE_DECODER_H_

#include <memory>
#include <stdexcept>
#include <vector>
#include "dali/core/any.h"
#include "dali/core/span.h"
#include "dali/core/tensor_shape.h"
#include "dali/imgcodec/image_format.h"
#include "dali/pipeline/data/sample_view.h"
#include "dali/pipeline/data/backend.h"

namespace dali {
class ThreadPool;

namespace imgcodec {
template <typename T, span_extent_t E = dynamic_extent>
using cspan = span<const T, E>;

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

struct ImageDecoderProperties {
  /**
   * @brief Whether the codec can decode a part of the image without storing the
   *        entire image in memory.
   */
  bool roi_support = false;

  /**
   * @brief A mask of supported input kinds
   */
  InputKind supported_input_kinds;

  // if true and the codec fails to decode an image,
  // an attempt will be made to use other compatible codecs
  bool fallback = true;
};

class DLL_PUBLIC ImageDecoderInstance {
 public:
  virtual ~ImageDecoderInstance() = default;
  /**
   * @brief Checks whether this codec can decode this encoded image with given parameters
   */
  virtual bool CanDecode(ImageSource *in, DecodeParams opts) = 0;

  /**
   * @brief Batch version of CanDecode
   */
  virtual std::vector<bool> CanDecode(cspan<ImageSource *> in, cspan<DecodeParams> opts) = 0;

  /**
   * @brief Decodes a single image to a host buffer
   */
  virtual DecodeResult Decode(SampleView<CPUBackend> out, ImageSource *in, DecodeParams opts) = 0;

  /**
   * @brief Decodes a batch of images to host buffers
   */
  virtual std::vector<DecodeResult> Decode(span<SampleView<CPUBackend>> out,
                                           cspan<ImageSource *> in, cspan<DecodeParams> opts) = 0;



  /**
   * @brief Decodes a single image to a device buffer
   */
  virtual DecodeResult Decode(SampleView<GPUBackend> out, ImageSource *in, DecodeParams opts) = 0;

  /**
   * @brief Decodes a single image to device buffers
   */
  virtual std::vector<DecodeResult> Decode(span<SampleView<GPUBackend>> out,
                                           cspan<ImageSource *> in, cspan<DecodeParams> opts) = 0;

  /**
   * @brief Sets a codec-specific parameter
   */
  virtual void SetParam(const char *key, const any &value) = 0;
  /**
   * @brief Gets a codec-specific parameter
   */
  virtual any GetParam(const char *key) const = 0;

  template <typename T>
  inline enable_if_t<!std::is_same<std::remove_reference_t<T>, any>::value>
  SetParam(const char *key, T value) {
    SetParam(key, any(value));
  }

  template <typename T>
  inline T GetParam(const char *key) const {
    return any_cast<T>(GetParam(key));
  }
};

class DLL_PUBLIC ImageDecoder {
 public:
  virtual ~ImageDecoder() = default;

  /**
   * @brief Gets the properties and capabilities of the codec
   */
  virtual ImageDecoderProperties GetProperties() const = 0;

  /**
   * @brief Checks whether the codec is supported on the specified device
   *
   * The result may differ depending on extra hardware modules (e.g. hardware JPEG decoder).
   * A negative device id means "cpu-only". Decoders requiring a GPU must return false in that case.
   */
  virtual bool IsSupported(int device_id) const = 0;

  /**
   * @brief Creates an instance of a codec
   *
   * Note: For decoders that carry no state, this may just increase reference count on a singleton.
   */
  virtual std::shared_ptr<ImageDecoderInstance> Create(int device_id, ThreadPool &tp) const = 0;
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_IMAGE_DECODER_H_
