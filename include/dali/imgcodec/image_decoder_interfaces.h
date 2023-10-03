// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_IMGCODEC_IMAGE_DECODER_INTERFACES_H_
#define DALI_IMGCODEC_IMAGE_DECODER_INTERFACES_H_

#include <any>
#include <map>
#include <memory>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>
#include "dali/core/span.h"
#include "dali/core/tensor_shape.h"
#include "dali/imgcodec/image_format.h"
#include "dali/imgcodec/decode_results.h"
#include "dali/pipeline/data/sample_view.h"
#include "dali/pipeline/data/backend.h"

namespace dali {
class ThreadPool;

namespace imgcodec {

struct DecodeParams {
  DALIDataType  dtype   = DALI_UINT8;
  DALIImageType format  = DALI_RGB;
  bool          planar  = false;
  bool          use_orientation = true;
};

/**
 * @brief Region of interest
 *
 * Spatial coordinates of the ROI to decode. Channels shall not be included.
 *
 * If there are no coordinates for `begin` or `end` then no cropping is requried and the
 * ROI is considered to include the entire image.
 *
 * NOTE: If the orientation of the image is adjusted, these values are in the output space
 *       (after including the orientation).
 */
struct ROI {
  /**
   * @brief The beginning and end of the region-of-interest.
   *
   * If both begin and end are empty, the ROI denotes full image.
   */
  TensorShape<> begin, end;

  bool use_roi() const {
    return begin.sample_dim() || end.sample_dim();
  }

  explicit operator bool() const { return use_roi(); }

  /**
   * @brief Returns the extent of the region of interest as (end - begin)
   */
  TensorShape<> shape() const {
    TensorShape<> out = end;
    for (int d = 0; d < begin.sample_dim(); d++)
      out[d] -= begin[d];
    return out;
  }
};

struct ImageDecoderProperties {
  /**
   * @brief Whether the decoder can decode a region of interest without decoding the entire image
   */
  bool supports_partial_decoding = false;

  /**
   * @brief A mask of supported input kinds
   */
  InputKind supported_input_kinds;

  /**
   * @brief Whether the output of the decoder is in GPU memory
   */
  bool gpu_output = false;

  /**
   * @brief if true and the codec fails to decode an image,
   *        an attempt will be made to use other compatible codecs
   */
  bool fallback = true;
};

struct DecodeContext {
  DecodeContext() = default;
  DecodeContext(ThreadPool *tp, cudaStream_t stream) : tp(tp), stream(stream) {}
  ThreadPool *tp = nullptr;
  cudaStream_t stream = cudaStream_t(-1);
};

class DLL_PUBLIC ImageDecoderInstance {
 public:
  virtual ~ImageDecoderInstance() = default;
  /**
   * @brief Checks whether this codec can decode this encoded image with given parameters
   */
  virtual bool CanDecode(DecodeContext ctx,
                         ImageSource *in,
                         DecodeParams opts,
                         const ROI &roi = {}) = 0;

  /**
   * @brief Batch version of CanDecode
   */
  virtual std::vector<bool> CanDecode(DecodeContext ctx,
                                      cspan<ImageSource *> in,
                                      DecodeParams opts,
                                      cspan<ROI> rois = {}) = 0;

  /**
   * @brief Decodes a single image to a host buffer
   */
  virtual DecodeResult Decode(DecodeContext ctx,
                              SampleView<CPUBackend> out,
                              ImageSource *in,
                              DecodeParams opts,
                              const ROI &roi = {}) = 0;

  /**
   * @brief Schedules decoding of an image to a host buffer
   */
  virtual FutureDecodeResults ScheduleDecode(DecodeContext ctx,
                                             SampleView<CPUBackend> out,
                                             ImageSource *in,
                                             DecodeParams opts,
                                             const ROI &roi = {}) = 0;

  /**
   * @brief Decodes a batch of images to host buffers
   */
  virtual std::vector<DecodeResult> Decode(DecodeContext ctx,
                                           span<SampleView<CPUBackend>> out,
                                           cspan<ImageSource *> in,
                                           DecodeParams opts,
                                           cspan<ROI> rois = {}) = 0;

  /**
   * @brief Schedules decoding of a batch of images to host buffers
   */
  virtual FutureDecodeResults ScheduleDecode(DecodeContext ctx,
                                             span<SampleView<CPUBackend>> out,
                                             cspan<ImageSource *> in,
                                             DecodeParams opts,
                                             cspan<ROI> rois = {}) = 0;

  /**
   * @brief Decodes a single image to a device buffer
   */
  virtual DecodeResult Decode(DecodeContext ctx,
                              SampleView<GPUBackend> out,
                              ImageSource *in,
                              DecodeParams opts,
                              const ROI &roi = {}) = 0;


  /**
   * @brief Schedules decoding of a single image to a device buffer
   */
  virtual FutureDecodeResults ScheduleDecode(DecodeContext ctx,
                                             SampleView<GPUBackend> out,
                                             ImageSource *in,
                                             DecodeParams opts,
                                             const ROI &roi = {}) = 0;

  /**
   * @brief Decodes a batch of images to device buffers
   */
  virtual std::vector<DecodeResult> Decode(DecodeContext ctx,
                                           span<SampleView<GPUBackend>> out,
                                           cspan<ImageSource *> in,
                                           DecodeParams opts,
                                           cspan<ROI> rois = {}) = 0;

  /**
   * @brief Scheduls decoding of a batch of images to device buffers
   */
  virtual FutureDecodeResults ScheduleDecode(DecodeContext ctx,
                                             span<SampleView<GPUBackend>> out,
                                             cspan<ImageSource *> in,
                                             DecodeParams opts,
                                             cspan<ROI> rois = {}) = 0;
  /**
   * @brief Sets a codec-specific parameter
   *
   * @return  true, if the parameter is relevant for this decoder, false otherwise
   *
   * @remarks The function may throw if the parameter name is valid for this decoder, but
   *          the value is incorrect.
   */
  virtual bool SetParam(const char *key, const std::any &value) = 0;

  /**
   * @brief Sets codec-specific parameters.
   *
   * @return  The number of parameters that were relevant for this decoder.
   *
   * @remarks The function may throw if the parameter name is valid for this decoder, but
   *          the value is incorrect.
   */
  virtual int SetParams(const std::map<std::string, std::any> &params) = 0;

  /**
   * @brief Gets a codec-specific parameter
   */
  virtual std::any GetParam(const char *key) const = 0;

  template <typename T>
  inline enable_if_t<!std::is_same<std::remove_reference_t<T>, std::any>::value, bool>
  SetParam(const char *key, T &&value) {
    return SetParam(key, std::any(std::forward<T>(value)));
  }

  template <typename T>
  inline T GetParam(const char *key) const {
    return std::any_cast<T>(GetParam(key));
  }
};

class DLL_PUBLIC ImageDecoderFactory {
 public:
  virtual ~ImageDecoderFactory() = default;

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
   */
  virtual std::shared_ptr<ImageDecoderInstance>
  Create(int device_id, const std::map<std::string, std::any> &params = {}) const = 0;
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_IMAGE_DECODER_INTERFACES_H_
