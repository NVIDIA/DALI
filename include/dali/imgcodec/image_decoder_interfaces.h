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

#ifndef DALI_IMGCODEC_IMAGE_DECODER_INTERFACES_H_
#define DALI_IMGCODEC_IMAGE_DECODER_INTERFACES_H_

#include <memory>
#include <utility>
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

template <typename OutShape>
void OutputShape(OutShape &out_shape,
                 const ImageInfo &info, const DecodeParams &params, const ROI &roi) {
  int ndim = info.shape.sample_dim();
  resize_if_possible(out_shape, ndim);
  assert(size(out_shape) == ndim);  // check the size, in case we couldn't resize

  int spatial_ndim = ndim - 1;
  int in_channel_dim = ndim - 1;
  int num_channels = NumberOfChannels(params.format, info.shape[in_channel_dim]);
  int out_channel_dim = params.planar ? 0 : ndim - 1;

  out_shape[out_channel_dim] = num_channels;

  for (int d = 0; d < spatial_ndim; d++) {
    int in_d = d;
    if (params.use_orientation && ((info.orientation.rotate / 90) & 1)) {
      if (spatial_ndim != 2)
        throw std::logic_error("Orientation only applied to 2D images.");
      in_d = 1 - d;
    }
    int extent = info.shape[in_d];
    if (d < roi.end.size())
      extent = roi.end[d];
    if (d < roi.begin.size())
      extent -= roi.begin[d];
    out_shape[d + (d >= out_channel_dim)] = extent;
  }
}

struct DecodeResult {
  bool success;
  std::exception_ptr exception;
};

struct ImageDecoderProperties {
  /**
   * @brief Whether the codec can decode a region of interest without decoding the entire image
   */
  bool supports_partial_decoding = false;

  /**
   * @brief A mask of supported input kinds
   */
  InputKind supported_input_kinds;

  /**
   * @brief if true and the codec fails to decode an image,
   *        an attempt will be made to use other compatible codecs
   */
  bool fallback = true;
};

class DLL_PUBLIC DeferredDecodeResults {
 public:
  explicit DeferredDecodeResults(int num_samples);
  ~DeferredDecodeResults();

  DeferredDecodeResults(const DeferredDecodeResults &) = delete;
  DeferredDecodeResults(DeferredDecodeResults &&other) : impl_(other.impl_) {
    other.impl_ = nullptr;
  }

  DeferredDecodeResults &operator=(const DeferredDecodeResults &) = delete;
  DeferredDecodeResults &operator=(DeferredDecodeResults &&other) {
    std::swap(impl_, other.impl_);
    return *this;
  }

  /**
   * @brief Waits for all the results to be ready
   */
  void wait_all() const;

  /**
   * @brief Waits for a particular result
   */
  void wait(int index) const;

  /**
   * @brief Get the all results
   */
  span<DecodeResult> get_all() const;

  /**
   * @brief Get the a particular result
   */
  DecodeResult get(int index) const;

  /**
   * @brief Set the all the results
   */
  void set_all(span<const DecodeResult> res);
   
  /**
   * @brief Set a particular result
   */
  void set(int index, DecodeResult res);

  /**
   * @brief Number of samples
   */
  int num_samples() const;

 private:
  DecodeResult get_no_wait(int index) const;
  class Impl;
  Impl *impl_ = nullptr;
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
  virtual bool CanDecode(ImageSource *in, DecodeParams opts, const ROI &roi = {}) = 0;

  /**
   * @brief Batch version of CanDecode
   */
  virtual std::vector<bool> CanDecode(cspan<ImageSource *> in,
                                      DecodeParams opts,
                                      cspan<ROI> rois = {}) = 0;

  /**
   * @brief Decodes a single image to a host buffer
   */
  virtual DecodeResult Decode(SampleView<CPUBackend> out,
                              ImageSource *in,
                              DecodeParams opts,
                              const ROI &roi = {}) = 0;

  /* virtual DeferredDecodeResults SheduleDecode(DecodeContext &ctx,
                                              SampleView<CPUBackend> out,
                                              ImageSource *in,
                                              DecodeParams opts,
                                              const ROI &roi = {}) = 0; */

  /**
   * @brief Decodes a batch of images to host buffers
   */
  virtual std::vector<DecodeResult> Decode(/* DecodeContext &ctx, */
                                           span<SampleView<CPUBackend>> out,
                                           cspan<ImageSource *> in,
                                           DecodeParams opts,
                                           cspan<ROI> rois = {}) = 0;

  /**
   * @brief Decodes a batch of images to host buffers
   */
  /* virtual DeferredDecodeResults ScheduleDecode(DecodeContext &ctx,
                                               span<SampleView<CPUBackend>> out,
                                               cspan<ImageSource *> in,
                                               DecodeParams opts,
                                               cspan<ROI> rois = {}) = 0; */

  /**
   * @brief Decodes a single image to a device buffer
   */
  virtual DecodeResult Decode(cudaStream_t stream,
                              SampleView<GPUBackend> out,
                              ImageSource *in,
                              DecodeParams opts,
                              const ROI &roi = {}) = 0;

  /**
   * @brief Decodes a single image to device buffers
   */
  virtual std::vector<DecodeResult> Decode(cudaStream_t stream,
                                           span<SampleView<GPUBackend>> out,
                                           cspan<ImageSource *> in,
                                           DecodeParams opts,
                                           cspan<ROI> rois = {}) = 0;
  /**
   * @brief Sets a codec-specific parameter
   */
  virtual bool SetParam(const char *key, const any &value) = 0;
  /**
   * @brief Gets a codec-specific parameter
   */
  virtual any GetParam(const char *key) const = 0;

  template <typename T>
  inline enable_if_t<!std::is_same<std::remove_reference_t<T>, any>::value, bool>
  SetParam(const char *key, T &&value) {
    return SetParam(key, any(std::forward<T>(value)));
  }

  template <typename T>
  inline T GetParam(const char *key) const {
    return any_cast<T>(GetParam(key));
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
   *
   * Note: For decoders that carry no state, this may just increase reference count on a singleton.
   */
  virtual std::shared_ptr<ImageDecoderInstance> Create(int device_id, ThreadPool &tp) const = 0;
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_IMAGE_DECODER_INTERFACES_H_
