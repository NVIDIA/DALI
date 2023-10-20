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

#ifndef DALI_IMGCODEC_IMAGE_DECODER_H_
#define DALI_IMGCODEC_IMAGE_DECODER_H_

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>
#include "dali/imgcodec/image_decoder_interfaces.h"
#include "dali/imgcodec/image_format.h"
#include "dali/pipeline/data/tensor_vector.h"

namespace dali {
namespace imgcodec {

inline bool is_cpu_decoder(ImageDecoderFactory *factory) {
  const auto &properties = factory->GetProperties();
  return !properties.gpu_output && !(properties.supported_input_kinds & InputKind::DeviceMemory);
}

class DLL_PUBLIC ImageDecoder : public ImageDecoderInstance, public ImageParser {
 public:
  ImageDecoder(int device_id,
               bool lazy_init,
               const std::map<std::string, std::any> &params = {})
  : ImageDecoder(
      device_id,
      lazy_init,
      params,
      device_id == CPU_ONLY_DEVICE_ID ? is_cpu_decoder
                                      : [](ImageDecoderFactory *) { return true; }) {}

  /**
   * @brief Constructs a new ImageDecoder, allowing the user to filter the decoders
   *
   * @param device_id       CUDA device ordinal or -1 for current device or CPU_ONLY_DEVICE_ID
   * @param lazy_init       if true, the construction of sub-decoders is deferred until they are
   *                        needed - this may be usefule when decoding datasets with
   *                        few image formats
   * @param params          parameters passed to the construction of the decoders;
   *                        when using lazy initialization, the parameters can be set later
   * @param decoder_filter  a predicate applied to decoder factories, which allows the caller
   *                        to select a subset with desired properties
   */
  ImageDecoder(int device_id,
               bool lazy_init,
               const std::map<std::string, std::any> &params,
               std::function<bool(ImageDecoderFactory *)> decoder_filter);

  ~ImageDecoder();

  bool CanParse(ImageSource *encoded) const override;

  ImageInfo GetInfo(ImageSource *encoded) const override;

  std::vector<bool> GetInfo(span<ImageInfo> info, span<ImageSource*> sources) const;

  ImageFormatRegistry &FormatRegistry() const {
    // Use a global format registry.
    return ImageFormatRegistry::instance();
  }

  /**
   * @brief Stubbed; returns true.
   */
  bool CanDecode(DecodeContext ctx,
                 ImageSource *in,
                 DecodeParams opts,
                 const ROI &roi = {}) override;

  void CalculateOutputShape(TensorListShape<> &shape,
                            span<SampleView<CPUBackend>> out,
                            cspan<ImageSource *> in,
                            DecodeParams opts,
                            cspan<ROI> rois = {}) const;

  /**
   * @brief Stubbed; returns true for all images in the batch.
   */
  std::vector<bool> CanDecode(DecodeContext ctx,
                              cspan<ImageSource *> in,
                              DecodeParams opts,
                              cspan<ROI> rois = {}) override;

  // Single image API

  DecodeResult Decode(DecodeContext ctx,
                      SampleView<CPUBackend> out,
                      ImageSource *in,
                      DecodeParams opts,
                      const ROI &roi = {}) override;

  DecodeResult Decode(DecodeContext ctx,
                      SampleView<GPUBackend> out,
                      ImageSource *in,
                      DecodeParams opts,
                      const ROI &roi = {}) override;

  FutureDecodeResults ScheduleDecode(DecodeContext ctx,
                                     SampleView<CPUBackend> out,
                                     ImageSource *in,
                                     DecodeParams opts,
                                     const ROI &roi = {}) override;

  FutureDecodeResults ScheduleDecode(DecodeContext ctx,
                                     SampleView<GPUBackend> out,
                                     ImageSource *in,
                                     DecodeParams opts,
                                     const ROI &roi = {}) override;


  // Batch API

  std::vector<DecodeResult> Decode(DecodeContext ctx,
                                   span<SampleView<CPUBackend>> out,
                                   cspan<ImageSource *> in,
                                   DecodeParams opts,
                                   cspan<ROI> rois = {}) override;

  std::vector<DecodeResult> Decode(DecodeContext ctx,
                                   span<SampleView<GPUBackend>> out,
                                   cspan<ImageSource *> in,
                                   DecodeParams opts,
                                   cspan<ROI> rois = {}) override;

  FutureDecodeResults ScheduleDecode(DecodeContext ctx,
                                     span<SampleView<CPUBackend>> out,
                                     cspan<ImageSource *> in,
                                     DecodeParams opts,
                                     cspan<ROI> rois) override;

  FutureDecodeResults ScheduleDecode(DecodeContext ctx,
                                     span<SampleView<GPUBackend>> out,
                                     cspan<ImageSource *> in,
                                     DecodeParams opts,
                                     cspan<ROI> rois) override;


  std::vector<DecodeResult> Decode(DecodeContext ctx,
                                   TensorList<CPUBackend> &out,
                                   cspan<ImageSource *> in,
                                   DecodeParams opts,
                                   cspan<ROI> rois = {});

  std::vector<DecodeResult> Decode(DecodeContext ctx,
                                   TensorList<GPUBackend> &out,
                                   cspan<ImageSource *> in,
                                   DecodeParams opts,
                                   cspan<ROI> rois = {});
  /**
   * @brief Sets a value of a parameter.
   *
   * It sets a value of a parameter for all existing sub-decoders as well ones that will be
   * created in the future.
   *
   * This function succeeds even if no sub-decoder recognizes the key.
   * If the value is incorrect for one of the decoders, but that decoder has not yet
   * been constructed, an error may be thrown at a later time, when the decoder is instantiated.
   */
  bool SetParam(const char *key, const std::any &value) override;

  int SetParams(const std::map<std::string, std::any> &params) override;

  /**
   * @brief Gets a value previously passed to `SetParam` with the given key.
   */
  std::any GetParam(const char *key) const override;

 private:
  struct ScheduledWork;
  // Scheduled work pool

  std::mutex work_mutex_;
  std::unique_ptr<ScheduledWork> free_work_items_;

  std::unique_ptr<ScheduledWork> new_work(DecodeContext ctx,
                                          DecodeResultsPromise results,
                                          DecodeParams params);

  void recycle_work(std::unique_ptr<ScheduledWork> work);

  void combine_work(ScheduledWork &target, std::unique_ptr<ScheduledWork> source);

  static void move_to_fallback(ScheduledWork *fallback,
                               ScheduledWork &work,
                               const vector<bool> &keep);

  static void filter(ScheduledWork &work, const vector<bool> &keep) {
    move_to_fallback(nullptr, work, keep);
  }

  // Decoder workers
  class DecoderWorker;

  void InitWorkers(bool lazy);

  void DistributeWork(std::unique_ptr<ScheduledWork> work);

  // Miscellanous

  static void copy(SampleView<GPUBackend> &out,
                   const ConstSampleView<CPUBackend> &in,
                   cudaStream_t stream);

  int device_id_ = CPU_ONLY_DEVICE_ID;
  std::map<std::string, std::any> params_;
  std::map<const ImageDecoderFactory*, std::unique_ptr<DecoderWorker>> workers_;
  std::multimap<const ImageFormat*, ImageDecoderFactory*> filtered_;
  struct TempImageSource {
    // TODO(michalz): store auxiliary image sources here
  };
  std::map<InputKind, std::vector<TempImageSource>> tmp_sources_;
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_IMAGE_DECODER_H_
