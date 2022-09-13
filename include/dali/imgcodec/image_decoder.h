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

#include <condition_variable>
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
               const std::map<std::string, any> &params = {})
  : ImageDecoder(
      device_id,
      lazy_init,
      params,
      device_id == CPU_ONLY_DEVICE_ID ? is_cpu_decoder
                                      : [](ImageDecoderFactory *) { return true; }) {}

  /**
   * @brief Construct a new ImageDecode, allowing the user to filter the decoders
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
               const std::map<std::string, any> &params,
               std::function<bool(ImageDecoderFactory *)> decoder_filter);

  ~ImageDecoder();

  bool CanParse(ImageSource *encoded) const override {
    auto *f = FormatRegistry().GetImageFormat(encoded);
    return f && filtered_.find(f) != filtered_.end();
  }

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
  bool SetParam(const char *key, const any &value) override;

  int SetParams(const std::map<std::string, any> &params) override;

  /**
   * @brief Gets a value previously passed to `SetParam` with the given key.
   */
  any GetParam(const char *key) const override;

 private:
  struct ScheduledWork {
    ScheduledWork(DecodeContext ctx, DecodeResultsPromise results)
    : ctx(std::move(ctx)), results(std::move(results)) {}

    void clear() {
      indices.clear();
      sources.clear();
      cpu_outputs.clear();
      gpu_outputs.clear();
      rois.clear();
    }

    int num_samples() const {
      return indices.size();
    }

    bool empty() const {
      return indices.empty();
    }

    void resize(int num_samples) {
      indices.resize(num_samples);
      sources.resize(num_samples);
      if (!cpu_outputs.empty())
        cpu_outputs.resize(num_samples);
      if (!gpu_outputs.empty())
        gpu_outputs.resize(num_samples);
      if (!rois.empty())
        rois.resize(num_samples);
    }

    void move_entry(ScheduledWork &from, int which);

    void alloc_temp_cpu_outputs(ImageDecoder &owner);

    DecodeContext ctx;
    // The original promise
    DecodeResultsPromise results;
    // The indices in the original request
    BatchVector<int> indices;
    BatchVector<ImageSource *> sources;
    BatchVector<SampleView<CPUBackend>> cpu_outputs;
    BatchVector<SampleView<GPUBackend>> gpu_outputs;
    DecodeParams params;
    BatchVector<ROI> rois;

    std::unique_ptr<ScheduledWork> next;
  };

  std::mutex work_mutex_;
  std::unique_ptr<ScheduledWork> free_work_items_;

  std::unique_ptr<ScheduledWork> new_work(DecodeContext ctx, DecodeResultsPromise results);

  void recycle_work(std::unique_ptr<ScheduledWork> work);

  void combine_work(ScheduledWork &target, std::unique_ptr<ScheduledWork> source);

  static void move_to_fallback(ScheduledWork *fallback,
                               ScheduledWork &work,
                               const vector<bool> &keep);

  static void filter(ScheduledWork &work, const vector<bool> &keep) {
    move_to_fallback(nullptr, work, keep);
  }

  static void copy(SampleView<GPUBackend> &out,
                   const ConstSampleView<CPUBackend> &in,
                   cudaStream_t stream);

  class DecoderWorker {
   public:
    DecoderWorker(ImageDecoder *owner, const ImageDecoderFactory *factory, bool start) {
      owner_ = owner;
      factory_ = factory;
      if (start)
        this->start();
    }
    ~DecoderWorker();

    void start();
    void stop();
    void add_work(std::unique_ptr<ScheduledWork> work);

    void set_fallback(DecoderWorker *fallback) {
      fallback_ = fallback;
    }

    ImageDecoderInstance *decoder(bool create_if_null = true);

   private:
    std::mutex mtx_;
    std::condition_variable cv_;

    std::unique_ptr<ScheduledWork> work_;
    std::thread worker_;
    bool stop_requested_ = false;
    std::once_flag started_;

    ImageDecoder *owner_ = nullptr;
    const ImageDecoderFactory *factory_ = nullptr;
    std::shared_ptr<ImageDecoderInstance> decoder_;
    bool produces_gpu_output_ = false;

    void process_batch(std::unique_ptr<ScheduledWork> work);

    void run();

    DecoderWorker *fallback_ = nullptr;
  };

  void InitWorkers(bool lazy);

  DecoderWorker &GetWorker(ImageDecoderFactory *factory);

  void DistributeWork(std::unique_ptr<ScheduledWork> work);

  void Decode(ScheduledWork *work);

  int device_id_ = CPU_ONLY_DEVICE_ID;
  std::map<std::string, any> params_;
  std::map<const ImageDecoderFactory*, std::unique_ptr<DecoderWorker>> workers_;
  std::multimap<const ImageFormat*, ImageDecoderFactory*> filtered_;
  std::vector<std::shared_ptr<void>> temp_buffers_;
  struct TempImageSource {
    // TODO(michalz): store auxiliary image sources here
  };
  std::map<InputKind, std::vector<TempImageSource>> tmp_sources_;
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_IMAGE_DECODER_H_
