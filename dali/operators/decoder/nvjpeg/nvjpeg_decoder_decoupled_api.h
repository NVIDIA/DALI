// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_DECODER_NVJPEG_NVJPEG_DECODER_DECOUPLED_API_H_
#define DALI_OPERATORS_DECODER_NVJPEG_NVJPEG_DECODER_DECOUPLED_API_H_

#include <nvjpeg.h>
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <numeric>
#include <atomic>
#include "dali/pipeline/operator/operator.h"
#include "dali/operators/decoder/nvjpeg/nvjpeg_helper.h"
#include "dali/operators/decoder/nvjpeg/nvjpeg_memory.h"
#include "dali/operators/decoder/nvjpeg/nvjpeg2k_helper.h"
#include "dali/operators/decoder/cache/cached_decoder_impl.h"
#include "dali/kernels/alloc.h"
#include "dali/util/image.h"
#include "dali/util/ocv.h"
#include "dali/image/image_factory.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/core/device_guard.h"
#include "dali/core/dev_buffer.h"
#include "dali/operators/decoder/nvjpeg/permute_layout.h"

namespace dali {

using ImageInfo = EncodedImageInfo<int>;

class nvJPEGDecoder : public Operator<MixedBackend>, CachedDecoderImpl {
 public:
  explicit nvJPEGDecoder(const OpSpec& spec) :
    Operator<MixedBackend>(spec),
    CachedDecoderImpl(spec),
    output_image_type_(spec.GetArgument<DALIImageType>("output_type")),
    hybrid_huffman_threshold_(spec.GetArgument<unsigned int>("hybrid_huffman_threshold")),
    use_fast_idct_(spec.GetArgument<bool>("use_fast_idct")),
    output_shape_(max_batch_size_, kOutputDim),
    pinned_buffers_(num_threads_*2),
    jpeg_streams_(num_threads_*2),
#if NVJPEG2K_ENABLED
    nvjpeg2k_intermediate_buffer_(),
    nvjpeg2k_dev_alloc_(nvjpeg_memory::GetDeviceAllocatorNvJpeg2k()),
    nvjpeg2k_pin_alloc_(nvjpeg_memory::GetPinnedAllocatorNvJpeg2k()),
    nvjpeg2k_streams_(max_batch_size_),
#endif  // NVJPEG2K_ENABLED
    device_buffers_(num_threads_),
    streams_(num_threads_),
    decode_events_(num_threads_),
    thread_page_ids_(num_threads_),
    device_id_(spec.GetArgument<int>("device_id")),
    device_allocator_(nvjpeg_memory::GetDeviceAllocator()),
    pinned_allocator_(nvjpeg_memory::GetPinnedAllocator()),
    thread_pool_(num_threads_,
                 spec.GetArgument<int>("device_id"),
                 spec.GetArgument<bool>("affine") /* pin threads */),
    nvjpeg2k_thread_(1,
                     spec.GetArgument<int>("device_id"),
                     spec.GetArgument<bool>("affine")) {
#if NVJPEG_VER_MAJOR >= 11
    // if hw_decoder_load is not present in the schema (crop/sliceDecoder) then it is not supported
    bool try_init_hw_decoder = false;
    if (spec_.GetSchema().HasArgument("hw_decoder_load")) {
      hw_decoder_load_ = spec.GetArgument<float>("hw_decoder_load");
      try_init_hw_decoder = true;
    } else {
      hw_decoder_load_ = 0;
    }

    if (try_init_hw_decoder &&
        nvjpegCreate(NVJPEG_BACKEND_HARDWARE, NULL, &handle_) == NVJPEG_STATUS_SUCCESS) {
      LOG_LINE << "Using NVJPEG_BACKEND_HARDWARE" << std::endl;
      NVJPEG_CALL(nvjpegJpegStateCreate(handle_, &state_hw_batched_));
      hw_decoder_images_staging_.set_pinned(true);
      hw_decoder_images_staging_.SetGrowthFactor(2);
      // assume close the worst case size 300kb per image
      auto shapes = uniform_list_shape(CalcHwDecoderBatchSize(hw_decoder_load_, max_batch_size_),
                                       TensorShape<1>{300*1024});
      hw_decoder_images_staging_.Resize(shapes, TypeInfo::Create<uint8_t>());
#if defined(NVJPEG_PREALLOCATE_API)
      // TODO(awolant): How to expose chroma subsampling to the user?
      if (spec.GetArgument<int>("preallocate_width_hint") > 0 &&
          spec.GetArgument<int>("preallocate_height_hint") > 0) {
        LOG_LINE << "Using NVJPEG_PREALLOCATE_API" << std::endl;
        NVJPEG_CALL(nvjpegDecodeBatchedPreAllocate(
          handle_,
          state_hw_batched_,
          CalcHwDecoderBatchSize(hw_decoder_load_, max_batch_size_),
          spec.GetArgument<int>("preallocate_width_hint"),
          spec.GetArgument<int>("preallocate_height_hint"),
          NVJPEG_CSS_444,
          NVJPEG_OUTPUT_RGB));
      }
#endif
      using_hw_decoder_ = true;
      in_data_.reserve(max_batch_size_);
      in_lengths_.reserve(max_batch_size_);
      nvjpeg_destinations_.reserve(max_batch_size_);
    } else {
      LOG_LINE << "NVJPEG_BACKEND_HARDWARE is either disabled or not supported" << std::endl;
      NVJPEG_CALL(nvjpegCreateSimple(&handle_));
    }
#else
    NVJPEG_CALL(nvjpegCreateSimple(&handle_));
#endif

    size_t device_memory_padding = spec.GetArgument<Index>("device_memory_padding");
    size_t host_memory_padding = spec.GetArgument<Index>("host_memory_padding");
    size_t device_memory_padding_jpeg2k = spec.GetArgument<Index>("device_memory_padding_jpeg2k");
    size_t host_memory_padding_jpeg2k = spec.GetArgument<Index>("host_memory_padding_jpeg2k");
    NVJPEG_CALL(nvjpegSetDeviceMemoryPadding(device_memory_padding, handle_));
    NVJPEG_CALL(nvjpegSetPinnedMemoryPadding(host_memory_padding, handle_));

    nvjpegDevAllocator_t *device_allocator_ptr = &device_allocator_;
    nvjpegPinnedAllocator_t *pinned_allocator_ptr = &pinned_allocator_;

    nvjpeg_memory::SetEnableMemStats(spec.GetArgument<bool>("memory_stats"));

    for (auto thread_id : thread_pool_.GetThreadIds()) {
      if (device_memory_padding > 0) {
        nvjpeg_memory::AddBuffer(thread_id, kernels::AllocType::GPU, device_memory_padding);
      }
      if (host_memory_padding > 0) {
        nvjpeg_memory::AddBuffer(thread_id, kernels::AllocType::Pinned, host_memory_padding);
        nvjpeg_memory::AddBuffer(thread_id, kernels::AllocType::Pinned, host_memory_padding);
      }
    }

    // GPU
    // create the handles, streams and events we'll use
    // We want to use nvJPEG default device allocator
    for (auto &stream : jpeg_streams_) {
      NVJPEG_CALL(nvjpegJpegStreamCreate(handle_, &stream));
    }
    NVJPEG_CALL(nvjpegJpegStreamCreate(handle_, &hw_decoder_jpeg_stream_));

    for (auto &buffer : pinned_buffers_) {
      NVJPEG_CALL(nvjpegBufferPinnedCreate(handle_, pinned_allocator_ptr, &buffer));
    }
    for (auto &buffer : device_buffers_) {
      NVJPEG_CALL(nvjpegBufferDeviceCreate(handle_, device_allocator_ptr, &buffer));
    }
    for (auto &stream : streams_) {
      CUDA_CALL(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking,
                                             default_cuda_stream_priority_));
    }
    CUDA_CALL(cudaStreamCreateWithPriority(
      &hw_decode_stream_, cudaStreamNonBlocking, default_cuda_stream_priority_));

    for (auto &event : decode_events_) {
      CUDA_CALL(cudaEventCreate(&event));
      CUDA_CALL(cudaEventRecord(event, streams_[0]));
    }

    CUDA_CALL(cudaEventCreate(&hw_decode_event_));
    CUDA_CALL(cudaEventRecord(hw_decode_event_, hw_decode_stream_));

#if NVJPEG2K_ENABLED
    auto nvjpeg2k_thread_id = nvjpeg2k_thread_.GetThreadIds()[0];
    nvjpeg2k_thread_.AddWork([this, device_memory_padding_jpeg2k, host_memory_padding_jpeg2k,
                              nvjpeg2k_thread_id](int) {
      nvjpeg2k_handle_ = NvJPEG2KHandle(&nvjpeg2k_dev_alloc_, &nvjpeg2k_pin_alloc_);

      // nvJPEG2k doesn't support deprecated sm while DALI does. In this case, NvJPEG2KHandle may
      // be empty and we need to fall back to the CPU implementation
      if (!nvjpeg2k_handle_) {
        return;
      }
      if (device_memory_padding_jpeg2k > 0) {
        // Adding smaller buffers that are allocated by nvjpeg2k on startup.
        // The sizes were obtained empirically.
        nvjpeg_memory::AddBuffer(nvjpeg2k_thread_id, kernels::AllocType::GPU, 1024);
        nvjpeg_memory::AddBuffer(nvjpeg2k_thread_id, kernels::AllocType::GPU, 4 * 1024);
        nvjpeg_memory::AddBuffer(nvjpeg2k_thread_id, kernels::AllocType::GPU, 16 * 1024);
        nvjpeg2k_intermediate_buffer_.resize(device_memory_padding_jpeg2k / 8);
        nvjpeg_memory::AddBuffer(nvjpeg2k_thread_id, kernels::AllocType::GPU,
                                 device_memory_padding_jpeg2k);
        nvjpeg_memory::AddBuffer(nvjpeg2k_thread_id, kernels::AllocType::GPU,
                                 device_memory_padding_jpeg2k);
      }
      if (host_memory_padding_jpeg2k > 0) {
        nvjpeg_memory::AddBuffer(nvjpeg2k_thread_id, kernels::AllocType::Pinned,
                                 host_memory_padding_jpeg2k);
      }
      nvjpeg2k_decoder_ = NvJPEG2KDecodeState(nvjpeg2k_handle_);
    });
    nvjpeg2k_thread_.RunAll();

    CUDA_CALL(cudaStreamCreateWithPriority(&nvjpeg2k_cu_stream_, cudaStreamNonBlocking,
                                           default_cuda_stream_priority_));
    CUDA_CALL(cudaEventCreate(&nvjpeg2k_decode_event_));
    CUDA_CALL(cudaEventRecord(nvjpeg2k_decode_event_, nvjpeg2k_cu_stream_));

    for (auto &stream : nvjpeg2k_streams_) {
      stream = NvJPEG2KStream::Create();
    }
#endif  // NVJPEG2K_ENABLED

    ReserveSampleContainers();

    RegisterTestCounters();
  }

  ~nvJPEGDecoder() override {
    try {
      DeviceGuard g(device_id_);

      sample_data_.clear();

      for (auto &stream : streams_) {
        CUDA_CALL(cudaStreamSynchronize(stream));
      }

      for (auto &stream  : jpeg_streams_) {
        NVJPEG_CALL(nvjpegJpegStreamDestroy(stream));
      }
      NVJPEG_CALL(nvjpegJpegStreamDestroy(hw_decoder_jpeg_stream_));

      for (auto &buffer : pinned_buffers_) {
        NVJPEG_CALL(nvjpegBufferPinnedDestroy(buffer));
      }
      for (auto &buffer : device_buffers_) {
        NVJPEG_CALL(nvjpegBufferDeviceDestroy(buffer));
      }
      for (auto &event : decode_events_) {
        CUDA_CALL(cudaEventDestroy(event));
      }
      CUDA_CALL(cudaEventDestroy(hw_decode_event_));

      for (auto &stream : streams_) {
        CUDA_CALL(cudaStreamDestroy(stream));
      }
      CUDA_CALL(cudaStreamDestroy(hw_decode_stream_));

      NVJPEG_CALL(nvjpegDestroy(handle_));

      // Free any remaining buffers and remove the thread entry from the global map
      for (auto thread_id : thread_pool_.GetThreadIds()) {
        nvjpeg_memory::DeleteAllBuffers(thread_id);
      }

#if NVJPEG2K_ENABLED
      CUDA_CALL(cudaEventDestroy(nvjpeg2k_decode_event_));
      CUDA_CALL(cudaStreamDestroy(nvjpeg2k_cu_stream_));
      for (auto thread_id : nvjpeg2k_thread_.GetThreadIds()) {
        nvjpeg_memory::DeleteAllBuffers(thread_id);
      }
#endif  // NVJPEG2K_ENABLED

      nvjpeg_memory::PrintMemStats();
    } catch (const std::exception &e) {
      // If destroying nvJPEG resources failed we are leaking something so terminate
      std::cerr << "Fatal error: exception in ~nvJPEGDecoder():\n" << e.what() << std::endl;
      std::terminate();
    }
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const MixedWorkspace &ws) override {
    auto curr_batch_size = ws.GetInputBatchSize(0);
    hw_decoder_bs_ = CalcHwDecoderBatchSize(hw_decoder_load_, curr_batch_size);
    return false;
  }

  using dali::OperatorBase::Run;
  void Run(MixedWorkspace &ws) override {
    SetupSharedSampleParams(ws);
    ParseImagesInfo(ws);
    ProcessImages(ws);
  }

 protected:
  virtual CropWindowGenerator GetCropWindowGenerator(int data_idx) const {
    return {};
  }

  enum class DecodeMethod {
    Host,
    NvjpegCuda,
    NvjpegHw,
#if NVJPEG2K_ENABLED
    Nvjpeg2k,
#endif  // NVJPEG2K_ENABLED
    Cache,
  };

  struct DecoderData {
    nvjpegJpegDecoder_t decoder = nullptr;
    nvjpegJpegState_t state = nullptr;
  };

  struct SampleData {
    int sample_idx;
    int64_t encoded_length = 0;
    bool is_progressive = false;
    std::string file_name;
    TensorShape<> shape;
    CropWindow roi;
    DecodeMethod method = DecodeMethod::Host;
    nvjpegDecodeParams_t params;
    nvjpegChromaSubsampling_t subsampling = NVJPEG_CSS_UNKNOWN;

    // enough to access by nvjpegBackend_t (index 0 not used)
    std::array<DecoderData, 4> decoders = {};

    DecoderData *selected_decoder = nullptr;

    SampleData(int idx, nvjpegHandle_t &handle, DALIImageType img_type)
        : sample_idx(idx) {
      NVJPEG_CALL(nvjpegDecodeParamsCreate(handle, &params));
      NVJPEG_CALL(nvjpegDecodeParamsSetOutputFormat(params, GetFormat(img_type)));
      NVJPEG_CALL(nvjpegDecodeParamsSetAllowCMYK(params, true));

      for (auto backend : {NVJPEG_BACKEND_HYBRID, NVJPEG_BACKEND_GPU_HYBRID}) {
        auto &decoder = decoders[backend].decoder;
        NVJPEG_CALL(nvjpegDecoderCreate(handle, backend, &decoder));
        NVJPEG_CALL(nvjpegDecoderStateCreate(handle, decoder, &decoders[backend].state));
      }
    }

    ~SampleData() {
      try {
        NVJPEG_CALL(nvjpegDecodeParamsDestroy(params));
        for (auto &decoder_data : decoders) {
          auto &state = decoder_data.state;
          if (state) {
            NVJPEG_CALL(nvjpegJpegStateDestroy(state));
            state = nullptr;
          }

          auto &decoder = decoder_data.decoder;
          if (decoder) {
            NVJPEG_CALL(nvjpegDecoderDestroy(decoder));
            decoder = nullptr;
          }
        }
      } catch (const std::exception &e) {
        // If destroying nvJPEG resources failed we are leaking something, so terminate
        std::cerr << "Error while releasing nvjpeg resources: " << e.what() << std::endl;
        std::terminate();
      }
    }
  };

  std::vector<SampleData> sample_data_;

  std::vector<SampleData*> samples_cache_;
  std::vector<SampleData*> samples_host_;
  std::vector<SampleData*> samples_hw_batched_;
  std::vector<SampleData*> samples_single_;
  std::vector<SampleData*> samples_jpeg2k_;

  nvjpegJpegState_t state_hw_batched_ = nullptr;

  static int64_t subsampling_score(nvjpegChromaSubsampling_t subsampling) {
    switch (subsampling) {
      case NVJPEG_CSS_444:
        return 8;
      case NVJPEG_CSS_422:
        return 7;
      case NVJPEG_CSS_440:
        return 6;
      case NVJPEG_CSS_420:
        return 5;
      case NVJPEG_CSS_411:
        return 4;
      case NVJPEG_CSS_410:
        return 3;
      case NVJPEG_CSS_GRAY:
        return 2;
      case NVJPEG_CSS_UNKNOWN:
      default:
        return 1;
    }
  }

  void RebalanceAndSortSamples() {
    static const char *sort_method_env = getenv("SORT_METHOD");
    enum {
      SORT_METHOD_SUBSAMPLING_AND_SIZE = 0,
      SORT_METHOD_NO_SORTING = 1
    };

    static const int sort_method = sort_method_env == nullptr ?
        SORT_METHOD_SUBSAMPLING_AND_SIZE : atoi(sort_method_env);

    DALI_ENFORCE(sort_method >= SORT_METHOD_SUBSAMPLING_AND_SIZE &&
                 sort_method <= SORT_METHOD_NO_SORTING);
    auto sample_order = [](SampleData *lhs, SampleData *rhs) {
      if (lhs->subsampling != rhs->subsampling)
        return subsampling_score(lhs->subsampling) > subsampling_score(rhs->subsampling);

      if (lhs->shape[0] == rhs->shape[0])
        return lhs->shape[1] > rhs->shape[1];

      return lhs->shape[0] > rhs->shape[0];
    };

    if (sort_method != SORT_METHOD_NO_SORTING)
      std::sort(samples_hw_batched_.begin(), samples_hw_batched_.end(), sample_order);

    assert(hw_decoder_bs_ >= 0 && hw_decoder_bs_ <= max_batch_size_);

    // If necessary trim HW batch size and push the remaining samples to the single API batch
    while (samples_hw_batched_.size() > static_cast<size_t>(hw_decoder_bs_)) {
      samples_single_.push_back(samples_hw_batched_.back());
      samples_hw_batched_.pop_back();

      auto &data = *samples_single_.back();
      data.method = DecodeMethod::NvjpegCuda;
      int64_t sz = data.roi
        ? data.roi.shape[1] * (data.roi.anchor[0] + data.roi.shape[0])
        : data.shape[0] * data.shape[1];
      if (sz > hybrid_huffman_threshold_ && !data.is_progressive) {
        data.selected_decoder = &data.decoders[NVJPEG_BACKEND_GPU_HYBRID];
      } else {
        data.selected_decoder = &data.decoders[NVJPEG_BACKEND_HYBRID];
      }
    }

    if (sort_method != SORT_METHOD_NO_SORTING) {
      std::sort(samples_single_.begin(), samples_single_.end(), sample_order);
      std::sort(samples_host_.begin(), samples_host_.end(), sample_order);
    }
  }

  bool ParseNvjpeg2k(SampleData &data, span<const uint8_t> input) {
#if NVJPEG2K_ENABLED
    if (!nvjpeg2k_handle_) {
      return false;
    }
    const auto &jpeg2k_stream = nvjpeg2k_streams_[data.sample_idx];
    auto ret = nvjpeg2kStreamParse(nvjpeg2k_handle_, input.data(), input.size(),
                                  0, 0, jpeg2k_stream);
    if (ret == NVJPEG2K_STATUS_SUCCESS) {
      nvjpeg2kImageInfo_t image_info;
      NVJPEG2K_CALL(nvjpeg2kStreamGetImageInfo(jpeg2k_stream, &image_info));
      nvjpeg2kImageComponentInfo_t comp;
      NVJPEG2K_CALL(nvjpeg2kStreamGetImageComponentInfo(jpeg2k_stream, &comp, 0));
      uint32_t height = comp.component_height;
      uint32_t width = comp.component_width;
      for (uint32_t c = 1; c < image_info.num_components; ++c) {
        NVJPEG2K_CALL(nvjpeg2kStreamGetImageComponentInfo(jpeg2k_stream, &comp, c));
        DALI_ENFORCE(height == comp.component_height &&
                      width == comp.component_width,
                      make_string("Components dimensions do not match: "
                                  "component 0 has a shape of {", comp.component_height, ", ",
                                  comp.component_width, "} and component ", c, " has a shape of {",
                                  height, ", ", width, "}"));
      }
      data.shape = {height, width, image_info.num_components};
      data.method = DecodeMethod::Nvjpeg2k;
      samples_jpeg2k_.push_back(&data);
      return true;
    }
#endif  // NVJPEG2K_ENABLED
    return false;
  }

  void ParseImagesInfo(MixedWorkspace &ws) {
    auto curr_batch_size = ws.GetInputBatchSize(0);
    output_shape_.resize(curr_batch_size);
    samples_cache_.clear();
    samples_host_.clear();
    samples_hw_batched_.clear();
    samples_single_.clear();
#if NVJPEG2K_ENABLED
    samples_jpeg2k_.clear();
#endif  // NVJPEG2K_ENABLED

    for (int i = 0; i < curr_batch_size; i++) {
      const auto &in = ws.Input<CPUBackend>(0, i);
      const auto *input_data = in.data<uint8_t>();
      const auto in_size = in.size();

      SampleData &data = sample_data_[i];
      data.file_name = in.GetSourceInfo();
      assert(data.sample_idx == i);
      data.encoded_length = in_size;
      data.selected_decoder = nullptr;

      auto cached_shape = CacheImageShape(data.file_name);
      if (volume(cached_shape) > 0) {
        data.method = DecodeMethod::Cache;
        data.shape = cached_shape;
        output_shape_.set_tensor_shape(i, data.shape);
        samples_cache_.push_back(&data);
        continue;
      }

      int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT], c;
      nvjpegChromaSubsampling_t subsampling;
      nvjpegStatus_t ret = nvjpegGetImageInfo(handle_, input_data, in_size, &c,
                                              &subsampling, widths, heights);

      auto crop_generator = GetCropWindowGenerator(i);
      if (ret == NVJPEG_STATUS_SUCCESS) {
        bool hw_decode = false;
#if NVJPEG_VER_MAJOR >= 11
        if (!crop_generator && state_hw_batched_ != nullptr) {
          NVJPEG_CALL(nvjpegJpegStreamParseHeader(handle_, input_data, in_size,
                                                  hw_decoder_jpeg_stream_));
          int is_supported = -1;
          NVJPEG_CALL(nvjpegDecodeBatchedSupported(handle_, hw_decoder_jpeg_stream_,
                                                   &is_supported));
          hw_decode = is_supported == 0;
          if (!hw_decode) {
            LOG_LINE << "Sample \"" << data.file_name
                     << "\" can't be handled by the HW decoder and shall be processed by the CUDA "
                        "hybrid decoder"
                     << std::endl;
          }
        }
#endif
        data.shape = {heights[0], widths[0], c};
        data.subsampling = subsampling;

        if (hw_decode) {
          data.method = DecodeMethod::NvjpegHw;
          samples_hw_batched_.push_back(&data);
        } else {
          data.method = DecodeMethod::NvjpegCuda;
          samples_single_.push_back(&data);
        }
        if (output_image_type_ != DALI_ANY_DATA) {
          data.shape[2] = NumberOfChannels(output_image_type_);
        } else {
          data.shape[2] = 3;  // just output RGB
        }
      } else if (crop_generator || !ParseNvjpeg2k(data, span<const uint8_t>(input_data, in_size))) {
        try {
          data.method = DecodeMethod::Host;
          auto image = ImageFactory::CreateImage(input_data, in_size, output_image_type_);
          data.shape = image->PeekShape();
          if (output_image_type_ != DALI_ANY_DATA) {
            data.shape[2] = NumberOfChannels(output_image_type_);
          }
          samples_host_.push_back(&data);
        } catch (const std::runtime_error &e) {
          DALI_FAIL(e.what() + ". File: " + data.file_name);
        }
      }

      if (crop_generator) {
        TensorShape<> dims{data.shape[0], data.shape[1]};
        int nchannels = data.shape[2];
        data.roi = crop_generator(dims, "HW");
        DALI_ENFORCE(data.roi.IsInRange(dims));
        output_shape_.set_tensor_shape(i, {data.roi.shape[0], data.roi.shape[1], nchannels});
        NVJPEG_CALL(nvjpegDecodeParamsSetROI(data.params, data.roi.anchor[1], data.roi.anchor[0],
                                             data.roi.shape[1], data.roi.shape[0]));
      } else {
        output_shape_.set_tensor_shape(i, data.shape);
        NVJPEG_CALL(nvjpegDecodeParamsSetROI(data.params, 0, 0, -1, -1));
      }

      data.is_progressive = IsProgressiveJPEG(input_data, in_size);
      if (data.method == DecodeMethod::NvjpegCuda) {
        int64_t sz = data.roi
          ? data.roi.shape[1] * (data.roi.anchor[0] + data.roi.shape[0])
          : data.shape[0] * data.shape[1];
        if (sz > hybrid_huffman_threshold_ && !data.is_progressive) {
          data.selected_decoder = &data.decoders[NVJPEG_BACKEND_GPU_HYBRID];
        } else {
          data.selected_decoder = &data.decoders[NVJPEG_BACKEND_HYBRID];
        }
      }
    }

    // Makes sure that the HW decoder balance is right and sorts the image by chroma subsampling and
    // dimensions
    RebalanceAndSortSamples();
  }

  void ProcessImagesCache(MixedWorkspace &ws) {
    auto& output = ws.Output<GPUBackend>(0);
    for (auto *sample : samples_cache_) {
      assert(sample);
      auto i = sample->sample_idx;
      auto *output_data = output.mutable_tensor<uint8_t>(i);
      DALI_ENFORCE(DeferCacheLoad(sample->file_name, output_data));
    }
    LoadDeferred(ws.stream());
  }

  void ProcessImagesCuda(MixedWorkspace &ws) {
    auto& output = ws.Output<GPUBackend>(0);
    for (auto *sample : samples_single_) {
      assert(sample);
      auto i = sample->sample_idx;
      auto *output_data = output.mutable_tensor<uint8_t>(i);
      const auto &in = ws.Input<CPUBackend>(0, i);
      ImageCache::ImageShape shape = output_shape_[i].to_static<3>();
      thread_pool_.AddWork(
        [this, sample, &in, output_data, shape](int tid) {
          SampleWorker(sample->sample_idx, sample->file_name, in.size(), tid,
            in.data<uint8_t>(), output_data, streams_[tid]);
          CacheStore(sample->file_name, output_data, shape, streams_[tid]);
        }, task_priority_seq_--);  // FIFO order, since the samples were already ordered
    }
  }

#if NVJPEG2K_ENABLED
  void DecodeJpeg2k(uint8_t* output_data, const SampleData *sample,
                    span<const uint8_t> input_data) {
    if (output_image_type_ != DALI_ANY_DATA) {
      auto requested_channel_num = NumberOfChannels(output_image_type_);
      DALI_ENFORCE(requested_channel_num == sample->shape[2],
                   make_string("Requested number of output channels ", requested_channel_num,
                   ", but the image ", sample->file_name, " has ", sample->shape[2], "."));
    }
    CUDA_CALL(cudaEventSynchronize(nvjpeg2k_decode_event_));
    bool single_channel = sample->shape[2] == 1;
    auto &buffer = nvjpeg2k_intermediate_buffer_;
    buffer.clear();
    if (!single_channel)
      buffer.resize(volume(sample->shape), nvjpeg2k_cu_stream_);
    const auto &jpeg2k_stream = nvjpeg2k_streams_[sample->sample_idx];
    void *pixel_data[NVJPEG_MAX_COMPONENT] = {};
    size_t pitch_in_bytes[NVJPEG_MAX_COMPONENT] = {};
    auto i_out = single_channel ? output_data : buffer.data();
    for (uint32_t c = 0; c < sample->shape[2]; ++c) {
      pixel_data[c] = i_out + c * sample->shape[0] * sample->shape[1];
      pitch_in_bytes[c] = sample->shape[1];
    }
    nvjpeg2kImage_t output_image;
    output_image.pixel_data = pixel_data;
    output_image.pitch_in_bytes = pitch_in_bytes;
    output_image.pixel_type = NVJPEG2K_UINT8;
    output_image.num_components = sample->shape[2];
    auto ret = nvjpeg2kDecode(nvjpeg2k_handle_, nvjpeg2k_decoder_,
                              jpeg2k_stream, &output_image, nvjpeg2k_cu_stream_);
    if (ret == NVJPEG2K_STATUS_SUCCESS) {
      if (!single_channel) {
        auto comp_size = sample->shape[0] * sample->shape[1];
        PermuteToInterleaved(output_data, i_out, comp_size, sample->shape[2], nvjpeg2k_cu_stream_);
      }
      CUDA_CALL(cudaEventRecord(nvjpeg2k_decode_event_, nvjpeg2k_cu_stream_));
    } else if (ret == NVJPEG2K_STATUS_BAD_JPEG || ret == NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED) {
      HostFallback<StorageGPU>(input_data.data(), input_data.size(), output_image_type_,
                                output_data, nvjpeg2k_cu_stream_, sample->file_name,
                                sample->roi, use_fast_idct_);

    } else {
      NVJPEG2K_CALL_EX(ret, sample->file_name);
    }
  }
#endif  // NVJPEG2K_ENABLED

  void ProcessImagesJpeg2k(MixedWorkspace &ws) {
#if NVJPEG2K_ENABLED
    if (!nvjpeg2k_handle_) {
      return;
    }
    nvjpeg2k_thread_.AddWork([this, &ws](int) {
      auto &output = ws.OutputRef<GPUBackend>(0);
      auto &input = ws.InputRef<CPUBackend>(0);
      for (auto *sample : samples_jpeg2k_) {
        assert(sample);
        auto i = sample->sample_idx;
        auto output_data = output.mutable_tensor<uint8_t>(i);
        auto in = span<const uint8_t>(input[i].data<uint8_t>(), input[i].size());
        ImageCache::ImageShape shape = output_shape_[i].to_static<3>();
        DecodeJpeg2k(output_data, sample, in);
        CacheStore(sample->file_name, output_data, shape, nvjpeg2k_cu_stream_);
      }
    });
#endif  // NVJPEG2K_ENABLED
  }

  void ProcessImagesHost(MixedWorkspace &ws) {
    auto& output = ws.Output<GPUBackend>(0);
    for (auto *sample : samples_host_) {
      auto i = sample->sample_idx;
      auto *output_data = output.mutable_tensor<uint8_t>(i);
      const auto &in = ws.Input<CPUBackend>(0, i);
      ImageCache::ImageShape shape = output_shape_[i].to_static<3>();
      thread_pool_.AddWork(
        [this, sample, &in, output_data, shape](int tid) {
          HostFallback<StorageGPU>(in.data<uint8_t>(), in.size(), output_image_type_, output_data,
                                   streams_[tid], sample->file_name, sample->roi, use_fast_idct_);
          CacheStore(sample->file_name, output_data, shape, streams_[tid]);
        }, task_priority_seq_--);  // FIFO order, since the samples were already ordered
    }
  }

  void ProcessImagesHw(MixedWorkspace &ws) {
    auto& output = ws.Output<GPUBackend>(0);
    if (!samples_hw_batched_.empty()) {
      nvjpegJpegState_t &state = state_hw_batched_;
      assert(state != nullptr);
      int max_cpu_threads = 4;

      nvjpegOutputFormat_t format = GetFormat(output_image_type_);
      NVJPEG_CALL(nvjpegDecodeBatchedInitialize(handle_, state, samples_hw_batched_.size(),
                                                max_cpu_threads, format));

      in_data_.resize(samples_hw_batched_.size());
      in_lengths_.resize(samples_hw_batched_.size());
      nvjpeg_destinations_.resize(samples_hw_batched_.size());
      memset(nvjpeg_destinations_.data(), 0, nvjpeg_destinations_.size() * sizeof(nvjpegImage_t));

      int j = 0;
      TensorVector<CPUBackend> tv(samples_hw_batched_.size());

      for (auto *sample : samples_hw_batched_) {
        assert(!sample->roi);
        int i = sample->sample_idx;
        const auto &in = ws.Input<CPUBackend>(0, i);
        const auto &out_shape = output_shape_.tensor_shape(i);

        tv[j].ShareData(const_cast<Tensor<CPUBackend>*>(&in));
        in_lengths_[j] = in.size();
        nvjpeg_destinations_[j].channel[0] = output.mutable_tensor<uint8_t>(i);
        nvjpeg_destinations_[j].pitch[0] = out_shape[1] * out_shape[2];
        j++;
      }

      CUDA_CALL(cudaEventSynchronize(hw_decode_event_));
      // it is H2H copy so the stream doesn't metter much as we don't use cudaMemcpy but
      // maybe someday...
      hw_decoder_images_staging_.Copy(tv, hw_decode_stream_);
      for (size_t k = 0; k < samples_hw_batched_.size(); ++k) {
        in_data_[k] = hw_decoder_images_staging_.mutable_tensor<uint8_t>(k);
      }
      NVJPEG_CALL(nvjpegDecodeBatched(handle_, state, in_data_.data(), in_lengths_.data(),
                                      nvjpeg_destinations_.data(), hw_decode_stream_));
      for (auto *sample : samples_hw_batched_) {
        int i = sample->sample_idx;
        CacheStore(sample->file_name, output.mutable_tensor<uint8_t>(i),
                   output_shape_.tensor_shape(i).to_static<3>(), hw_decode_stream_);
      }
      CUDA_CALL(cudaEventRecord(hw_decode_event_, hw_decode_stream_));
    }
  }

  void ProcessImages(MixedWorkspace &ws) {
    auto &output = ws.Output<GPUBackend>(0);
    TypeInfo type = TypeTable::GetTypeInfoFromStatic<uint8_t>();
    output.set_type(type);
    assert(output_shape_.num_samples() ==
           ws.GetInputBatchSize(0));  // If fails: Incorrect number of samples in shape
    output.Resize(output_shape_);
    output.SetLayout("HWC");

    UpdateTestCounters(samples_hw_batched_.size(), samples_single_.size(), samples_host_.size(),
                       samples_jpeg2k_.size());

    // Reset the task priority. Subsequent tasks will use decreasing numbers to ensure the
    // expected order of execution.
    task_priority_seq_ = 0;
    ProcessImagesCache(ws);

    ProcessImagesCuda(ws);
    ProcessImagesHost(ws);
    ProcessImagesJpeg2k(ws);
    thread_pool_.RunAll(false);  // don't block
    nvjpeg2k_thread_.RunAll(false);

    ProcessImagesHw(ws);

    thread_pool_.WaitForWork();
    nvjpeg2k_thread_.WaitForWork();
    // wait for all work in workspace main stream
    for (int tid = 0; tid < num_threads_; tid++) {
      CUDA_CALL(cudaEventRecord(decode_events_[tid], streams_[tid]));
      CUDA_CALL(cudaStreamWaitEvent(ws.stream(), decode_events_[tid], 0));
    }
    CUDA_CALL(cudaEventRecord(hw_decode_event_, hw_decode_stream_));
    CUDA_CALL(cudaStreamWaitEvent(ws.stream(), hw_decode_event_, 0));
#if NVJPEG2K_ENABLED
    CUDA_CALL(cudaEventRecord(nvjpeg2k_decode_event_, nvjpeg2k_cu_stream_));
    CUDA_CALL(cudaStreamWaitEvent(ws.stream(), nvjpeg2k_decode_event_, 0));
#endif  // NVJPEG2K_ENABLED
  }

  inline int GetNextBufferIndex(int thread_id) {
    const int page = thread_page_ids_[thread_id];
    thread_page_ids_[thread_id] ^= 1;  // negate LSB
    return 2*thread_id + page;
  }

  // Per sample worker called in a thread of the thread pool.
  // It decodes the encoded image `input_data` (host mem) into `output_data` (device mem) with
  // nvJPEG. If nvJPEG can't handle the image, it falls back to CPU decoder implementation
  // with libjpeg.
  void SampleWorker(int sample_idx, string file_name, int in_size, int thread_id,
                    const uint8_t* input_data, uint8_t* output_data, cudaStream_t stream) {
    SampleData &data = sample_data_[sample_idx];
    assert(data.method != DecodeMethod::Host);

    const int buff_idx = GetNextBufferIndex(thread_id);
    const int jpeg_stream_idx = buff_idx;

    // At this point sample data should have a valid selected decoder
    auto &decoder = data.selected_decoder->decoder;
    assert(decoder != nullptr);

    auto &state = data.selected_decoder->state;
    assert(state != nullptr);

    NVJPEG_CALL(nvjpegStateAttachPinnedBuffer(state, pinned_buffers_[buff_idx]));

    nvjpegStatus_t ret = nvjpegJpegStreamParse(handle_, input_data, in_size, false, false,
                                               jpeg_streams_[jpeg_stream_idx]);

    // If nvjpegJpegStreamParse failed we can skip nvjpeg's host decode step and
    // rely on the host decoder fallback
    if (ret == NVJPEG_STATUS_SUCCESS) {
      ret = nvjpegDecodeJpegHost(handle_, decoder, state, data.params,
                                 jpeg_streams_[jpeg_stream_idx]);
    }

    // If image is somehow not supported try host decoder
    if (ret != NVJPEG_STATUS_SUCCESS) {
      data.method = DecodeMethod::Host;
      HostFallback<StorageGPU>(input_data, in_size, output_image_type_, output_data,
                               stream, file_name, data.roi, use_fast_idct_);
      if (ret != NVJPEG_STATUS_JPEG_NOT_SUPPORTED && ret != NVJPEG_STATUS_BAD_JPEG) {
         auto warning_msg = make_string("NVJPEG error \"", static_cast<int>(ret), "\" : ",
                            nvjpeg_parse_error_code(ret), " ", file_name);
         DALI_WARN(warning_msg);
      }
      return;
    }
    NVJPEG_CALL_EX(ret, file_name);

    assert(data.method == DecodeMethod::NvjpegCuda);
    if (data.method == DecodeMethod::NvjpegCuda) {
      const auto &out_shape = output_shape_.tensor_shape(sample_idx);
      nvjpegImage_t nvjpeg_image;
      nvjpeg_image.channel[0] = output_data;
      nvjpeg_image.pitch[0] = out_shape[1] * out_shape[2];

      CUDA_CALL(cudaEventSynchronize(decode_events_[thread_id]));
      NVJPEG_CALL_EX(nvjpegStateAttachDeviceBuffer(state, device_buffers_[thread_id]), file_name);

      NVJPEG_CALL_EX(nvjpegDecodeJpegTransferToDevice(handle_, decoder, state,
                                                      jpeg_streams_[jpeg_stream_idx], stream),
                     file_name);

      ret = nvjpegDecodeJpegDevice(handle_, decoder, state, &nvjpeg_image, stream);
      // if nvJPEG fails try HostDecoder
      if (ret != NVJPEG_STATUS_SUCCESS) {
        auto warning_msg = make_string("NVJPEG error \"", static_cast<int>(ret), "\" : ",
                           nvjpeg_parse_error_code(ret), " ", file_name);
        DALI_WARN(warning_msg);
        HostFallback<StorageGPU>(input_data, in_size, output_image_type_, output_data,
                                stream, file_name, data.roi, use_fast_idct_);
        return;
      }
      CUDA_CALL(cudaEventRecord(decode_events_[thread_id], stream));
    }
  }


  USE_OPERATOR_MEMBERS();
  nvjpegHandle_t handle_;

  // output colour format
  DALIImageType output_image_type_;

  unsigned int hybrid_huffman_threshold_;
  bool use_fast_idct_;

  TensorListShape<> output_shape_;

  // Per thread - double buffered
  std::vector<nvjpegBufferPinned_t> pinned_buffers_;
  std::vector<nvjpegJpegStream_t> jpeg_streams_;
  nvjpegJpegStream_t hw_decoder_jpeg_stream_;

#if NVJPEG2K_ENABLED
  // nvjpeg2k
  NvJPEG2KHandle nvjpeg2k_handle_{};
  NvJPEG2KDecodeState nvjpeg2k_decoder_{};
  DeviceBuffer<uint8_t> nvjpeg2k_intermediate_buffer_;
  cudaStream_t nvjpeg2k_cu_stream_;
  cudaEvent_t nvjpeg2k_decode_event_;
  nvjpeg2kDeviceAllocator_t nvjpeg2k_dev_alloc_;
  nvjpeg2kPinnedAllocator_t nvjpeg2k_pin_alloc_;

  // nvjpeg2k - per sample
  std::vector<NvJPEG2KStream> nvjpeg2k_streams_;
#endif  // NVJPEG2K_ENABLED

  // GPU
  // Per thread
  std::vector<nvjpegBufferDevice_t> device_buffers_;
  std::vector<cudaStream_t> streams_;
  cudaStream_t hw_decode_stream_;
  std::vector<cudaEvent_t> decode_events_;
  cudaEvent_t hw_decode_event_;
  std::vector<int> thread_page_ids_;  // page index for double-buffering

  int device_id_;

  bool using_hw_decoder_ = false;
  float hw_decoder_load_ = 0.0f;
  int hw_decoder_bs_ = 0;

  // Those are used to feed nvjpeg's batched API
  std::vector<const unsigned char*> in_data_;
  std::vector<size_t> in_lengths_;
  std::vector<nvjpegImage_t> nvjpeg_destinations_;

  // Allocators
  nvjpegDevAllocator_t device_allocator_;
  nvjpegPinnedAllocator_t pinned_allocator_;

  ThreadPool thread_pool_;
  ThreadPool nvjpeg2k_thread_;
  static constexpr int kOutputDim = 3;

  TensorList<CPUBackend> hw_decoder_images_staging_;

 private:
  void UpdateTestCounters(int nsamples_hw, int nsamples_cuda,
                          int nsamples_host, int nsamples_nvjpeg2k) {
    nsamples_hw_ += nsamples_hw;
    nsamples_cuda_ += nsamples_cuda;
    nsamples_host_ += nsamples_host;
    nsamples_nvjpeg2k_ += nsamples_nvjpeg2k;
  }

  /**
   * Registers counters, used only for unit-test reasons.
   */
  void RegisterTestCounters() {
    RegisterDiagnostic("nsamples_hw", &nsamples_hw_);
    RegisterDiagnostic("nsamples_cuda", &nsamples_cuda_);
    RegisterDiagnostic("nsamples_host", &nsamples_host_);
    RegisterDiagnostic("nsamples_nvjpeg2k", &nsamples_nvjpeg2k_);
    RegisterDiagnostic("using_hw_decoder", &using_hw_decoder_);
  }

  int CalcHwDecoderBatchSize(float hw_decoder_load, int curr_batch_size) {
    if (hw_decoder_load == 0.f) return 0;
    auto hw_batch_size = static_cast<int>(std::round(hw_decoder_load * curr_batch_size));

    constexpr int kNumHwDecoders = 5;
    int tail = hw_batch_size % kNumHwDecoders;
    if (tail > 0) {
      hw_batch_size = hw_batch_size + kNumHwDecoders - tail;
    }
    if (hw_batch_size > curr_batch_size) {
      hw_batch_size = curr_batch_size;
    }

    return hw_batch_size;
  }

  void ReserveSampleContainers() {
    sample_data_.reserve(max_batch_size_);
    for (int i = 0; i < max_batch_size_; i++) {
      sample_data_.emplace_back(i, handle_, output_image_type_);
    }
    samples_cache_.reserve(max_batch_size_);
    samples_host_.reserve(max_batch_size_);
    samples_hw_batched_.reserve(max_batch_size_);
    samples_single_.reserve(max_batch_size_);
#if NVJPEG2K_ENABLED
    samples_jpeg2k_.reserve(max_batch_size_);
#endif  // NVJPEG2K_ENABLED
  }

  // HW/CUDA Utilization test counters
  int64_t nsamples_hw_ = 0, nsamples_cuda_ = 0, nsamples_host_ = 0, nsamples_nvjpeg2k_ = 0;

  // Used to ensure the work in the thread pool is picked FIFO
  int64_t task_priority_seq_ = 0;
};

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_NVJPEG_NVJPEG_DECODER_DECOUPLED_API_H_
