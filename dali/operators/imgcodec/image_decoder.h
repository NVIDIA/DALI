// Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// limitations under the License.

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "dali/core/call_at_exit.h"
#include "dali/core/mm/memory.h"
#include "dali/operators.h"
#include "dali/operators/decoder/cache/cached_decoder_impl.h"
#include "dali/operators/generic/slice/slice_attr.h"
#include "dali/operators/image/crop/crop_attr.h"
#include "dali/operators/image/crop/random_crop_attr.h"
#include "dali/operators/imgcodec/util/convert.h"
#include "dali/operators/imgcodec/util/convert_gpu.h"
#include "dali/operators/imgcodec/util/convert_utils.h"
#include "dali/operators/imgcodec/util/nvimagecodec_types.h"
#include "dali/operators/imgcodec/util/output_shape.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

#if not(WITH_DYNAMIC_NVIMGCODEC_ENABLED)
nvimgcodecStatus_t get_libjpeg_turbo_extension_desc(nvimgcodecExtensionDesc_t* ext_desc);
nvimgcodecStatus_t get_libtiff_extension_desc(nvimgcodecExtensionDesc_t* ext_desc);
nvimgcodecStatus_t get_opencv_extension_desc(nvimgcodecExtensionDesc_t* ext_desc);
nvimgcodecStatus_t get_nvjpeg_extension_desc(nvimgcodecExtensionDesc_t* ext_desc);
nvimgcodecStatus_t get_nvjpeg2k_extension_desc(nvimgcodecExtensionDesc_t* ext_desc);
#endif


#ifndef DALI_OPERATORS_IMGCODEC_IMAGE_DECODER_H_
#define DALI_OPERATORS_IMGCODEC_IMAGE_DECODER_H_

namespace dali {
namespace imgcodec {

template <typename Backend>
struct OutBackend {
  using type = GPUBackend;
};

template <>
struct OutBackend<CPUBackend> {
  using type = CPUBackend;
};


constexpr uint32_t verbosity_to_severity(int verbose) {
  uint32_t result = 0;
  if (verbose >= 1)
    result |= NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_FATAL | NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_ERROR;
  if (verbose >= 2)
    result |= NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_WARNING;
  if (verbose >= 3)
    result |= NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_INFO;
  if (verbose >= 4)
    result |= NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEBUG;
  if (verbose >= 5)
    result |= NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_TRACE;
  return result;
}

static constexpr size_t kDevAlignment = 256;  // warp alignment for 32x64-bit
static constexpr size_t kHostAlignment = 64;  // cache alignment

inline int static_dali_device_malloc(void *ctx, void **ptr, size_t size, cudaStream_t stream) {
  auto *mr = static_cast<mm::device_async_resource *>(ctx);
  try {
    *ptr = mr->allocate_async(size, kDevAlignment, stream);
    return cudaSuccess;
  } catch (const std::bad_alloc &) {
    *ptr = nullptr;
    return cudaErrorMemoryAllocation;
  } catch (const CUDAError &e) {
    return e.is_rt_api() ? e.rt_error() : cudaErrorUnknown;
  } catch (...) {
    *ptr = nullptr;
    return cudaErrorUnknown;
  }
}

inline int static_dali_device_free(void *ctx, void *ptr, size_t size, cudaStream_t stream) {
  auto *mr = static_cast<mm::device_async_resource *>(ctx);
  mr->deallocate_async(ptr, size, kDevAlignment, stream);
  return cudaSuccess;
}

inline int static_dali_pinned_malloc(void *ctx, void **ptr, size_t size, cudaStream_t stream) {
  auto *mr = static_cast<mm::pinned_async_resource *>(ctx);
  try {
    *ptr = mr->allocate_async(size, kHostAlignment, stream);
    return cudaSuccess;
  } catch (const std::bad_alloc &) {
    *ptr = nullptr;
    return cudaErrorMemoryAllocation;
  } catch (const CUDAError &e) {
    return e.is_rt_api() ? e.rt_error() : cudaErrorUnknown;
  } catch (...) {
    *ptr = nullptr;
    return cudaErrorUnknown;
  }
}

inline int static_dali_pinned_free(void *ctx, void *ptr, size_t size, cudaStream_t stream) {
  auto *mr = static_cast<mm::pinned_async_resource *>(ctx);
  mr->deallocate_async(ptr, size, kHostAlignment, stream);
  return cudaSuccess;
}

inline void get_nvimgcodec_version(int* major, int *minor, int* patch) {
  nvimgcodecProperties_t properties{NVIMGCODEC_STRUCTURE_TYPE_PROPERTIES,
                                    sizeof(nvimgcodecProperties_t), 0};
  if (NVIMGCODEC_STATUS_SUCCESS != nvimgcodecGetProperties(&properties))
    throw std::runtime_error("Failed to check the version of nvimgcodec.");
  int version = static_cast<int>(properties.version);
  *major = NVIMGCODEC_MAJOR_FROM_SEMVER(version);
  *minor = NVIMGCODEC_MINOR_FROM_SEMVER(version);
  *patch = NVIMGCODEC_PATCH_FROM_SEMVER(version);
}

template <typename Backend>
class ImageDecoder : public StatelessOperator<Backend> {
 public:
  ~ImageDecoder() override {
#if not(WITH_DYNAMIC_NVIMGCODEC_ENABLED)
    decoder_.reset();  // first stop the decoder
    for (auto& extension : extensions_) {
      nvimgcodecExtensionDestroy(extension);
    }
#endif
  }

 protected:
  using Operator<Backend>::spec_;

  struct ParsedSample {
    NvImageCodecCodeStream encoded_stream = {};
    nvimgcodecImageInfo_t nvimgcodec_img_info = {};
    nvimgcodecJpegImageInfo_t nvimgcodec_jpeg_info = {};
    ImageInfo dali_img_info = {};
    DALIDataType orig_dtype = {};
  };

  struct SampleState {
    ParsedSample parsed_sample = {};
    NvImageCodecImage image = {};
    nvimgcodecImageInfo_t image_info = {};
    TensorShape<> out_shape = {};
    bool need_processing = true;

    TensorLayout req_layout;
    DALIImageType orig_img_type;
    DALIImageType req_img_type;
    float dyn_range_multiplier = 1.0f;

    mm::uptr<uint8_t> host_buf;
    mm::async_uptr<uint8_t> device_buf;

    SampleView<CPUBackend> out_cpu;
    SampleView<GPUBackend> out_gpu;

    SampleView<CPUBackend> decode_out_cpu;
    SampleView<GPUBackend> decode_out_gpu;
  };

  struct nvImagecodecOpts {
    template <typename T>
    void add_global_option(const std::string &key, const T &value) {
      opts_.emplace_back(":" + key, std::to_string(value));
    }

    template <typename T>
    void add_module_option(const std::string &module, const std::string &key, const T &value) {
      opts_.emplace_back(module + ":" + key, std::to_string(value));
    }

    std::string to_string() {
      std::stringstream ss;
      bool first = true;
      for (auto &[key, value] : opts_) {
        if (!first)
          ss << " ";
        else
          first = false;
        ss << key << "=" << value;
      }
      return ss.str();
    }

    std::vector<std::pair<std::string, std::string>> opts_;
  };
  nvImagecodecOpts opts_;


  explicit ImageDecoder(const OpSpec &spec) : StatelessOperator<Backend>(spec) {
    device_id_ = std::is_same<CPUBackend, Backend>::value ? CPU_ONLY_DEVICE_ID :
                                                            spec.GetArgument<int>("device_id");
    format_ = spec.GetArgument<DALIImageType>("output_type");
    dtype_ = spec.GetArgument<DALIDataType>("dtype");
    use_orientation_ = spec.GetArgument<bool>("adjust_orientation");
    max_batch_size_ = spec.GetArgument<int>("max_batch_size");
    num_threads_ = spec.GetArgument<int>("num_threads");
    GetDecoderSpecificArguments(spec);

    if (std::is_same<MixedBackend, Backend>::value) {
      thread_pool_ = std::make_unique<ThreadPool>(num_threads_, device_id_,
                                                  spec.GetArgument<bool>("affine"), "MixedDecoder");

      if (spec_.HasArgument("cache_size"))
        cache_ = std::make_unique<CachedDecoderImpl>(spec_);
    }

    EnforceMinimumNvimgcodecVersion();

    nvimgcodecDeviceAllocator_t *dev_alloc_ptr = nullptr;
    nvimgcodecPinnedAllocator_t *pinned_alloc_ptr = nullptr;
    if (device_id_ >= 0) {
      dev_alloc_.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DEVICE_ALLOCATOR;
      dev_alloc_.struct_size = sizeof(nvimgcodecDeviceAllocator_t);
      dev_alloc_.struct_next = nullptr;
      dev_alloc_.device_malloc = static_dali_device_malloc;
      dev_alloc_.device_free = static_dali_device_free;
      dev_alloc_.device_ctx = mm::GetDefaultResource<mm::memory_kind::device>();
      dev_alloc_.device_mem_padding = spec.GetArgument<int64_t>("device_memory_padding");
      dev_alloc_ptr = &dev_alloc_;

      pinned_alloc_.struct_type = NVIMGCODEC_STRUCTURE_TYPE_PINNED_ALLOCATOR;
      pinned_alloc_.struct_size = sizeof(nvimgcodecPinnedAllocator_t);
      pinned_alloc_.struct_next = nullptr;
      pinned_alloc_.pinned_malloc = static_dali_pinned_malloc;
      pinned_alloc_.pinned_free = static_dali_pinned_free;
      pinned_alloc_.pinned_ctx = mm::GetDefaultResource<mm::memory_kind::pinned>();
      pinned_alloc_.pinned_mem_padding = spec.GetArgument<int64_t>("host_memory_padding");
      pinned_alloc_ptr = &pinned_alloc_;
    }

    nvimgcodecInstanceCreateInfo_t instance_create_info{
        NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t),
        nullptr};

    const char *log_lvl_env = std::getenv("DALI_NVIMGCODEC_LOG_LEVEL");
    int log_lvl = log_lvl_env ? clamp(atoi(log_lvl_env), 1, 5): 2;

    instance_create_info.load_extension_modules = static_cast<int>(WITH_DYNAMIC_NVIMGCODEC_ENABLED);
    instance_create_info.load_builtin_modules = static_cast<int>(true);
    instance_create_info.extension_modules_path = nullptr;
    instance_create_info.create_debug_messenger = static_cast<int>(true);
    instance_create_info.debug_messenger_desc = nullptr;
    instance_create_info.message_severity = verbosity_to_severity(log_lvl);
    instance_create_info.message_category = NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL;

    instance_ = NvImageCodecInstance::Create(&instance_create_info);

    // If we link statically, we need to initialize the extensions manually
#if not(WITH_DYNAMIC_NVIMGCODEC_ENABLED)
    auto load_ext = [&](nvimgcodecExtensionModuleEntryFunc_t func) {
      extensions_descs_.push_back(
          {NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), nullptr});
      extensions_.emplace_back();
      func(&extensions_descs_.back());
      nvimgcodecExtensionCreate(instance_, &extensions_.back(), &extensions_descs_.back());
    };

    load_ext(get_opencv_extension_desc);

#if LIBJPEG_TURBO_ENABLED
    load_ext(get_libjpeg_turbo_extension_desc);
#endif

#if LIBTIFF_ENABLED
    load_ext(get_libtiff_extension_desc);
#endif

#if NVJPEG_ENABLED
    load_ext(get_nvjpeg_extension_desc);
#endif

#if NVJPEG2K_ENABLED
    load_ext(get_nvjpeg2k_extension_desc);
#endif

#endif

    std::stringstream opts_ss;
    float hw_load = 0.9f;

    std::vector<std::pair<std::string, std::string>> option_strs;
    for (auto &[key, value] : decoder_params_) {
      if (key == "use_fast_idct") {
        opts_.add_module_option("libjpeg_turbo_decoder", "fast_idct", std::any_cast<bool>(value));
      } else if (key == "hybrid_huffman_threshold") {
        opts_.add_module_option("nvjpeg_cuda_decoder", "hybrid_huffman_threshold",
                                std::any_cast<size_t>(value));
      } else if (key == "hw_decoder_load") {
        hw_load = std::any_cast<float>(value);
      } else if (key == "jpeg_fancy_upsampling") {
        // TODO(janton): Make fancy_upsampling default `true` and let the option control the CPU
        // decoder as well. For backward compatibility reasons, we don't set the fancy upsampling
        // flag for CPU decoders (old decoder always uses fancy_upsampling with CPU decoders)
        opts_.add_module_option("nvjpeg_cuda_decoder", "fancy_upsampling",
                                std::any_cast<bool>(value));
        opts_.add_module_option("nvjpeg_hw_decoder", "fancy_upsampling",
                                std::any_cast<bool>(value));
      } else if (key == "preallocate_width_hint") {
        opts_.add_module_option("nvjpeg_hw_decoder", "preallocate_width_hint",
                                std::max(1, std::any_cast<int>(value)));
      } else if (key == "preallocate_height_hint") {
        opts_.add_module_option("nvjpeg_hw_decoder", "preallocate_height_hint",
                                std::max(1, std::any_cast<int>(value)));
      } else {
        continue;
      }
    }

    // Batch size
    opts_.add_module_option("nvjpeg_hw_decoder", "preallocate_batch_size",
                            std::max(1, max_batch_size_));
    // Nvjpeg2k parallel tiles
    opts_.add_module_option("nvjpeg2k_cuda_decoder", "num_parallel_tiles", 16);

    int nvimgcodec_device_id =
        device_id_ == CPU_ONLY_DEVICE_ID ? NVIMGCODEC_DEVICE_CPU_ONLY : device_id_;

    exec_params_.device_id = nvimgcodec_device_id;
    backends_.clear();
    backends_.reserve(4);
    if (nvimgcodec_device_id != NVIMGCODEC_DEVICE_CPU_ONLY) {
      backends_.push_back(
          nvimgcodecBackend_t{NVIMGCODEC_STRUCTURE_TYPE_BACKEND,
                              sizeof(nvimgcodecBackend_t),
                              nullptr,
                              NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY,
                              {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS,
                               sizeof(nvimgcodecBackendParams_t), nullptr, hw_load}});
      backends_.push_back(nvimgcodecBackend_t{NVIMGCODEC_STRUCTURE_TYPE_BACKEND,
                                              sizeof(nvimgcodecBackend_t),
                                              nullptr,
                                              NVIMGCODEC_BACKEND_KIND_GPU_ONLY,
                                              {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS,
                                               sizeof(nvimgcodecBackendParams_t), nullptr, 1.0f}});
      backends_.push_back(nvimgcodecBackend_t{NVIMGCODEC_STRUCTURE_TYPE_BACKEND,
                                              sizeof(nvimgcodecBackend_t),
                                              nullptr,
                                              NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU,
                                              {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS,
                                               sizeof(nvimgcodecBackendParams_t), nullptr, 1.0f}});
    }
    backends_.push_back(nvimgcodecBackend_t{NVIMGCODEC_STRUCTURE_TYPE_BACKEND,
                                            sizeof(nvimgcodecBackend_t),
                                            nullptr,
                                            NVIMGCODEC_BACKEND_KIND_CPU_ONLY,
                                            {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS,
                                             sizeof(nvimgcodecBackendParams_t), nullptr, 1.0f}});

    // Forcing allocations, so that we have memory available in the pools when nvimgcodec requests
    // it. This should not be needed when nvjpegBufferPinnedResize/nvjpegBufferDeviceResize is
    // functional
    if (nvimgcodec_device_id != NVIMGCODEC_DEVICE_CPU_ONLY) {
      std::vector<mm::uptr<uint8_t>> tmp_buffs;
      for (int i = 0; i < num_threads_; i++) {
        tmp_buffs.push_back(mm::alloc_raw_unique<uint8_t, mm::memory_kind::device>(
            dev_alloc_ptr->device_mem_padding));
        if (!RestrictPinnedMemUsage()) {
          tmp_buffs.push_back(mm::alloc_raw_unique<uint8_t, mm::memory_kind::pinned>(
              pinned_alloc_ptr->pinned_mem_padding));
          tmp_buffs.push_back(mm::alloc_raw_unique<uint8_t, mm::memory_kind::pinned>(
              pinned_alloc_ptr->pinned_mem_padding));
        }
      }
      tmp_buffs.clear();  // return all memory to the pools
    }

    exec_params_.backends = backends_.data();
    exec_params_.num_backends = backends_.size();
    exec_params_.device_allocator = dev_alloc_ptr;
    exec_params_.pinned_allocator = pinned_alloc_ptr;
    exec_params_.executor = &executor_;
    exec_params_.max_num_cpu_threads = num_threads_;
    exec_params_.pre_init = 1;
    decoder_ = NvImageCodecDecoder::Create(instance_, &exec_params_, opts_.to_string());
  }

  nvimgcodecStatus_t launch(int device_id, int sample_idx, void *task_context,
                            void (*task)(int thread_id, int sample_idx, void *task_context)) {
    assert(tp_);
    tp_->AddWork([=](int tid) { task(tid, sample_idx, task_context); }, 0, true);
    return NVIMGCODEC_STATUS_SUCCESS;
  }

  int get_num_threads() const {
    // For the host backend, the thread pool is not available until we see the first batch.
    // this allows nvimgcodec accessing number of threads before the thread pool is available
    return num_threads_;
  }

  static nvimgcodecStatus_t static_launch(void *instance, int device_id, int sample_idx,
                                          void *task_context,
                                          void (*task)(int thread_id, int sample_idx,
                                                       void *task_context)) {
    auto *handle = static_cast<ImageDecoder<Backend> *>(instance);
    return handle->launch(device_id, sample_idx, task_context, task);
  }

  static int static_get_num_threads(void *instance) {
    auto *handle = static_cast<ImageDecoder<Backend> *>(instance);
    return handle->get_num_threads();
  }

  virtual void SetupRoiGenerator(const OpSpec &spec, const Workspace &ws) {}

  virtual ROI GetRoi(const OpSpec &spec, const Workspace &ws, std::size_t data_idx,
                     TensorShape<> shape) {
    return {};
  }

  template <typename T>
  void GetDecoderSpecificArgument(
      const OpSpec &spec, const std::string &name,
      std::function<bool(const T &)> validator = [](T x) { return true; }) {
    T value;
    if (spec.TryGetArgument(value, name)) {
      auto value = spec.GetArgument<T>(name);
      if (!validator(value)) {
        DALI_FAIL(make_string("Invalid value for decoder-specific parameter ", name));
      }
      decoder_params_[name] = value;
    }
  }

  void GetDecoderSpecificArguments(const OpSpec &spec) {
    GetDecoderSpecificArgument<uint64_t>(spec, "hybrid_huffman_threshold");
    GetDecoderSpecificArgument<int>(spec, "device_memory_padding");
    GetDecoderSpecificArgument<int>(spec, "host_memory_padding");
    GetDecoderSpecificArgument<int>(spec, "device_memory_padding_jpeg2k");
    GetDecoderSpecificArgument<int>(spec, "host_memory_padding_jpeg2k");
    GetDecoderSpecificArgument<float>(spec, "hw_decoder_load");
    GetDecoderSpecificArgument<int>(spec, "preallocate_width_hint");
    GetDecoderSpecificArgument<int>(spec, "preallocate_height_hint");
    GetDecoderSpecificArgument<bool>(spec, "use_fast_idct");
    GetDecoderSpecificArgument<bool>(spec, "jpeg_fancy_upsampling");
    GetDecoderSpecificArgument<int>(spec, "num_threads");
    // Make sure we set the default that DALI expects
    if (decoder_params_.count("jpeg_fancy_upsampling") == 0)
      decoder_params_["jpeg_fancy_upsampling"] = false;
  }

  bool CanInferOutputs() const override {
    return true;
  }

  void ParseSample(ParsedSample &parsed_sample, span<const uint8_t> encoded) {
    parsed_sample.encoded_stream =
        NvImageCodecCodeStream::FromHostMem(instance_, encoded.data(), encoded.size());
    parsed_sample.nvimgcodec_jpeg_info = nvimgcodecJpegImageInfo_t{
        NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), nullptr};
    parsed_sample.nvimgcodec_img_info =
        nvimgcodecImageInfo_t{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t),
                              static_cast<void *>(&parsed_sample.nvimgcodec_jpeg_info)};

    CHECK_NVIMGCODEC(nvimgcodecCodeStreamGetImageInfo(parsed_sample.encoded_stream,
                                                      &parsed_sample.nvimgcodec_img_info));

    // nvjpeg lossless backend (used by nvimgcodec) can only decode to uint16 (no uint8)
    if (parsed_sample.nvimgcodec_jpeg_info.encoding == NVIMGCODEC_JPEG_ENCODING_LOSSLESS_HUFFMAN) {
      for (uint32_t p = 0; p < parsed_sample.nvimgcodec_img_info.num_planes; p++) {
        parsed_sample.nvimgcodec_img_info.plane_info[p].sample_type =
            NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
      }
    }

    auto &info = parsed_sample.dali_img_info = to_dali_img_info(parsed_sample.nvimgcodec_img_info);

    auto sample_type = parsed_sample.nvimgcodec_img_info.plane_info[0].sample_type;
    parsed_sample.orig_dtype = to_dali_dtype(sample_type);
    if (parsed_sample.orig_dtype == DALI_NO_TYPE)
      throw std::runtime_error(make_string("Invalid sample_type: ", sample_type));
  }

  ThreadPool *GetThreadPool(const Workspace &ws) {
    return std::is_same<MixedBackend, Backend>::value ? thread_pool_.get() : &ws.GetThreadPool();
  }


  bool SetupImpl(std::vector<OutputDesc> &output_descs, const Workspace &ws) override {
    DomainTimeRange tr("Setup", DomainTimeRange::kOrange);
    tp_ = GetThreadPool(ws);
    assert(tp_ != nullptr);
    auto auto_cleanup = AtScopeExit([&] {
      tp_ = nullptr;
    });

    output_descs.resize(1);
    auto &input = ws.template Input<CPUBackend>(0);
    int nsamples = input.num_samples();

    SetupRoiGenerator(spec_, ws);
    TensorListShape<> shapes;
    shapes.resize(nsamples, 3);
    while (static_cast<int>(state_.size()) < nsamples)
      state_.push_back(std::make_unique<SampleState>());
    rois_.resize(nsamples);
    bool is_planar_layout = false;


    const bool use_cache = cache_ && cache_->IsCacheEnabled() && dtype_ == DALI_UINT8;
    auto get_task = [&](int block_idx, int nblocks) {
      return [&, block_idx, nblocks](int tid) {
        int i_start = nsamples * block_idx / nblocks;
        int i_end = nsamples * (block_idx + 1) / nblocks;
        for (int i = i_start; i < i_end; i++) {
          auto *st = state_[i].get();
          assert(st != nullptr);
          const auto &input_sample = input[i];

          auto src_info = input.GetMeta(i).GetSourceInfo();
          if (use_cache && cache_->IsInCache(src_info)) {
            auto cached_shape = cache_->CacheImageShape(src_info);
            auto roi = GetRoi(spec_, ws, i, cached_shape);
            if (!roi.use_roi()) {
              shapes.set_tensor_shape(i, cached_shape);
              continue;
            }
          }
          ParseSample(st->parsed_sample,
                      span<const uint8_t>{static_cast<const uint8_t *>(input_sample.raw_data()),
                                          volume(input_sample.shape())});
          ROI &roi = rois_[i] = GetRoi(spec_, ws, i, st->parsed_sample.dali_img_info.shape);
          OutputShape(st->out_shape, st->parsed_sample.dali_img_info, format_,
                      is_planar_layout, use_orientation_, roi);
          shapes.set_tensor_shape(i, st->out_shape);
        }
      };
    };

    int nblocks = tp_->NumThreads() + 1;
    if (nsamples > nblocks * 4) {
      int block_idx = 0;
      for (; block_idx < tp_->NumThreads(); block_idx++) {
        tp_->AddWork(get_task(block_idx, nblocks), -block_idx);
      }
      tp_->RunAll(false);                // start work but not wait
      get_task(block_idx, nblocks)(-1);  // run last block
      tp_->WaitForWork();                // wait for the other threads
    } else {                             // not worth parallelizing
      get_task(0, 1)(-1);                // run all in current thread
    }

    output_descs[0] = {std::move(shapes), dtype_};
    return true;
  }

  /**
   * @brief nvImageCodec up to 0.2 doesn't synchronize with the user stream before decoding.
   * Because of that, we need to host synchronize before passing the async allocated buffer
   * to the decoding function
   */
  bool need_host_sync_alloc() {
    static bool need_sync = [] {
      int major, minor, patch;
      get_nvimgcodec_version(&major, &minor, &patch);
      return major == 0 && minor <= 2;
    }();
    return need_sync;
  }

  template <typename OutBackend>
  void PrepareOutput(SampleState &st, SampleView<OutBackend> out, const ROI &roi,
                     const Workspace &ws) {
    // Make a copy of the parsed img info. We might modify it
    // (for example, request planar vs. interleaved, etc)
    st.image_info = st.parsed_sample.nvimgcodec_img_info;
    st.req_layout = "HWC";
    st.req_img_type = format_;
    auto info = st.parsed_sample.dali_img_info;
    int64_t &nchannels = st.out_shape[2];

    // Decode to format
    st.image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
    if (format_ == DALI_ANY_DATA) {
      st.image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED;
      st.orig_img_type = DALI_ANY_DATA;
    } else if (nchannels == 1 || format_ == DALI_GRAY) {
      st.image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_Y;
      st.orig_img_type = DALI_GRAY;
      nchannels = 1;
    } else {
      st.image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
      st.orig_img_type = DALI_RGB;
      nchannels = 3;
    }

    st.image_info.cuda_stream = std::is_same<MixedBackend, Backend>::value ? ws.stream() : nullptr;

    st.image_info.region.ndim = roi.use_roi() ? 2 : 0;
    if (roi.use_roi()) {
      st.image_info.region.start[0] = roi.begin[0];
      st.image_info.region.start[1] = roi.begin[1];
      st.image_info.region.end[0] = roi.end[0];
      st.image_info.region.end[1] = roi.end[1];
    }

    // At the moment we are not dealing with floating point outputs in nvimagecodec
    assert(IsIntegral(st.parsed_sample.orig_dtype));

    int precision = st.image_info.plane_info[0].precision;
    if (precision == 0)
      precision = PositiveBits(st.parsed_sample.orig_dtype);
    bool need_dynamic_range_scaling =
        NeedDynamicRangeScaling(precision, st.parsed_sample.orig_dtype);
    st.dyn_range_multiplier = need_dynamic_range_scaling ?
                                  DynamicRangeMultiplier(precision, st.parsed_sample.orig_dtype) :
                                  1.0f;

    st.need_processing = st.req_img_type != st.orig_img_type ||
                         dtype_ != st.parsed_sample.orig_dtype || need_dynamic_range_scaling;

    st.image_info.buffer_kind = std::is_same<MixedBackend, Backend>::value ?
                                    NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE :
                                    NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
    int64_t image_buffer_size =
        volume(st.out_shape) * TypeTable::GetTypeInfo(st.parsed_sample.orig_dtype).size();
    st.image_info.buffer_size = image_buffer_size;

    st.decode_out_cpu = {};
    st.decode_out_gpu = {};

    if (st.need_processing) {
      if constexpr (std::is_same<MixedBackend, Backend>::value) {
        st.device_buf = mm::alloc_raw_async_unique<uint8_t, mm::memory_kind::device>(
            image_buffer_size, st.image_info.cuda_stream, st.image_info.cuda_stream);
        st.image_info.buffer = st.device_buf.get();
        st.decode_out_gpu = {st.image_info.buffer, st.out_shape, st.parsed_sample.orig_dtype};
      } else {
        st.host_buf = mm::alloc_raw_unique<uint8_t, mm::memory_kind::host>(image_buffer_size);
        st.image_info.buffer = st.host_buf.get();
        st.decode_out_cpu = {st.image_info.buffer, st.out_shape, st.parsed_sample.orig_dtype};
      }
    } else {
      st.image_info.buffer = static_cast<uint8_t *>(out.raw_mutable_data());
    }

    st.image_info.num_planes = 1;
    st.image_info.plane_info[0].row_stride =
        st.out_shape[1] * nchannels * TypeTable::GetTypeInfo(st.parsed_sample.orig_dtype).size();
    st.image_info.plane_info[0].height = st.out_shape[0];
    st.image_info.plane_info[0].width = st.out_shape[1];
    st.image_info.plane_info[0].num_channels = nchannels;
    st.image = NvImageCodecImage::Create(instance_, &st.image_info);
  }

  void RunImplImpl(Workspace &ws) {
    const auto &input = ws.Input<CPUBackend>(0);
    auto &output = ws.template Output<typename OutBackend<Backend>::type>(0);
    output.SetLayout("HWC");
    int nsamples = input.num_samples();
    assert(output.num_samples() == nsamples);

    tp_ = GetThreadPool(ws);
    assert(tp_ != nullptr);
    auto auto_cleanup = AtScopeExit([&] {
      tp_ = nullptr;
    });

    bool has_any_roi = false;
    for (auto &roi : rois_)
      has_any_roi |= roi.use_roi();

    nvimgcodecDecodeParams_t decode_params = {NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS,
                                              sizeof(nvimgcodecDecodeParams_t), nullptr};
    decode_params.apply_exif_orientation = static_cast<int>(use_orientation_);
    decode_params.enable_roi = static_cast<int>(has_any_roi);

    assert(static_cast<int>(state_.size()) == nsamples);
    batch_encoded_streams_.clear();
    batch_encoded_streams_.reserve(nsamples);
    batch_images_.clear();
    batch_images_.reserve(nsamples);
    decode_sample_idxs_.clear();
    decode_sample_idxs_.reserve(nsamples);

    // TODO(janton): consider extending cache to different dtype as well
    const bool use_cache = cache_ && cache_->IsCacheEnabled() && dtype_ == DALI_UINT8;
    if (use_cache) {
      int samples_to_load = 0;
      DomainTimeRange tr(make_string("CacheLoad"), DomainTimeRange::kOrange);
      for (int orig_idx = 0; orig_idx < nsamples; orig_idx++) {
        auto src_info = input.GetMeta(orig_idx).GetSourceInfo();
        // To simplify things, we do not allow caching ROIs
        bool has_roi = rois_[orig_idx].use_roi();
        if (cache_->IsInCache(src_info) && !has_roi) {
          cache_->DeferCacheLoad(src_info, output.template mutable_tensor<uint8_t>(orig_idx));
          samples_to_load++;
        } else {
          decode_sample_idxs_.push_back(orig_idx);
        }
      }
      if (samples_to_load > 0)
        cache_->LoadDeferred(ws.stream());
    } else {
      decode_sample_idxs_.resize(nsamples);
      std::iota(decode_sample_idxs_.begin(), decode_sample_idxs_.end(), 0);
    }

    int decode_nsamples = decode_sample_idxs_.size();
    {
      DomainTimeRange tr(make_string("Prepare descs"), DomainTimeRange::kOrange);
      auto get_task = [&](int block_idx, int nblocks) {
        return [&, block_idx, nblocks](int tid) {
          int i_start = decode_nsamples * block_idx / nblocks;
          int i_end = decode_nsamples * (block_idx + 1) / nblocks;
          for (int i = i_start; i < i_end; i++) {
            int orig_idx = decode_sample_idxs_[i];
            PrepareOutput(*state_[orig_idx], output[orig_idx], rois_[orig_idx], ws);
          }
        };
      };

      int nblocks = tp_->NumThreads() + 1;
      if (decode_nsamples > nblocks * 4) {
        int block_idx = 0;
        for (; block_idx < tp_->NumThreads(); block_idx++) {
          tp_->AddWork(get_task(block_idx, nblocks), -block_idx);
        }
        tp_->RunAll(false);                 // start work but not wait
        get_task(block_idx, nblocks)(-1);  // run last block
        tp_->WaitForWork();                 // wait for the other threads
      } else {                             // not worth parallelizing
        get_task(0, 1)(-1);                // run all in current thread
      }

      for (int orig_idx : decode_sample_idxs_) {
        auto &st = *state_[orig_idx];
        batch_encoded_streams_.push_back(st.parsed_sample.encoded_stream);
        batch_images_.push_back(st.image);
      }
    }

    // This is a workaround for nvImageCodec <= 0.2
    auto any_need_processing = [&]() {
      for (auto &st : state_) {
        assert(ws.stream() == st->image_info.cuda_stream);  // assuming this is true
        if (st->need_processing)
          return true;
      }
      return false;
    };
    if (ws.has_stream() && need_host_sync_alloc() && any_need_processing()) {
      DomainTimeRange tr("alloc sync", DomainTimeRange::kOrange);
      CUDA_CALL(cudaStreamSynchronize(ws.stream()));
    }

    {
      DomainTimeRange tr("Decode", DomainTimeRange::kOrange);
      nvimgcodecFuture_t future;
      decode_status_.resize(decode_nsamples);
      size_t status_size = 0;
      CHECK_NVIMGCODEC(nvimgcodecDecoderDecode(decoder_, batch_encoded_streams_.data(),
                                               batch_images_.data(), decode_nsamples,
                                               &decode_params, &future));
      CHECK_NVIMGCODEC(
          nvimgcodecFutureGetProcessingStatus(future, decode_status_.data(), &status_size));
      if (static_cast<int>(status_size) != decode_nsamples)
        throw std::logic_error("Failed to retrieve processing status");
      CHECK_NVIMGCODEC(nvimgcodecFutureDestroy(future));

      for (int i = 0; i < decode_nsamples; i++) {
        if (decode_status_[i] != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
          int orig_idx = decode_sample_idxs_[i];
          throw std::runtime_error(make_string("Failed to decode sample #", orig_idx, " : ",
                                               input.GetMeta(orig_idx).GetSourceInfo()));
        }
      }
    }

    for (int orig_idx : decode_sample_idxs_) {
      auto st_ptr = state_[orig_idx].get();
      if (st_ptr->need_processing) {
        tp_->AddWork(
            [&, out = output[orig_idx], st_ptr, orig_idx](int tid) {
              DomainTimeRange tr(make_string("Convert #", orig_idx), DomainTimeRange::kOrange);
              auto &st = *st_ptr;
              if constexpr (std::is_same<MixedBackend, Backend>::value) {
                ConvertGPU(out, st.req_layout, st.req_img_type, st.decode_out_gpu, st.req_layout,
                           st.orig_img_type, ws.stream(), ROI{}, nvimgcodecOrientation_t{},
                           st.dyn_range_multiplier);
                st.device_buf.reset();
              } else {
                assert(st.dyn_range_multiplier == 1.0f);  // TODO(janton): enable
                ConvertCPU(out, st.req_layout, st.req_img_type, st.decode_out_cpu, st.req_layout,
                           st.orig_img_type, ROI{}, nvimgcodecOrientation_t{});
                st.host_buf.reset();
              }
            },
            -orig_idx);
      }
    }
    tp_->RunAll();

    if (use_cache) {
      DomainTimeRange tr(make_string("CacheStore"), DomainTimeRange::kOrange);
      for (int orig_idx : decode_sample_idxs_) {
        // We don't store ROIs
        bool has_roi = rois_[orig_idx].use_roi();
        if (has_roi)
          continue;
        // We only store RGB HWC data. The cache doesn't have format information, and we could have
        // another decoder with different configuration
        auto &st = *state_[orig_idx];
        if (st.req_img_type != DALI_RGB || st.req_layout != "HWC")
          continue;
        auto src_info = input.GetMeta(orig_idx).GetSourceInfo();
        auto *out_data = output.template mutable_tensor<uint8_t>(orig_idx);
        const auto &out_shape = output.tensor_shape(orig_idx);
        cache_->CacheStore(src_info, out_data, out_shape, ws.stream());
      }
    }
  }

  std::unique_ptr<ThreadPool> thread_pool_;
  std::unique_ptr<CachedDecoderImpl> cache_;

  NvImageCodecInstance instance_ = {};
  NvImageCodecDecoder decoder_ = {};

  nvimgcodecExecutorDesc_t executor_{NVIMGCODEC_STRUCTURE_TYPE_EXECUTOR_DESC,
                                     sizeof(nvimgcodecExecutorDesc_t),
                                     nullptr,
                                     this,
                                     &static_launch,
                                     &static_get_num_threads};
  nvimgcodecDeviceAllocator_t dev_alloc_ = {};
  nvimgcodecPinnedAllocator_t pinned_alloc_ = {};
  std::vector<nvimgcodecBackend_t> backends_;
  nvimgcodecExecutionParams_t exec_params_{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS,
                                           sizeof(nvimgcodecExecutionParams_t), nullptr};

  std::map<std::string, std::any> decoder_params_;
  std::vector<ROI> rois_;
  int device_id_;
  DALIImageType format_;
  DALIDataType dtype_;
  TensorLayout layout_ = "HWC";
  bool use_orientation_ = true;
  int max_batch_size_ = 1;
  int num_threads_ = -1;
  ThreadPool* tp_ = nullptr;
  std::vector<std::unique_ptr<SampleState>> state_;
  std::vector<nvimgcodecCodeStream_t> batch_encoded_streams_;
  std::vector<nvimgcodecImage_t> batch_images_;
  std::vector<nvimgcodecProcessingStatus_t> decode_status_;

  // In case of cache, the batch we send to the decoder might have fewer samples than the full batch
  // This vector is used to get the original index of the decoded samples
  std::vector<size_t> decode_sample_idxs_;

  // Manually loaded extensions
  std::vector<nvimgcodecExtensionDesc_t> extensions_descs_;
  std::vector<nvimgcodecExtension_t> extensions_;
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_OPERATORS_IMGCODEC_IMAGE_DECODER_H_
