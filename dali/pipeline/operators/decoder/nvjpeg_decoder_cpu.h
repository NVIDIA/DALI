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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_CPU_H_
#define DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_CPU_H_

#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dali/pipeline/operators/decoder/nvjpeg_helper.h"

#include "dali/image/image_factory.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/util/image.h"
#include "dali/util/ocv.h"

namespace dali {

namespace mem {
// TODO(spanev): move in separate file
class BasicPinnedAllocator {
 public:
  static void PreallocateBuffers(size_t element_size_hint, size_t n_elements_hint) {
    std::lock_guard<std::mutex> l(m_);

    if (element_size_hint_ == 0) {
      element_size_hint_ = element_size_hint;
    } else if (element_size_hint_ != element_size_hint) {
      DALI_FAIL("All instances of nvJPEGDecoder should have the same host_memory_padding.");
    }
    free_buffers_pool_.reserve(free_buffers_pool_.size() + n_elements_hint);
    for (size_t i = 0; i < n_elements_hint; ++i) {
      void* buffer;
      CUDA_CALL(cudaHostAlloc(&buffer, element_size_hint, 0));
      free_buffers_pool_.push_back(buffer);
    }
  }

  static void FreeBuffers(size_t n_elements_hint) {
    std::lock_guard<std::mutex> l(m_);
    for (size_t i = 0; i < n_elements_hint; ++i) {
      // Will call std::terminate
      DALI_ENFORCE(free_buffers_pool_.size() > 0, "Some buffers were not returned by nvJPEG.");
      CUDA_CALL(cudaFreeHost(free_buffers_pool_.back()));
      free_buffers_pool_.pop_back();
    }
  }

  static int Alloc(void** ptr, size_t size, unsigned int flags) {
    std::lock_guard<std::mutex> l(m_);
    // Non managed buffer
    if (size > element_size_hint_ || free_buffers_pool_.empty()) {
      return cudaHostAlloc(ptr, size, flags) == cudaSuccess ? 0 : 1;
    }

    // Managed buffer, adding it to the allocated set
    void* buffer = free_buffers_pool_.back();
    free_buffers_pool_.pop_back();
    *ptr = buffer;
    allocated_buffers_.insert(buffer);
    return 0;
  }

  static int Free(void* ptr) {
    std::lock_guard<std::mutex> l(m_);
    if (allocated_buffers_.find(ptr) == allocated_buffers_.end()) {
      // Non managed buffer... just free
      return cudaFreeHost(ptr) == cudaSuccess ? 0 : 1;
    }
    // Managed buffer
    allocated_buffers_.erase(ptr);
    free_buffers_pool_.push_back(ptr);
    return 0;
  }

 private:
  static std::vector<void*> free_buffers_pool_;
  static size_t element_size_hint_;
  static std::unordered_set<void*> allocated_buffers_;

  static std::mutex m_;
};

class ChunkPinnedAllocator {
 public:
  static void PreallocateBuffers(size_t element_size_hint, size_t n_elements_hint) {
    std::lock_guard<std::mutex> l(m_);

    counter_++;

    if (element_size_hint_ == 0) {
      element_size_hint_ = element_size_hint;
    } else if (element_size_hint_ != element_size_hint) {
      DALI_FAIL("All instances of nvJPEGDecoder should have the same host_memory_padding.");
    }

    Chunk chunk;
    CUDA_CALL(cudaHostAlloc(&chunk.memory, element_size_hint * n_elements_hint, 0));
    chunk.free_metadata.resize(n_elements_hint, true);
    chunk.first_free = 0;
    chunk.size = n_elements_hint;
    chunks_.push_back(chunk);
  }

  static void FreeBuffers(size_t n_elements_hint) {
    std::lock_guard<std::mutex> l(m_);
    counter_--;
    if (counter_ == 0) {
      for (auto chunk : chunks_) {
        // Will call std::terminate if CUDA_CALL fails
        CUDA_CALL(cudaFreeHost(chunk.memory));
      }
      chunks_.clear();
    }
  }

  static int Alloc(void** ptr, size_t size, unsigned int flags) {
    std::lock_guard<std::mutex> l(m_);
    // Non managed buffer
    if (size > element_size_hint_) {
      return cudaHostAlloc(ptr, size, flags) == cudaSuccess ? 0 : 1;
    }

    for (size_t c_idx = 0; c_idx < chunks_.size(); ++c_idx) {
      auto& chunk = chunks_[c_idx];
      size_t first_free = chunk.first_free;
      // This chunk is full
      if (chunk.first_free == chunk.size)
        continue;

      *ptr = static_cast<void*>(
              static_cast<uint8_t*>(chunk.memory) + (element_size_hint_ * first_free));
      allocated_buffers_[*ptr] = std::make_pair(c_idx, first_free);

      // Update free metadata and find next first_free
      chunk.free_metadata[first_free] = false;
      while (first_free < chunk.size && !chunk.free_metadata[first_free]) {
        first_free++;
      }
      chunk.first_free = first_free;
      return 0;
    }

    return cudaHostAlloc(ptr, size, flags) == cudaSuccess ? 0 : 1;
  }

  static int Free(void* ptr) {
    std::lock_guard<std::mutex> l(m_);
    auto it = allocated_buffers_.find(ptr);
    if (it == allocated_buffers_.end()) {
      // Non managed buffer... just free
      return cudaFreeHost(ptr) == cudaSuccess ? 0 : 1;
    }
    size_t c_idx, buff_idx;
    std::tie(c_idx, buff_idx) = it->second;
    auto& chunk = chunks_[c_idx];
    chunk.free_metadata[buff_idx] = true;
    if (chunk.first_free > buff_idx) {
      chunk.first_free = buff_idx;
    }
    allocated_buffers_.erase(it);
    return 0;
  }

 private:
  struct Chunk {
    void* memory;
    std::vector<int> free_metadata;
    size_t first_free;
    size_t size;
  };
  static std::vector<Chunk> chunks_;
  static size_t element_size_hint_;
  // ptr to chunk-offset pos
  static std::unordered_map<void*, std::pair<size_t, size_t>> allocated_buffers_;

  static int counter_;
  static std::mutex m_;
};
}  // namespace mem

using ImageInfo = EncodedImageInfo<unsigned int>;

using PinnedAllocator = mem::ChunkPinnedAllocator;

class nvJPEGDecoderCPUStage : public Operator<CPUBackend> {
 public:
  explicit nvJPEGDecoderCPUStage(const OpSpec& spec) :
    Operator<CPUBackend>(spec),
    output_image_type_(spec.GetArgument<DALIImageType>("output_type")),
    hybrid_huffman_threshold_(spec.GetArgument<unsigned int>("hybrid_huffman_threshold")),
    decode_params_(batch_size_) {
    NVJPEG_CALL(nvjpegCreateSimple(&handle_));

    // Do we really need both in both stages ops?
    size_t device_memory_padding = spec.GetArgument<Index>("device_memory_padding");
    size_t host_memory_padding = spec.GetArgument<Index>("host_memory_padding");
    NVJPEG_CALL(nvjpegSetDeviceMemoryPadding(device_memory_padding, handle_));
    NVJPEG_CALL(nvjpegSetPinnedMemoryPadding(host_memory_padding, handle_));

    NVJPEG_CALL(nvjpegDecoderCreate(handle_, NVJPEG_BACKEND_HYBRID, &decoder_host_));
    NVJPEG_CALL(nvjpegDecoderCreate(handle_, NVJPEG_BACKEND_GPU_HYBRID, &decoder_hybrid_));

    for (int i = 0; i < batch_size_; i++) {
      NVJPEG_CALL(nvjpegDecodeParamsCreate(handle_, &decode_params_[i]));
      NVJPEG_CALL(nvjpegDecodeParamsSetOutputFormat(decode_params_[i],
                                                    GetFormat(output_image_type_)));
      NVJPEG_CALL(nvjpegDecodeParamsSetAllowCMYK(decode_params_[i], true));
    }

    int nbuffers = spec.GetArgument<int>("cpu_prefetch_queue_depth") * batch_size_;
    PinnedAllocator::PreallocateBuffers(host_memory_padding, nbuffers);
    pinned_allocator_.pinned_malloc = &PinnedAllocator::Alloc;
    pinned_allocator_.pinned_free = &PinnedAllocator::Free;
  }

  virtual ~nvJPEGDecoderCPUStage() noexcept(false) {
    NVJPEG_CALL(nvjpegDecoderDestroy(decoder_host_));
    NVJPEG_CALL(nvjpegDecoderDestroy(decoder_hybrid_));
    NVJPEG_CALL(nvjpegDestroy(handle_));
    int nbuffers = spec_.GetArgument<int>("cpu_prefetch_queue_depth") * batch_size_;
    PinnedAllocator::FreeBuffers(nbuffers);
  }

  void RunImpl(SampleWorkspace *ws, const int idx) {
    const int data_idx = ws->data_idx();
    const auto& in = ws->Input<CPUBackend>(0);
    const auto *input_data = in.data<uint8_t>();
    const auto in_size = in.size();
    const auto file_name = in.GetSourceInfo();

    // We init and store the images and nvJPEG state in the CPU operator outputs
    // Since this CPU operator is instatiated on the GPU/Mixed boundary, it allows us to use
    // the Executor CPU output multiple-buffering

    // Output #0 contains general infomation about the sample
    // Output #1 contains nvJPEG state, buffer and information
    // Output #2 optionally contains OpenCV if nvJPEG cannot decode the sample

    ImageInfo* info;
    StateNvJPEG* state_nvjpeg;
    std::tie(info, state_nvjpeg) = InitAndGet(ws->Output<CPUBackend>(0),
                                              ws->Output<CPUBackend>(1));

    ws->Output<CPUBackend>(0).SetSourceInfo(file_name);

    nvjpegStatus_t ret = nvjpegJpegStreamParse(handle_,
                                                static_cast<const unsigned char*>(input_data),
                                                in_size,
                                                false,
                                                false,
                                                state_nvjpeg->jpeg_stream);
    info->nvjpeg_support = ret == NVJPEG_STATUS_SUCCESS;
    auto crop_generator = GetCropWindowGenerator(data_idx);
    if (!info->nvjpeg_support) {
      try {
        const auto image = ImageFactory::CreateImage(static_cast<const uint8 *>(input_data),
                                                     in_size);
        const auto dims = image->GetImageDims();
        info->heights[0] = std::get<0>(dims);
        info->widths[0] = std::get<1>(dims);
        if (crop_generator) {
          info->crop_window = crop_generator(info->heights[0], info->widths[0]);
          DALI_ENFORCE(info->crop_window.IsInRange(info->heights[0], info->widths[0]));
          info->widths[0] = info->crop_window.w;
          info->heights[0] = info->crop_window.h;
        }
        auto& out = ws->Output<CPUBackend>(2);
        out.set_type(TypeInfo::Create<uint8_t>());
        const auto c = static_cast<Index>(NumberOfChannels(output_image_type_));
        out.Resize({info->heights[0], info->widths[0], c});
        auto *output_data = out.mutable_data<uint8_t>();

        HostFallback<kernels::StorageCPU>(input_data, in_size, output_image_type_, output_data, 0,
                                          file_name, info->crop_window);
      } catch (const std::runtime_error& e) {
        DALI_FAIL(e.what() + "File: " + file_name);
      }
    } else {
      NVJPEG_CALL(nvjpegJpegStreamGetFrameDimensions(state_nvjpeg->jpeg_stream,
                                                     info->widths,
                                                     info->heights));
      NVJPEG_CALL(nvjpegJpegStreamGetComponentsNum(state_nvjpeg->jpeg_stream,
                                                   &info->c));
      state_nvjpeg->nvjpeg_backend =
                    ShouldUseHybridHuffman(*info, input_data, in_size, hybrid_huffman_threshold_)
                    ? NVJPEG_BACKEND_GPU_HYBRID : NVJPEG_BACKEND_HYBRID;

      if (crop_generator) {
        info->crop_window = crop_generator(info->heights[0], info->widths[0]);
        auto &crop_window = info->crop_window;
        DALI_ENFORCE(crop_window.IsInRange(info->heights[0], info->widths[0]));
        nvjpegDecodeParamsSetROI(decode_params_[data_idx],
          crop_window.x, crop_window.y, crop_window.w, crop_window.h);
        info->widths[0] = crop_window.w;
        info->heights[0] = crop_window.h;
      }

      nvjpegJpegState_t state = GetNvjpegState(*state_nvjpeg);
      NVJPEG_CALL(nvjpegStateAttachPinnedBuffer(state,
                                                state_nvjpeg->pinned_buffer));
      nvjpegStatus_t ret = nvjpegDecodeJpegHost(
          handle_,
          GetDecoder(state_nvjpeg->nvjpeg_backend),
          state,
          decode_params_[data_idx],
          state_nvjpeg->jpeg_stream);
      if (ret != NVJPEG_STATUS_SUCCESS) {
        if (ret == NVJPEG_STATUS_JPEG_NOT_SUPPORTED || ret == NVJPEG_STATUS_BAD_JPEG) {
          info->nvjpeg_support = false;
        } else {
          NVJPEG_CALL_EX(ret, file_name);
        }
      } else {
        // TODO(spanev): free Output #2
      }
    }
  }

 protected:
  USE_OPERATOR_MEMBERS();

  virtual CropWindowGenerator GetCropWindowGenerator(int data_idx) const {
    return {};
  }

  inline std::pair<ImageInfo*, StateNvJPEG*>
  InitAndGet(Tensor<CPUBackend>& info_tensor, Tensor<CPUBackend>& state_tensor) {
    if (info_tensor.size() == 0) {
      TypeInfo type;
      // we need to set a arbitrary to be able to access to call raw_data()
      type.SetType<uint8_t>();

      std::shared_ptr<ImageInfo> info_p(new ImageInfo());
      info_tensor.ShareData(info_p, 1, {1});
      info_tensor.set_type(type);

      std::shared_ptr<StateNvJPEG> state_p(new StateNvJPEG(),
        [](StateNvJPEG* s) {
          NVJPEG_CALL(nvjpegJpegStreamDestroy(s->jpeg_stream));
          NVJPEG_CALL(nvjpegBufferPinnedDestroy(s->pinned_buffer));
          NVJPEG_CALL(nvjpegJpegStateDestroy(s->decoder_host_state));
          NVJPEG_CALL(nvjpegJpegStateDestroy(s->decoder_hybrid_state));
      });

      // We want to use nvJPEG default pinned allocator
      NVJPEG_CALL(nvjpegBufferPinnedCreate(handle_, &pinned_allocator_, &state_p->pinned_buffer));
      NVJPEG_CALL(nvjpegDecoderStateCreate(handle_,
                                        decoder_host_,
                                        &state_p->decoder_host_state));
      NVJPEG_CALL(nvjpegDecoderStateCreate(handle_,
                                        decoder_hybrid_,
                                        &state_p->decoder_hybrid_state));
      NVJPEG_CALL(nvjpegJpegStreamCreate(handle_, &state_p->jpeg_stream));

      state_tensor.ShareData(state_p, 1, {1});
      state_tensor.set_type(type);
    }

    ImageInfo* info = reinterpret_cast<ImageInfo*>(info_tensor.raw_mutable_data());
    StateNvJPEG* nvjpeg_state = reinterpret_cast<StateNvJPEG*>(state_tensor.raw_mutable_data());
    return std::make_pair(info, nvjpeg_state);
  }

  inline nvjpegJpegDecoder_t GetDecoder(nvjpegBackend_t backend) const {
    switch (backend) {
      case NVJPEG_BACKEND_HYBRID:
        return decoder_host_;
      case NVJPEG_BACKEND_GPU_HYBRID:
        return decoder_hybrid_;
      default:
        DALI_FAIL("Unknown nvjpegBackend_t " + std::to_string(backend));
        return nullptr;
    }
  }

  // output colour format
  DALIImageType output_image_type_;

  unsigned int hybrid_huffman_threshold_;

  // Common handles
  nvjpegHandle_t handle_;
  nvjpegJpegDecoder_t decoder_host_;
  nvjpegJpegDecoder_t decoder_hybrid_;

  // TODO(spanev): add huffman hybrid decode
  std::vector<nvjpegDecodeParams_t> decode_params_;


  nvjpegPinnedAllocator_t pinned_allocator_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_CPU_H_
