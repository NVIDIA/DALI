// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_C_API_2_DATA_OBJECTS_H_
#define DALI_C_API_2_DATA_OBJECTS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include "dali/dali.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/c_api_2/ref_counting.h"
#include "dali/core/tensor_shape_print.h"

struct _DALITensorList {};

namespace dali {
namespace c_api {

class TensorListInterface : public _DALITensorList, public RefCountedObject {
 public:
  virtual ~TensorListInterface() = default;

  virtual void Resize(
      int num_samples,
      int ndim,
      daliDataType_t dtype,
      const int64_t *shapes) = 0;

  virtual void AttachBuffer(
      int num_samples,
      int ndim,
      daliDataType_t dtype,
      const char *layout,
      const int64_t *shapes,
      void *data,
      const ptrdiff_t *sample_offsets,
      daliDeleter_t deleter) = 0;

  virtual void AttachSamples(
      int num_samples,
      int ndim,
      daliDataType_t dtype,
      const char *layout,
      const daliTensorDesc_t *samples,
      const daliDeleter_t *sample_deleters) = 0;

  virtual daliBufferPlacement_t GetBufferPlacement() const = 0;

  virtual const char *GetLayout() const = 0;

  virtual void SetLayout(const char *layout) = 0;

  virtual void SetStream(std::optional<cudaStream_t> stream, bool synchronize) = 0;

  virtual std::optional<cudaStream_t> GetStream() const = 0;

  virtual std::optional<cudaEvent_t> GetReadyEvent() const = 0;

  virtual cudaEvent_t GetOrCreateReadyEvent() = 0;

  static RefCountedPtr<TensorListInterface> Create(daliBufferPlacement_t placement);
};

struct BufferDeleter {
  daliDeleter_t deleter;
  AccessOrder deletion_order;

  void operator()(void *data) {
    if (deleter.delete_buffer) {
      cudaStream_t stream = deletion_order.stream();
      deleter.delete_buffer(deleter.deleter_ctx, data,
                            deletion_order.is_device() ? &stream : nullptr);
    }
    if (deleter.destroy_context) {
      deleter.destroy_context(deleter.deleter_ctx);
    }
  }
};

template <typename Backend>
class TensorListWrapper : public TensorListInterface {
 public:
  TensorListWrapper(std::shared_ptr<TensorList<Backend>> tl) : tl_(std::move(tl)) {};

  void Resize(
      int num_samples,
      int ndim,
      daliDataType_t dtype,
      const int64_t *shapes) override {
    std::vector<int64_t> shape_data(shapes, shapes + ndim * num_samples);
    tl_->Resize(TensorListShape<>(shape_data, num_samples, ndim), dtype);
  }

  void AttachBuffer(
      int num_samples,
      int ndim,
      daliDataType_t dtype,
      const char *layout,
      const int64_t *shapes,
      void *data,
      const ptrdiff_t *sample_offsets,
      daliDeleter_t deleter) override {

    if (num_samples < 0)
      throw std::invalid_argument("The number of samples must not be negative.");
    if (ndim < 0)
      throw std::invalid_argument("The number of dimensions must not be negative.");
    if (!shapes && ndim >= 0)
      throw std::invalid_argument("The `shapes` are required for non-scalar (ndim>=0) samples.");
    if (!data && num_samples > 0) {
      for (int i = 0; i < num_samples; i++) {
        auto sample_shape = make_cspan(&shapes[i*ndim], ndim);
        for (int j = 0; j < ndim; j++)
          if (sample_shape[j] < 0)
            throw std::invalid_argument(make_string(
              "Negative extent encountered in the shape of sample ", i, ". Offending shape: ",
              TensorShape<-1>(sample_shape)));
        if (volume(sample_shape) > 0)
          throw std::invalid_argument(
            "The pointer to the data buffer must not be null for a non-empty tensor list.");
        if (sample_offsets && sample_offsets[i])
          throw std::invalid_argument(
            "All sample_offsets must be zero when the data pointer is NULL.");
      }
    }

    TensorLayout new_layout = {};

    if (!layout) {
      if (ndim == tl_->sample_dim())
        new_layout = tl_->GetLayout();
    } else {
      new_layout = layout;
      if (new_layout.ndim() != ndim)
        throw std::invalid_argument(make_string(
          "The layout '", new_layout, "' cannot describe ", ndim, "-dimensional data."));
    }

    tl_->Reset();
    tl_->SetSize(num_samples);
    tl_->set_sample_dim(ndim);
    tl_->SetLayout(new_layout);
    ptrdiff_t next_offset = 0;
    auto type_info = TypeTable::GetTypeInfo(dtype);
    auto element_size = type_info.size();

    std::shared_ptr<void> buffer;
    if (!deleter.delete_buffer && !deleter.destroy_context) {
      buffer = std::shared_ptr<void>(data, [](void *){});
    } else {
      buffer = std::shared_ptr<void>(data, BufferDeleter{deleter, tl_->order()});
    }

    for (int i = 0; i < num_samples; i++) {
      TensorShape<> sample_shape(make_cspan(&shapes[i*ndim], ndim));
      void *sample_data;
      size_t sample_bytes = volume(sample_shape) * element_size;
      if (sample_offsets) {
        sample_data = static_cast<char *>(data) + sample_offsets[i];
      } else {
        sample_data = static_cast<char *>(data) + next_offset;
        next_offset += sample_bytes;
      }
      tl_->SetSample(
        i,
        std::shared_ptr<void>(buffer, sample_data),
        sample_bytes,
        tl_->is_pinned(),
        sample_shape,
        dtype,
        tl_->device_id(),
        tl_->order(),
        new_layout);
    }
  }

  virtual void AttachSamples(
      int num_samples,
      int ndim,
      daliDataType_t dtype,
      const char *layout,
      const daliTensorDesc_t *samples,
      const daliDeleter_t *sample_deleters) {
    if (num_samples < 0)
      throw std::invalid_argument("The number of samples must not be negative.");
    if (num_samples > 0 && !samples)
      throw std::invalid_argument("The pointer to sample descriptors must not be NULL.");
    if (ndim < 0) {
      if (num_samples == 0)
        throw std::invalid_argument(
          "The number of dimensions must not be negative when num_samples is 0.");
      else
        ndim = samples[0].ndim;
    }

    for (int i = 0; i < num_samples; i++) {
      if (samples[i].ndim != ndim)
        throw std::invalid_argument(make_string(
            "Invalid `ndim` at sample ", i, ": got ", samples[i].ndim, ", expected ", ndim, "."));
      if (ndim && !samples[i].shape)
        throw std::invalid_argument(make_string("Got NULL shape in sample ", i, "."));

      for (int j = 0; j < ndim; j++)
        if (samples[i].shape[j] < 0) {
          TensorShape<> sample_shape(make_cspan(samples[i].shape, samples[i].ndim));
          throw std::invalid_argument(make_string(
            "Negative extent encountered in the shape of sample ", i, ". Offending shape: ",
            sample_shape));
        }

      if (!samples[i].data && volume(make_cspan(samples[i].shape, ndim)))
        throw std::invalid_argument(make_string(
            "Got NULL data pointer in a non-empty sample ", i, "."));
    }

    TensorLayout new_layout = {};

    if (!layout) {
      if (ndim == tl_->sample_dim())
        new_layout = tl_->GetLayout();
    } else {
      new_layout = layout;
      if (new_layout.ndim() != ndim)
        throw std::invalid_argument(make_string(
          "The layout '", new_layout, "' cannot describe ", ndim, "-dimensional data."));
    }

    tl_->Reset();
    tl_->SetSize(num_samples);
    tl_->set_sample_dim(ndim);
    tl_->SetLayout(new_layout);

    auto deletion_order = tl_->order();

    auto type_info = TypeTable::GetTypeInfo(dtype);
    auto element_size = type_info.size();
    for (int i = 0; i < num_samples; i++) {
      TensorShape<> sample_shape(make_cspan(samples[i].shape, samples[i].ndim));
      size_t sample_bytes = volume(sample_shape) * element_size;
      std::shared_ptr<void> sample_ptr;
      if (sample_deleters) {
        sample_ptr = std::shared_ptr<void>(
          samples[i].data,
          BufferDeleter{sample_deleters[i], deletion_order});
      } else {
        sample_ptr = std::shared_ptr<void>(samples[i].data, [](void*) {});
      }

      tl_->SetSample(
        i,
        sample_ptr,
        sample_bytes,
        tl_->is_pinned(),
        sample_shape,
        dtype,
        tl_->device_id(),
        tl_->order(),
        new_layout);
    }
  }

  daliBufferPlacement_t GetBufferPlacement() const override {
    daliBufferPlacement_t placement;
    placement.device_id = tl_->device_id();
    StorageDevice dev = backend_to_storage_device<Backend>::value;
    placement.device_type = static_cast<daliStorageDevice_t>(dev);
    placement.pinned = tl_->is_pinned();
    return placement;
  }

  void SetStream(std::optional<cudaStream_t> stream, bool synchronize) override {
    tl_->set_order(stream.has_value() ? AccessOrder(*stream) : AccessOrder::host(), synchronize);
  }

  void SetLayout(const char *layout_string) {
    if (layout_string) {
      TensorLayout layout(layout_string);
      if (layout.ndim() != tl_->sample_dim())
        throw std::invalid_argument(make_string(
          "The layout '", layout, "' cannot describe ", tl_->sample_dim(), "-dimensional data."));
      tl_->SetLayout(layout);
    } else {
      tl_->SetLayout("");
    }
  }

  const char *GetLayout() const override {
    auto &layout = tl_->GetLayout();
    return !layout.empty() ? layout.data() : nullptr;
  }

  std::optional<cudaStream_t> GetStream() const override {
    auto o = tl_->order();
    if (o.is_device())
      return o.stream();
    else
      return std::nullopt;
  }

  std::optional<cudaEvent_t> GetReadyEvent() const override {
    auto &e = tl_->ready_event();
    if (e)
      return e.get();
    else
      return std::nullopt;
  }

  cudaEvent_t GetOrCreateReadyEvent() override {
    auto &e = tl_->ready_event();
    if (e)
      return e.get();
    int device_id = tl_->device_id();
    if (device_id < 0)
      throw std::runtime_error("The tensor list is not associated with a CUDA device.");
    tl_->set_ready_event(CUDASharedEvent::Create(device_id));
    return tl_->ready_event().get();
  }
 private:
  std::shared_ptr<TensorList<Backend>> tl_;
};

template <typename Backend>
RefCountedPtr<TensorListWrapper<Backend>> Wrap(std::shared_ptr<TensorList<Backend>> tl) {
  return RefCountedPtr<TensorListWrapper<Backend>>(new TensorListWrapper<Backend>(std::move(tl)));
}

TensorListInterface *ToPointer(daliTensorList_h handle);

}  // namespace c_api
}  // namespace dali

#endif  // DALI_C_API_2_DATA_OBJECTS_H_
