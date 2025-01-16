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

#include <optional>
#include "dali/dali.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/c_api_2/ref_counting.h"

namespace dali {
namespace c_api {

class TensorListInterface : public RefCountedObject {
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
      const int64_t *shapes,
      void *data,
      const ptrdiff_t *sample_offsets,
      daliDeleter_t deleter) = 0;

  virtual void AttachSamples(
      int num_samples,
      int ndim,
      daliDataType_t dtype,
      const daliTensorDesc_t *samples,
      const daliDeleter_t *sample_deleters) = 0;

  virtual daliBufferPlacement_t GetBufferPlacement() const = 0;

  virtual void SetStream(std::optional<cudaStream_t> stream, bool synchronize) = 0;

  virtual std::optional<cudaStream_t> GetStream() const = 0;

  virtual std::optional<cudaEvent_t> GetReadyEvent() const() = 0;

  virtual cudaEvent_t GetOrCreateReadyEvent() = 0;
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
    tl_->Resize(TensorListShape<>(make_cspan(shapes, num_samples*ndim), num_samples, ndim), dtype);
  }

  void AttachBuffer(
      int num_samples,
      int ndim,
      daliDataType_t dtype,
      const int64_t *shapes,
      void *data,
      const ptrdiff_t *sample_offsets,
      daliDeleter_t deleter) override {
    tl_->Reset();
    tl_->SetSize(num_samples);
    tl_->set_sample_dim(ndim);
    ptridff_t next_offset = 0;
    auto type_info = TypeTable::GetTypeInfo(dtype);
    auto element_size = type_info.size();
    std::shared_ptr<void *> buffer;
    if (!deleter.delete_buffer && !deleter.destroy_context) {
      buffer.reset(buffer, [](void *){});
    } else {
      buffer.reset(buffer, [deleter](void *p) {
        if (deleter.delete_buffer)
          deleter.delete_buffer(deleter.deleter_ctx, p, nullptr);
      });
    }
    for (int i = 0; i < num_samples; i++) {
      TensorShape<> sample_shape(make_cspan(&shapes[i*ndim]. ndim));
      void *sample_data;
      if (sample_offsets) {
        sample_data = static_cast<char *>(data) + sample_offsets[i];
      } else {
        sample_data = static_cast<char *>(data) + next_offset;
        next_offset += volme(sample_shape) * element_size;
      }

    }

  }

  virtual void AttachSamples(
      int num_samples,
      int ndim,
      daliDataType_t dtype,
      const daliTensorDesc_t *samples,
      const daliDeleter_t *sample_deleters) = 0;

  virtual daliBufferPlacement_t GetBufferPlacement() const = 0;

  virtual void SetStream(std::optional<cudaStream_t> stream, bool synchronize) = 0;

  virtual std::optional<cudaStream_t> GetStream() const = 0;

  virtual std::optional<cudaEvent_t> GetReadyEvent() const() = 0;

  virtual cudaEvent_t GetOrCreateReadyEvent() = 0;
 private:
  std::shared_ptr<TensorList<Backend>> impl_;
};

}  // namespace c_api
}  // namespace dali

#endif  // DALI_C_API_2_DATA_OBJECTS_H_
