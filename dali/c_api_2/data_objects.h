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

namespace dali {
namespace c_api {

class TensorListInterface : public RefCountedObject {
 public:
  virtual ~TensorListInterface() = default;

  int IncRef() { return ++ref_; }
  int DecRef() {
    int new_ref = --ref_;
    if (new_ref == 0) {
      delete this;
      return 0;
    } else {
      return new_ref;
    }
  }

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

 private:
  std::atomic_int ref_{1};
};

}  // namespace c_api
}  // namespace dali

#endif  // DALI_C_API_2_DATA_OBJECTS_H_
