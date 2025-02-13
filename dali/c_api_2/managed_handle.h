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

#ifndef DALI_C_API_2_MANAGED_HANDLE_H_
#define DALI_C_API_2_MANAGED_HANDLE_H_

#include <cassert>
#include <stdexcept>
#include <utility>
#define DALI_ALLOW_NEW_C_API
#include "dali/dali.h"
#include "dali/core/unique_handle.h"

namespace dali::c_api {

template <typename HandleType, typename Actual>
class RefCountedHandle {
 public:
  using handle_type = HandleType;
  static constexpr handle_type null_handle() { return 0; }

  constexpr RefCountedHandle() : handle_(Actual::null_handle()) {}
  constexpr explicit RefCountedHandle(handle_type h) : handle_(h) {}
  ~RefCountedHandle() { reset(); }

  RefCountedHandle(const RefCountedHandle &h) {
    handle_ = h.handle_;
    if (*this)
        Actual::IncRef(handle_);
  }

  RefCountedHandle(RefCountedHandle &&h) noexcept {
    handle_ = h.handle_;
    h.handle_ = Actual::null_handle();
  }

  RefCountedHandle &operator=(const RefCountedHandle &other) {
    if (handle_ == other.handle_)
      return *this;
    if (other.handle_) {
      Actual::IncRef(other.handle_);
    }
    reset();
    handle_ = other.handle_;
    return *this;
  }

  RefCountedHandle &operator=(RefCountedHandle &&other) noexcept {
    if (&other == this)  // cannot move to self
      return *this;
    std::swap(handle_, other.handle_);
    other.reset();
    return *this;
  }

  void reset() noexcept {
    if (*this)
        Actual::DecRef(handle_);
    handle_ = Actual::null_handle();
  }

  [[nodiscard]] handle_type release() noexcept {
    auto h = handle_;
    handle_ = Actual::null_handle();
    return h;
  }

  handle_type get() const noexcept { return handle_; }
  operator handle_type() const noexcept { return get(); }

  explicit operator bool() const noexcept { return handle_ != Actual::null_handle(); }

 private:
  handle_type handle_;
};

#define DALI_C_UNIQUE_HANDLE(Resource) \
class Resource##Handle : public dali::UniqueHandle<dali##Resource##_h, Resource##Handle> { \
 public: \
  using UniqueHandle<dali##Resource##_h, Resource##Handle>::UniqueHandle; \
  static void DestroyHandle(dali##Resource##_h h) { \
    auto result = dali##Resource##Destroy(h); \
    if (result != DALI_SUCCESS) { \
        throw std::runtime_error(daliGetLastErrorMessage()); \
    } \
  } \
}

#define DALI_C_REF_HANDLE(Resource) \
class Resource##Handle \
: public dali::c_api::RefCountedHandle<dali##Resource##_h, Resource##Handle> { \
 public: \
  using RefCountedHandle<dali##Resource##_h, Resource##Handle>::RefCountedHandle; \
  static int IncRef(dali##Resource##_h h) { \
    int ref = 0; \
    auto result = dali##Resource##IncRef(h, &ref); \
    if (result != DALI_SUCCESS) { \
        throw std::runtime_error(daliGetLastErrorMessage()); \
    } \
    return ref; \
  } \
  static int DecRef(dali##Resource##_h h) { \
    int ref = 0; \
    auto result = dali##Resource##DecRef(h, &ref); \
    if (result != DALI_SUCCESS) { \
        throw std::runtime_error(daliGetLastErrorMessage()); \
    } \
    return ref; \
  } \
}

DALI_C_UNIQUE_HANDLE(Pipeline);
DALI_C_UNIQUE_HANDLE(PipelineOutputs);
DALI_C_REF_HANDLE(TensorList);
DALI_C_REF_HANDLE(Tensor);


}  // namespace dali::c_api

#endif  // DALI_C_API_2_MANAGED_HANDLE_H_
