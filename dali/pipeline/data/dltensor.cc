// Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime_api.h>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include "dali/pipeline/data/dltensor.h"
#include "dali/core/error_handling.h"
#include "dali/core/mm/detail/aux_alloc.h"
#include "dali/core/static_switch.h"

namespace dali {

class DLTensorGraveyard {
 public:
  ~DLTensorGraveyard() {
    shutdown();
  }

  void enqueue(std::shared_ptr<void> mem) {
    {
      std::lock_guard g(mtx_);
      if (exit_requested_)
        return;  // we don't prolong the life of memory on shutdown
      if (!started_)
        start();
      pending_.push_back(std::move(mem));
    }
    cv_.notify_one();
  }

  static DLTensorGraveyard &instance(int dev) {
    static std::vector<DLTensorGraveyard> inst = []() {
      int ndev = 0;
      CUDA_CALL(cudaGetDeviceCount(&ndev));
      std::vector<DLTensorGraveyard> ret(ndev);
      for (int i = 0; i < ndev; i++)
        ret[i].device_id_ = i;
      return ret;
    }();
    return inst[dev];
  }

 private:
  void start() {
    assert(!exit_requested_);
    assert(!started_);
    worker_ = std::thread([this]() { run(); });
    started_ = true;
  }

  void shutdown() {
    {
      std::lock_guard g(mtx_);
      exit_requested_ = true;
    }
    cv_.notify_one();
    if (worker_.joinable())
      worker_.join();
  }

  void run() {
    CUDA_CALL(cudaSetDevice(device_id_));
    std::unique_lock lock(mtx_);
    for (;;) {
      cv_.wait(lock, [&]() {
        return !pending_.empty() || exit_requested_;
      });
      if (exit_requested_)
        break;
      list_t tmp = std::move(pending_);
      lock.unlock();
      auto ret = cudaDeviceSynchronize();
      if (ret == cudaErrorCudartUnloading)
        break;
      CUDA_CALL(ret);
      tmp.clear();
      lock.lock();
    }
  }

  std::mutex mtx_;
  std::condition_variable cv_;
  std::thread worker_;
  using list_alloc_t = mm::detail::object_pool_allocator<std::shared_ptr<void>>;
  using list_t = std::list<std::shared_ptr<void>, list_alloc_t>;
  list_t pending_;
  int device_id_ = -1;
  bool started_ = false;
  bool exit_requested_ = false;
};

void EnqueueForDeletion(std::shared_ptr<void> data, int device_id) {
  DLTensorGraveyard::instance(device_id).enqueue(std::move(data));
}

DLDataType ToDLType(DALIDataType type) {
  DLDataType dl_type{};
  TYPE_SWITCH(type, type2id, T, (DALI_NUMERIC_TYPES_FP16, bool), (
    dl_type.bits = sizeof(T) * 8;
      dl_type.lanes = 1;
      if constexpr (dali::is_fp_or_half<T>::value) {
        dl_type.code = kDLFloat;
      } else if constexpr (std::is_same_v<T, bool>) {
        dl_type.code = kDLBool;
      } else if constexpr (std::is_unsigned_v<T>) {
        dl_type.code = kDLUInt;
      } else if constexpr (std::is_integral_v<T>) {
        dl_type.code = kDLInt;
      } else {
        DALI_FAIL(make_string("This data type (", type, ") cannot be handled by DLTensor."));
      }
  ), (DALI_FAIL(make_string("The element type ", type, " is not supported."))));  // NOLINT
  return dl_type;
}

void DLMTensorPtrDeleter(DLManagedTensor* dlm_tensor_ptr) {
  if (dlm_tensor_ptr && dlm_tensor_ptr->deleter) {
    dlm_tensor_ptr->deleter(dlm_tensor_ptr);
  }
}

DALIDataType ToDALIType(const DLDataType &dl_type) {
  DALI_ENFORCE(dl_type.lanes == 1,
               "DALI Tensors do not support types with the number of lanes other than 1");
  switch (dl_type.code) {
    case kDLUInt: {
      switch (dl_type.bits) {
        case 8: return DALI_UINT8;
        case 16: return DALI_UINT16;
        case 32: return DALI_UINT32;
        case 64: return DALI_UINT64;
      }
      break;
    }
    case kDLInt: {
      switch (dl_type.bits) {
        case 8: return DALI_INT8;
        case 16: return DALI_INT16;
        case 32: return DALI_INT32;
        case 64: return DALI_INT64;
      }
      break;
    }
    case kDLFloat: {
      switch (dl_type.bits) {
        case 16: return DALI_FLOAT16;
        case 32: return DALI_FLOAT;
        case 64: return DALI_FLOAT64;
      }
      break;
    }
    case kDLBool: {
      return DALI_BOOL;
      break;
    }
  }
  DALI_FAIL("Could not convert DLPack tensor of unsupported type " + to_string(dl_type));
}

}  // namespace dali
