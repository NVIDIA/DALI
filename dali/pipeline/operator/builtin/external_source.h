// Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATOR_BUILTIN_EXTERNAL_SOURCE_H_
#define DALI_PIPELINE_OPERATOR_BUILTIN_EXTERNAL_SOURCE_H_

#include <atomic>
#include <string>
#include <vector>
#include <memory>
#include <list>
#include <mutex>
#include <condition_variable>
#include <utility>

#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/worker_thread.h"

namespace dali {

namespace detail {

struct CudaEventWrapper {
  CudaEventWrapper() {
    auto res = cudaEventCreate(&event);
    if (res != cudaSuccess) {
      DALI_ERROR("Fatal error: " + to_string(res) + " in CudaEventWrapper()");
      std::terminate();
    }
  }

  ~CudaEventWrapper() {
    auto res = cudaEventDestroy(event);
    if (res != cudaSuccess) {
      DALI_ERROR("Fatal error:: " + to_string(res) + " in ~CudaEventWrapper()");
      std::terminate();
    }
  }

  DEFAULT_COPY_MOVE_ASSIGN(CudaEventWrapper);

  operator cudaEvent_t() const {  // NOLINT (non-explicit op)
    return event;
  }

  cudaEvent_t event;
};

}  // namespace detail

namespace detail {

/**
 * CachingList differs from std::List by the ability to recycle empty elements. When allocating memory
 * is expensive it is better to store already allocated but no longer needed element in the list of the free
 * elements, than to free the memory and allocate it again later. CachingList supports the following operations:
 * - GetEmpty moves an empty element of type T, either allocate it or use one from the free list
 * - PopFront moves the element from the front and removes it from the full list, the behavior
 * is undefined when the list is empty
 * - Recycle moves passed element to the free list
 * - AddBack moves element to the full list
 * - IsEmpty checks if the full list is empty
 * All functions operate on one element list as transferring elements between list is a very low cost
 * operation, which doesn't involve any memory allocation, while adding an element to the list requires
 * allocation of the memory for the storage in the list.
 */
template <typename T>
class CachingList {
 public:
  bool IsEmpty() const {
    return full_data_.empty();
  }

  std::list<T> PopFront() {
    std::list<T> tmp;
    tmp.splice(tmp.begin(), full_data_, full_data_.begin());
    return tmp;
  }

  void Recycle(std::list<T> &elm) {
    empty_data_.splice(empty_data_.end(), elm, elm.begin());
  }

  std::list<T> GetEmpty() {
    std::list<T> tmp;
    if (empty_data_.empty()) {
      tmp.emplace_back(std::make_unique<typename T::element_type>());
    } else {
      tmp.splice(tmp.begin(), empty_data_, empty_data_.begin());
    }
    return tmp;
  }

  void AddBack(std::list<T> &elm) {
    full_data_.splice(full_data_.end(), elm, elm.begin());
  }

 private:
  std::list<T> full_data_;
  std::list<T> empty_data_;
};

}  // namespace detail

/**
 * @brief Provides in-graph access to data fed in from outside of dali.
 * For now, we do a copy from the passed in data into our data to avoid
 * potential scoping and data corruption issues.
 * Please note, that it is not allowed to call this concurrently as it
 * may mix the order of inputted data.
 */
template <typename Backend>
class ExternalSource : public Operator<Backend> {
  using uptr_tl_type = std::unique_ptr<TensorList<Backend>>;
  using uptr_vt_type = std::unique_ptr<std::vector<Tensor<Backend>>>;
  using uptr_cuda_event_type = std::unique_ptr<detail::CudaEventWrapper>;

 public:
  inline explicit ExternalSource(const OpSpec &spec) :
    Operator<Backend>(spec),
    sync_worker_(spec.GetArgument<int>("device_id"), false) {
    output_name_ = spec.Output(0);
    sync_worker_.WaitForInit();
  }

  inline ~ExternalSource() {
    try {
      sync_worker_.ForceStop();
      sync_worker_.Shutdown();
    } catch (const std::exception &) {
      // Something went terribly wrong while releasing resources. We'd better die right now.
      std::terminate();
    }
  }

  inline string name() const override {
    return "ExternalSource (" + output_name_ + ")";
  }


  /**
   * @brief Sets the data that should be passed out of the op
   * on the next iteration.
   */
  template<typename SrcBackend>
  inline void SetDataSource(const TensorList<SrcBackend> &tl, cudaStream_t stream = 0) {
    DALI_ENFORCE(OperatorBase::batch_size_ == static_cast<int>(tl.ntensor()),
                 "Data list provided to ExternalSource needs to have batch_size length.");
    // Note: If we create a GPU source, we will need to figure
    // out what stream we want to do this copy in. CPU we can
    // pass anything as it is ignored.
    std::list<uptr_tl_type> data;
    std::list<uptr_cuda_event_type> copy_to_gpu_event;
    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      data = tl_data_.GetEmpty();
      copy_to_gpu_event = copy_to_gpu_events_.GetEmpty();
    }

    data.front()->Copy(tl, stream);
    if (std::is_same<SrcBackend, GPUBackend>::value) {
      cudaEventRecord(*copy_to_gpu_event.front(), stream);
    }

    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      tl_data_.AddBack(data);
      copy_to_gpu_events_.AddBack(copy_to_gpu_event);
      data_in_tl_.push_back(true);
    }
    cv_.notify_one();
  }


  /**
   * @brief Sets the data that should be passed out of the op
   * on the next iteration.
   */
  template<typename SrcBackend>
  inline void SetDataSource(const vector<Tensor<SrcBackend>> &t, cudaStream_t stream = 0) {
    DALI_ENFORCE(OperatorBase::batch_size_ == static_cast<int>(t.size()),
                 "Data list provided to ExternalSource needs to have batch_size length.");
    // Note: If we create a GPU source, we will need to figure
    // out what stream we want to do this copy in. CPU we can
    // pass anything as it is ignored.
    std::list<uptr_vt_type> data;
    std::list<uptr_cuda_event_type> copy_to_gpu_event;
    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      data = t_data_.GetEmpty();
      copy_to_gpu_event = copy_to_gpu_events_.GetEmpty();
    }

    data.front()->resize(t.size());
    for (size_t i = 0; i < t.size(); ++i) {
      (*(data.front()))[i].Copy(t[i], stream);
    }
    if (std::is_same<SrcBackend, GPUBackend>::value) {
      cudaEventRecord(*copy_to_gpu_event.front(), stream);
    }

    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      t_data_.AddBack(data);
      copy_to_gpu_events_.AddBack(copy_to_gpu_event);
      data_in_tl_.push_back(false);
    }
    cv_.notify_one();
  }

  DISABLE_COPY_MOVE_ASSIGN(ExternalSource);

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    return false;
  }

  /*
   * So that compiler wouldn't complain, that
   * "overloaded virtual function `dali::Operator<dali::CPUBackend>::RunImpl` is only partially
   * overridden in class `dali::[...]<dali::CPUBackend>`"
   */
  using Operator<Backend>::RunImpl;

  void RunImpl(workspace_t<Backend> &ws) override;

  void RecycleHelper(std::list<uptr_tl_type> &data) {
    tl_data_.Recycle(data);
  }

  void RecycleHelper(std::list<uptr_vt_type> &data) {
    t_data_.Recycle(data);
  }

  // pass cuda_event by pointer to allow default, nullptr value, with the
  // reference it is not that easy
  template<typename DataType>
  void RecycleBuffer(DataType &data,
                     std::list<uptr_cuda_event_type> *cuda_event = nullptr,
                     std::list<uptr_cuda_event_type> *copy_to_gpu = nullptr) {
    if (cuda_event) {
      cudaEventSynchronize(cuda_event->front()->event);
    }
    // No need to synchronize on copy_to_gpu - it was already synchronized before
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    RecycleHelper(data);
    if (cuda_event) {
      cuda_events_.Recycle(*cuda_event);
    }
    if (copy_to_gpu) {
      copy_to_gpu_events_.Recycle(*copy_to_gpu);
    }
  }

  string output_name_;
  detail::CachingList<uptr_tl_type> tl_data_;
  detail::CachingList<uptr_vt_type> t_data_;
  detail::CachingList<uptr_cuda_event_type> cuda_events_, copy_to_gpu_events_;
  std::list<bool> data_in_tl_;
  struct RecycleFunctor;

  std::mutex busy_m_;
  std::condition_variable cv_;

  WorkerThread sync_worker_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BUILTIN_EXTERNAL_SOURCE_H_
