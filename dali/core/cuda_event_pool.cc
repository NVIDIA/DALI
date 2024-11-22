// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <cassert>
#include <mutex>
#include <utility>
#include "dali/core/cuda_event_pool.h"
#include "dali/core/cuda_error.h"

namespace dali {

CUDAEventPool::~CUDAEventPool() {
  Purge();
}

CUDAEventPool::CUDAEventPool(unsigned event_flags) {
  event_flags_ = event_flags;
  int num_devices = 0;
  CUDA_CALL(cudaGetDeviceCount(&num_devices));
  dev_events_.resize(num_devices);
}

CUDAEvent CUDAEventPool::Get(int device_id) {
  if (device_id == -1)
    CUDA_CALL(cudaGetDevice(&device_id));

  CUDAEvent ev = GetFromPool(device_id);
  if (!ev)
    return CUDAEvent::CreateWithFlags(event_flags_, device_id);
  return ev;
}

void CUDAEventPool::Purge() {
  std::lock_guard<spinlock> guard(lock_);
  DeleteList(unused_);
  for (auto &list : dev_events_)
    DeleteList(list);
}

void CUDAEventPool::DeleteList(EventEntry *&head) {
  while (head) {
    auto *next = head->next;
    delete head;
    head = next;
  }
}

CUDAEvent CUDAEventPool::GetFromPool(int device_id) {
  assert(device_id >= 0 && device_id < static_cast<int>(dev_events_.size()));
  std::lock_guard<spinlock> guard(lock_);
  EventEntry *e = Pop(dev_events_[device_id]);
  if (!e)
    return {};
  CUDAEvent ev = std::move(e->event);
  Push(unused_, e);
  return ev;
}

void CUDAEventPool::Put(CUDAEvent &&event, int device_id) {
  if (device_id < 0)
    cudaGetDevice(&device_id);

  assert(device_id >= 0 && device_id < static_cast<int>(dev_events_.size()));

  std::unique_lock<spinlock> lock(lock_);
  EventEntry *e = Pop(unused_);
  if (!e) {
    lock.unlock();
    e = new EventEntry(std::move(event));
    lock.lock();
  } else {
    e->event = std::move(event);
  }
  Push(dev_events_[device_id], e);
}

CUDAEventPool &CUDAEventPool::instance() {
  static CUDAEventPool instance;
  return instance;
}

}  // namespace dali
