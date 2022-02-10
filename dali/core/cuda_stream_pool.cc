// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/cuda_error.h"

namespace dali {

CUDAStreamPool::~CUDAStreamPool() {
  Purge();
}

CUDAStreamPool::CUDAStreamPool() {
  dev_streams_.reserve(128);  // to avoid allocation in 1st call
}

CUDAStreamLease CUDAStreamPool::Get(int device_id) {
  if (device_id < 0)
    CUDA_CALL(cudaGetDevice(&device_id));

  CUDAStream s = GetFromPool(device_id);
  if (!s)
    s = CUDAStream::Create(true, device_id);
  return { std::move(s), device_id, this };
}

void CUDAStreamPool::Purge() {
  std::lock_guard<spinlock> guard(lock_);
  DeleteList(unused_);
  for (auto &list : dev_streams_)
    DeleteList(list);
}

void CUDAStreamPool::DeleteList(StreamEntry *&head) {
  while (head) {
    auto *next = head->next;
    delete head;
    head = next;
  }
}

void CUDAStreamPool::Init() {
  int num_devices = 0;
  auto e = cudaGetDeviceCount(&num_devices);
  (void)cudaGetLastError();  // clear the error
  if (e != cudaSuccess && e != cudaErrorNoDevice && e != cudaErrorInsufficientDriver)
    throw CUDAError(e);
  dev_streams_.resize(num_devices);
}

CUDAStream CUDAStreamPool::GetFromPool(int device_id) {
  std::lock_guard<spinlock> guard(lock_);
  if (dev_streams_.empty())
    Init();
  assert(device_id >= 0 && device_id < static_cast<int>(dev_streams_.size()));
  StreamEntry *e = Pop(dev_streams_[device_id]);
  if (!e)
    return {};
  CUDAStream ev = std::move(e->stream);
  Push(unused_, e);
  return ev;
}

void CUDAStreamPool::Put(CUDAStream &&stream, int device_id) {
  if (!stream)
    throw std::invalid_argument("Cannot put a null stream in the pool.");
  if (device_id < 0) {
    try {
      device_id = stream.GetDevice();
    } catch (const CUDAError &e) {
      if (e.is_unloading()) {
        stream.reset();
        return;
      } else {
        throw;
      }
    }
  }

  std::unique_lock<spinlock> lock(lock_);
  if (dev_streams_.empty())
    Init();
  assert(device_id >= 0 && device_id < static_cast<int>(dev_streams_.size()));

  StreamEntry *e = Pop(unused_);
  if (!e) {
    lock.unlock();
    e = new StreamEntry(std::move(stream));
    lock.lock();
  } else {
    e->stream = std::move(stream);
  }
  Push(dev_streams_[device_id], e);
}

CUDAStreamPool &CUDAStreamPool::instance() {
  static CUDAStreamPool instance;
  return instance;
}

}  // namespace dali
