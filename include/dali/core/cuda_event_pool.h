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

#ifndef DALI_CORE_CUDA_EVENT_POOL_H_
#define DALI_CORE_CUDA_EVENT_POOL_H_

#include <vector>
#include <utility>
#include "dali/core/cuda_event.h"
#include "dali/core/spinlock.h"

namespace dali {

class DLL_PUBLIC CUDAEventPool {
 public:
  ~CUDAEventPool();
  explicit CUDAEventPool(unsigned event_flags = cudaEventDisableTiming);

  /**
   * @brief Get an event for given device.
   *
   * @param device_id   CUDA runtime API device ordinal. If negative, calling thread's
   *                    current device is used.
   *
   * @return A CUDA event wrapper object. If there were any events in the pools, the
   *         event is taken from it, otherwise a new event is created.
   */
  CUDAEvent Get(int device_id = -1);

  /**
   * @brief Place an event for given device in the pool.
   *
   * @param event     CUDA event wrapper object. The caller relinquishes ownership of the event.
   * @param device_id CUDA runtime API device ordinal of the device for which the event was
   *                  created. If negative, calling thread's current device is used.
   *
   * @remarks It is an error to misstate the device_id. Placing an event with improper deviceid
   *          will render the event pool unusable.
   */
  void Put(CUDAEvent &&event, int device_id = -1);

  /**
   * @brief Removes all events currently in the pool and deletes auxiliary data structures.
   */
  void Purge();

  /**
   * @brief Returns a reference to the singleton instance.
   *
   * @remarks Using the singleton instance is only possible if:
   * - the enclosing library is compiled and used as a shared object,
   * - the enclosing library is static, but matching calls to Get and Put calls are
   *   contained within one shared object.
   */
  static CUDAEventPool &instance();

 private:
  CUDAEvent GetFromPool(int device_id);

  unsigned event_flags_ = cudaEventDisableTiming;

  struct EventEntry {
    EventEntry() = default;
    explicit EventEntry(CUDAEvent event, EventEntry *next = nullptr)
    : event(std::move(event)), next(next) {}
    CUDAEvent event;
    EventEntry *next = nullptr;
  };

  EventEntry *unused_ = nullptr;

  vector<EventEntry *> dev_events_;
  spinlock lock_;

  static EventEntry *Pop(EventEntry *&head) {
    auto *e = head;
    if (e)
      head = head->next;
    return e;
  }

  static void Push(EventEntry *&head, EventEntry *new_entry) {
    new_entry->next = head;
    head = new_entry;
  }

  void DeleteList(EventEntry *&head);
};

}  // namespace dali

#endif  // DALI_CORE_CUDA_EVENT_POOL_H_
