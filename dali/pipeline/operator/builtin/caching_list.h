// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATOR_BUILTIN_CACHING_LIST_H_
#define DALI_PIPELINE_OPERATOR_BUILTIN_CACHING_LIST_H_

#include <stdexcept>
#include <list>
#include <memory>

namespace dali {
/**
 * CachingList differs from std::List by the ability to recycle empty elements. When allocating
 * memory is expensive it is better to store already allocated but no longer needed element in the
 * list of the free elements, than to free the memory and allocate it again later. CachingList
 * supports the following operations:
 * - GetEmpty moves an empty element of type T, either allocate it or use one from the free list
 * - PopFront moves the element from the front and removes it from the full list, the behavior
 * is undefined when the list is empty
 * - Recycle moves passed element to the free list
 * - PushBack moves element to the full list
 * - IsEmpty checks if the full list is empty
 * All functions operate on one element list as transferring elements between list is a very low
 * cost operation, which doesn't involve any memory allocation, while adding an element to the list
 * requires allocation of the memory for the storage in the list.
 *
 * Additionally, CachingList has a Prophet feature. This is an unidirectional iterator,
 * that travels over the data (asynchronously w.r.t. current Front and Back). The Prophet
 * allows to peek a list element and maintains the order even when elements are Pushed
 * and Popped in/out.
 * Use PeekProphet() and AdvanceProphet() to control the prophet.
 * In case there's an illegal access to the list, std::out_of_range will be thrown.
 */
template<typename T>
class CachingList {
 public:
  CachingList() : prophet_(full_data_.end()) {}


  bool IsEmpty() const {
    return full_data_.empty();
  }


  const T &PeekFront() {
    return full_data_.front();
  }


  std::list<T> PopFront() {
    assert(!full_data_.empty());  // Can't pop from an empty list
    std::list<T> tmp;
    tmp.splice(tmp.begin(), full_data_, full_data_.begin());
    if (tmp.begin() == prophet_)
      prophet_ = full_data_.begin();
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


  void PushBack(std::list<T> &elm) {
    full_data_.splice(full_data_.end(), elm, elm.begin());
    /*
     * When the prophet is dead and needs to be resurrected,
     * he shall be resurrected by the apprentice.
     * In the special scenario, when prophet is dead and the data list is empty
     * (hence the apprentice is dead too), the prophet will be resurrected
     * from scratch, by assigning him to the element that was just added to the data list.
     * Sic mundus creatus est.
     */
    if (resurrect_prophet_) {
      if (full_data_.size() == 1) {
        prophet_ = full_data_.begin();
      } else {
        prophet_ = std::next(apprentice_);
      }
      resurrect_prophet_ = false;
    }
  }


  const T &PeekProphet() {
    if (prophet_ == full_data_.end())
      throw std::out_of_range(
              "Attempted to peek the data batch that doesn't exist. Add more elements to the DALI"
              " input operator.");
    return *prophet_;
  }


  void AdvanceProphet() {
    if (!CanProphetAdvance())
      throw std::out_of_range(
              "Attempted to move to the data batch that doesn't exist. Add more elements to"
              " the DALI input operator.");
    apprentice_ = prophet_++;
    resurrect_prophet_ = prophet_ == full_data_.end();
  }


  bool CanProphetAdvance() {
    return prophet_ != full_data_.end();
  }

 private:
  std::list<T> full_data_;
  std::list<T> empty_data_;

  /**
   * Prophet dies when he hits the end() iterator of the list with the data.
   * Prophet can be resurrected, iff there is a data record for him, i.e.
   * when user calls PushBack and therefore inserts the data at the end
   * of the CachingList
   */
  bool resurrect_prophet_ = true;

  /**
   * The apprentice follows the prophet and is always one step behind him.
   * Apprentice is used to resurrect the prophet, so that the prophet might
   * again point to the last actual element of the list.
   */
  typename std::list<T>::iterator prophet_, apprentice_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BUILTIN_CACHING_LIST_H_
