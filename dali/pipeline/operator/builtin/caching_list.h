// Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <list>
#include <memory>
#include <utility>
#include <stdexcept>

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

  class Item {
   public:
    Item() = default;
    T &operator*() const & noexcept { return l_.front(); }
    T &&operator*() && noexcept { return l_.front(); }

    T *operator->() const & noexcept { return &l_.front(); }
   private:
    explicit Item(std::list<T> &&l) : l_(std::move(l)) {}
    mutable std::list<T> l_;
    friend class CachingList<T>;
  };


  bool IsEmpty() const {
    return full_data_.empty();
  }


  const T &PeekFront() {
    return full_data_.front();
  }


  Item PopFront() {
    if (full_data_.empty())
      throw std::out_of_range("Cannot pop an item from an empty list");
    std::list<T> tmp;
    tmp.splice(tmp.begin(), full_data_, full_data_.begin());
    if (tmp.begin() == prophet_)
      prophet_ = full_data_.begin();
    assert(tmp.size() == 1u);
    return Item(std::move(tmp));
  }


  void Recycle(Item &&elm) {
    empty_data_.splice(empty_data_.end(), elm.l_, elm.l_.begin(), elm.l_.end());
  }


  Item GetEmpty() {
    std::list<T> tmp;
    if (empty_data_.empty()) {
      tmp.emplace_back();
    } else {
      tmp.splice(tmp.begin(), empty_data_, empty_data_.begin());
    }
    return Item(std::move(tmp));
  }


  void PushBack(Item &&elm) {
    if (elm.l_.empty())
      throw std::logic_error("The element is empty - has it been moved out?");

    // If the "prophet" is at the end of the list, we'll need to restore it to point to the
    // beginning of the newly appended item.
    if (prophet_ == full_data_.end() || full_data_.empty())
      prophet_ = elm.l_.begin();
    full_data_.splice(full_data_.end(), elm.l_, elm.l_.begin(), elm.l_.end());
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
    ++prophet_;
  }


  bool CanProphetAdvance() {
    return prophet_ != full_data_.end();
  }

 private:
  std::list<T> full_data_;
  std::list<T> empty_data_;

  // The "prophet" is a separate lookahead pointer into the list, used for peeking into
  // future items without altering the contents of the list.
  typename std::list<T>::iterator prophet_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BUILTIN_CACHING_LIST_H_
