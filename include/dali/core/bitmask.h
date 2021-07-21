// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_BITMASK_H_
#define DALI_CORE_BITMASK_H_

#include <cassert>
#include "dali/core/util.h"
#include "dali/core/small_vector.h"

namespace dali {

/**
 * @brief A vector of bits with a utility for quickly searching for set/cleared bits.
 */
class bitmask {
 public:
  using bit_storage_t = uint64_t;

  static constexpr const int storage_bits = sizeof(bit_storage_t) * 8;
  static constexpr const int storage_bits_log = ilog2(storage_bits);

  static constexpr int bit_idx(ptrdiff_t idx) {
    return idx & (storage_bits - 1);
  }

  static constexpr ptrdiff_t word_idx(ptrdiff_t idx) {
    return idx >> storage_bits_log;
  }

  /**
   * @brief Find the first bit with given value, starting at start
   *
   * @return The index of the first bit found or the size of the bit mask if not found.
   */
  ptrdiff_t find(bool value, ptrdiff_t start = 0) const {
    assert(start >= 0);
    if (start >= size_)
      return size_;
    ptrdiff_t index = word_idx(start);
    int bit = bit_idx(start);
    bit_storage_t flip = 0;
    if (!value)
      flip = ~flip;

    // we're not starting at a storage word boundary - look at the first word first
    if (bit > 0) {
      auto word = storage_[index] ^ flip;
      word >>= bit;  // skip bits before start
      if (word)
        return start + ctz(word);
      index++;
    }

    ptrdiff_t num_words = storage_.size();
    for (; index < num_words; index++) {
      auto word = storage_[index] ^ flip;
      if (word) {  // we've found something...
        return (index << storage_bits_log) | ctz(word);
      }
    }

    return size_;
  }

  /**
   * @brief Fill a range of bits with given value
   */
  void fill(ptrdiff_t start, ptrdiff_t end, bool value) {
    if (start >= end)
      return;
    assert(start >= 0 && start <= ssize());
    assert(end   >= 0 && end   <= ssize());
    ptrdiff_t start_idx = word_idx(start);
    ptrdiff_t end_idx = word_idx(end);
    bit_storage_t ones = ~bit_storage_t(0);
    bit_storage_t start_mask = ones;
    bit_storage_t end_mask = 0;
    if (bit_idx(start)) {
        start_mask <<= bit_idx(start);
    }
    if (bit_idx(end)) {
        end_mask = ones >> (storage_bits - bit_idx(end));
    }
    if (start_idx == end_idx) {
        start_mask &= end_mask;
    }
    ptrdiff_t idx = start_idx;
    if (value)
        storage_[idx++] |= start_mask;
    else
        storage_[idx++] &= ~start_mask;
    if (value) {
        for (; idx < end_idx; idx++)
            storage_[idx] = ones;
    } else {
        for (; idx < end_idx; idx++)
            storage_[idx] = 0;
    }
    if (end_mask && start_idx != end_idx) {
      if (value)
        storage_[end_idx] |= end_mask;
      else
        storage_[end_idx] &= ~end_mask;
    }
  }

  /**
   * @brief Fill the entire mask with given value.
   */
  void fill(bool value) {
    size_t s = size();
    clear();
    resize(s, value);
  }

  /**
   * @brief A pseudoreference to a single bit in the mask
   */
  struct bitref {
    bitref &operator=(bool value) {
      if (value)
        *storage |= mask;
      else
        *storage &= ~mask;
      return *this;
    }
    bitref &operator|=(bool value) {
      if (value)
        *storage |= mask;
      return *this;
    }
    bitref &operator&=(bool value) {
      if (!value)
        *storage &= ~mask;
      return *this;
    }
    bitref &operator^=(bool value) {
      if (value)
        *storage ^= mask;
      return *this;
    }
    constexpr operator bool() const {
      return *storage & mask;
    }

   private:
    friend class bitmask;
    constexpr bitref(bit_storage_t *array, ptrdiff_t index)
    : storage(&array[word_idx(index)])
    , mask(bit_storage_t(1) << bit_idx(index)) {}

    bit_storage_t *storage;
    bit_storage_t mask;
  };

  using reference_type = bitref;
  using value_type = bool;

  void push_back(bool value) {
    size_t req_words = (size_ + storage_bits) >> storage_bits_log;
    if (req_words > storage_.size())
      storage_.push_back(value ? 1 : 0);  // put LSB in the storage vector
    else
      (*this)[size_] = value;  // set the last bit
    size_++;
  }

  void pop_back() {
    assert(!empty());
    (*this)[--size_] = 0;
    if (bit_idx(size_) == 0)
      storage_.pop_back();
  }

  constexpr bool empty() const noexcept { return size_ == 0; }

  void reserve(size_t capacity) {
    storage_.reserve((capacity + storage_bits - 1) >> storage_bits_log);
  }

  size_t capacity() const noexcept {
    return storage_.capacity() * storage_bits;
  }

  void resize(size_t new_size, bool value = false) {
    if (new_size == size())
      return;
    bit_storage_t ones = ~bit_storage_t(0);
    bit_storage_t fill = value ? ones : 0;
    size_t prev_words = storage_.size();
    storage_.resize((new_size + storage_bits - 1) >> storage_bits_log, fill);
    if (storage_.size() == prev_words && new_size > size()) {
      if (value) {
        storage_.back() |= ones << bit_idx(size_);
      }
    }
    if (bit_idx(new_size)) {
      bit_storage_t mask = ones;
      mask = (mask << bit_idx(new_size));  // remove leading ones
      mask = ~mask;  // flip values to keep only the leading ones
      storage_.back() &= mask;
    }
    size_ = new_size;
  }

  void clear() {
    storage_.clear();
    size_ = 0;
  }

  bit_storage_t *data() noexcept {
    return storage_.data();
  }

  const bit_storage_t *data() const noexcept {
    return storage_.data();
  }

  constexpr size_t size() const noexcept {
    return size_;
  }
  constexpr ssize_t ssize() const noexcept {
    return size_;
  }

  bitref operator[](ptrdiff_t index) {
    return { storage_.data(), index };
  }

  bool operator[](const ptrdiff_t index) const {
    bit_storage_t storage = storage_[word_idx(index)];
    bit_storage_t mask = bit_storage_t(1) << bit_idx(index);
    return storage & mask;
  }

  void append(const bitmask &other) {
    if (empty()) {
      *this = other;
      return;
    }

    int first_free_bit_idx = bit_idx(size_);
    if (first_free_bit_idx == 0) {
      size_t num_words = storage_.size();
      storage_.resize(num_words + other.storage_.size());
      for (size_t i = 0; i < other.storage_.size(); i++)
        storage_[i + num_words] = other.storage_[i];
    } else {
      ptrdiff_t idx = size_;
      ptrdiff_t widx = word_idx(idx);
      ptrdiff_t total_words = (size_ + other.size_ + storage_bits - 1) >> storage_bits_log;
      storage_.resize(total_words);
      int lshift = first_free_bit_idx;
      int rshift = storage_bits - lshift;

      ptrdiff_t i = 0;
      ptrdiff_t other_words = other.storage_.size();
      bit_storage_t prev = storage_[widx];

      for (; i < other_words; i++) {
        storage_[widx++] = prev | (other.storage_[i] << lshift);
        prev = other.storage_[i] >> rshift;
      }
      if (widx < total_words) {
        storage_[widx] = prev;
      }
    }
    size_ += other.size_;
  }

 private:
  // vector of bit storage words - the bits that are outside of the mask are 0
  SmallVector<bit_storage_t, 1> storage_;
  // number of bits in the mask
  ptrdiff_t size_ = 0;
};

}  // namespace dali

#endif  // DALI_CORE_BITMASK_H_
