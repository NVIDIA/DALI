// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_CORE_MM_DETAIL_FREE_LIST_H_
#define DALI_CORE_MM_DETAIL_FREE_LIST_H_

#include <cassert>
#include <utility>
#include <map>
#include <set>
#include <functional>
#include "dali/core/util.h"
#include "dali/core/mm/detail/align.h"
#include "dali/core/mm/detail/aux_alloc.h"
#include "dali/core/mm/detail/aux_collections.h"

namespace dali {
namespace mm {

namespace free_list_impl {

// Deletes a unidirectional list
template <typename Block>
void delete_list(Block *&b) {
  while (b) {
    Block *n = b->next;
    delete b;
    b = n;
  }
}

}  // namespace free_list_impl

/**
 * @brief Maintains a list of free memory blocks of uniform size.
 *
 * This object is used for managing a list of perfectly interchangeable memory regions.
 * As such, it does not need to store size of the blocks, since they will, by definition,
 * have equal size and the list doesn't need to know what that size is.
 *
 * This is a non-intrusive free list and, as such, it needs to allocate the storage for the block
 * descriptors. The block descriptors are not freed until the object is destroyed.
 *
 * If a block is taken from the free list, it's removed from the main list and stored in an
 * auxiliary list of unused block descriptors. These blocks will be reused when something
 * is added to the list to avoid interaction with the heap.
 */
class uniform_free_list {
 public:
  struct block {
    block *next;
    void *mem;
  };

  uniform_free_list() = default;
  uniform_free_list(uniform_free_list &&other) {
    swap(other);
  }

  void swap(uniform_free_list &other) {
    std::swap(head_, other.head_);
    std::swap(unused_blocks_, other.unused_blocks_);
  }

  uniform_free_list &operator=(uniform_free_list &&other) {
    if (this != &other) {
      clear();
      swap(other);
    }
    return *this;
  }

  ~uniform_free_list() {
    clear();
  }

  void clear() {
    free_list_impl::delete_list(head_);
    free_list_impl::delete_list(unused_blocks_);
  }

  void *get() {
    if (block *blk = head_) {
      void *mem = blk->mem;
      head_ = blk->next;
      blk->next = unused_blocks_;
      blk->mem  = nullptr;
      unused_blocks_ = blk;
      return mem;
    } else {
      return nullptr;
    }
  }

  void put(void *ptr) {
    block *blk = unused_blocks_;
    if (blk) {
      unused_blocks_ = blk->next;
      blk->next = head_;
      blk->mem = ptr;
    } else {
      blk = new block{head_, ptr};
    }
    head_ = blk;
  }

 private:
  block *head_ = nullptr;
  block *unused_blocks_ = nullptr;
};

/**
 * @brief Maintains a list of free memory blocks of variable size, returning free blocks
 *        with least margin.
 *
 * This object is used for managing a list of variable-sized memory regions.
 * It returns blocks on a best-fit basis. The remaining parts of the block are placed in the list.
 *
 * This list may be of utility when there's a limited range of object sizes to be allocated
 * - ideally, if these are powers of two. It suffers from fragmentation and therefore is not
 * intended for long term usage.
 */
class best_fit_free_list {
 public:
  best_fit_free_list() = default;
  best_fit_free_list(best_fit_free_list &&other) {
    swap(other);
  }

  void swap(best_fit_free_list &other) {
    std::swap(head_, other.head_);
    std::swap(unused_blocks_, other.unused_blocks_);
  }

  best_fit_free_list &operator=(best_fit_free_list &&other) {
    if (this != &other) {
      clear();
      swap(other);
    }
    return *this;
  }

  virtual ~best_fit_free_list() {
    clear();
  }

  void clear() {
    free_list_impl::delete_list(head_);
    free_list_impl::delete_list(unused_blocks_);
  }

  struct block {
    block *next;
    char  *start, *end;

    size_t fit(size_t requested, size_t alignment) const {
      char *base = static_cast<char *>(start);
      char *aligned = detail::align_ptr(base, alignment);
      return (end - aligned) - requested;  // intentional wrap into huge values
    }

    bool succeeds(const void *ptr, size_t bytes) const {
      return static_cast<const char *>(ptr) + bytes == start;
    }

    bool precedes(const void *ptr) const {
      return static_cast<const char *>(ptr) == end;
    }
  };

  /**
   * @brief Obtains a best-fit block from the list of free blocks.
   *
   * The function allocates smallest possible block. If the alignment requirements
   * are not met, the block pointer is aligned and the size is checked taking into account
   * any necessary padding.
   *
   * If there's padding at the beginning (due to alignment) or at the end (due to the block being
   * larger), new free blocks are created from the padding regions and stored in the list.
   *
   * @return Aligned pointer to a smallest suitable memory block that can fit the required number
   *         of bytes with specified alignment or nullptr, if not available.
   */
  void *get(size_t bytes, size_t alignment) {
    block **pbest = nullptr;
    size_t best_fit = (static_cast<size_t>(-1)) >> 1;  // clear MSB
    for (block **pptr = &head_; *pptr; pptr = &(*pptr)->next) {
      size_t fit = (*pptr)->fit(bytes, alignment);
      if (fit < best_fit) {
        best_fit = fit;
        pbest = pptr;
      }
    }
    if (!pbest)
      return nullptr;
    char *base = static_cast<char *>((*pbest)->start);
    char *aligned = detail::align_ptr(base, alignment);
    bool gap_lo = aligned > base;
    bool gap_hi = aligned + bytes < (*pbest)->end;
    if (gap_lo) {
      // there's space at the beginnning of the block due to alignment
      block *lo = *pbest;
      if (gap_hi) {
        // there's space at the end of the block - we need a new block to describe that
        block *hi = get_block();
        hi->next = lo->next;
        lo->next = hi;

        hi->start = aligned + bytes;
        hi->end = lo->end;
      }
      // the space at the beginning of the block is the difference in alignments
      lo->end = aligned;
    } else {
      if (gap_hi) {
        // there's space at the end of the block - update current block to describe it
        (*pbest)->start = aligned + bytes;
      } else {
        // the block was fully utilized - remove it from the free list
        remove(pbest);
      }
    }
    return aligned;
  }

  /**
   * @brief Places the free block in the list.
   *
   * The block is not merged with adjacent blocks. The fragmentation of the list is permanent.
   * For long term usage, when fragmentation is an issue, use coalescing_free_list or free_tree.
   */
  void put(void *ptr, size_t bytes) {
    block *blk = get_block();
    blk->start = static_cast<char*>(ptr);
    blk->end = blk->start + bytes;
    blk->next = head_;
    head_ = blk;
  }

 protected:
  /**
   * @brief Recycle an unused block descriptor or create a new one.
   */
  block *get_block() {
    if (unused_blocks_) {
      block *blk = unused_blocks_;
      unused_blocks_ = blk->next;
      blk->next = nullptr;
      return blk;
    } else {
      return new block();
    }
  }

  /**
   * @brief Remove a block from the list and place it in unused_blocks_
   *
   * @param ppblock pointer to a block pointer - this automatically updates head_
   */
  void remove(block **ppblock) {
    block *b = *ppblock;
    block *next = b->next;
    *ppblock = next;
    b->start = nullptr;
    b->end = nullptr;
    b->next = unused_blocks_;
    unused_blocks_ = b;
  }

  block *head_ = nullptr;
  block *unused_blocks_ = nullptr;
};


/**
 * @brief Maintains a list of free memory blocks of variable size, returning free blocks
 *        with least margin.
 *
 * This free list is always sorted and tries to connect each newly freed block to its immediate
 * predecessor and successor. This causes the list to represent true fragmentation - all contiguous
 * free blocks are stored as such.
 *
 * There is an extra cost of going through the list upon placing a memory region in the list,
 * but this should be offset by the list being typically shorter than a non-coalescing one.
 *
 * Free time is increased compared to non-coalescing variant, but allocation time is reduced
 * due to reduced element count.
 */
class coalescing_free_list : public best_fit_free_list {
 public:
  coalescing_free_list() = default;
  coalescing_free_list(coalescing_free_list &&other) {
    swap(other);
  }

  coalescing_free_list &operator=(coalescing_free_list &&other) {
    static_cast<best_fit_free_list&>(*this) = std::move(other);
    return *this;
  }

  /**
   * @brief Places the memory block in the tree, joining it with adjacent blocks, if possible.
   */
  void put(void *ptr, size_t bytes) {
    if (bytes == 0)
      return;
    // find blocks that immediately precede and succeed the freed block
    block **pwhere = &head_;
    block *prev = nullptr;

    for (; *pwhere; prev = *pwhere, pwhere = &(*pwhere)->next) {
      assert((*pwhere)->start != ptr && "Free list corruption: address already in the list");
      if ((*pwhere)->start > static_cast<const char *>(ptr))
        break;
    }
    // found both - glue them and remove one of the blocks
    block *next = *pwhere;
    assert((!next || next->start >= static_cast<char*>(ptr) + bytes) &&
          "Free list corruption: current block overlaps with next one.");
    assert((!prev || prev->end <= static_cast<char*>(ptr)) &&
          "Free list corruption: current block overlaps with previous one.");

    if (prev && prev->precedes(ptr)) {
      if (next && next->succeeds(ptr, bytes)) {
        prev->end = next->end;
        remove(&prev->next);
      } else {
        // found preceding block - move its end pointer
        prev->end += bytes;
      }
    } else if (next && next->succeeds(ptr, bytes)) {
      // found following block - move its start pointer
      next->start -= bytes;
    } else {
      block *blk = get_block();
      blk->start = static_cast<char *>(ptr);
      blk->end = blk->start + bytes;
      blk->next = next;
      *pwhere = blk;
    }
  }

  void merge(coalescing_free_list &&with) {
    if (!with.head_)
      return;
    if (!head_) {
      swap(with);
      return;
    }

    block **a = &head_;
    block **b = &with.head_;
    block *new_head = nullptr;

    auto next_block = [&]() {
      return (*a)->start < (*b)->start ? a : b;
    };

    block **src = next_block();
    new_head = *src;
    *src = (*src)->next;
    new_head->next = nullptr;
    block *tail = new_head;
    while (*a && *b) {
      src = next_block();
      block *curr = *src;
      *src = curr->next;  // advance the source
      curr->next = nullptr;

      if (curr->start == tail->end) {
        // coalesce
        tail->end = curr->end;
        // Current block is no longer needed - place it in unused blocks in source list.
        // This avoids growing the destination's unused blocks list indefinitely.
        curr->next = with.unused_blocks_;
        curr->start = curr->end = nullptr;
        with.unused_blocks_ = curr;
      } else {
        // attach current element to the resulting list
        tail->next = curr;
        // move tail
        tail = curr;
      }
    }
    // append whatever's left
    for (block **p : { a, b }) {
      if (*p) {
        if ((*p)->start == tail->end) {
          // coalesce
          tail->end = (*p)->end;
          with.remove(p);
        }
        // the rest of the list cannot be coalesced because all of it is in one list
        // and would have been coalesced anyway.
        tail->next = *p;
        *p = nullptr;
      }
    }
    // the source lists should be empty by now
    assert(head_ == nullptr);
    assert(with.head_ == nullptr);
    head_ = new_head;
  }
};

/**
 * @brief Maintains a list of free memory blocks of variable size, returning free blocks
 *        with least margin.
 *
 * The free_tree yields exactly the same results as coalescing_free_list, but the operations
 * are completed in log(n) time.
 *
 * When there are few elements, the additional constant overhead may favor the use of
 * coalescing_free_list, but for long lifetimes and large number of free blocks free_tree
 * yields superior performance.
 *
 * Merging of the free_trees is accomplished in k*(log(n)+log(k)) time where n is the number
 * of elements already in the list and k is the number of inserted elements.
 *
 * The tree nodes are allocated from a pooling allocator.
 */
class free_tree {
 public:
  void clear() {
    by_addr_.clear();
    by_size_.clear();
  }

  /**
   * @brief Obtains a best-fit block from the tree of free blocks.
   *
   * The function allocates smallest possible block. If the alignment requirements
   * are not met, the block pointer is aligned and the size is checked taking into account
   * any necessary padding.
   *
   * If there's padding at the beginning (due to alignment) or at the end (due to the block being
   * larger), new free blocks are created from the padding regions and stored in the tree.
   *
   * @return Aligned pointer to a smallest suitable memory block that can fit the required number
   *         of bytes with specified alignment or nullptr, if not available.
   */
  void *get(size_t size, size_t alignment) {
    for (auto it = by_size_.lower_bound({ size, nullptr }); it != by_size_.end(); ++it) {
      size_t block_size = it->first;
      char *base = it->second;
      char *aligned = detail::align_ptr(base, alignment);
      size_t front_padding = aligned - base;
      assert(static_cast<ptrdiff_t>(front_padding) >= 0);
      // NOTE: block_size - front_padding >= size  can overflow and fail - meh, unsigned size_t
      if (block_size >= size + front_padding) {
        by_size_.erase(it);
        size_t back_padding = block_size - size - front_padding;
        assert(static_cast<ptrdiff_t>(back_padding) >= 0);
        if (front_padding) {
          by_addr_[base] = front_padding;
          by_size_.insert({front_padding, base});
        } else {
          by_addr_.erase(base);
        }
        if (back_padding) {
          by_addr_.insert({ aligned + size, back_padding });
          by_size_.insert({ back_padding, aligned + size });
        }
        return aligned;
      }
    }
    return nullptr;
  }

  /**
   * @brief Places the memory block in the tree, joining it with adjacent blocks, if possible.
   */
  void put(void *ptr, size_t size) {
    char *addr = static_cast<char *>(ptr);
    if (by_addr_.empty()) {
      by_addr_.insert({ addr, size });
      by_size_.insert({ size, addr });
      return;
    }
    auto next = by_addr_.lower_bound(addr);
    auto prev = next != by_addr_.begin() ? std::prev(next) : next;
    bool join_prev = prev != next && prev->first + prev->second == addr;
    bool join_next = next != by_addr_.end() && next->first == addr + size;

    if (join_prev) {
      by_size_.erase({ prev->second, prev->first });
      prev->second += size;
      if (join_next) {
        by_size_.erase({ next->second, next->first });
        prev->second += next->second;
        by_addr_.erase(next);
      }
      by_size_.insert({ prev->second, prev->first });
    } else {
      if (join_next) {
        by_size_.erase({ next->second, next->first });
        size += next->second;
        by_addr_.erase(next);
      }
      by_addr_.insert({ addr, size });
      by_size_.insert({ size, addr });
    }
  }

  void merge(free_tree &&with) {
    with.by_size_.clear();
    // Erase the source list one by one - this reduces requirements on total auxiliary memory
    // for the maps.
    for (auto it = with.by_addr_.begin(); it != with.by_addr_.end(); it = with.by_addr_.erase(it)) {
      put(it->first, it->second);
    }
  }

 protected:
  detail::pooled_map<char *, size_t, true> by_addr_;
  detail::pooled_set<std::pair<size_t, char *>, true> by_size_;
};

}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_DETAIL_FREE_LIST_H_
