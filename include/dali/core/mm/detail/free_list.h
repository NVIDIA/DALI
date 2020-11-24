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

#ifndef DALI_CORE_MM_DETAIL_FREE_LIST_H_
#define DALI_CORE_MM_DETAIL_FREE_LIST_H_

#include <cassert>
#include <utility>
#include "dali/core/util.h"
#include "dali/core/mm/detail/align.h"

namespace dali {
namespace mm {

/**
 * @brief Maintains a list of free memory blocks of uniform size.
 *
 * This object is used for managing a list of perfectly interchangeable memory regions.
 * As such, it does not need to store size of the blocks, since they will, by definition,
 * have equal size and the list doesn need to know what that size is.
 *
 * This is a non-intrusive free list and, as such, it needs to allocate the storage for the block
 * descirptors. The block descirptors are not freed until the object is destroyed.
 *
 * If a block is taken from the free list, it's removed from the main list and stored in an
 * auxiliary list of unused block descirptors. These blocks will be reused when something
 * is added to the list to avoid interaction with the heap.
 */
class uniform_free_list {
 public:
  struct block {
    block *next;
    void *mem;
  };

  ~uniform_free_list() {
    clear();
  }

  void clear() {
    while (head) {
      block *n = head->next;
      delete head;
      head = n;
    }
    while (unused_blocks) {
      block *n = unused_blocks->next;
      delete unused_blocks;
      unused_blocks = n;
    }
  }

  void *get() {
    if (block *blk = head) {
      void *mem = blk->mem;
      head = blk->next;
      blk->next = unused_blocks;
      blk->mem  = nullptr;
      unused_blocks = blk;
      return mem;
    } else {
      return nullptr;
    }
  }

  void put(void *ptr) {
    block *blk = unused_blocks;
    if (blk) {
      unused_blocks = blk->next;
      blk->next = head;
      blk->mem = ptr;
    } else {
      blk = new block{head, ptr};
    }
    head = blk;
  }

 private:
  block *head = nullptr;
  block *unused_blocks = nullptr;
};

/**
 * @brief Maintains a list of free memory blocks of variable size, returning free blocks
 *        with least margin.
 *
 * This object is used for managing a list of variable-sized memory regions.
 * It returns blocks on a best-fit basis. The remaining remaining parts of the block are
 * placed in the list.
 *
 * This list may be of utility when there's a limited range of object sizes to be allocated
 * - ideally, if these are powers of two. It suffers from fragmentation and therefore is not
 * intended for long term usage.
 */
class best_fit_free_list {
 public:
  virtual ~best_fit_free_list() {
    clear();
  }

  void clear() {
    while (head) {
      block *n = head->next;
      delete head;
      head = n;
    }
    while (unused_blocks) {
      block *n = unused_blocks->next;
      delete unused_blocks;
      unused_blocks = n;
    }
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

  void *get(size_t bytes, size_t alignment) {
    block **pbest = nullptr;
    size_t best_fit = (static_cast<size_t>(-1)) >> 1;  // clear MSB
    for (block **pptr = &head; *pptr; pptr = &(*pptr)->next) {
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

  void put(void *ptr, size_t bytes) {
    block *blk = get_block();
    blk->start = static_cast<char*>(ptr);
    blk->end = blk->start + bytes;
    blk->next = head;
    head = blk;
  }

 protected:
  /**
   * @brief Recycle an unused block descirptor or create a new one.
   */
  block *get_block() {
    if (unused_blocks) {
      block *blk = unused_blocks;
      unused_blocks = blk->next;
      blk->next = nullptr;
      return blk;
    } else {
      return new block();
    }
  }

  /**
   * @brief Remove a block from the list and place it in unused_blocks
   *
   * @param ppblock pointer to a block pointer - this automatically updates head
   */
  void remove(block **ppblock) {
    block *b = *ppblock;
    block *next = b->next;
    *ppblock = next;
    b->start = nullptr;
    b->end = nullptr;
    b->next = unused_blocks;
    unused_blocks = b;
  }

  block *head = nullptr;
  block *unused_blocks = nullptr;
};


/**
 * @brief Maintains a list of free memory blocks of variable size, returning free blocks
 *        with least margin.
 *
 * This free list tries to connect a free block to its immediate predecessor and successor.
 * This causes the list to represent true fragmentation - all contiguous free blocks are stored
 * as such.
 *
 * There is an extra cost of going through the list upon placing a memory region in the list,
 * but this should be offset by the list being typically shorter than a non-coalescing one.
 *
 * Free time is increased compared to non-coalescing variant, but allocation time is reduced
 * due to reduced element count.
 */
class coalescing_free_list : public best_fit_free_list {
 public:
  void put(void *ptr, size_t bytes) {
    // find blocks that immediatele precede and succeed the freed block
    block *pred = nullptr;
    block **succ = nullptr;
    for (block **pb = &head; *pb; pb = &(*pb)->next) {
      if ((*pb)->precedes(ptr)) {
        pred = *pb;
        if (succ)
          break;
      }
      if ((*pb)->succeeds(ptr, bytes))  {
        succ = pb;
        if (pred)
          break;
      }
    }
    // found both - glue them and remove one of the blocks
    if (pred && succ) {
      pred->end = (*succ)->end;
      remove(succ);
    } else if (pred) {
      // found preceding block - move its end pointer
      pred->end += bytes;
    } else if (succ) {
      // found following block - move its start pointer
      (*succ)->start -= bytes;
    } else {
      // not found - add a new free block, we'll hopefully coalesce it later
      best_fit_free_list::put(ptr, bytes);
    }
  }
};

}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_DETAIL_FREE_LIST_H_
