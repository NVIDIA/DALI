/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Templates implementing loading of tiles from pitch-linear rank=2 tensors.

    This iterator uses masks to guard out-of-bounds accesses and visits the last "residue" tile
    first, with the objective of minimizing predicate mask updates during steady-state operation.

    A precomputed "Params" object minimizes the amount of state that must be stored in registers,
    and integer addition is used to advance the pointer through memory.
*/

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_THREADBLOCK_PREDICATED_TILE_ITERATOR_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_THREADBLOCK_PREDICATED_TILE_ITERATOR_H_

#include "cutlass/arch/memory.h"

#include "dali/kernels/imgproc/convolution/cutlass/threadblock/predicated_tile_access_iterator.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////
///
/// Generate Convolution Matrix operand on the fly.
/// We always generate it pretending to be Row-Major, when we get redirected to Pitch-linear
/// implementation, the innermost convolution will be right-hand operand (AdvanceRank = 1),
/// (we iterate over strided-rows), and for outer-convolution the AdvanceRank = 0 (we iterate over
/// contiguous column dim).
///
/// PositionPredicatedTileIterator
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
/// Regular tile iterator using a precomputed control structure to minimize register liveness
/// and integer arithmetic.
///
/// Layout is assumed to be invariant at the time the precomputed "Params" object is constructed.
///
/// Base pointer and tensor extents may be specified at the time the iterator is constructed.
/// Subsequently, they are assumed to be immutable.
///
/// Adding a logical coordinate offset may be performed at the time the iterator is constructed.
/// Subsequent additions to logical coordinate offset may be performed but are relatively expensive.
///
/// Visitation order is intended to first visit a "residual" tile that may be partially full in
/// both the advance dimension and the steady-state dimension. This is assumed to be the last
/// tile in the iteration sequence. Advancing an iterator that has just been constructed moves to
/// the first tile that is full in the advance dimension and recomputes predicates. Subsequent
/// accesses may be performed without updating internal predicates and are efficient in terms of
/// live register state and pointer arithmetic instructions.
///
/// To be efficient, this assumes the iterator will be dereferenced and advanced at least once
/// outside any looping structure to minimize integer arithmetic.
///
/// Acceses out of bounds are safe so long as `clear_mask()` is called prior to dereferencing
/// the iterator.
///
///
/// Example:
///
/// An efficient pipeline structure may be constructed as follows:
///
// template <typename Iterator>
// __global__ void kernel(
//   typename Iterator::Params params,
//   typename Iterator::Element *ptr,
//   TensorCoord extent) {
//
//   typename Iterator::Fragment fragment;
//
//   TensorCoord threadblock_offset(0, 0);
//
//   Iterator iter(params, ptr, extent, threadIdx.x, threadblock_offsets);
//
//
//   fragment = *iter;        // load "residue" tile first
//   ++iter;                  // advance to first "steady state" tile and update internal masks
//
//
//   #pragma unroll
//   for (int i = Remaining - 1; i >= 0; --i) {
//
//     f(fragment);
//
//     if (!i) {
//       // light-weight operation to clear masks - subsequent loads become NO-OPs.
//       iter.clear_mask();
//
//     }
//
//     fragment = *iter;      // load tile during "steady state" phase
//     ++iter;                // advance to next tile - lightweight due to steady-state masks
//   }
// }
//
// void host(TensorView<Element, 2, layout::PitchLinear> view) {
//
//   using Iterator = transform::threadblock::PositionPredicatedTileIterator;
//
//   typename Iterator::Params params(view.layout());
//
//   kernel<Iterator>(params, view.data());
// }
//
//
/// This iterator is responsible for generating matrix used for computing convolution
/// of inner or outer axis with provided kernel/window. Matrix is generated on the fly
/// from a window stored in shared memory (should also work with global memory).
/// Let's call this a "kernel matrix".
///
/// Note: The coordinates used here are following the formula for correlation:
/// out[i] = \sum_{k \in [0, window_size)} in[i + k] * window[i + k + anchor]
/// The window is in fact reversed during preparation phase.
///
/// Note that the "kernel matrix" can be generated by applying the convolution with the convolution
/// kernel to an indentity matrix of desired size.
///
/// For inner convolution, we need to handle channels. For simplicity, first let's consider
/// case where channels = 1, let's take window_size = 5 and disregard the border handling.
/// We also assume, that the windows are centered over the inputs, that is:
/// ``anchor = window_size / 2``.
/// For now, we consider a window as having elements of indexes in range:
/// [-window_size / 2, window_size / 2], in this example it's [-2, 2].
///
/// We want to convolve gray image, HWC along 'W' axis. For this we multiply the image
/// from the right by following matrix, placing the convolution window in columns.
/// (Note that the window center is always on the diagonal of the matrix):
///
///  0, -1, -2,  x,  x,  x,  x,  x,
///  1,  0, -1, -2,  x,  x,  x,  x,
///  2,  1,  0, -1, -2,  x,  x,  x,
///  x,  2,  1,  0, -1, -2,  x,  x,
///  x,  x,  2,  1,  0, -1, -2,  x,
///  x,  x,  x,  2,  1,  0, -1, -2,
///  x,  x,  x,  x,  2,  1,  0, -1,
///  x,  x,  x,  x,  x,  2,  1,  0,
///
/// where number stands for particular kernel element, and x for actual 0 value
/// Now, let's consider border handling as "reflect 101". If we extended both image and
/// the generated matrix by the border elements, we would get:
///
/// -2,  x,  x,  x,  x,  x,  x,  x   // -2
/// -1, -2,  x,  x,  x,  x,  x,  x   // -1
///  ------------------------------
///  0, -1, -2,  x,  x,  x,  x,  x,  //  0
///  1,  0, -1, -2,  x,  x,  x,  x,  //  1
///  2,  1,  0, -1, -2,  x,  x,  x,  //  2
///  x,  2,  1,  0, -1, -2,  x,  x,
///  x,  x,  2,  1,  0, -1, -2,  x,
///  x,  x,  x,  2,  1,  0, -1, -2,
///  x,  x,  x,  x,  2,  1,  0, -1,
///  x,  x,  x,  x,  x,  2,  1,  0,
///  ------------------------------
///  x,  x,  x,  x,  x,  x,  2,  1,
///  x,  x,  x,  x,  x,  x,  x,  2,
///
/// Row "-2" of "kernel matrix" would multiply column "-2" of the image, which is equal to column
/// "2" of the image. By the distributive properties of multiplication, we can add the coefficients
/// of row -2 and 2 to get the same result without the need to extend the matrices,
/// thus the matrix will contain a sum of given kernel elements in given positions:
///
///  0,    -1,    -2,     x,     x,     x,     x,    x,
///  1 -1,  0 -2, -1,    -2,     x,     x,     x,    x,
///  2 -2,  1,     0,    -1,    -2,     x,     x,    x,
///  x,     2,     1,     0,    -1,    -2,     x,    x,
///  x,     x,     2,     1,     0,    -1,    -2,    x,
///  x,     x,     x,     2,     1,     0,    -1,   -2 2,
///  x,     x,     x,     x,     2,     1,     0 2, -1 1,
///  x,     x,     x,     x,     x,     2,     1,    0,
///
/// To generate specific coordinate (row, col), we check how far it is from diagonal:
/// dist_diag = row - col <- negative means we're above, that number maps to the base window element
/// we need to use.
/// When handling the borders, we need to check the distance to the top and bottom of the matrix.
/// For the "top" border, we would use elements that are twice the distance to the top from
/// currently loaded "base element", than those that are twice the distance to the bottom,
/// and repeat the process until we go outside of the window size. Similarly for the "bottom"
/// border.
///
/// Examples:
/// (1, 0): dist_diag = 1   -> we load the kernel at coordinate "1".
/// The distance to the top is equal 1 (row coordinate), so moving twice the distance up
/// gives us 1 - 2 * 1 = -1 -> we need to add the kernel at coordinate "-1".
/// Distance to the bottom is 6, and -1 - 2 * 6 = -13 is out of our kernel size so we stop.
/// Same when going to the bottom, 1 + 2 * 6 = 13 - also out.
/// Note that, when the distance is 0, we don't add the additional coefficients in this border mode.
///
/// Making the channel number > 1 will cause additional gaps in the kernel when looking at the
/// columns of "kernel matrix" for example (again without border) for channels = 3,
/// part of the matrix:
///
///  0,  x,  x, -1,  x,  x, -2
///  x,  0,  x,  x, -1,  x,  x
///  x,  x,  0,  x,  x, -1,  x
///  1,  x,  x,  0,  x,  x, -1
///  x,  1,  x,  x,  0,  x,  x
///  x,  x,  1,  x,  x,  0,  x
///  2,  x,  x,  1,  x,  x,  0
///  x,  2,  x,  x,  1,  x,  x
///  x,  x,  2,  x,  x,  1,  x,
///  x,  x,  x,  2,  x,  x,  1,
///  x,  x,  x,  x,  2,  x,  x,
///  x,  x,  x,  x,  x,  2,  x,
///  x,  x,  x,  x,  x,  x,  2
///
/// We need to consider a case, when we're using aligned vector loads through an AlignedArray.
/// Same principles will apply, but we can observe that the window element indexes decrease
/// along the rows. We place the kernel in reverse in memory (with the necessary empty
/// spaces for channels) so we can utilize the vector loads:
///
/// -2, x, x, -1, x, x, 0, x, x, 1, x, x, 2
///
/// In case of outer convolution, the "kernel matrix" is placed on the left
/// and we can ignore the channels - it's the same matrix as for inner convolution but transposed,
/// but when used with aligned vector loads it requires a bit different masking (instead of
/// ignoring first and last row, we need to mask the first and last column).
/// The kernel is placed in the rows, again with the centers (in general case anchors)
/// on the diagonal.
///
///  0,  1,  2,  x,  x,  x,  x,  x,
/// -1,  0,  1,  2,  x,  x,  x,  x,
/// -2, -1,  0,  1,  2,  x,  x,  x,
///  x, -2, -1,  0,  1,  2,  x,  x,
///  x,  x, -2, -1,  0,  1,  2,  x,
///  x,  x,  x, -2, -1,  0,  1,  2,
///  x,  x,  x,  x, -2, -1,  0,  1,
///  x,  x,  x,  x,  x, -2, -1,  0,
///
/// Additionally when using vector loads with border handling for the outer kernel matrix,
/// we need in turns mirrored window and regular window order.
///
/// In general case, instead of the center of the window placed on the diagonal, we have
/// the element specified by the window anchor. By adding the window_anchor value to the indices
/// described earlier, we can convert those window indices
/// from [-window_anchor, window_size - window_anchor - 1] range to [0, window_size - 1] range,
/// where 0 is the leftmost element of the kernel.
///
template <typename Shape, typename Element, typename Layout, int AdvanceRank, typename ThreadMap,
          typename ConvWindowConfiguration, int AccessSize = ThreadMap::kElementsPerAccess>
class PositionPredicatedTileIterator;

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PositionPredicatedTileIterator for pitch-linear data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank, typename ThreadMap_,
          typename ConvWindowConfiguration, int AccessSize>
class PositionPredicatedTileIterator<Shape_, Element_, layout::PitchLinear, AdvanceRank, ThreadMap_,
                                     ConvWindowConfiguration, AccessSize> {
 public:
  static_assert(AdvanceRank == 0 || AdvanceRank == 1,
                "Specialization for pitch-linear iterator may along advance along the "
                "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::PitchLinear;
  static int const kAdvanceRank = AdvanceRank;

  // Advance over strided rows means right hand side operand (we always use row-major)
  static int const kIsInnerConv = kAdvanceRank == 1;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  /// Type used for internal memory accesses
  using AccessType =
      AlignedArray<Element, AccessSize, (AccessSize * sizeof_bits<Element>::value / 8)>;

  /// Underlying iterator to compute the addresses
  using TileAccessIterator =
      PositionPredicatedTileAccessIterator<Shape, Element, Layout, kAdvanceRank,
                                   ThreadMap, AccessType>;

  static int const kAccessesPerVector = TileAccessIterator::kAccessesPerVector;

  /// Fragment object to be loaded or stored
  using Fragment =
      cutlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename TileAccessIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   public:
    friend PositionPredicatedTileIterator;
    int window_size_;
    int window_anchor_;
    int channels_;

   private:
    /// Parameters object
    typename TileAccessIterator::Params params_;

   public:
    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout, int window_size, int window_anchor, int channels)
        : window_size_(window_size),
          window_anchor_(window_anchor),
          channels_(channels),
          params_(layout) {}

    CUTLASS_HOST_DEVICE
    Params() {}
  };

 private:
  /// Internal pointer type permits fast address arithmetic
  using BytePointer = char *;

 private:
  //
  // Data members
  //

  /// Data member to the tile access iterator
  TileAccessIterator address_iterator_;
  Pointer pointer_;
  int window_size_;
  int window_anchor_;
  int channels_ = 1;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to the start of the Window
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : address_iterator_(params.params_, pointer, extent, thread_id, threadblock_offset),
        pointer_(pointer),
        window_size_(params.window_size_),
        window_anchor_(params.window_anchor_),
        channels_(params.channels_) {}

  /// Construct a PositionPredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator(Params const &params,  ///< Precomputed parameters object
                                 Pointer pointer,       ///< Pointer to start of tensor
                                 TensorCoord extent,    ///< Extent of tensor
                                 int thread_id          ///< ID of each participating thread
                                 )
      : PositionPredicatedTileIterator(params, pointer, extent, thread_id, make_Coord(0, 0), 0) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    address_iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator &operator++() {
    if (kAdvanceRank)
      address_iterator_.add_tile_offset({0, 1});
    else
      address_iterator_.add_tile_offset({1, 0});

    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator operator++(int) {
    PositionPredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask() {
    address_iterator_.clear_mask();
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    address_iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) {
    address_iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
    address_iterator_.get_mask(mask);
  }

  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    load_with_byte_offset(frag, pointer_offset * sizeof_bits<Element>::value / 8);
  }

  CUTLASS_DEVICE
  void load_access_elements(AccessType *frag_ptr, int idx) {
    // This calculates the logical coordinate of the beggining of access
    TensorCoord current_coord = address_iterator_.get_current_coord();

    // Distance from diagonal in the dim in which we place window (inner - column, outer - row)
    int dist_diag = 0;
    // Distance from first row/column for inner/outer convolution
    int dist_first = 0;
    // Distance from last row/column for inner/outer convolution
    int dist_last = 0;
    int row = current_coord.strided();
    int col = current_coord.contiguous();
    int height = address_iterator_.get_extent().strided();
    int width = address_iterator_.get_extent().contiguous();
    if (kIsInnerConv) {
      // we generate matrix by placing windows vertically, so the row is the major coordinate
      dist_diag = row - col;  // negative above coordinate
      dist_first = row;
      dist_last = height - 1 - row;
    } else {
      // we generate matrix by placing windows horizontally and there is no channel-based spacing
      dist_diag = col - row;
      dist_first = col;
      dist_last = width - 1 - col;
    }
    // this is the starting element of the window, for inner-conv vector load goes in negative order
    // for default case, it's distance from center, for general it's distance from anchor
    int window_element = dist_diag;  // offset from anchor
    int absolute_window_element = to_absolute(window_element);  // index starting from 0

    bool is_used = any_access_within_window<kIsInnerConv>(absolute_window_element) &&
                   address_iterator_.valid();
    if (is_used) {
      AccessType dst;
      load_vec<kIsInnerConv>(dst, absolute_window_element);
      add_border_reflect_101(dst, absolute_window_element, dist_first, dist_last);
      frag_ptr[idx] = dst;
    } else {
      for (int i = 0; i < AccessSize; i++) {
        frag_ptr[idx][i] = static_cast<Element>(0);
      }
    }
  }

  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, LongIndex byte_offset) {
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {
          int idx = v + kAccessesPerVector * (c + s * ThreadMap::Iterations::kContiguous);
          // This is to mark the iteration number as a compile time constant
          address_iterator_.set_iteration_index(idx);

          load_access_elements(frag_ptr, idx);

          ++address_iterator_;
        }
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_byte_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    store_with_byte_offset(frag, pointer_offset * sizeof_bits<Element>::value / 8);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, LongIndex byte_offset) {
    address_iterator_.set_iteration_index(0);
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {
          int idx = v + kAccessesPerVector * (c + s * ThreadMap::Iterations::kContiguous);

          char *byte_ptr = reinterpret_cast<char *>(address_iterator_.get()) + byte_offset;
          AccessType *access_ptr = reinterpret_cast<AccessType *>(byte_ptr);

          if (address_iterator_.valid()) {
            *access_ptr = frag_ptr[idx];
          }
          ++address_iterator_;
        }
      }
    }
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_byte_offset(frag, 0);
  }

 private:
  /**
   * @brief Fill the dst with elements of lo[offset...AccessSize-1], hi[0...AccessSize - 1 - offset]
   */
  CUTLASS_DEVICE
  void mux(AccessType &dst, const AccessType &lo, const AccessType &hi, int offset) {
    // offset is limited to 0..AccessSize
    offset = ::max(0, ::min(offset, AccessSize));
#pragma unroll
    for (int i = 0; i < AccessSize - offset; i++) {
      dst[i] = static_cast<Element>(lo[i + offset]);
    }
#pragma unroll
    for (int i = AccessSize - offset; i < AccessSize; i++) {
      dst[i] = static_cast<Element>(hi[i - AccessSize + offset]);
    }
  }

  /**
   * @brief Add to the dst elements from lo[offset...AccessSize-1], hi[0...AccessSize - 1 - offset]
   */
  CUTLASS_DEVICE
  void mux_add(AccessType &dst, const AccessType &lo, const AccessType &hi, int offset) {
    // offset is limited to 0..AccessSize
    offset = ::max(0, ::min(offset, AccessSize));
#pragma unroll
    for (int i = 0; i < AccessSize - offset; i++) {
      // There is an issue with half, that prohibits me from doing:
      // dst[i] += static_cast<Element>(lo[i + offset]);
      Element tmp = static_cast<Element>(dst[i]) + static_cast<Element>(lo[i + offset]);
      dst[i] = tmp;
    }
#pragma unroll
    for (int i = AccessSize - offset; i < AccessSize; i++) {
      Element tmp =
          static_cast<Element>(dst[i]) + static_cast<Element>(hi[i - AccessSize + offset]);
      dst[i] = tmp;
    }
  }

  /**
   * @brief Calculate the element for lookup in window that is stored in reversed order
   *
   * This mirroring is channel-aware, we need to calculate what channel the abs_window_element
   * points to, remove that component, mirror the coordinate and revert to the proper channel,
   * otherwise we would swap channels while doing so.
   */
  CUTLASS_DEVICE int get_mirrored_element(int abs_window_element) {
    int current_channel = abs_window_element % Channels();
    current_channel = current_channel < 0 ? current_channel + Channels() : current_channel;
    int non_channel_element = abs_window_element - current_channel;
    return (window_size_ - 1) * Channels() - non_channel_element + current_channel;
  }

  /**
   * @brief Get the offset for loading the window_element, that includes mirroring the coordinates,
   * that is aligned and is before the non-aligned access to `window_element`
   *
   * Uses absolute window coordinates
   */
  template <bool mirrored>
  CUTLASS_DEVICE int get_aligned_window_element(int abs_window_element) {
    // for Inner Convolution the window is in reverse when traversing in contiguous axis
    if (mirrored) {
      abs_window_element = get_mirrored_element(abs_window_element);
    }
    if (abs_window_element >= 0) {
      return (abs_window_element / AccessSize) * AccessSize;
    }
    return ((abs_window_element - (AccessSize - 1)) / AccessSize) * AccessSize;
  }

  /**
   * @brief Get distance from aligned `lo_offset` to current `window_element`
   *
   * `lo_offset` should be already mirrored
   */
  template <bool mirrored>
  CUTLASS_DEVICE int get_distance(int abs_window_element, int lo_offset) {
    if (mirrored) {
      return get_mirrored_element(abs_window_element) - lo_offset;
    } else {
      return abs_window_element - lo_offset;
    }
  }

  struct aligned_offset_data {
    int aligned_element;  /// aligned to multiple of AccessSize window element
    int offset;           /// positive offset to original element from the aligned access
  };

  template <bool mirrored>
  CUTLASS_DEVICE aligned_offset_data get_aligned_offset_data(int window_element) {
    static_assert(!kIsInnerConv || mirrored, "All lookups are mirrored for Inner Conv.");
    int lo_element = get_aligned_window_element<mirrored>(window_element);
    int offset = get_distance<mirrored>(window_element, lo_element);
    return {lo_element, offset};
  }

  /**
   * @brief Load dst with kernel elements starting at `window_element`
   *
   * @tparam mirrored - whether elements should be increasing or not.
   * Note that it's always mirrored=true for Inner conv.
   */
  template <bool mirrored = true>
  CUTLASS_DEVICE void load_vec(AccessType &dst, int abs_window_element) {
    auto aligned_offset = get_aligned_offset_data<mirrored>(abs_window_element);
    constexpr int window_start = ConvWindowConfiguration::template getWindowStart<mirrored>();
    const auto *access_ptr = reinterpret_cast<AccessType const *>(pointer_ + window_start +
                                                                  aligned_offset.aligned_element);
    mux(dst, access_ptr[0], access_ptr[1], aligned_offset.offset);
  }

  /**
   * @brief Load and add kernel elements from abs_window_element - Inner Conv variant
   *
   * In inner conv, we skip first and last row for border handling,
   * as the loads are row-wise, this can be a no-op if mask_first or mask_last is true.
   */
  template <bool mirrored = true, bool IsInnerConv_ = kIsInnerConv>
  CUTLASS_DEVICE std::enable_if_t<IsInnerConv_> add_vec(AccessType &dst, int abs_window_element,
                                                      bool mask_first, bool mask_last) {
    static_assert(mirrored, "All accesses should be mirrored for inner conv.");
    if (mask_first || mask_last) {
      return;
    }
    auto aligned_offset = get_aligned_offset_data<mirrored>(abs_window_element);
    constexpr int window_start = ConvWindowConfiguration::template getWindowStart<mirrored>();
    const auto *access_ptr = reinterpret_cast<AccessType const *>(pointer_ + window_start +
                                                                  aligned_offset.aligned_element);
    mux_add(dst, access_ptr[0], access_ptr[1], aligned_offset.offset);
  }

  /**
   * @brief Load and add kernel elements from abs_window_element - Outer Conv variant
   *
   * In inner conv, we skip first and last column for border handling,
   * so we sometimes need to mask the addition of first or last vector element.
   * As for "vector aligned" loads, the loaded data consist of several elements, we only mask
   * the first or last element of AccessType.
   */
  template <bool mirrored, bool IsInnerConv_ = kIsInnerConv>
  CUTLASS_DEVICE std::enable_if_t<!IsInnerConv_> add_vec(AccessType &dst, int abs_window_element,
                                                         bool mask_first, bool mask_last) {
    // In outer conv we skip columns
    auto aligned_offset = get_aligned_offset_data<mirrored>(abs_window_element);
    constexpr int window_start = ConvWindowConfiguration::template getWindowStart<mirrored>();
    const auto *access_ptr = reinterpret_cast<AccessType const *>(pointer_ + window_start +
                                                                  aligned_offset.aligned_element);
    Element tmp = static_cast<Element>(0);

    if (mask_first) {
      tmp = static_cast<Element>(dst[0]);
    } else if (mask_last) {
      tmp = static_cast<Element>(dst[AccessSize - 1]);
    }

    mux_add(dst, access_ptr[0], access_ptr[1], aligned_offset.offset);

    if (mask_first) {
      dst[0] = static_cast<Element>(tmp);
    } else if (mask_last) {
      dst[AccessSize - 1] = static_cast<Element>(tmp);
    }
  }

  /**
   * @brief Channel information is necessary only for inner convolution, for outer it's always 1
   */
  CUTLASS_DEVICE
  int Channels() {
    return kIsInnerConv ? channels_ : 1;
  }

  /**
   * @brief Whether to skip first row/column in border handling
   *
   * Note: Border reflect 1001 would always return false here
   */
  CUTLASS_DEVICE
  bool mask_first(int dist_first) {
    if (kIsInnerConv) {
      return dist_first < Channels();
    } else {
      return dist_first == 0;
    }
  }

  /**
   * @brief Whether to skip last row/column in border handling
   */
  CUTLASS_DEVICE
  bool mask_last(int dist_last) {
    if (kIsInnerConv) {
      return dist_last < Channels();
    } else {
      return dist_last == AccessSize - 1;
    }
  }

  /**
   * @brief Converts from anchor-relative indexing inside the window to the absolute offset starting
   * from 0.
   */
  CUTLASS_DEVICE int to_absolute(int window_element) {
    return window_element + window_anchor_ * Channels();
  }

  /**
   * @brief Check if [a0, a1] overlaps with [b0, b1]
   *
   * Assumes that a0 <= a1 and b0 <= b1
   */
  CUTLASS_DEVICE bool range_overlap(int a0, int a1, int b0, int b1) {
    return a0 <= b1 && b0 <= a1;
  }

  /**
   * @brief Check if range [a0, a1] overlaps with window span.
   *
   * Assumes that a0 <= a1, window range includes the channel stride.
   */
  CUTLASS_DEVICE bool range_overlap_with_window(int a0, int a1) {
    return range_overlap(a0, a1, 0, window_size_ * Channels() - 1);
  }

  /**
   * @brief Check whether any of the `AccesType` elements to be loaded lies within the kernel
   *
   * Uses absolute window indexing starting from 0
   */
  template <bool mirrored>
  CUTLASS_DEVICE bool any_access_within_window(int abs_window_element) {
    if (mirrored) {
      // We will try to access [abs_window_element, abs_window_element - 1, ...,
      // abs_window_element - AccessSize + 1]
      // So, a range: [abs_window_element - AccessSize + 1, abs_window_element]
      return range_overlap_with_window(abs_window_element - AccessSize + 1, abs_window_element);
    } else {
      // We will try to access [abs_window_element, abs_window_element + 1, ...,
      // abs_window_element + AccessSize - 1]
      // so, a range: [abs_window_element, abs_window_element + AccessSize - 1]
      return range_overlap_with_window(abs_window_element, abs_window_element + AccessSize - 1);
    }
  }

  /**
   * @brief Emulate border reflect 101 and add necessary elements to AccessTypes
   * based on window_elment loaded for current position and distance to first and last row/column
   * for the outer/inner convolution (windows are placed as rows/columns).
   */
  CUTLASS_DEVICE
  void add_border_reflect_101(AccessType &dst, int abs_window_element, int dist_first,
                              int dist_last) {
    // add all negative coordinates, pattern is twice the dist to first, twice the dist to last
    if (kIsInnerConv) {
      // Border handling, eliminate the remainder (channel offset from the position calculation,
      // so it is not repeated by every addition - effectively we would change the channel)
      dist_first = (dist_first / Channels()) * Channels();
      dist_last = (dist_last / Channels()) * Channels();
    }
    // prevent infinite loop for input of size 1, see FillAlignedWindows for details
    if (dist_first == 0 && dist_last == 0) {
      return;
    }
    int neg_abs_element = abs_window_element;
    while (true) {
      neg_abs_element -= 2 * dist_first;
      if (any_access_within_window<true>(neg_abs_element)) {
        add_vec<true>(dst, neg_abs_element, mask_first(dist_first), false);
      } else {
        break;
      }
      neg_abs_element -= 2 * dist_last;
      if (any_access_within_window<kIsInnerConv>(neg_abs_element)) {
        add_vec<kIsInnerConv>(dst, neg_abs_element, false, mask_last(dist_last));
      } else {
        break;
      }
    }
    // add all positive coordinates
    int pos_abs_element = abs_window_element;
    // twice the dist to last, twice the dist to first
    while (true) {
      pos_abs_element += 2 * dist_last;
      if (any_access_within_window<true>(pos_abs_element)) {
        add_vec<true>(dst, pos_abs_element, false, mask_last(dist_last));
      } else {
        break;
      }
      pos_abs_element += 2 * dist_first;
      if (any_access_within_window<kIsInnerConv>(pos_abs_element)) {
        add_vec<kIsInnerConv>(dst, pos_abs_element, mask_first(dist_first), false);
      } else {
        break;
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PositionPredicatedTileIterator for column-major data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank, typename ThreadMap_,
          typename ConvWindowConfiguration, int AccessSize>
class PositionPredicatedTileIterator<Shape_, Element_, layout::ColumnMajor, AdvanceRank, ThreadMap_,
                                     ConvWindowConfiguration, AccessSize> {
 public:
  static_assert(AdvanceRank == 0 || AdvanceRank == 1,
                "Specialization for pitch-linear iterator may along advance along the "
                "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator =
      PositionPredicatedTileIterator<layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, Element,
                                     layout::PitchLinear, (kAdvanceRank == 0 ? 0 : 1), ThreadMap,
                                     ConvWindowConfiguration, AccessSize>;

  using AccessType = typename UnderlyingIterator::AccessType;

  /// Fragment object to be loaded or stored
  using Fragment =
      cutlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PositionPredicatedTileIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:
    CUTLASS_HOST_DEVICE
    Params() {}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    explicit Params(Layout const &layout, int window_size, int window_anchor, int channels)
        : params_(layout::PitchLinear(layout.stride(0)), window_size, window_anchor, channels) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset, and thread ID
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator(Params const &params,          ///< Precomputed parameters object
                         Pointer pointer,                       ///< Pointer to start of tensor
                         TensorCoord extent,                    ///< Extent of tensor
                         int thread_id,                         ///< ID of each participating thread
                         TensorCoord const &threadblock_offset  ///< Initial offset of threadblock
                         )
      : iterator_(params.params_, pointer, layout::PitchLinearCoord(extent.row(), extent.column()),
                  thread_id,
                  layout::PitchLinearCoord(threadblock_offset.row(), threadblock_offset.column())) {
  }

  /// Construct a PositionPredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator(Params const &params,  ///< Precomputed parameters object
                                 Pointer pointer,       ///< Pointer to start of tensor
                                 TensorCoord extent,    ///< Extent of tensor
                                 int thread_id          ///< ID of each participating thread
                                 )
      : PositionPredicatedTileIterator(params, pointer, extent, thread_id, make_Coord(0, 0)) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator operator++(int) {
    PositionPredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask() {
    iterator_.clear_mask();
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) {
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
    iterator_.get_mask(mask);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, LongIndex byte_offset) {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, LongIndex byte_offset) {
    iterator_.store_with_byte_offset(frag, byte_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PositionPredicatedTileIterator for row-major data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank, typename ThreadMap_,
          typename ConvWindowConfiguration, int AccessSize>
class PositionPredicatedTileIterator<Shape_, Element_, layout::RowMajor, AdvanceRank, ThreadMap_,
                                     ConvWindowConfiguration, AccessSize> {
 public:
  static_assert(AdvanceRank == 0 || AdvanceRank == 1,
                "Specialization for pitch-linear iterator may along advance along the "
                "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator =
      PositionPredicatedTileIterator<layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element,
                                     layout::PitchLinear, (kAdvanceRank == 0 ? 1 : 0), ThreadMap,
                                     ConvWindowConfiguration, AccessSize>;

  using AccessType = typename UnderlyingIterator::AccessType;

  /// Fragment object to be loaded or stored
  using Fragment =
      cutlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PositionPredicatedTileIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:
    CUTLASS_HOST_DEVICE
    Params() {}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout, int window_size, int window_anchor, int channels)
        : params_(layout::PitchLinear(layout.stride(0)), window_size, window_anchor, channels) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset, and thread ID
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator(
      Params const &params,                  ///< Precomputed parameters object
      Pointer pointer,                       ///< Pointer to start of tensor
      TensorCoord extent,                    ///< Extent of tensor
      int thread_id,                         ///< ID of each participating thread
      TensorCoord const &threadblock_offset  ///< Initial offset of threadblock
      )
      : iterator_(params.params_, pointer, layout::PitchLinearCoord(extent.column(), extent.row()),
                  thread_id,
                  layout::PitchLinearCoord(threadblock_offset.column(), threadblock_offset.row())) {
  }

  /// Construct a PositionPredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator(Params const &params,  ///< Precomputed parameters object
                                 Pointer pointer,       ///< Pointer to start of tensor
                                 TensorCoord extent,    ///< Extent of tensor
                                 int thread_id          ///< ID of each participating thread
                                 )
      : PositionPredicatedTileIterator(params, pointer, extent, thread_id, make_Coord(0, 0)) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator operator++(int) {
    PositionPredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask() {
    iterator_.clear_mask();
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) {
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
    iterator_.get_mask(mask);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, LongIndex byte_offset) {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, LongIndex byte_offset) {
    iterator_.store_with_byte_offset(frag, byte_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PositionPredicatedTileIterator for interleaved data.  It is mapped
/// to the congruous layout.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///

template <typename Shape_, typename Element_, int AdvanceRank, typename ThreadMap_,
          typename ConvWindowConfiguration, int AccessSize, int InterleavedK>
class PositionPredicatedTileIterator<Shape_, Element_, layout::ColumnMajorInterleaved<InterleavedK>,
                                     AdvanceRank, ThreadMap_, ConvWindowConfiguration, AccessSize> {
 public:
  static_assert(AdvanceRank == 0 || AdvanceRank == 1,
                "Specialization for pitch-linear iterator may along advance along the "
                "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  static int const kInterleavedK = InterleavedK;
  using Layout = layout::ColumnMajorInterleaved<kInterleavedK>;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PositionPredicatedTileIterator<
      layout::PitchLinearShape<Shape::kRow * kInterleavedK, Shape::kColumn / kInterleavedK>,
      Element, layout::PitchLinear, (kAdvanceRank == 0 ? 0 : 1), ThreadMap, ConvWindowConfiguration,
      AccessSize>;

  using AccessType = typename UnderlyingIterator::AccessType;

  /// Fragment object to be loaded or stored
  using Fragment =
      cutlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PositionPredicatedTileIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:
    CUTLASS_HOST_DEVICE
    Params() {}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    explicit Params(Layout const &layout, int window_size, int window_anchor, int channels)
        : params_(layout::PitchLinear(layout.stride(0)), window_size, window_anchor, channels) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : iterator_(
            params.params_, pointer,
            layout::PitchLinearCoord(extent.row() * kInterleavedK, extent.column() / kInterleavedK),
            thread_id,
            layout::PitchLinearCoord(threadblock_offset.row() * kInterleavedK,
                                     threadblock_offset.column() / kInterleavedK)) {}

  /// Construct a PositionPredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator(Params const &params,  ///< Precomputed parameters object
                                 Pointer pointer,       ///< Pointer to start of tensor
                                 TensorCoord extent,    ///< Extent of tensor
                                 int thread_id          ///< ID of each participating thread
                                 )
      : PositionPredicatedTileIterator(params, pointer, extent, thread_id, make_Coord(0, 0)) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator operator++(int) {
    PositionPredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask() {
    iterator_.clear_mask();
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) {
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
    iterator_.get_mask(mask);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PositionPredicatedTileIterator for interleaved-32 data.  It is
/// mapped to the congruous layout.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank, typename ThreadMap_,
          typename ConvWindowConfiguration, int AccessSize, int InterleavedK>
class PositionPredicatedTileIterator<Shape_, Element_, layout::RowMajorInterleaved<InterleavedK>,
                                     AdvanceRank, ThreadMap_, ConvWindowConfiguration, AccessSize> {
 public:
  static_assert(AdvanceRank == 0 || AdvanceRank == 1,
                "Specialization for pitch-linear iterator may along advance along the "
                "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  static int const kInterleavedK = InterleavedK;
  using Layout = layout::RowMajorInterleaved<kInterleavedK>;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PositionPredicatedTileIterator<
      layout::PitchLinearShape<Shape::kColumn * kInterleavedK, Shape::kRow / kInterleavedK>,
      Element, layout::PitchLinear, (kAdvanceRank == 0 ? 1 : 0), ThreadMap, ConvWindowConfiguration,
      AccessSize>;

  using AccessType = typename UnderlyingIterator::AccessType;

  /// Fragment object to be loaded or stored
  using Fragment =
      cutlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PositionPredicatedTileIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:
    CUTLASS_HOST_DEVICE
    Params() {}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    explicit Params(Layout const &layout, int window_size, int window_anchor, int channels)
        : params_(layout::PitchLinear(layout.stride(0)), window_size, window_anchor, channels) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : iterator_(
            params.params_, pointer,
            layout::PitchLinearCoord(extent.column() * kInterleavedK, extent.row() / kInterleavedK),
            thread_id,
            layout::PitchLinearCoord(threadblock_offset.column() * kInterleavedK,
                                     threadblock_offset.row() / kInterleavedK)) {}

  /// Construct a PositionPredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator(Params const &params,  ///< Precomputed parameters object
                                 Pointer pointer,       ///< Pointer to start of tensor
                                 TensorCoord extent,    ///< Extent of tensor
                                 int thread_id          ///< ID of each participating thread
                                 )
      : PositionPredicatedTileIterator(params, pointer, extent, thread_id, make_Coord(0, 0)) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator operator++(int) {
    PositionPredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask() {
    iterator_.clear_mask();
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) {
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
    iterator_.get_mask(mask);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace transform
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_THREADBLOCK_PREDICATED_TILE_ITERATOR_H_
