/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
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
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_KERNEL_GEMM_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_KERNEL_GEMM_H_

#include <vector>

#include "cutlass/cutlass.h"

#include "cutlass/array.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_iterator.h"

#include "dali/kernels/imgproc/convolution/cutlass/utility.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Mma_,                 ///! Threadblock-scoped matrix multiply-accumulate
          typename Epilogue_,            ///! Epilogue
          typename ThreadblockSwizzle_,  ///! Threadblock swizzling function
          bool SplitKSerial  ///! If true, code supporting split-K via serial reduction is enabled.
          >
struct Conv {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static bool const kSplitKSerial = SplitKSerial;

  static int const kIsInnerConv = Mma::kIsInnerConv;

  /// Linear buffer with window values
  using WindowElement = typename Mma::IteratorWindow::Element;
  using WindowLayout = layout::PitchLinear;
  using WindowRef = TensorRef<WindowElement, layout::PitchLinear>;
  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Parameters structure
  struct SampleParams {
    int channels;
    int window_size;
    int window_anchor;
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord sample_grid_tiled_shape;
    typename Mma::IteratorIn::Params params_In;
    typename Mma::IteratorIn::TensorRef ref_In;
    WindowRef ref_conv_Window;
    typename Mma::IteratorWindow::Params params_Window;  ///< This are parameters for iterator
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::TensorRef ref_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::OutputTileIterator::TensorRef ref_D;
    typename OutputOp::Params output_op;
    int gemm_k_size;
    int planes = 1;
    int plane_stride = 0;

    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    SampleParams() : gemm_k_size(0) {}

    CUTLASS_HOST_DEVICE
    SampleParams(int channels, int window_anchor, cutlass::gemm::GemmCoord const &problem_size,
                 cutlass::gemm::GemmCoord const &sample_grid_tiled_shape,
                 typename Mma::IteratorIn::TensorRef ref_In, WindowRef ref_conv_Window,
                 typename Epilogue::OutputTileIterator::TensorRef ref_C,
                 typename Epilogue::OutputTileIterator::TensorRef ref_D,
                 typename OutputOp::Params output_op = typename OutputOp::Params(),
                 int planes = 1, int plane_stride = 0)
        : channels(channels),
          window_size(ref_conv_Window.stride(0)),
          window_anchor(window_anchor),
          problem_size(problem_size),  // actual problem size
          // the grid used for the run (m, n, k), we assume k = 1
          sample_grid_tiled_shape(sample_grid_tiled_shape),
          params_In(ref_In.layout()),
          ref_In(ref_In),
          // do not pass explicit window, we construct it later
          ref_conv_Window(ref_conv_Window),
          params_Window(layout::RowMajor(kIsInnerConv ? problem_size.n() : problem_size.m()),
                        window_size, window_anchor, channels),
          params_C(ref_C.layout()),
          ref_C(ref_C),
          params_D(ref_D.layout()),
          ref_D(ref_D),
          output_op(output_op),
          planes(planes),
          plane_stride(plane_stride) {
      gemm_k_size = calc_gemm_k_size(problem_size, sample_grid_tiled_shape);
    }

    CUTLASS_HOST_DEVICE
    static int calc_gemm_k_size(gemm::GemmCoord const &problem_size,
                                GemmCoord const &grid_tiled_shape) {
      // total tiles that cover the k-dim
      int total_gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
      // We could split the k across grids, but we're using it to iterate over samples
      return total_gemm_k_iterations * Mma::Shape::kK;
    }
  };

  using HostParams = std::vector<SampleParams>;
  struct Params {
    int num_samples = 0;
    SampleParams *params = nullptr;
    cutlass::gemm::GemmCoord grid_tiled_shape;

    CUTLASS_HOST_DEVICE
    Params() {}
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Conv() {}

  /// Determines whether kernel satisfies alignment
  static Status can_implement(Array<int, 2> const &matrix_size,
                              typename Mma::IteratorIn::TensorRef ref_In,
                              typename Epilogue::OutputTileIterator::TensorRef ref_C,
                              typename Epilogue::OutputTileIterator::TensorRef ref_D) {
    static int const kAlignmentIn = Mma::IteratorIn::AccessType::kElements;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    if (!TensorRef_aligned(ref_In, kAlignmentIn)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_C, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_D, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if ((matrix_size[0] % kAlignmentIn) || (matrix_size[1] % kAlignmentIn) ||
        (matrix_size[0] % kAlignmentC) || (matrix_size[1] % kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    return Status::kSuccess;
  }

  ////////////
  /// Copy the window from global mem to smem for matrix bulding lookups
  /// Load from params.ref_conv_Window to shared_storage.main_loop.operand_Window
  CUTLASS_DEVICE
  void transfer_conv_window(SampleParams const &params,
                            typename Mma::TensorRefWindow &window_smem) {
    using WindowShape = layout::PitchLinearShape<kThreadCount, 1>;

    // ThreadMaps define how threads are mapped to a given tile. The PitchLinearStripminedThreadMap
    // stripmines a pitch-linear tile among a given number of threads, first along the contiguous
    // dimension then along the strided dimension.
    using WindowThreadMap = transform::PitchLinearStripminedThreadMap<WindowShape, kThreadCount>;

    // Define the PredicateTileIterator, using TileShape, Element, Layout, and ThreadMap types
    using WindowIteratorGmem =
        transform::threadblock::PredicatedTileIterator<WindowShape, WindowElement, WindowLayout, 0,
                                                       WindowThreadMap>;

    using WindowIteratorSmem =
        transform::threadblock::PredicatedTileIterator<WindowShape, WindowElement, WindowLayout, 0,
                                                       WindowThreadMap>;

    cutlass::Coord<2> window_extent = cutlass::make_Coord(1024, 1);
    int iterations = (1024 + WindowShape::kContiguous - 1) / WindowShape::kContiguous;

    WindowIteratorGmem src_iterator(params.ref_conv_Window.layout(), params.ref_conv_Window.data(),
                                    window_extent, threadIdx.x);
    WindowIteratorSmem dst_iterator(window_smem.layout(), window_smem.data(), window_extent,
                                    threadIdx.x);

    typename WindowIteratorGmem::Fragment fragment;
    fragment.clear();

    src_iterator.load(fragment);
    dst_iterator.store(fragment);
    ++src_iterator;
    ++dst_iterator;

    for (; iterations > 1; --iterations) {
      src_iterator.load(fragment);
      dst_iterator.store(fragment);
      ++src_iterator;
      ++dst_iterator;
    }
    __syncthreads();
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params_vec, SharedStorage &shared_storage) {
    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    // we index into samples via the z coordinate, should match sample_grid_tiled_shape.k()
    int sample_idx = threadblock::RematerializeBlockIdxZ();

    SampleParams params = params_vec.params[sample_idx];

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.sample_grid_tiled_shape);

    for (int plane_batch = 0; plane_batch < params.planes; plane_batch++) {
      // Early exit if CTA is out of range
      if (params.sample_grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
          params.sample_grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
        return;
      }

      // Compute initial location in logical coordinates

      // the offset to the resulting matrix
      cutlass::MatrixCoord tb_offset_C{threadblock_tile_offset.m() * Mma::Shape::kM,
                                       threadblock_tile_offset.n() * Mma::Shape::kN};

      // effective span of the window in the generated matrix
      // (excluding element corresponding to current coordinate - center for default case)
      int left_span = params.window_anchor * params.channels;
      int right_span = (params.window_size - params.window_anchor - 1) * params.channels;

      // We need to start at aligned tile, otherwise tensor ops aren't happy.
      // Take this into account when calculating the non-zero region
      // For right side conv-matrix the non-zero regions starts at (n() - left_span, n()),
      // for the left side it's (m(), m() - left_span)
      int conv_diag_position = kIsInnerConv ? threadblock_tile_offset.n() * Mma::Shape::kN
                                          : threadblock_tile_offset.m() * Mma::Shape::kM;
      constexpr int tile_extent = kIsInnerConv ? Mma::Shape::kN : Mma::Shape::kM;

      // in this row/column we cover a rectangle starting at (diag - left_span), and ending at
      // (diag + tile_extent + right_span)

      // this handles the border, we don't extend the matrix shapes, where the data starts
      int k_skipped_offset = max(0, conv_diag_position - left_span);
      // where we should start on tile boundary
      k_skipped_offset = (k_skipped_offset / Mma::Shape::kK) * Mma::Shape::kK;

      cutlass::MatrixCoord tb_offset_A{threadblock_tile_offset.m() * Mma::Shape::kM,
                                       k_skipped_offset};

      cutlass::MatrixCoord tb_offset_B{k_skipped_offset,
                                       threadblock_tile_offset.n() * Mma::Shape::kN};

      // Problem size is a function of threadblock index in the K dimension
      int problem_size_k =
          min(params.problem_size.k(), (threadblock_tile_offset.k() + 1) * params.gemm_k_size);
      int total_gemm_k_iterations = (problem_size_k + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // where the data ends
      int k_end_offset = min(problem_size_k - 1, conv_diag_position + tile_extent + right_span - 1);
      int k_last_iter = ((k_end_offset + Mma::Shape::kK - 1) / Mma::Shape::kK);

      // this is how many iterations we need if we start at the offset
      int gemm_k_iterations =
          (problem_size_k - k_skipped_offset + Mma::Shape::kK - 1) / Mma::Shape::kK;
      int skip_last_iterations = max(total_gemm_k_iterations - k_last_iter - 1, 0);

      // Compute position within threadblock
      int thread_idx = threadIdx.x;

      auto window_smem = shared_storage.main_loop.operand_Window_ref(params.window_size);

      // Transfer window from GMEM to SMEM
      transfer_conv_window(params, window_smem);

      void *in_data = params.ref_In.data();
      void *window_data = window_smem.data();

      // Construct iterators to A and B operands
      // One is proper GMEM iterator, the other generates the matrix on the fly from conv window
      typename Mma::IteratorA iterator_A(
          select_A<kIsInnerConv>(params.params_In, params.params_Window),
          select_A<kIsInnerConv>(params.ref_In.data(), window_smem.data()),
          {params.problem_size.m(), problem_size_k}, thread_idx, tb_offset_A);

      typename Mma::IteratorB iterator_B(
          select_B<kIsInnerConv>(params.params_In, params.params_Window),
          select_B<kIsInnerConv>(params.ref_In.data(), window_smem.data()),
          {problem_size_k, params.problem_size.n()}, thread_idx, tb_offset_B);

      // Broadcast the warp_id computed by lane 0 to ensure dependent code
      // is compiled as warp-uniform.
      int warp_idx = __shfl_sync(0xffffffffu, threadIdx.x / 32, 0);
      int lane_idx = threadIdx.x % 32;

      //
      // Main loop
      //

      // Construct thread-scoped matrix multiply
      Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

      typename Mma::FragmentC accumulators;

      accumulators.clear();

      if (!kSplitKSerial || gemm_k_iterations > 0) {
          // Compute threadblock-scoped matrix multiply-add
          mma(gemm_k_iterations, skip_last_iterations, accumulators, iterator_A, iterator_B,
              accumulators);
        }

      //
      // Epilogue
      //

      OutputOp output_op(params.output_op);

      //
      // Masked tile iterators constructed from members
      //

      threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.sample_grid_tiled_shape);

      // assume identity swizzle
      MatrixCoord threadblock_offset(threadblock_tile_offset.m() * Mma::Shape::kM,
                                    threadblock_tile_offset.n() * Mma::Shape::kN);

      int block_idx = threadblock_tile_offset.m() +
                        threadblock_tile_offset.n() * params.sample_grid_tiled_shape.m();

      // Tile iterator loading from source tensor.
      typename Epilogue::OutputTileIterator iterator_C(params.params_C, params.ref_C.data(),
                                                      params.problem_size.mn(), thread_idx,
                                                      threadblock_offset);

      // Tile iterator writing to destination tensor.
      typename Epilogue::OutputTileIterator iterator_D(params.params_D, params.ref_D.data(),
                                                      params.problem_size.mn(), thread_idx,
                                                      threadblock_offset);

      Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

      // Execute the epilogue operator to update the destination tensor.
      epilogue(output_op, iterator_D, accumulators, iterator_C);

      // Move to next plane
      params.ref_In.add_pointer_offset(params.plane_stride);
      params.ref_C.add_pointer_offset(params.plane_stride);
      params.ref_D.add_pointer_offset(params.plane_stride);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_KERNEL_GEMM_H_
