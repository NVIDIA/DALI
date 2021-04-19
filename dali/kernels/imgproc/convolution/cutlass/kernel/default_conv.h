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
    \brief
      Default kernel-level GEMM definitions combine threadblock-scoped matrix multiply-add with
      the appropriate threadblock-scoped epilogue.

      Note, CUTLASS epilogues universally target row-major outputs. Column-major outputs are
      accommodated by exchanging A and B operands and assuming transposed layouts. Partial
      specializations here choose 'device::GemmTransposed' to implement this functionality.
*/

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_KERNEL_DEFAULT_CONV_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_KERNEL_DEFAULT_CONV_H_

#include "cutlass/cutlass.h"

#include "cutlass/arch/wmma.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/gemm_pipelined.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

#include "dali/kernels/imgproc/convolution/cutlass/kernel/gemm.h"
#include "dali/kernels/imgproc/convolution/cutlass/threadblock/default_conv_mma.h"
#include "dali/kernels/imgproc/convolution/cutlass/threadblock/predicated_tile_iterator.h"



#if defined(CUTLASS_ARCH_WMMA_ENABLED)
#include "cutlass/epilogue/threadblock/default_epilogue_wmma_tensor_op.h"
#endif  // CUTLASS_ARCH_WMMA_ENABLED

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Element type for A matrix operand (input in Gmem)
    typename ElementA,
    /// Element type for A matrix operand for computation
    typename ElementCastA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand (input in Gmem)
    typename ElementB,
    /// Element type for B matrix operand for computation
    typename ElementCastB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Convolution window storage configuration
    typename ConvWindowConfiguration,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator,
    /// If the convolution is computed in the innermost or outer dimension
    bool IsInnerConv = true>
struct DefaultConv {
  using UnderlyingConv =
      DefaultGemm<ElementCastA, LayoutA, kAlignmentA, ElementCastB, LayoutB, kAlignmentB, ElementC,
                  layout::RowMajor, ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape,
                  WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, Stages,
                  SplitKSerial, Operator>;

  using Mma = typename cutlass::gemm::threadblock::SpecializedConvMma<
      ElementA, ElementCastA, LayoutA, kAlignmentA, ElementB, ElementCastB, LayoutB, kAlignmentB,
      ConvWindowConfiguration, ElementAccumulator, LayoutC, OperatorClass, ArchTag,
      ThreadblockShape, WarpShape, InstructionShape, Stages, Operator, false,
      IsInnerConv>::ThreadblockMma;

  /// Define the epilogue
  using Epilogue = typename UnderlyingConv::Epilogue;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::Conv<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_KERNEL_DEFAULT_CONV_H_
