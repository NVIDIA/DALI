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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_DEVICE_GEMM_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_DEVICE_GEMM_H_

#include <vector>

#include "cutlass/cutlass.h"

#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/numeric_types.h"

#include "dali/kernels/imgproc/convolution/cutlass/conv_window_configuration.h"
#include "dali/kernels/imgproc/convolution/cutlass/device/default_conv_configuration.h"
#include "dali/kernels/imgproc/convolution/cutlass/kernel/default_conv.h"
#include "dali/kernels/imgproc/convolution/cutlass/kernel/gemm.h"
#include "dali/kernels/imgproc/convolution/cutlass/utility.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

/*! Conv device-level operator. This is an interface to efficient CUTLASS GEMM kernels that may
  be invoked from host code.

  The contributions of this class are:

    1. At compile time, it maps data types and high-level structural parameters onto
       specific CUTLASS components.

    2. At runtime, it maps logical arguments to GEMM problems to kernel parameters.

    3. At runtime, it launches kernels on the device.

  The intent is to provide a convenient mechanism for interacting with most plausible GEMM
  configurations for each supported architecture. Consequently, not all parameters are exposed
  to the top-level interface. Rather, sensible defaults at each level of the CUTLASS hierarchy
  are selected to tradeoff simplicity of the interface with flexibility. We expect
  most configurations to be specified at this level. Applications with more exotic requirements
  may construct their kernels of interest using CUTLASS components at the threadblock, warp,
  and thread levels of abstraction.

  CUTLASS exposes computations using the functor design pattern in which objects compose some
  internal state with an overloaded function call operator. This enables decoupling of
  initialization from execution, possibly reducing overhead during steady state phases of
  application execution.

  CUTLASS device-level operators expose an Arguments structure encompassing each logical
  input to the computation. This is distinct from the kernel-level Params structure pattern
  which contains application-specific precomputed state needed by the device code.

  Example of a CUTLASS GEMM operator implementing the functionality of cuBLAS's SGEMM NN
  is as follows:

    //
    // Instantiate the CUTLASS GEMM operator.
    //

    cutlass::gemm::device::Conv<
      float,
      cutlass::layout::ColumnMajor,
      float,
      cutlass::layout::ColumnMajor,
      float,
      cutlass::layout::ColumnMajor
    > gemm_op;

    //
    // Launch the GEMM operation on the device
    //

    cutlass::Status status = gemm_op({
      {m, n, k},                          // GemmCoord problem_size,
      {A, lda},                           // TensorRef<float, layout::ColumnMajor> ref_In,
      {B, ldb},                           // TensorRef<float, layout::ColumnMajor> ref_Windows,
      {C, ldc},                           // TensorRef<float, layout::ColumnMajor> ref_C,
      {D, ldd},                           // TensorRef<float, layout::ColumnMajor> ref_D,
      {alpha, beta}                       // EpilogueOutputOp::Params epilogue_op_params
    });


  A simplified view of the template is listed below.

    template <
      /// Element type for Input matrix operand
      typename ElementIn,

      /// Layout type for Input matrix operand
      typename LayoutIn,

      /// Element type for window operands
      typename ElementWindow,

      /// Layout type for window operands
      typename LayoutWindow,

      /// Element type for C and D matrix operands
      typename ElementOut,

      /// Layout type for C and D matrix operands
      typename LayoutOut,

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
      int Stages
    >
    class Conv;
*/
template <
    /// Element type for input matrix operand
    typename ElementIn_,
    /// Element type for input matrix operand
    typename ElementCastIn_,
    /// Layout type for input matrix operand
    typename LayoutIn_,
    /// Element type for window operands
    typename ElementWindow_,
    /// Element type for window operands
    typename ElementCastWindow_,
    /// Element type for C and D matrix operands
    typename ElementOut_,
    /// Layout type for C and D matrix operands
    typename LayoutOut_,
    /// Type of convolution
    bool IsInnerConv = true,
    /// Convolution window storage configuration
    typename ConvWindowConfiguration_ = ConvWindowConfiguration<1024, IsInnerConv>,
    /// Element type for internal accumulation
    typename ElementAccumulator_ = ElementOut_,
    /// Operator class tag
    typename OperatorClass_ = arch::OpClassSimt,
    /// Tag indicating architecture to tune for
    typename ArchTag_ = arch::Sm70,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_ = typename DefaultConvConfiguration<
        OperatorClass_, ArchTag_,
        select_A_t<IsInnerConv, ElementIn_, ElementWindow_>,
        select_B_t<IsInnerConv, ElementIn_, ElementWindow_>,
        ElementOut_, ElementAccumulator_>::ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_ = typename DefaultConvConfiguration<
        OperatorClass_, ArchTag_,
        select_A_t<IsInnerConv, ElementIn_, ElementWindow_>,
        select_B_t<IsInnerConv, ElementIn_, ElementWindow_>,
        ElementOut_, ElementAccumulator_>::WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_ = typename DefaultConvConfiguration<
        OperatorClass_, ArchTag_,
        select_A_t<IsInnerConv, ElementIn_, ElementWindow_>,
        select_B_t<IsInnerConv, ElementIn_, ElementWindow_>,
        ElementOut_, ElementAccumulator_>::InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp_ = typename DefaultConvConfiguration<
        OperatorClass_, ArchTag_,
        select_A_t<IsInnerConv, ElementIn_, ElementWindow_>,
        select_B_t<IsInnerConv, ElementIn_, ElementWindow_>,
        ElementOut_, ElementAccumulator_>::EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_ = typename threadblock::GemmIdentityThreadblockSwizzle<>,
    /// Number of stages used in the pipelined mainloop
    int Stages = DefaultConvConfiguration<OperatorClass_, ArchTag_,
                                          select_A_t<IsInnerConv, ElementIn_, ElementWindow_>,
                                          select_B_t<IsInnerConv, ElementIn_, ElementWindow_>,
                                          ElementOut_, ElementAccumulator_>::kStages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA = DefaultConvConfiguration<OperatorClass_, ArchTag_,
                                              select_A_t<IsInnerConv, ElementIn_, ElementWindow_>,
                                              select_B_t<IsInnerConv, ElementIn_, ElementWindow_>,
                                              ElementOut_, ElementAccumulator_>::kAlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB = DefaultConvConfiguration<OperatorClass_, ArchTag_,
                                              select_A_t<IsInnerConv, ElementIn_, ElementWindow_>,
                                              select_B_t<IsInnerConv, ElementIn_, ElementWindow_>,
                                              ElementOut_, ElementAccumulator_>::kAlignmentB,
    /// If true, kernel supports split-K with serial reduction
    bool SplitKSerial = false,
    /// Operation performed by GEMM
    typename Operator_ = typename DefaultConvConfiguration<
        OperatorClass_, ArchTag_,
        select_A_t<IsInnerConv, ElementIn_, ElementWindow_>,
        select_B_t<IsInnerConv, ElementIn_, ElementWindow_>,
        ElementOut_, ElementAccumulator_>::Operator>
class Conv {
 public:
  using ElementIn = ElementIn_;
  using ElementCastIn = ElementCastIn_;
  using LayoutIn = LayoutIn_;
  using ElementWindow = ElementWindow_;
  using ElementCastWindow = ElementCastWindow_;
  using LayoutWindow = layout::RowMajor;  // placeholder
  using ConvWindowConfiguration = ConvWindowConfiguration_;
  using ElementOut = ElementOut_;
  using LayoutOut = LayoutOut_;
  using TensorRefC = TensorRef<ElementOut const, LayoutOut>;
  using TensorRefD = TensorRef<ElementOut, LayoutOut>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Operator = Operator_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static bool const kSplitKSerial = SplitKSerial;
  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;
  static_assert(kSplitKSerial == false, "Only basic options are supported");
  // Set this to 1, we don't split k, instead we iterate over samples
  static int const split_k_slices = 1;
  static int const kAxes = 2;
  static bool const kIsInnerConv = IsInnerConv;
  // static size_t const kSampleParamsSizeof = sizeof(typename ConvKernel::SampleParams);

    /// Define the kernel
  using ConvKernel = typename kernel::DefaultConv<
    select_A_t<IsInnerConv, ElementIn_, ElementWindow_>,
    select_A_t<IsInnerConv, ElementCastIn_, ElementCastWindow_>,
    select_A_t<IsInnerConv, LayoutIn, LayoutWindow>,
    kAlignmentA,
    select_B_t<IsInnerConv, ElementIn_, ElementWindow_>,
    select_B_t<IsInnerConv, ElementCastIn_, ElementCastWindow_>,
    select_B_t<IsInnerConv, LayoutIn, LayoutWindow>,
    kAlignmentB,
    ConvWindowConfiguration,
    ElementOut,
    LayoutOut,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    kStages,
    kSplitKSerial,
    Operator,
    kIsInnerConv
  >::GemmKernel;

  using SampleParams = typename ConvKernel::SampleParams;

  /// Argument structure
  struct SampleArguments {
    //
    // Data members
    //
    Array<int, kAxes> matrix_size;
    int window_size;
    int window_anchor;
    int channels;
    TensorRef<ElementIn const, LayoutIn> ref_In;
    ElementWindow *ref_Window;
    TensorRef<ElementOut const, LayoutOut> ref_C;
    TensorRef<ElementOut, LayoutOut> ref_D;
    typename EpilogueOutputOp::Params epilogue;
    int planes = 1;
    int plane_stride = 0;

    //
    // Methods
    //

    /// Constructs an Arguments structure
    CUTLASS_HOST_DEVICE
    SampleArguments(
        Array<int, kAxes> matrix_size_, int window_size_, int window_anchor_, int channels_,
        TensorRef<ElementIn const, LayoutIn> ref_In_, ElementWindow *ref_Window_,
        TensorRef<ElementOut const, LayoutOut> ref_C_, TensorRef<ElementOut, LayoutOut> ref_D_,
        typename EpilogueOutputOp::Params epilogue_ = typename EpilogueOutputOp::Params(),
        int planes_ = 1, int plane_stride_ = 0)
        : matrix_size(matrix_size_),
          window_size(window_size_),
          window_anchor(window_anchor_),
          channels(channels_),
          ref_In(ref_In_),
          ref_Window(ref_Window_),
          ref_C(ref_C_),
          ref_D(ref_D_),
          epilogue(epilogue_),
          planes(planes_),
          plane_stride(plane_stride_) {}
  };

  struct Arguments {
    SampleParams *device_params_ptr;
    std::vector<SampleArguments> sample_arguments;
  };

 private:
  /// Kernel parameters object
  typename ConvKernel::Params params_;
  typename ConvKernel::HostParams host_params_;

 public:
  /// Constructs the GEMM.
  Conv() {}

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {
    if (!kSplitKSerial && split_k_slices > 1) {
      return Status::kErrorInvalidProblem;
    }
    for (const auto &arg : args.sample_arguments) {
      if (arg.window_size * arg.channels > ConvWindowConfiguration::kMaxWindowSize) {
        return Status::kErrorInvalidProblem;
      }
      Status status = ConvKernel::can_implement(arg.matrix_size, arg.ref_In.non_const_ref(),
                                                arg.ref_C.non_const_ref(), arg.ref_D);
      if (status != Status::kSuccess) {
        return status;
      }
    }

    return Status::kSuccess;
  }

  /// Prepare convolution layout (in host memory) to use required layout
  template <typename T>
  void prepare_window(dali::span<T, ConvWindowConfiguration::kTotalAlignedSize> dst,
                      dali::span<T> src, int num_channels = 1) {
    ConvWindowConfiguration::prepare_window(dst, src, num_channels);
  }

  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const &args, cudaStream_t stream = nullptr) {
    params_.params = args.device_params_ptr;
    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    // Find the biggest grid necessary among the samples,
    // smaller samples skip unused blocks on entry
    int max_m = 0, max_n = 0;
    for (auto &arg : args.sample_arguments) {
      max_m = std::max(max_m, arg.matrix_size[0]);
      max_n = std::max(max_n, arg.matrix_size[1] * arg.channels);
    }
    // The basic threadblock swizzle takes only M and N dims into account here
    GemmCoord max_problem_size(max_m, max_n, 1);

    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
        max_problem_size, {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        split_k_slices);

    assert(grid_shape.k() == 1);

    // Assign the number of samples to gridDim.z to iterate over them
    grid_shape[2] = args.sample_arguments.size();

    // Initialize the Params structure
    for (auto &arg : args.sample_arguments) {
      GemmCoord sample_size = GetProblemSize(arg.matrix_size, arg.channels, kIsInnerConv);
      cutlass::gemm::GemmCoord sample_grid_shape = threadblock_swizzle.get_tiled_shape(
          sample_size, {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
          split_k_slices);
      if (!kIsInnerConv) {
        assert(arg.channels == 1 && "For outer convolution channels should be set to 1.");
      }

      // TODO(klecki): the kernel part uses Params that we create based on Arguments
      // we need to provide them for all samples, for now we just fill a temporary vector
      // and copy it to device, we might try to persist the memory
      host_params_.push_back(typename ConvKernel::SampleParams{
          arg.channels,
          arg.window_anchor,
          sample_size,
          sample_grid_shape,
          arg.ref_In.non_const_ref(),
          {arg.ref_Window, {arg.window_size}},  // build window ref on the fly
          arg.ref_C.non_const_ref(),
          arg.ref_D,
          arg.epilogue,
          arg.planes,
          arg.plane_stride});
    }
    params_.grid_tiled_shape = grid_shape;

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(ConvKernel::kThreadCount, 1, 1);

    cudaError_t result;

    int smem_size = static_cast<int>(sizeof(typename ConvKernel::SharedStorage));
    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(Kernel<ConvKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }

      result = cudaFuncSetAttribute(Kernel<ConvKernel>,
                                    cudaFuncAttributePreferredSharedMemoryCarveout, 100);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    size_t params_sizeof = host_params_.size() * sizeof(SampleParams);
    result = cudaMemcpyAsync(params_.params, host_params_.data(), params_sizeof,
                             cudaMemcpyHostToDevice, stream);

    if (result != cudaSuccess) {
      return Status::kErrorInternal;
    }

    cutlass::Kernel<ConvKernel><<<grid, block, smem_size, stream>>>(params_);

    result = cudaGetLastError();

    return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(Arguments const &args, cudaStream_t stream = nullptr) {
    Status status = initialize(args);

    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }

  GemmCoord GetProblemSize(const Array<int, kAxes> &matrix_size, int channels, bool inner) {
    if (inner) {
      // (m, n, n) where n = width * channels
      return {matrix_size[0], matrix_size[1] * channels, matrix_size[1] * channels};
    } else {
      // m, n, m where n = width * channels
      return {matrix_size[0], matrix_size[1] * channels, matrix_size[0]};
    }
  }
};

////////////////////////////////////////////////////////////////////////////////


}  // namespace device
}  // namespace gemm
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_DEVICE_GEMM_H_
