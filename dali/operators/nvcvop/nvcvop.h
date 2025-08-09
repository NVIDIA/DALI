// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_NVCVOP_NVCVOP_H_
#define DALI_OPERATORS_NVCVOP_NVCVOP_H_

#include <nvcv/DataType.h>
#include <nvcv/BorderType.h>
#include <cvcuda/Types.h>
#include <nvcv/alloc/Allocator.hpp>
#include <cvcuda/Workspace.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorBatch.hpp>
#include <nvcv/ImageBatch.hpp>

#include <optional>
#include <string>
#include <vector>

#include "dali/core/call_at_exit.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/sequence_operator.h"
#include "dali/core/cuda_event_pool.h"


namespace dali::nvcvop {

/**
 * @brief Get the nvcv border mode from name
 *
 * @param border_mode border mode name
 */
NVCVBorderType GetBorderMode(std::string_view border_mode);

/**
 * @brief Get the nvcv interpolation type from name
 *
 * @param interpolation_type interpolation type name
 */
NVCVInterpolationType GetInterpolationType(DALIInterpType interpolation_type);

/**
 * @brief Get nvcv data kind of a given data type
 */
nvcv::DataKind GetDataKind(DALIDataType dtype);

/**
 * @brief Construct a DataType object with a given number of channels and given channel type
 */
nvcv::DataType GetDataType(DALIDataType dtype, int num_channels = 1);

/**
 * @brief Construct a DataType object with a given number of channels and given channel type
 *
 * @tparam T channel type
 */
template <typename T>
nvcv::DataType GetDataType(int num_channels = 1) {
  return GetDataType(TypeTable::GetTypeId<T>(), num_channels);
}


/**
 * @brief Get image format for an image with a given data type and number of channels.
 *
 * The memory layout of the image is assumed to be contiguous, with interleaved, equally-sized channels.
 */
nvcv::ImageFormat GetImageFormat(DALIDataType dtype, int num_channels);

/**
 * @brief Wrap a sample view into an nvcv Image
 *
 * @param sample sample view
 * @param format image format. It needs to match the shape and data type of the sample view
 */
nvcv::Image AsImage(const SampleView<GPUBackend> &sample, const nvcv::ImageFormat &format);

/**
 * @brief Wrap a const sample view into an nvcv Image
 *
 * @param sample sample view
 * @param format image format. It needs to match the shape and data type of the sample view
 */
nvcv::Image AsImage(const ConstSampleView<GPUBackend> &sample, const nvcv::ImageFormat &format);

/**
 * @brief Wrap a DALI tensor as an NVCV Tensor
 *
 * @param tensor DALI tensor
 * @param layout layout of the resulting nvcv::Tensor.
 * If not provided, layout of the DALI tensor is used.
 * @param reshape shape of the resulting nvcv::Tensor.
 * Its volume must match the volume of the original tensor.
 */
nvcv::Tensor AsTensor(const Tensor<GPUBackend> &tensor, TensorLayout layout = "",
                      const std::optional<TensorShape<>> &reshape = std::nullopt);

nvcv::Tensor AsTensor(SampleView<GPUBackend> sample, TensorLayout layout = "",
                      const std::optional<TensorShape<>> &reshape = std::nullopt);

nvcv::Tensor AsTensor(ConstSampleView<GPUBackend> sample, TensorLayout layout = "",
                      const std::optional<TensorShape<>> &reshape = std::nullopt);

nvcv::Tensor AsTensor(void *data, const TensorShape<> &shape, DALIDataType dtype,
                      TensorLayout layout);

/**
 * @brief Allocates an image batch using a dynamic scratchpad.
 * Allocated images have the shape and data type matching the samples in the given TensorList.
 *
 * It can be used to allocate a workspace batch that matches a given input batch.
 *
 * The output images are valid only for the lifetime of the scratchpad.
 */
void AllocateImagesLike(nvcv::ImageBatchVarShape &output, const TensorList<GPUBackend> &t_list,
                        kernels::Scratchpad &scratchpad);

/**
 * @brief Wrap samples from the given TensorList into nvcv Images and push them to an output Image
 * batch The resulting images cannot outlive the TensorList.
 */
void PushImagesToBatch(nvcv::ImageBatchVarShape &batch, const TensorList<GPUBackend> &t_list);


/**
 * @brief Push a range of frames from the input TensorList as samples in the output TensorBatch.
 *
 * The input TensorList is interpreted as sequence of frames where innermost dimensions
 * starting from `first_spatial_dim` are the frames' dimensions.
 *
 * The range of frames is determined by the `starting_sample`, `frame_offset`
 * and `num_frames` arguments.
 * `starting_sample` is an index of the first source sample from the input TensorList. All the samples before that are skipped.
 * `frame_offset` is an index of a first frame in the starting sample to be taken.
 * `num_frames` is the total number of frames that will be pushed to the output TensorBatch.
 *
 * @param batch output TensorBatch
 * @param t_list input TensorList
 * @param layout layout of the output TensorBatch
 */
void PushFramesToBatch(nvcv::TensorBatch &batch, const TensorList<GPUBackend> &t_list,
                       int first_spatial_dim, int64_t starting_sample, int64_t frame_offset,
                       int64_t num_frames, const TensorLayout &layout);


class NVCVOpWorkspace {
 public:
  NVCVOpWorkspace() {
    CUDA_CALL(cudaGetDevice(&device_id_));
    auto &eventPool = CUDAEventPool::instance();
    workspace_.hostMem.ready = eventPool.Get(device_id_).release();
    workspace_.pinnedMem.ready = eventPool.Get(device_id_).release();
    workspace_.cudaMem.ready = eventPool.Get(device_id_).release();
  }

  cvcuda::Workspace Allocate(const cvcuda::WorkspaceRequirements &reqs,
                             kernels::Scratchpad &scratchpad);

  ~NVCVOpWorkspace() {
    CUDA_DTOR_CALL(cudaEventSynchronize(workspace_.hostMem.ready));
    CUDA_DTOR_CALL(cudaEventSynchronize(workspace_.pinnedMem.ready));
    CUDA_DTOR_CALL(cudaEventSynchronize(workspace_.cudaMem.ready));

    auto &eventPool = CUDAEventPool::instance();
    eventPool.Put(CUDAEvent(workspace_.hostMem.ready), device_id_);
    eventPool.Put(CUDAEvent(workspace_.pinnedMem.ready), device_id_);
    eventPool.Put(CUDAEvent(workspace_.cudaMem.ready), device_id_);
  }

 private:
  cvcuda::Workspace workspace_{};
  int device_id_{};
};

/**
 * @brief A base class for the CVCUDA operators.
 * It adds convenience methods to access inputs, outputs and arguments as nvcv types.
 */
template <typename BaseOp>
class NVCVOperator: public BaseOp {
 public:
  using BaseOp::spec_;
  using BaseOp::BaseOp;

  /**
   * @brief Get an operator argument as an nvcv Tensor.
   * This method uses the dynamic scratchpad to allocate the (pinned and device) buffers
   * used by the output tensor.
   *
   * The scratchpad should return its memory in an order that is compatible with ws.stream().
   * The output is valid only for the lifetime of the scratchpad.
   *
   * @tparam T data type of the DALI operator argument
   * @tparam DTYPE expected nvcv Tensor data type
   * @param arg_shape shape of the DALI operator argument
   * @param reshape shape of the resulting Tensor. Volume of this shape must match
   * the volume of the argument (arg_shape).
   */
  template <typename T, int ndim>
  nvcv::Tensor AcquireTensorArgument(Workspace &ws, kernels::Scratchpad &scratchpad,
                                     ArgValue<T, ndim> &arg, const TensorShape<> &arg_shape,
                                     nvcv::DataType dtype = GetDataType<T>(),
                                     TensorLayout layout = "",
                                     const std::optional<TensorShape<>> &reshape = {}) {
    int num_samples = ws.GetInputBatchSize(0);
    int arg_sample_vol = volume(arg_shape);
    arg.Acquire(spec_, ws, num_samples, arg_shape);
    T *staging_buffer =
        scratchpad.AllocatePinned<T>(num_samples * arg_sample_vol, dtype.alignment());
    for (int s = 0; s < num_samples; ++s) {
      std::copy(arg[s].data, arg[s].data + arg_sample_vol,
                staging_buffer + s * arg_sample_vol);
    }
    T *device_buffer = scratchpad.AllocateGPU<T>(num_samples * arg_sample_vol, dtype.alignment());
    MemCopy(device_buffer, staging_buffer, num_samples * arg_sample_vol * sizeof(T), ws.stream());

    TensorShape<> shape_data;
    if (reshape.has_value()) {
      DALI_ENFORCE(arg_sample_vol == volume(*reshape),
                   make_string("Cannot reshape ", arg_shape, " to ", *reshape, "."));
      shape_data = CalcTensorShape(*reshape, num_samples, dtype);
    } else {
      shape_data = CalcTensorShape(arg_shape, num_samples, dtype);
    }

    nvcv::TensorDataStridedCuda::Buffer inBuf;
    inBuf.basePtr = reinterpret_cast<NVCVByte*>(device_buffer);
    inBuf.strides[shape_data.size() - 1] = dtype.strideBytes();
    for (int d = shape_data.size() - 2; d >= 0; --d) {
      inBuf.strides[d] = shape_data[d + 1] * inBuf.strides[d + 1];
    }
    TensorLayout out_layout = layout.empty() ? layout : "N" + layout;
    nvcv::TensorShape shape(shape_data.data(), shape_data.size(),
                            nvcv::TensorLayout(out_layout.c_str()));
    nvcv::TensorDataStridedCuda inData(shape, dtype, inBuf);
    return nvcv::TensorWrapData(inData);
  }

  TensorShape<> CalcTensorShape(const TensorShape<> &arg_shape, int num_samples,
                                const nvcv::DataType &dtype) {
    int ndim = arg_shape.sample_dim();
    int64_t inner_dim = arg_shape[ndim - 1];
    DALI_ENFORCE(dtype.numChannels() == 1 || inner_dim == dtype.numChannels(),
                 make_string("Invalid argument shape. Inner dimension should match "
                 "the number of channels in the data type: ", dtype.numChannels()));
    std::vector<int64_t> shape_data(ndim + 1);
    shape_data[0] = num_samples;
    for (int d = 0; d < ndim; ++d) {
      shape_data[d + 1] = arg_shape[d];
    }
    // If number of channels in the data type is greater than 1,
    // we remove the inner dimension from the DALI shape
    if (dtype.numChannels() > 1) {
      shape_data.pop_back();
    }
    return shape_data;
  }

  /**
   * @brief Get input image batch.
   * This method allocates the ImageBatchVarShape and wraps the input data as nvcv images.
   *
   * The output should be used only within the RunImpl method.
   */
  const nvcv::ImageBatchVarShape &GetInputBatch(Workspace &ws, size_t input_idx) {
    if (inputs_.size() < input_idx + 1) {
      inputs_.resize(input_idx + 1, nullptr);
    }
    const auto &tl_input = ws.Input<GPUBackend>(input_idx);
    auto &input = inputs_[input_idx];
    int curr_cap = input ? input.capacity() : 0;
    if (curr_cap < tl_input.num_samples()) {
      input = nvcv::ImageBatchVarShape(std::max(curr_cap * 2, tl_input.num_samples()));
    }
    PushImagesToBatch(input, tl_input);
    return input;
  }

  /**
   * @brief Get output image batch.
   * This method allocates the ImageBatchVarShape and wraps the input data as nvcv images.
   *
   * The output should be used only within the RunImpl method.
   */
  nvcv::ImageBatchVarShape &GetOutputBatch(Workspace &ws, size_t output_idx) {
    if (outputs_.size() < output_idx + 1) {
      outputs_.resize(output_idx + 1, nullptr);
    }
    const auto &tl_output = ws.Output<GPUBackend>(output_idx);
    auto &output = outputs_[output_idx];
    int curr_cap = output ? output.capacity() : 0;
    if (curr_cap < tl_output.num_samples()) {
      output = nvcv::ImageBatchVarShape(std::max(curr_cap * 2, tl_output.num_samples()));
    }
    PushImagesToBatch(output, tl_output);
    return output;
  }

  void Run(Workspace &ws) override {
    auto atexit = AtScopeExit([this]() { ClearBatches(); });
    BaseOp::Run(ws);
  }


 private:
  void ClearBatches() {
    for (auto &inp : inputs_) {
      inp.clear();
    }

    for (auto &out : outputs_) {
      out.clear();
    }
  }

  std::vector<nvcv::ImageBatchVarShape> inputs_;
  std::vector<nvcv::ImageBatchVarShape> outputs_;
};

template <template<typename> typename BaseOp>
using NVCVSequenceOperator = NVCVOperator<SequenceOperator<GPUBackend, BaseOp>>;

}  // namespace dali::nvcvop

#endif  // DALI_OPERATORS_NVCVOP_NVCVOP_H_
