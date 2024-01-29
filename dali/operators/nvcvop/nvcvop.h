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
#include <string>
#include <vector>
#include <nvcv/Tensor.hpp>
#include <nvcv/ImageBatch.hpp>

#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/sequence_operator.h"

namespace dali::nvcvop {

/**
 * @brief Get the nvcv border mode from name
 *
 * @param border_mode border mode name
 */
NVCVBorderType GetBorderMode(const std::string &border_mode);

/**
 * @brief Get nvcv data kind of a given data type
 */
nvcv::DataKind GetDataKind(DALIDataType dtype);

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
nvcv::Image AsImage(SampleView<GPUBackend> sample, const nvcv::ImageFormat &format);

/**
 * @brief Wrap a const sample view into an nvcv Image
 *
 * @param sample sample view
 * @param format image format. It needs to match the shape and data type of the sample view
 */
nvcv::Image AsImage(ConstSampleView<GPUBackend> sample, const nvcv::ImageFormat &format);

/**
 * @brief Allocates an image batch using a dynamic scratchpad.
 * Allocated images have the shape and data type matching the samples in the given TensorList.
 *
 * It can be used to allocate a workspace batch that matches a given input batch.
 */
void AllocateImagesLike(const TensorList<GPUBackend> &t_list,
                        kernels::DynamicScratchpad &scratchpad, nvcv::ImageBatchVarShape &output);

/**
 * @brief Wrap samples from the given TensorList into nvcv Images and push them to an output Image batch
 */
void PushImagesToBatch(const TensorList<GPUBackend> &t_list, nvcv::ImageBatchVarShape &batch);

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
   *  used by the output tensor.
   *
   * @tparam T data type of the DALI operator argument
   * @tparam DTYPE expected nvcv Tensor data type
   * @param arg_shape shape of the DALI operator argument
   */
  template <typename T, NVCVDataType DTYPE>
  nvcv::Tensor AcquireTensorArgument(Workspace &ws, kernels::DynamicScratchpad &scratchpad,
                                     ArgValue<T, 1> &arg, const TensorShape<> &arg_shape) {
    auto dtype = nvcv::DataType(DTYPE);
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
    int ndim = arg_shape.sample_dim();
    auto inner_dim = arg_shape[ndim - 1];
    DALI_ENFORCE(inner_dim % dtype.numChannels() == 0, "Invalid argument shape.");
    std::vector<int64_t> shape_data(ndim);
    shape_data[0] = num_samples;
    for (int d = 0; d < ndim; ++d) {
      shape_data[d + 1] = arg_shape[d];
    }
    if (inner_dim != dtype.numChannels()) {
      shape_data.push_back(inner_dim / dtype.numChannels());
    }
    nvcv::TensorDataStridedCuda::Buffer inBuf;
    inBuf.basePtr = reinterpret_cast<NVCVByte*>(device_buffer);
    inBuf.strides[shape_data.size() - 1] = dtype.strideBytes();
    for (int d = shape_data.size() - 2; d >= 0; --d) {
      inBuf.strides[d] = shape_data[d + 1] * inBuf.strides[d + 1];
    }
    nvcv::TensorShape shape(shape_data.data(), shape_data.size(), nvcv::TensorLayout(""));
    nvcv::TensorDataStridedCuda inData(shape, dtype, inBuf);
    return nvcv::TensorWrapData(inData);
  }

  /**
   * @brief Get input image batch.
   * This method allocates the ImageBatchVarShape and wraps the input data as nvcv images.
   */
  const nvcv::ImageBatchVarShape &GetInputBatch(Workspace &ws, size_t input_idx) {
    if (inputs_.size() < input_idx + 1) {
      inputs_.resize(input_idx + 1, nvcv::ImageBatchVarShape{1});
    }
    const auto &tl_input = ws.Input<GPUBackend>(input_idx);
    auto &input = inputs_[input_idx];
    if (input.capacity() < tl_input.num_samples()) {
      input = nvcv::ImageBatchVarShape(std::max(input.capacity() * 2, tl_input.num_samples()));
    }
    input.clear();
    PushImagesToBatch(tl_input, input);
    return input;
  }

  /**
   * @brief Get output image batch.
   * This method allocates the ImageBatchVarShape and wraps the input data as nvcv images.
   */
  nvcv::ImageBatchVarShape &GetOutputBatch(Workspace &ws, size_t output_idx) {
    if (outputs_.size() < output_idx + 1) {
      outputs_.resize(output_idx + 1, nvcv::ImageBatchVarShape{1});
    }
    const auto &tl_output = ws.Output<GPUBackend>(output_idx);
    auto &output = outputs_[output_idx];
    if (output.capacity() < tl_output.num_samples()) {
      output = nvcv::ImageBatchVarShape(std::max(output.capacity() * 2, tl_output.num_samples()));
    }
    output.clear();
    PushImagesToBatch(tl_output, output);
    return output;
  }

  std::vector<nvcv::ImageBatchVarShape> inputs_;
  std::vector<nvcv::ImageBatchVarShape> outputs_;
};

template <template<typename> typename BaseOp>
using NVCVSequenceOperator = NVCVOperator<SequenceOperator<GPUBackend, BaseOp>>;

}  // namespace dali::nvcvop

#endif  // DALI_OPERATORS_NVCVOP_NVCVOP_H_
