// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include "dali/operators/numba_function/numba_func.h"
#include <vector>
#include "dali/core/cuda_rt_utils.h"
#include "dali/kernels/common/utils.h"

namespace dali {

template <typename GPUBackend>
NumbaFuncImpl<GPUBackend>::NumbaFuncImpl(const OpSpec &spec) : Base(spec) {
  run_fn_ = spec.GetArgument<uint64_t>("run_fn");
  setup_fn_ = spec.GetArgument<uint64_t>("setup_fn");
  batch_processing_ = spec.GetArgument<bool>("batch_processing");
  DALI_ENFORCE(batch_processing_ == false,
    make_string("Currently batch processing for GPU is not supported."));

  out_types_ = spec.GetRepeatedArgument<DALIDataType>("out_types");
  DALI_ENFORCE(out_types_.size() <= 6,
    make_string("Trying to specify ", out_types_.size(), " outputs. "
    "This operator can have at most 6 outputs."));
  in_types_ = spec.GetRepeatedArgument<DALIDataType>("in_types");
  DALI_ENFORCE(in_types_.size() <= 6,
    make_string("Trying to specify ", in_types_.size(), " inputs. "
      "This operator can have at most 6 inputs."));

  outs_ndim_ = spec.GetRepeatedArgument<int>("outs_ndim");
  DALI_ENFORCE(outs_ndim_.size() == out_types_.size(), make_string("Size of `outs_ndim` "
    "should match size of `out_types`."));
  for (size_t i = 0; i < outs_ndim_.size(); i++) {
    DALI_ENFORCE(outs_ndim_[i] >= 0, make_string(
      "All dimensions should be non negative. Value specified in `outs_ndim` at index ",
        i, " is negative."));
  }

  ins_ndim_ = spec.GetRepeatedArgument<int>("ins_ndim");
  DALI_ENFORCE(ins_ndim_.size() == in_types_.size(), make_string(
    "Size of `ins_dnim` should match size of `in_types`."));
  for (size_t i = 0; i < ins_ndim_.size(); i++) {
    DALI_ENFORCE(ins_ndim_[i] >= 0, make_string(
      "All dimensions should be non negative. Value specified in "
      "`ins_ndim` at index ", i, " is negative."));
  }

  if (!setup_fn_) {
    DALI_ENFORCE(out_types_.size() == in_types_.size(), make_string(
      "Size of `out_types` should match size of `in_types` if the custom `setup_fn` function ",
      "is not provided. Provided ", in_types_.size(), " inputs and ", out_types_.size(),
      " outputs."));
  }

  blocks_ = spec.GetRepeatedArgument<int>("blocks");
  DALI_ENFORCE(blocks_.size() == 3, make_string(
    "`blocks` array should contain 3 numbers, while received: ", blocks_.size()));
  for (size_t i = 0; i < blocks_.size(); i++) {
    DALI_ENFORCE(blocks_[i] >= 0, make_string(
      "All dimensions should be positive. Value specified in "
      "`blocks` at index ", i, " is nonpositive: ", blocks_[i]));
  }

  threads_per_block_ = spec.GetRepeatedArgument<int>("threads_per_block");
  DALI_ENFORCE(threads_per_block_.size() == 3, make_string(
    "`threads_per_block` array should contain 3 numbers, while received: ",
    threads_per_block_.size()));
  for (size_t i = 0; i < threads_per_block_.size(); i++) {
    DALI_ENFORCE(threads_per_block_[i] >= 0, make_string(
      "All dimensions should be positive. Value specified in "
      "`blocks` at index ", i, " is nonpositive: ", threads_per_block_[i]));
  }

  int blocks_per_sm;
  CUfunction cu_func = reinterpret_cast<CUfunction>(run_fn_);
  int blockDim = volume(threads_per_block_);
  CUDA_CALL(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, cu_func, blockDim, 0));
  DALI_ENFORCE(blocks_per_sm, "Too many threads per block specified for the Numba provided GPU "
                                "kernel");
  auto number_of_blocks = GetSmCount() * blocks_per_sm;
  if (number_of_blocks > volume(blocks_)) {
    DALI_WARN(make_string("It is recommended that the grid volume: ", volume(blocks_),
                          " for the Numba provided GPU kernel is at least: ",
                          number_of_blocks));
  }
}


template <>
void NumbaFuncImpl<GPUBackend>::OutputsSetupFn(std::vector<OutputDesc> &output_desc,
                                               int noutputs, int ninputs, int nsamples) {
  out_shapes_.resize(noutputs);
  for (int i = 0; i < noutputs; i++) {
    out_shapes_[i].resize(nsamples, outs_ndim_[i]);
    output_desc[i].type = static_cast<DALIDataType>(out_types_[i]);
  }

  input_shape_ptrs_.resize(nsamples * ninputs);
  for (int in_id = 0; in_id < ninputs; in_id++) {
    for (int i = 0; i < nsamples; i++) {
      input_shape_ptrs_[nsamples * in_id + i] =
        reinterpret_cast<uintptr_t>(in_shapes_[in_id].tensor_shape_span(i).data());
    }
  }

  output_shape_ptrs_.resize(nsamples * noutputs);
  for (int out_id = 0; out_id < noutputs; out_id++) {
    for (int i = 0; i < nsamples; i++) {
      output_shape_ptrs_[nsamples * out_id + i] =
        reinterpret_cast<uintptr_t>(out_shapes_[out_id].tensor_shape_span(i).data());
    }
  }

  ((void (*)(void*, const void*, int32_t, const void*, const void*, int32_t, int32_t))setup_fn_)(
    output_shape_ptrs_.data(), outs_ndim_.data(), noutputs,
    input_shape_ptrs_.data(), ins_ndim_.data(), ninputs, nsamples);

  for (int out_id = 0; out_id < noutputs; out_id++) {
    output_desc[out_id].shape = out_shapes_[out_id];
    for (int i = 0; i < nsamples; i++) {
      auto out_shape_span = output_desc[out_id].shape.tensor_shape_span(i);
      for (int d = 0; d < outs_ndim_[out_id]; d++) {
        DALI_ENFORCE(out_shape_span[d] >= 0, make_string(
          "A shape of a tensor cannot contain negative extents. The setup function produced an "
          "invalid shape of output ", out_id, " at sample ", i, ":\n", output_desc[out_id].shape));
      }
    }
  }

  out_strides_.clear();
  out_strides_.reserve(noutputs * nsamples);
  out_arrays_.clear();
  out_arrays_.reserve(noutputs * nsamples);
  for (int out_id = 0; out_id < noutputs; out_id++) {
    for (int i = 0; i < nsamples; i++) {
      out_strides_.push_back(kernels::GetStrides(out_shapes_[out_id][i]));
      out_arrays_.push_back(
        NumbaDevArray(out_shapes_[out_id].tensor_shape_span(i),
                      make_span(out_strides_.back()), out_types_[out_id]));
    }
  }
}


template <>
void NumbaFuncImpl<GPUBackend>::OutputsSetupNoFn(std::vector<OutputDesc> &output_desc,
                                                 int noutputs, int ninputs, int nsamples) {
  assert(ninputs == noutputs);
  out_arrays_ = in_arrays_;

  for (int i = 0; i < noutputs; i++) {
    output_desc[i] = {in_shapes_[i], in_types_[i]};
  }
}

template <>
bool NumbaFuncImpl<GPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
    const Workspace &ws) {
  int ninputs = ws.NumInput();
  int noutputs = out_types_.size();
  DALI_ENFORCE(in_types_.size() == static_cast<size_t>(ninputs), make_string(
    "Expected ", in_types_.size(), " inputs (basing on `in_types`), but got ", ninputs));
  DALI_ENFORCE(ins_ndim_.size() == static_cast<size_t>(ninputs), make_string(
    "Expected ", ins_ndim_.size(), " inputs (basing on `ins_ndim`), but got ", ninputs));

  int64_t N = ws.Input<GPUBackend>(0).shape().num_samples();

  in_shapes_.resize(ninputs);
  in_strides_.clear();
  in_strides_.reserve(ninputs * N);
  in_arrays_.clear();
  in_arrays_.reserve(ninputs * N);
  for (int in_id = 0; in_id < ninputs; in_id++) {
    auto& in = ws.Input<GPUBackend>(in_id);
    in_shapes_[in_id] = in.shape();
    DALI_ENFORCE(in_shapes_[in_id].sample_dim() == ins_ndim_[in_id], make_string(
      "Number of dimensions passed in `ins_ndim` at index ", in_id,
      " doesn't match the number of dimensions of the input data: ",
      in_shapes_[in_id].sample_dim(), " != ", ins_ndim_[in_id]));

    DALI_ENFORCE(in.type() == in_types_[in_id], make_string(
      "Data type passed in `in_types` at index ", in_id, " doesn't match type of the input data: ",
      in.type(), " != ", in_types_[in_id]));
    for (int64_t i = 0; i < N; ++i) {
      in_strides_.push_back(kernels::GetStrides(in.shape()[i]));
      in_arrays_.push_back(
        NumbaDevArray(in.shape().tensor_shape_span(i),
                      make_span(in_strides_.back()), in.type()));
    }
  }

  output_desc.resize(out_types_.size());

  if (!setup_fn_) {
    OutputsSetupNoFn(output_desc, noutputs, ninputs, N);
  } else {
    OutputsSetupFn(output_desc, noutputs, ninputs, N);
  }
  return true;
}


template <>
void NumbaFuncImpl<GPUBackend>::RunImpl(Workspace &ws) {
  auto N = ws.Input<GPUBackend>(0).shape().num_samples();
  int ninputs = ws.NumInput();

  for (size_t out_id = 0; out_id < out_types_.size(); out_id++) {
    auto& out = ws.Output<GPUBackend>(out_id);
    for (int i = 0; i < N; i++) {
      out_arrays_[N * out_id + i].data = out.raw_mutable_tensor(i);
    }
  }
  for (size_t in_id = 0; in_id < in_types_.size(); in_id++) {
    auto& in = ws.Input<GPUBackend>(in_id);
    for (int i = 0; i < N; i++) {
      in_arrays_[N * in_id + i].data = const_cast<void*>(in.raw_tensor(i));
    }
  }

  for (int i = 0; i < N; i++) {
    vector<void*> args;

    for (size_t out_id = 0; out_id < out_types_.size(); out_id++) {
      auto &dev_array = out_arrays_[out_id * N + i];
      dev_array.PushArgs(args);
    }

    for (size_t in_id = 0; in_id < in_types_.size(); in_id++) {
      auto &dev_array = in_arrays_[in_id * N + i];
      dev_array.PushArgs(args);
    }
    CUfunction cu_func = reinterpret_cast<CUfunction>(run_fn_);
    CUDA_CALL(cuLaunchKernel(cu_func,
                             blocks_[0], blocks_[1], blocks_[2],
                             threads_per_block_[0], threads_per_block_[1], threads_per_block_[2],
                             0,
                             ws.stream(),
                             args.data(),
                             nullptr));
  }
}

DALI_REGISTER_OPERATOR(NumbaFuncImpl, NumbaFuncImpl<GPUBackend>, GPU);

}  // namespace dali

