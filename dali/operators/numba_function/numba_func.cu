// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace dali {

template <typename GPUBackend>
NumbaFuncImpl<GPUBackend>::NumbaFuncImpl(const OpSpec &spec) : Base(spec) {
  // run_fn_cuda_ = spec.GetArgument<void*>("run_fn_cuda");
  run_fn_ = spec.GetArgument<uint64_t>("run_fn");
  setup_fn_ = spec.GetArgument<uint64_t>("setup_fn");
  batch_processing_ = spec.GetArgument<bool>("batch_processing");

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
  if (!setup_fn_) {
    DALI_ENFORCE(out_types_.size() == in_types_.size(),
      "Size of `out_types` should match size of `in_types` when `setup_fn` isn't provided.");
  }

  ins_ndim_ = spec.GetRepeatedArgument<int>("ins_ndim");
  DALI_ENFORCE(ins_ndim_.size() == in_types_.size(), make_string(
    "Size of `ins_dnim` should match size of `in_types`."));
  for (size_t i = 0; i < ins_ndim_.size(); i++) {
    DALI_ENFORCE(ins_ndim_[i] >= 0, make_string(
      "All dimensions should be non negative. Value specified in "
      "`ins_ndim` at index ", i, " is negative."));
  }
}

template <>
bool NumbaFuncImpl<GPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
    const workspace_t<GPUBackend> &ws) {
  int ninputs = ws.NumInput();
  int noutputs = out_types_.size();
  DALI_ENFORCE(in_types_.size() == static_cast<size_t>(ninputs), make_string(
    "Expected ", in_types_.size(), " inputs (basing on `in_types`), but got ", ninputs));
  DALI_ENFORCE(ins_ndim_.size() == static_cast<size_t>(ninputs), make_string(
    "Expected ", ins_ndim_.size(), " inputs (basing on `ins_ndim`), but got ", ninputs));

  output_desc.resize(out_types_.size());
  in_shapes_.resize(ninputs);
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
  }
  auto N = in_shapes_[0].num_samples();
  input_shape_ptrs_.resize(N * ninputs);
  for (int in_id = 0; in_id < ninputs; in_id++) {
    for (int i = 0; i < N; i++) {
      input_shape_ptrs_[N * in_id + i] =
        reinterpret_cast<uint64_t>(in_shapes_[in_id].tensor_shape_span(i).data());
    }
  }

  for (int i = 0; i < noutputs; i++) {
    const auto &in = ws.Input<GPUBackend>(i);
    output_desc[i] = {in.shape(), in.type()};
  }
  return true;
}

template <>
void NumbaFuncImpl<GPUBackend>::RunImpl(workspace_t<GPUBackend> &ws) {
  auto N = ws.Input<GPUBackend>(0).shape().num_samples();

  std::vector<uint64_t> out_ptrs;
  std::vector<uint64_t> in_ptrs;
  out_ptrs.resize(N * out_types_.size());
  in_ptrs.resize(N * in_types_.size());
  for (size_t out_id = 0; out_id < out_types_.size(); out_id++) {
    auto& out = ws.Output<GPUBackend>(out_id);
    for (int i = 0; i < N; i++) {
      out_ptrs[N * out_id + i] = reinterpret_cast<uint64_t>(out.raw_mutable_tensor(i));
    }
  }
  for (size_t in_id = 0; in_id < in_types_.size(); in_id++) {
    auto& in = ws.Input<GPUBackend>(in_id);
    for (int i = 0; i < N; i++) {
      in_ptrs[N * in_id + i] = reinterpret_cast<uint64_t>(in.raw_tensor(i));
    }
  }

  // const auto &input_ref = ws.template Input<GPUBackend>(0);
  // auto &output_ref = ws.template Output<GPUBackend>(0);
  // output_ref.SetLayout(input_ref.GetLayout());
  // auto input = view<const float, 2>(input_ref);
  // auto output = view<float, 2>(output_ref);
  // void* p_v = static_cast<void*>(out_ptrs.data());
  std::vector<void*> out_ptrs_v {NULL};
  void** data = NULL;
  dim3 block(10, 10);
  dim3 grid(1, 1);
  // auto fun = *reinterpret_cast<void**>(run_fn_);
  cudaError_t result = cudaLaunchKernel(reinterpret_cast<void*>(run_fn_), grid, block, data, 0, ws.stream());
  // cudaError_t result = cudaLaunchKernel(run_fn_cuda_, grid, block, data, 0, ws.stream());
  printf("Result: %d \n", result);
  // CUresult result = cuLaunchKernel(static_cast<CUfunction>(run_fn_), 1, 1, 1, 1, 1, 1, 0, 0, (void**)(out_ptrs_v.data()), NULL);
  // if (result != 0) {
    // const char *msg = cudaGetErrorString(result);
    // printf("error: %s failed with error %s\n", result, msg);
  // }
}

DALI_REGISTER_OPERATOR(NumbaFuncImpl, NumbaFuncImpl<GPUBackend>, GPU);

}  // namespace dali

