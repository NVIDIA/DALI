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

namespace dali {

vector<ssize_t> calc_sizes(DALIDataType type, TensorShape<-1> shape) {
  vector<ssize_t> args;

  size_t nitems = volume(shape);
  ssize_t item_size = TypeTable::GetTypeInfo(type).size();
  args.push_back(nitems);
  args.push_back(item_size);

  for (int i = 0; i < shape.size(); i++) {
    args.push_back(shape[i]);
  }

  ssize_t s1 = item_size;
  ssize_t s2 = static_cast<ssize_t>(shape[1] * s1);
  args.push_back(s1);
  args.push_back(s2);

  return args;
}


vector<void*> prepare_args(vector<void*> &memory_ptrs,
    vector<ssize_t> &sizes, uint64_t *ptr) {
  // The order and structure of arguments is specified in the numba source code:
  // https://github.com/numba/numba/blob/b1be2f12c83c01f57fe34fab9a9d77334f9baa1d/numba/cuda/dispatcher.py#L325
  vector<void*> args;
  for (size_t i = 0; i < memory_ptrs.size(); i++) {
    args.push_back(reinterpret_cast<void*>(&memory_ptrs[i]));
  }

  for (size_t i = 0; i < sizes.size(); i++) {
    args.push_back(reinterpret_cast<void*>(&sizes[i]));
  }
  args.insert(args.begin()+4, static_cast<void*>(ptr));
  return args;
}


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

  DALI_ENFORCE(out_types_.size() == in_types_.size(),
    "Size of `out_types` should match size of `in_types`."
    "Currently different sizes of `out_types` and `in_types` aren't supported.");

  ins_ndim_ = spec.GetRepeatedArgument<int>("ins_ndim");
  DALI_ENFORCE(ins_ndim_.size() == in_types_.size(), make_string(
    "Size of `ins_dnim` should match size of `in_types`."));
  for (size_t i = 0; i < ins_ndim_.size(); i++) {
    DALI_ENFORCE(ins_ndim_[i] >= 0, make_string(
      "All dimensions should be non negative. Value specified in "
      "`ins_ndim` at index ", i, " is negative."));
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

  output_desc.resize(out_types_.size());
  in_shapes_.resize(ninputs);
  for (int in_id = 0; in_id < ninputs; in_id++) {
    auto& in = ws.Input<GPUBackend>(in_id);
    in_shapes_[in_id] = in.shape();
    DALI_ENFORCE(in_shapes_[in_id].sample_dim() == ins_ndim_[in_id], make_string(
      "Number of dimensions passed in `ins_ndim` at index ", in_id,
      " doesn't match the number of dimensions of the input data: ",
      in_shapes_[in_id].sample_dim(), " != ", ins_ndim_[in_id]));

    for (int i = 1; i < in_shapes_[in_id].num_samples(); i++) {
      DALI_ENFORCE(in_shapes_[in_id][0] == in_shapes_[in_id][i], make_string(
        "Shape of input ", in_id, ", sample at index ", i,
        " doesn't match the shape of first sample: ",
        in_shapes_[in_id][0], " != ", in_shapes_[in_id][i]));
    }

    DALI_ENFORCE(in.type() == in_types_[in_id], make_string(
      "Data type passed in `in_types` at index ", in_id, " doesn't match type of the input data: ",
      in.type(), " != ", in_types_[in_id]));
  }

  for (size_t in_id = 0; in_id < in_types_.size(); in_id++) {
    vector<ssize_t> sizes = calc_sizes(in_types_[in_id], in_shapes_[in_id][0]);
    in_sizes_.push_back(sizes);
    in_memory_ptrs_.push_back({nullptr, nullptr});
  }

  for (size_t out_id = 0; out_id < out_types_.size(); out_id++) {
    // For now we assume that inputs and outputs have the same shapes and types
    vector<ssize_t> sizes = calc_sizes(in_types_[out_id], in_shapes_[out_id][0]);
    out_sizes_.push_back(sizes);
    out_memory_ptrs_.push_back({nullptr, nullptr});
  }

  for (int i = 0; i < noutputs; i++) {
    const auto &in = ws.Input<GPUBackend>(i);
    output_desc[i] = {in.shape(), in.type()};
  }
  return true;
}


template <>
void NumbaFuncImpl<GPUBackend>::RunImpl(Workspace &ws) {
  auto N = ws.Input<GPUBackend>(0).shape().num_samples();
  int ninputs = ws.NumInput();

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

  for (int i = 0; i < N; i++) {
    vector<void*> args;
    for (size_t in_id = 0; in_id < in_types_.size(); in_id++) {
      vector<void*> args_local = prepare_args(
        in_memory_ptrs_[in_id],
        in_sizes_[in_id],
        &in_ptrs[N * in_id + i]);
      args.insert(
        args.end(),
        make_move_iterator(args_local.begin()),
        make_move_iterator(args_local.end()));
    }

    for (size_t out_id = 0; out_id < out_types_.size(); out_id++) {
      vector<void*> args_local = prepare_args(
        out_memory_ptrs_[out_id],
        out_sizes_[out_id],
        &out_ptrs[N * out_id + i]);
      args.insert(
        args.end(),
        make_move_iterator(args_local.begin()),
        make_move_iterator(args_local.end()));
    }

    CUfunction cufunc = (CUfunction)run_fn_;
    CUresult result = cuLaunchKernel(
      cufunc,
      blocks_[0], blocks_[1], blocks_[2],
      threads_per_block_[0], threads_per_block_[1], threads_per_block_[2],
      0,
      ws.stream(),
      static_cast<void**>(args.data()),
      NULL);
    cudaResultCheck(result);
  }
}

DALI_REGISTER_OPERATOR(NumbaFuncImpl, NumbaFuncImpl<GPUBackend>, GPU);

}  // namespace dali

