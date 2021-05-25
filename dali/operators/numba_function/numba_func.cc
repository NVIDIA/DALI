// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

DALI_SCHEMA(NumbaFunction)
  .DocStr(R"code(Invokes a njit compiled Numba function.

The run function should be a Python function that can be compiled in Numba ``nopython`` mode.
A function taking a single input and producing a single output should follow the following definition:

.. code-block:: python

    def run_fn(out0, in0)

where ``out0`` and ``in0`` are numpy array views of the input and output tensors.
If the operator is configured to run in batch mode, then the first dimension of the arrays is the sample index.

Note that the function can take at most 6 inputs and 6 outputs.

Additionally, an optional setup function calculating the shape of the output so DALI can allocate memory
for the output with the following definition:

.. code-block:: python

    def setup_fn(outs, ins)

The setup function is invoked once for the whole batch. The first dimension of ``outs``, ``ins`` is the number of
outputs/inputs, respectively. The second dimension is the sample index. For example, the first sample on the second
output can be accessed by ``outs[1][0]``.

If no setup function provided, the output shape and data type will be the same as the input.

.. note::
    This operator is experimental and its API might change without notice.

**Example 1:**

The following example shows a simple setup function which permutes the order of dimensions in the shape.

.. code-block:: python

    def setup_change_out_shape(outs, ins):
        out0 = outs[0]
        in0 = ins[0]
        perm = [1, 0, 2]
        for sample_idx in range(len(out0)):
            for d in range(len(perm)):
                out0[sample_idx][d] = in0[sample_idx][perm[d]]

Since the setup function is running for the whole batch, we need to iterate and permute each sample's shape individually.
For ``shapes = [(10, 20, 30), (20, 10, 30)]`` it will produce output with ``shapes = [(20, 10, 30), (10, 20, 30)]``.

Also lets provide run function:

.. code-block:: python

    def run_fn(out0, in0):
        for i in range(in0.shape[0]):
            for j in range(in0.shape[1]):
                out0[j, i] = in0[i, j]

The run function can work per-sample or per-batch, depending on the ``batch_processing`` argument.

A run function working per-batch may look like this:

.. code-block:: python

    def run_fn(out0_samples, in0_samples):
        for out0, in0 in zip(out0_samples, in0_samples):
            for i in range(in0.shape[0]):
                for j in range(in0.shape[1]):
                    out0[j, i] = in0[i, j]

A run function working per-sample may look like this:

.. code-block:: python

    def run_fn(out0, in0):
        for i in range(in0.shape[0]):
            for j in range(in0.shape[1]):
                out0[j, i] = in0[i, j]
)code")
  .NumInput(1, 6)
  .OutputFn([](const OpSpec &spec) { return spec.GetRepeatedArgument<int>("out_types").size(); })
  .AllowSequences()
  .Unserializable()
  .NoPrune()
  .AllowSequences()
  .SupportVolumetric()
  .AddArg("run_fn", R"code(Function to be invoked.
This function must work in Numba ``nopython`` mode.)code", DALI_PYTHON_OBJECT)
  .AddArg("out_types", R"code(Types of outputs.)code", DALI_DATA_TYPE_VEC)
  .AddArg("in_types", R"code(Types of inputs.)code", DALI_DATA_TYPE_VEC)
  .AddArg("outs_ndim", R"code(Number of dimensions which outputs shapes should have.)code", DALI_INT_VEC)
  .AddArg("ins_ndim", R"code(Number of dimensions which inputs shapes should have.)code", DALI_INT_VEC)
  .AddOptionalArg("setup_fn", R"code(Setup function setting shapes for outputs.
This function is invoked once per batch. Also this function must work in Numba ``nopython`` mode.)code",
                  DALI_PYTHON_OBJECT, nullptr)
  .AddOptionalArg("batch_processing", R"code(Determines whether the function is invoked once per batch or
separately for each sample in the batch.

When ``batch_processing`` is set to ``True``, the function processes the whole batch. It is necessary if the 
function has to perform cross-sample operations and may be beneficial if significant part of the work can 
be reused. For other use cases, specifying False and using per-sample processing function allows the operator 
to process samples in parallel.)code", false);

DALI_SCHEMA(NumbaFuncImpl)
  .DocStr("")
  .NumInput(1, 6)
  .OutputFn([](const OpSpec &spec) { return spec.GetRepeatedArgument<int>("out_types").size(); })
  .MakeInternal()
  .Unserializable()
  .NoPrune()
  .AllowSequences()
  .SupportVolumetric()
  .AddArg("run_fn", R"code(Address of function to be invoked.)code", DALI_INT64)
  .AddArg("out_types", R"code(DALI types of outputs.)code", DALI_DATA_TYPE_VEC)
  .AddArg("in_types", R"code(DALI types of inputs.)code", DALI_DATA_TYPE_VEC)
  .AddArg("outs_ndim", R"code(Number of dimensions which outputs shapes should have.)code", DALI_INT_VEC)
  .AddArg("ins_ndim", R"code(Number of dimensions which inputs shapes should have.)code", DALI_INT_VEC)
  .AddOptionalArg<int>("setup_fn", R"code(Address of setup function setting shapes for outputs.
This function is invoked once per batch.)code", 0)
  .AddOptionalArg("batch_processing", R"code(Determines whether the function is invoked once per batch or
separately for each sample in the batch.)code", false);

template <typename Backend>
NumbaFuncImpl<Backend>::NumbaFuncImpl(const OpSpec &spec) : Base(spec) {
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
bool NumbaFuncImpl<CPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
    const workspace_t<CPUBackend> &ws) {
  int ninputs = ws.NumInput();
  int noutputs = out_types_.size();
  DALI_ENFORCE(in_types_.size() == static_cast<size_t>(ninputs), make_string(
    "Expected ", in_types_.size(), " inputs (basing on `in_types`), but got ", ninputs));
  DALI_ENFORCE(ins_ndim_.size() == static_cast<size_t>(ninputs), make_string(
    "Expected ", ins_ndim_.size(), " inputs (basing on `ins_ndim`), but got ", ninputs));

  output_desc.resize(out_types_.size());
  in_shapes_.resize(ninputs);
  for (int in_id = 0; in_id < ninputs; in_id++) {
    auto& in = ws.InputRef<CPUBackend>(in_id);
    in_shapes_[in_id] = in.shape();
    DALI_ENFORCE(in_shapes_[in_id].sample_dim() == ins_ndim_[in_id], make_string(
      "Number of dimensions passed in `ins_ndim` at index ", in_id,
      " doesn't match the number of dimensions of the input data: ",
      in_shapes_[in_id].sample_dim(), " != ", ins_ndim_[in_id]));
    DALI_ENFORCE(in.type().id() == in_types_[in_id], make_string(
      "Data type passed in `in_types` at index ", in_id, " doesn't match type of the input data: ",
      in.type().id(), " != ", in_types_[in_id]));
  }
  auto N = in_shapes_[0].num_samples();
  input_shape_ptrs_.resize(N * ninputs);
  for (int in_id = 0; in_id < ninputs; in_id++) {
    for (int i = 0; i < N; i++) {
      input_shape_ptrs_[N * in_id + i] =
        reinterpret_cast<uint64_t>(in_shapes_[in_id].tensor_shape_span(i).data());
    }
  }

  if (!setup_fn_) {
    for (int i = 0; i < noutputs; i++) {
      const auto &in = ws.InputRef<CPUBackend>(i);
      output_desc[i] = {in.shape(), in.type()};
    }
    return true;
  }

  out_shapes_.resize(noutputs);
  for (int i = 0; i < noutputs; i++) {
    out_shapes_[i].resize(N, outs_ndim_[i]);
    output_desc[i].type = dali::TypeTable::GetTypeInfo(static_cast<DALIDataType>(out_types_[i]));
  }

  output_shape_ptrs_.resize(N * noutputs);
  for (int out_id = 0; out_id < noutputs; out_id++) {
    for (int i = 0; i < N; i++) {
      output_shape_ptrs_[N * out_id + i] =
        reinterpret_cast<uint64_t>(out_shapes_[out_id].tensor_shape_span(i).data());
    }
  }

  ((void (*)(void*, const void*, int32_t, const void*, const void*, int32_t, int32_t))setup_fn_)(
    output_shape_ptrs_.data(), outs_ndim_.data(), noutputs,
    input_shape_ptrs_.data(), ins_ndim_.data(), ninputs, N);

  for (int out_id = 0; out_id < noutputs; out_id++) {
    output_desc[out_id].shape = out_shapes_[out_id];
    for (int i = 0; i < N; i++) {
      auto out_shape_span = output_desc[out_id].shape.tensor_shape_span(i);
      for (int d = 0; d < outs_ndim_[out_id]; d++) {
        DALI_ENFORCE(out_shape_span[d] >= 0, make_string(
          "Shape of data should be non negative. ",
          "After setup function shape for output number ",
          out_id, " in sample ", i, " at dimension ", d, " is negative."));
      }
    }
  }
  return true;
}

template <>
void NumbaFuncImpl<CPUBackend>::RunImpl(workspace_t<CPUBackend> &ws) {
  auto N = ws.InputRef<CPUBackend>(0).shape().num_samples();

  std::vector<uint64_t> out_ptrs;
  std::vector<uint64_t> in_ptrs;
  out_ptrs.resize(N * out_types_.size());
  in_ptrs.resize(N * in_types_.size());
  for (size_t out_id = 0; out_id < out_types_.size(); out_id++) {
    auto& out = ws.OutputRef<CPUBackend>(out_id);
    for (int i = 0; i < N; i++) {
      out_ptrs[N * out_id + i] = reinterpret_cast<uint64_t>(out[i].raw_mutable_data());
    }
  }
  for (size_t in_id = 0; in_id < in_types_.size(); in_id++) {
    auto& in = ws.InputRef<CPUBackend>(in_id);
    for (int i = 0; i < N; i++) {
      in_ptrs[N * in_id + i] = reinterpret_cast<uint64_t>(in[i].raw_mutable_data());
    }
  }

  if (batch_processing_) {
    ((void (*)(void*, const void*, const void*, int32_t,
      const void*, const void*, const void*, int32_t, int32_t))run_fn_)(
        out_ptrs.data(),
        (setup_fn_ ? output_shape_ptrs_.data() : input_shape_ptrs_.data()),
        outs_ndim_.data(), outs_ndim_.size(),
        in_ptrs.data(), input_shape_ptrs_.data(),
        ins_ndim_.data(), ins_ndim_.size(), N);
    return;
  }

  auto &out = ws.OutputRef<CPUBackend>(0);
  auto out_shape = out.shape();
  auto &tp = ws.GetThreadPool();
  for (int sample_id = 0; sample_id < N; sample_id++) {
    tp.AddWork([&, sample_id](int thread_id) {
      SmallVector<uint64_t, 6> out_ptrs_per_sample;
      SmallVector<uint64_t, 6> out_shapes_per_sample;
      auto& out_shapes_ptrs = setup_fn_ ? output_shape_ptrs_ : input_shape_ptrs_;
      for (size_t out_id = 0; out_id < out_types_.size(); out_id++) {
        out_ptrs_per_sample[out_id] = out_ptrs[N * out_id + sample_id];
        out_shapes_per_sample[out_id] = out_shapes_ptrs[N * out_id + sample_id];
      }
      SmallVector<uint64_t, 6> in_ptrs_per_sample;
      SmallVector<uint64_t, 6> in_shapes_per_sample;
      for (size_t in_id = 0; in_id < in_types_.size(); in_id++) {
        in_ptrs_per_sample[in_id] = in_ptrs[N * in_id + sample_id];
        in_shapes_per_sample[in_id] = input_shape_ptrs_[N * in_id + sample_id];
      }

      ((void (*)(void*, const void*, const void*, int32_t,
      const void*, const void*, const void*, int32_t))run_fn_)(
        out_ptrs_per_sample.data(),
        out_shapes_per_sample.data(),
        outs_ndim_.data(),
        outs_ndim_.size(),
        in_ptrs_per_sample.data(),
        in_shapes_per_sample.data(),
        ins_ndim_.data(),
        ins_ndim_.size());
    }, out_shape.tensor_size(sample_id));
  }
  tp.RunAll();
}

DALI_REGISTER_OPERATOR(NumbaFuncImpl, NumbaFuncImpl<CPUBackend>, CPU);

}  // namespace dali

