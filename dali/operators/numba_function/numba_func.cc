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

DALI_SCHEMA(NumbaFunc)
  .DocStr(R"code()code")
  .NumInput(1, 6)
  .OutputFn([](const OpSpec &spec) { return spec.GetRepeatedArgument<int>("out_types").size(); })
  .AllowSequences()
  .Unserializable()
  .AddArg("run_fn", R"code()code", DALI_INT64)
  .AddArg("out_types", R"code()code", DALI_INT_VEC)
  .AddArg("in_types", R"code()code", DALI_INT_VEC)
  .AddArg("outs_ndim", R"code()code", DALI_INT_VEC)
  .AddArg("ins_ndim", R"code()code", DALI_INT_VEC)
  .AddOptionalArg<int>("setup_fn", R"code()code", 0)
  .AddOptionalArg("batch_processing", R"code()code", false);

DALI_SCHEMA(NumbaFuncImpl)
  .DocStr(R"code(Invokes a compiled Numba function passed as a pointer.

The run function should be a Numba C callback function (annotated with cfunc). This function is run on a 
per-sample basis and should have the following definition:

.. code-block:: python

    @cfunc(run_fn_sig([out_numba_types], [in_numba_types]), nopython=True)
    def callback_run_func(out1_ptr, out1_shape_ptr, out1_shape_ndim, in1_ptr, in1_shape_ptr, in1_shape_ndim)

``out_numba_types`` and ``in_numba_types`` refer to the numba data types (numba.types) 
of the output and input, respectively.

Additionally, an optional setup function calculating the shape of the output so DALI can allocate memory 
for the output with the following definition:

.. code-block:: python

    @cfunc(setup_fn_sig(num_outputs, num_inputs), nopython=True)
    def callback_setup_func(out1_shape_ptr, out1_ndim, out_dtype_ptr, in1_shape_ptr, in1_ndim, in1_dtype, num_samples)

The setup function is invoked once for the whole batch.

If no setup function provided, the output shape and data type will be the same as the input.

.. note::
    This operator is experimental and its API might change without notice.

**Example 1:**

The following example shows a simple setup function which permutes the order of dimensions in the shape and sets the output type to int32

.. code-block:: python

    dali_int32 = int(dali_types.INT32)
    @cfunc(dali_numba.setup_fn_sig(1, 1), nopython=True)
    def setup_change_out_shape(out_shape_ptr, out1_ndim, out_dtype, in_shape_ptr, in1_ndim, in_dtype, num_samples):
        in_shapes = carray(in_shape_ptr, (num_samples, in1_ndim))
        out_shapes = carray(out_shape_ptr, (num_samples, out1_ndim))
        perm = [1, 0, 2]
        for sample_idx in range(num_samples):
            for d in range(len(perm)):
                out_shapes[sample_idx][d] = in_shapes[sample_idx][perm[d]]
        out_type = carray(out_dtype, 1)
        out_type[0] = dali_int32

Since the setup function is running for the whole batch, we need to iterate and permute each sample's shape individually. 
For ``shapes = [(10, 20, 30), (20, 10, 30)]`` it will produce output with ``shapes = [(20, 10, 30), (10, 20, 30)]``.

Also lets provide run function:

.. code-block:: python

    @cfunc(dali_numba.run_fn_sig(types.int32, types.int64), nopython=True)
    def change_out_shape(out_ptr, out_shape_ptr, ndim_out, in_ptr, in_shape_ptr, ndim_in):
        out_shape = carray(out_shape_ptr, ndim_out)
        out_arr = carray(out_ptr, (out_shape[0], out_shape[1], out_shape[2]))
        in_shape = carray(in_shape_ptr, ndim_in)
        in_arr = carray(in_ptr, (in_shape[0], in_shape[1], in_shape[2]))
        for i in range(in_shape[0]):
            for j in range(in_shape[1]):
                out_arr[j, i] = in_arr[i, j]

The run function works on a per-sample basis, so we only need to handle one sample.
Notice that we are passing ``int32`` as the output data type, which we set in the setup function body.

)code")
  .NumInput(1, 6)
  .OutputFn([](const OpSpec &spec) { return spec.GetRepeatedArgument<int>("out_types").size(); })
  .Unserializable()
  .AddParent("NumbaFunc");

template <typename Backend>
NumbaFuncImpl<Backend>::NumbaFuncImpl(const OpSpec &spec) : Base(spec) {
  run_fn_ = spec.GetArgument<uint64_t>("run_fn");
  setup_fn_ = spec.GetArgument<uint64_t>("setup_fn");
  batch_processing_ = spec.GetArgument<bool>("batch_processing");

  out_types_ = spec.GetRepeatedArgument<int>("out_types");
  DALI_ENFORCE(!out_types_.empty(), "");
  in_types_ = spec.GetRepeatedArgument<int>("in_types");
  DALI_ENFORCE(!in_types_.empty(), "");

  outs_ndim_ = spec.GetRepeatedArgument<int>("outs_ndim");
  if (!setup_fn_) {
    DALI_ENFORCE(out_types_.size() == in_types_.size(), "");
  } else {
    DALI_ENFORCE(outs_ndim_.size() == out_types_.size(), "");
  }

  ins_ndim_ = spec.GetRepeatedArgument<int>("ins_ndim");
}

template <>
bool NumbaFuncImpl<CPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
    const workspace_t<CPUBackend> &ws) {
  output_desc.resize(out_types_.size());
  in_shapes_.resize(in_types_.size());
  for (size_t in_id = 0; in_id < in_types_.size(); in_id++) {
    in_shapes_[in_id] = ws.InputRef<CPUBackend>(in_id).shape();
  }
  auto N = in_shapes_[0].num_samples();
  input_shapes_.resize(N * in_types_.size());
  for (size_t in_id = 0; in_id < in_types_.size(); in_id++) {
    for (int i = 0; i < N; i++) {
      input_shapes_[N * in_id + i] = (uint64_t)in_shapes_[in_id].tensor_shape_span(i).data();
    }
  }

  if (!setup_fn_) {
    for (size_t i = 0; i < out_types_.size(); i++) {
      const auto &in = ws.InputRef<CPUBackend>(i);
      output_desc[i] = {in.shape(), in.type()};
    }
    return true;
  }

  for (size_t i = 0; i < out_types_.size(); i++) {
    output_desc[i].shape.resize(N, outs_ndim_[i]);
    output_desc[i].type = dali::TypeTable::GetTypeInfo(static_cast<DALIDataType>(out_types_[i]));
  }

  output_shapes_.resize(N * out_types_.size());
  for (size_t out_id = 0; out_id < out_types_.size(); out_id++) {
    for (int i = 0; i < N; i++) {
      output_shapes_[N * out_id + i] = (uint64_t)output_desc[out_id].shape.tensor_shape_span(i).data();
    }
  }

  ((void (*)(void*, const void*, int32_t, const void*, const void*, int32_t, int32_t))setup_fn_)(
    output_shapes_.data(), outs_ndim_.data(), outs_ndim_.size(), input_shapes_.data(), ins_ndim_.data(), ins_ndim_.size(), N);

  // TODO VALIDATION OF DATA?

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
    ((void (*)(void*, const void*, const void*, const void*, int32_t,
      const void*, const void*, const void*, const void*, int32_t, int32_t))run_fn_)(
        out_ptrs.data(), out_types_.data(), (setup_fn_ ? output_shapes_.data() : input_shapes_.data()),
        outs_ndim_.data(), outs_ndim_.size(),
        in_ptrs.data(), in_types_.data(), input_shapes_.data(), ins_ndim_.data(), ins_ndim_.size(), N);
    return;
  }

  auto &out = ws.OutputRef<CPUBackend>(0);
  auto out_shape = out.shape();
  auto &tp = ws.GetThreadPool();
  for (int sample_id = 0; sample_id < N; sample_id++) {
    tp.AddWork([&, sample_id](int thread_id) {
      std::vector<uint64_t> out_ptrs_per_sample(out_types_.size());
      std::vector<uint64_t> out_shapes_per_sample(out_types_.size());
      for (size_t out_id = 0; out_id < out_types_.size(); out_id++) {
        out_ptrs_per_sample[out_id] = out_ptrs[N * out_id + sample_id];
        out_shapes_per_sample[out_id] = (setup_fn_ ? output_shapes_[N * out_id + sample_id] : input_shapes_[N * out_id + sample_id]);
      }
      std::vector<uint64_t> in_ptrs_per_sample(in_types_.size());
      std::vector<uint64_t> in_shapes_per_sample(in_types_.size());
      for (size_t in_id = 0; in_id < in_types_.size(); in_id++) {
        in_ptrs_per_sample[in_id] = in_ptrs[N * in_id + sample_id];
        in_shapes_per_sample[in_id] = input_shapes_[N * in_id + sample_id];
      }

      ((void (*)(void*, const void*, const void*, const void*, int32_t,
      const void*, const void*, const void*, const void*, int32_t))run_fn_)(
        out_ptrs_per_sample.data(),
        out_types_.data(),
        out_shapes_per_sample.data(),
        outs_ndim_.data(),
        outs_ndim_.size(),
        in_ptrs_per_sample.data(),
        in_types_.data(),
        in_shapes_per_sample.data(),
        ins_ndim_.data(),
        ins_ndim_.size());
    }, out_shape.tensor_size(sample_id));
  }
  tp.RunAll();
}

DALI_REGISTER_OPERATOR(NumbaFuncImpl, NumbaFuncImpl<CPUBackend>, CPU);

}  // namespace dali

