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

DALI_SCHEMA(experimental__NumbaFunc)
  .DocStr(R"code(Invokes a compiled Numba function passed as a pointer.

The run function should be a Numba C callback function (annotated with cfunc). This function is run per
sample and should have following definition:

.. code-block:: python

    @cfunc(run_fn_sig([out_numba_types], [in_numba_types]), nopython=True)
    def callback_run_func(out1_ptr, out1_shape_ptr, out1_shape_ndim, in1_ptr, in1_shape_ptr, in1_shape_ndim)

Additionally, an optional setup function with the following definition:

.. code-block:: python

    @cfunc(setup_fn_sig([out_numba_types], [in_numba_types]), nopython=True)
    def callback_setup_func(out1_shape_ptr, out1_ndim, out_dtype_ptr, in1_shape_ptr, in1_ndim, in1_dtype, num_samples)

Setup function is invoked per batch. If no setup function provided, the output shape and data type
will be the same as the input.

``out_numba_types`` and ``in_numba_types`` refer to the numba data types (numba.types) 
of the output and input, respectively.

.. note::
    This operator is experimental and its API might change without notice.

**Example 1:**

We will show how to write easy setup function which reorders shapes of input.

.. code-block:: python

    dali_int32 = int(dali_types.INT32)
    @cfunc(setup_fn_sig(types.int64, types.int64), nopython=True)
    def setup_change_out_shape(out_shape_ptr, out1_ndim, out_dtype, in_shape_ptr, in1_ndim, in_dtype, num_samples):
        in_arr = carray(in_shape_ptr, num_samples * out1_ndim) # get input array from pointer
        out_arr = carray(out_shape_ptr, num_samples * in1_ndim) # get output array from pointer
        perm = [1, 2, 0, 5, 3, 4]
        for i in range(len(out_arr)):
            out_arr[i] = in_arr[perm[i]] # reorder shapes
        out_type = carray(out_dtype, 1)
        out_type[0] = dali_int32  # set output type

That's the definition of setup function. As setup function is running per batch, we are assuming that every batch
will contain two samples. For ``shapes`` = [(10, 20, 30), (20, 10, 30)] it will produce output with 
``shapes`` = [(20, 30, 10), (30, 20, 10)].

Also lets provide run function:

.. code-block:: python

    @cfunc(dali_numba.run_fn_sig(types.int32, types.int64), nopython=True)
    def change_out_shape(out_ptr, out_shape_ptr, ndim_out, in_ptr, in_shape_ptr, ndim_in):
        out_shape = carray(out_shape_ptr, ndim_out) # get output shape form pointer
        out_arr = carray(out_ptr, (out_shape[0], out_shape[1], out_shape[2])) # get output array from pointer
        out_arr[:] = 42 

Run function is running per sample so we don't need to handle whole batch in it. Notice that we are passing ``int32`` as an output type.
The same which we set in setup function for output.

)code")
  .NumInput(1)
  .NumOutput(1)
  .Unserializable()
  .AddArg("fn_ptr", R"code(Numba function pointer.)code", DALI_INT64)
  .AddOptionalArg<int>("setup_fn", R"code(Pointer to a function used to determine the 
output shape and data type based on the input.)code", 0);

template <typename Backend>
NumbaFunc<Backend>::NumbaFunc(const OpSpec &spec) : Base(spec) {
  fn_ptr_ = spec.GetArgument<uint64_t>("fn_ptr");
  setup_fn_ = spec.GetArgument<uint64_t>("setup_fn");
}

template <>
bool NumbaFunc<CPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
    const workspace_t<CPUBackend> &ws) {
  const auto &in = ws.InputRef<CPUBackend>(0);
  output_desc.resize(1);

  if (!setup_fn_) {
    output_desc[0] = {in.shape(), in.type()};
    return true;
  }

  auto in_shape = in.shape();
  auto N = in_shape.num_samples();
  auto ndim = in_shape.sample_dim();
  TensorListShape<> output_shape;
  output_shape.resize(N, ndim);
  DALIDataType out_type = DALIDataType::DALI_NO_TYPE;
  DALIDataType in_type = in.type().id();
  ((void (*)(void*, int32_t, void*, const void*, int32_t, int32_t, int32_t))setup_fn_)(
      output_shape.tensor_shape_span(0).data(), ndim, &out_type,
      in_shape.tensor_shape_span(0).data(), ndim, in_type, N);

  DALI_ENFORCE(out_type != DALIDataType::DALI_NO_TYPE,
    "Output type was not set by the custom setup function.");
  for (int i = 0; i < N; i++) {
    for (int d = 0; d < ndim; d++) {
      DALI_ENFORCE(output_shape.tensor_shape_span(i)[d] >= 0,
        make_string(d, "-th", " dimension of ", i, "-th sample's shape is negative."));
    }
  }
  output_desc[0].type = dali::TypeTable::GetTypeInfo(out_type);
  output_desc[0].shape = output_shape;

  return true;
}

template <>
void NumbaFunc<CPUBackend>::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &in = ws.InputRef<CPUBackend>(0);
  auto in_shape = in.shape();
  auto &out = ws.OutputRef<CPUBackend>(0);
  auto out_shape = out.shape();
  auto& tp = ws.GetThreadPool();

  for (int sample_id = 0; sample_id < in_shape.num_samples(); sample_id++) {
    tp.AddWork([&, fn_ptr = fn_ptr_, sample_id](int thread_id) {
      ((void (*)(void*, const void*, int32_t, const void*, const void*, int32_t))fn_ptr)(
        out[sample_id].raw_mutable_data(),
        out_shape.tensor_shape_span(sample_id).data(),
        out_shape.sample_dim(),
        in[sample_id].raw_data(),
        in_shape.tensor_shape_span(sample_id).data(),
        in_shape.sample_dim());
    }, out_shape.tensor_size(sample_id));
  }
  tp.RunAll();
}

DALI_REGISTER_OPERATOR(experimental__NumbaFunc, NumbaFunc<CPUBackend>, CPU);

}  // namespace dali
