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

#ifndef DALI_OPERATORS_PYTHON_FUNCTION_JAX_FUNCTION_H_
#define DALI_OPERATORS_PYTHON_FUNCTION_JAX_FUNCTION_H_

#include <dali/util/pybind.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "dali/core/span.h"
#include "dali/pipeline/data/dltensor_obj.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"

namespace dali {

namespace detail {

/**
 * @brief Exposes vector of tensors as a Python list of DLTensorObjs.
 *
 * The ownership of the tensors is transferred to the corresponding DLTensorObjs,
 * that will further transfer the ownership if the __dlpack__ method is called.
 *
 * @param tensors Tensors to expose as a Python list of DLTensorObjs.
 * @param producer_stream The stream on which the underlying tensors memory can be accessed.
 *                        The consumer's stream will wait for the completion of the work scheduled
 *                        on the `producer_stream` up to the moment of calling this function.
 * @return py::list of DlTensorObjs.
 */
template <typename Backend>
py::list TensorsAsDLTensorObjs(std::vector<Tensor<Backend>> &tensors,
                               std::optional<cudaStream_t> producer_stream) {
  py::list dl_tensor_objs;
  for (size_t i = 0; i < tensors.size(); ++i) {
    dl_tensor_objs.append(
        py::cast(DLTensorObj{GetSharedDLTensor(tensors[i]), producer_stream}));
  }
  return dl_tensor_objs;
}

template <typename Backend>
Tensor<Backend> AsContigiousTensor(TensorList<Backend> &tl, AccessOrder order) {
  TensorList<Backend> contigious_tl;
  contigious_tl.set_order(order);
  contigious_tl.set_device_id(tl.device_id());
  contigious_tl.set_pinned(tl.is_pinned());
  contigious_tl.Resize(tl.shape(), tl.type(), BatchContiguity::Contiguous);
  contigious_tl.Copy(tl);
  return contigious_tl.AsTensor();
}

inline bool IsDenseTensor(const DLTensor &dl_batch) {
  if (!dl_batch.strides) {
    return true;
  }
  int ndim = dl_batch.ndim;
  int64_t stride = 1;  // stride in number of elements (not bytes)
  for (int d = ndim - 1; d >= 0; --d) {
    if (stride != dl_batch.strides[d]) {
      return false;
    }
    stride *= dl_batch.shape[d];
  }
  return true;
}

inline TensorShape<> DenseStrides(const TensorShape<> &tensor_shape) {
  int ndim = tensor_shape.size();
  TensorShape strides;
  strides.resize(ndim);
  if (ndim > 0) {
    strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d) {
      strides[d] = strides[d + 1] * tensor_shape[d + 1];
    }
  }
  return strides;
}

inline TensorShape<> ParseValidateExternalOutput(const DLTensor &dl_batch, int expected_batch_size,
                                                 int out_idx, bool is_gpu) {
  DALI_ENFORCE(dl_batch.device.device_type == (is_gpu ? kDLCUDA : kDLCPU),
               make_string("The Python callback running as DALI ", (is_gpu ? "gpu" : "cpu"),
                           " operator must return tensors placed on the same device type. ",
                           "Got different device type for output at index ",
                           out_idx, "."));
  DALI_ENFORCE(dl_batch.ndim >= 1,
               make_string("The Python callback should return a tensor representing a batch of "
                           "samples. Got a 0D tensor for output at index ",
                           out_idx, "."));
  TensorShape<> dl_batch_shape(dl_batch.shape, dl_batch.shape + dl_batch.ndim);
  DALI_ENFORCE(
      dl_batch_shape[0] == expected_batch_size,
      make_string("The Python callback should return a tensor representing a batch of "
                  "samples. The leftmost dimension should match the current batch size of ",
                  expected_batch_size, ", but the output at index ", out_idx, " has shape of ",
                  dl_batch_shape, "."));
  DALI_ENFORCE(
      IsDenseTensor(dl_batch),
      make_string("The tensors returned by the Python callback should be dense, row-major "
                  "tensors. Got strided tensor as the output at index ",
                  out_idx, ". Expected strides: `", DenseStrides(dl_batch_shape), "`, actual `",
                  TensorShape<>(dl_batch.strides, dl_batch.strides + dl_batch.ndim), "`."));
  return dl_batch_shape;
}

}  // namespace detail


template <typename Backend>
class JaxFunction : public StatelessOperator<Backend> {
 public:
  inline explicit JaxFunction(const OpSpec &spec)
      : StatelessOperator<Backend>(spec),
        python_function_(py::reinterpret_borrow<py::object>(
            reinterpret_cast<PyObject *>(spec.GetArgument<int64_t>("function_id")))),
        output_layouts_{} {
    int num_outputs = spec.NumOutput();
    bool has_layouts_list = spec.TryGetRepeatedArgument(output_layouts_, "output_layouts");
    if (!has_layouts_list && spec.HasArgument("output_layouts")) {
      auto layout = spec.GetArgument<TensorLayout>("output_layouts");
      output_layouts_ = std::vector<TensorLayout>(num_outputs, layout);
    }
    int num_layouts = output_layouts_.size();
    DALI_ENFORCE(
        num_layouts == 0 || num_layouts == num_outputs,
        make_string("The number of `output_layouts` must match the `num_outputs` argument."));
  }

  ~JaxFunction() {
    auto interpreter_lock = py::gil_scoped_acquire();
    python_function_.dec_ref();
    python_function_.release();
  }

  bool HasContiguousOutputs() const override {
    return false;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    return false;
  }

  void RunImpl(Workspace &ws) override {
    ShareOutputs(ws, RunPythonCallback(ws, GetBatchedInputs(ws)));
    PropagateSourceInfo(ws);
  }

 protected:
  /**
   * @brief Returns the inputs to the operator as a vector of tensors.
   * The function checks if the input batches are uniform and fails if not.
   * The underlying memory is shared not copied if the operator inputs are contigious.
   * Otherwise, a new tensor is workspace-order allocated and the memory is copied.
   */
  std::vector<Tensor<Backend>> GetBatchedInputs(Workspace &ws) {
    std::vector<Tensor<Backend>> batched_inputs;
    batched_inputs.reserve(ws.NumInput());
    for (int input_idx = 0; input_idx < ws.NumInput(); ++input_idx) {
      auto &input =
          ws.UnsafeMutableInput<Backend>(input_idx);  // the AsTensor() is not marked const
      DALI_ENFORCE(
          is_uniform(input.shape()),
          make_string(
              "JAX Python function inputs must be batches of uniform shape, so that the input "
              "batch can be represented as a tensor. The input at index ",
              input_idx, " is a batch of samples with different shapes."));
      AccessOrder order = std::is_same_v<Backend, GPUBackend> ? ws.stream() : AccessOrder::host();
      if (input.IsContiguous()) {
        batched_inputs.push_back(input.AsTensor());
        batched_inputs.back().set_order(order);
      } else {
        batched_inputs.push_back(detail::AsContigiousTensor(input, order));
      }
    }
    return batched_inputs;
  }

  /**
   * @brief Runs the Python function.
   * This is the only place where the operator is expected to interact with Python
   * (either by calling the Python callback or interacting with python objects).
   */
  std::vector<DLMTensorPtr> RunPythonCallback(Workspace &ws,
                                              std::vector<Tensor<Backend>> &&batched_inputs) {
    py::gil_scoped_acquire interpreter_guard{};
    if constexpr (std::is_same_v<Backend, CPUBackend>) {
      auto dl_inputs = detail::TensorsAsDLTensorObjs(batched_inputs, std::nullopt);
      auto dl_outputs = python_function_(*dl_inputs);
      return ConsumePythonOutputs(ws, std::move(dl_outputs));
    } else if constexpr (!std::is_same_v<Backend, CPUBackend>) {  // NOLINT
      static_assert(std::is_same_v<Backend, GPUBackend>,
                    "The operator supports only CPU and GPU backends");
      cudaStream_t stream = ws.stream();
      auto dl_inputs = detail::TensorsAsDLTensorObjs(batched_inputs, stream);
      auto dl_outputs = python_function_(reinterpret_cast<int64_t>(stream), *dl_inputs);
      return ConsumePythonOutputs(ws, std::move(dl_outputs));
    }
  }

  /**
   * @brief Transforms the tuple of python capsules containing dlpacked tensors
   *        into a vector of DLManagedTensor unique pointers.
   *        From then on, the unique pointers own the dlpacked tensors.
   *
   * @param ws
   * @param py_outputs
   */
  std::vector<DLMTensorPtr> ConsumePythonOutputs(Workspace &ws, py::object &&py_outputs) {
    std::vector<DLMTensorPtr> outputs;
    int num_outputs = ws.NumOutput();
    auto as_tuple = [](py::object &py_outputs) -> pybind11::tuple {
      if (py::tuple::check_(py_outputs)) {
        return py_outputs;  // it's a tuple already
      } else if (!py_outputs.is_none()) {
        return py::make_tuple(py_outputs);  // it's a single element
      } else {
        return py::make_tuple();
      }
    };
    py::tuple dl_output_tuple = as_tuple(py_outputs);
    DALI_ENFORCE(
        dl_output_tuple.size() == static_cast<size_t>(num_outputs),
        make_string("The Python callback was expected to return (a tuple of) `num_outputs=",
                    num_outputs, "` outputs, but returned ", dl_output_tuple.size(), "."));
    outputs.reserve(num_outputs);
    for (int out_idx = 0; out_idx < num_outputs; ++out_idx) {
      py::capsule py_dl_batch = py::cast<py::capsule>(dl_output_tuple[out_idx]);
      // Here, we transfer the ownership of the dlpack-ed memory from the
      // python object to the DLMTensorPtr unique_ptr
      outputs.push_back(DLMTensorPtrFromCapsule(py_dl_batch));
    }
    return outputs;
  }

  /**
   * @brief Consumes the outputs coming from the Python callback and ties their
   *        lifespan to the output tensor list.
   * @param ws
   * @param outputs The vector of DLManagedTensor unique pointers.
   *                The tensor's underlying memory is expected to be safe to access
   *                in the host or ws.stream() order (for CPU and GPU backends respectively).
   */
  void ShareOutputs(Workspace &ws, std::vector<DLMTensorPtr> &&outputs) {
    assert(outputs.size() == static_cast<size_t>(ws.NumOutput()));
    int num_outputs = outputs.size();
    constexpr bool is_gpu = std::is_same_v<Backend, GPUBackend>;
    for (int out_idx = 0; out_idx < num_outputs; ++out_idx) {
      auto &dl_batch_ptr = outputs[out_idx];
      DLTensor &dl_batch = dl_batch_ptr->dl_tensor;

      const int batch_size = ws.GetRequestedBatchSize(out_idx);
      auto dl_batch_shape =
          detail::ParseValidateExternalOutput(dl_batch, batch_size, out_idx, is_gpu);
      TensorListShape<> batch_shape =
          uniform_list_shape(batch_size, dl_batch_shape.last(dl_batch_shape.size() - 1));

      auto dtype = ToDALIType(dl_batch.dtype);
      auto type_info = dali::TypeTable::GetTypeInfo(dtype);
      size_t size_of_dtype = type_info.size();
      int64_t bytes = dl_batch_shape.num_elements() * size_of_dtype;

      bool is_pinned = is_gpu;  // TODO(ktokarski) support consuming cuda pinned host memory
      int device_id = is_gpu ? dl_batch.device.device_id : CPU_ONLY_DEVICE_ID;
      AccessOrder access_order = is_gpu ? ws.stream() : AccessOrder::host();

      // Ties the lifespan of the incoming output tensor to the output tensor list.
      // This code assumes that the external dlpack-ed deleter
      // can be run without any Python locks.
      char *data_ptr = static_cast<char *>(dl_batch.data) + dl_batch.byte_offset;
      DALI_ENFORCE(reinterpret_cast<uintptr_t>(data_ptr) % size_of_dtype == 0,
                   "The tensor memory must be aligned at least up to the size of the data type.");
      std::shared_ptr<void> dl_data{data_ptr, [dl_batch_ptr = std::move(dl_batch_ptr)](
                                                  void *) mutable { dl_batch_ptr.reset(); }};
      ws.Output<Backend>(out_idx).ShareData(dl_data, bytes, is_pinned, batch_shape, dtype,
                                            device_id, access_order,
                                            GetOutputLayout(ws, out_idx, dl_batch.ndim - 1));
    }
  }

  TensorLayout GetOutputLayout(Workspace &ws, int out_idx, int out_ndim) {
    if (output_layouts_.size() > 0) {
      return output_layouts_[out_idx];
    } else if (out_idx < ws.NumInput() && ws.GetInputDim(out_idx) == out_ndim) {
      return ws.GetInputLayout(out_idx);
    } else {
      return "";
    }
  }

  void PropagateSourceInfo(Workspace &ws) {
    int batch_size = ws.GetRequestedBatchSize(0);
    for (int sample_idx = 0; sample_idx < batch_size; ++sample_idx) {
      std::stringstream out_source_info_ss;
      bool appended = false;
      for (int in_idx = 0; in_idx < ws.NumInput(); ++in_idx) {
        const auto &source_info = ws.Input<Backend>(in_idx).GetMeta(sample_idx).GetSourceInfo();
        if (!source_info.empty()) {
          if (appended) {
            out_source_info_ss << ";";
          }
          out_source_info_ss << source_info;
          appended = true;
        }
      }
      std::string source_info = out_source_info_ss.str();
      for (int out_idx = 0; out_idx < ws.NumOutput(); ++out_idx) {
        ws.Output<Backend>(out_idx).SetSourceInfo(sample_idx, source_info);
      }
    }
  }

  py::object python_function_;
  std::vector<TensorLayout> output_layouts_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_PYTHON_FUNCTION_JAX_FUNCTION_H_
