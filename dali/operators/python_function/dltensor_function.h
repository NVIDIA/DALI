// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_PYTHON_FUNCTION_DLTENSOR_FUNCTION_H_
#define DALI_OPERATORS_PYTHON_FUNCTION_DLTENSOR_FUNCTION_H_
#include <dali/util/pybind.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <vector>
#include <utility>
#include <string>
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/copy_with_stride.h"

namespace dali {

extern std::mutex operator_lock;

cudaStream_t GetCurrentStream();

void SetCurrentStream(cudaStream_t stream);

namespace detail {

template <typename Backend>
constexpr DLDeviceType Backend2DLDevice() {
  if (std::is_same<Backend, CPUBackend>::value) {
    return kDLCPU;
  } else {
    return kDLCUDA;
  }
}

template <typename Backend>
std::vector<DLMTensorPtr> CastToDLTensorList(py::list &list, Index exp_size, Index out_idx) {
  DALI_ENFORCE(list.size() == static_cast<size_t>(exp_size),
      "Function called by DLTensorPythonFunction returned tensor list of wrong size at idx "
      + std::to_string(out_idx) + ". Returned list is of a size "
      + std::to_string(list.size()) + " and should be of size " + std::to_string(exp_size));
  std::vector<DLMTensorPtr> result;
  result.reserve(exp_size);
  if (exp_size) {
    DALI_ENFORCE(py::capsule::check_(list[0]),
                 "Function called by DLTensorPythonFunction "
                 "should return a list of DLPack tensors.");
    auto caps = py::cast<py::capsule>(list[0]);
    result.push_back(DLMTensorPtrFromCapsule(caps));
    DALI_ENFORCE(result[0]->dl_tensor.device.device_type == Backend2DLDevice<Backend>(),
        "Wrong output backend");
    auto dtype = result[0]->dl_tensor.dtype;
    auto ndim = result[0]->dl_tensor.ndim;
    for (Index i = 1; i < exp_size; ++i) {
      auto caps = py::cast<py::capsule>(list[i]);
      result.push_back(DLMTensorPtrFromCapsule(caps));
      DALI_ENFORCE(result[i]->dl_tensor.device.device_type == Backend2DLDevice<Backend>(),
                   "Wrong output backend.");
      DALI_ENFORCE(DLToDALIType(result[i]->dl_tensor.dtype) == DLToDALIType(dtype),
                   "Output DLPack tensor list should have consistent data type.");
      DALI_ENFORCE(result[i]->dl_tensor.ndim == ndim,
                   "All samples in the batch should have the same number of dimensions.");
    }
  }
  return result;
}

TensorListShape<> GetDLTensorListShape(const std::vector<DLMTensorPtr> &dl_tensors);

template <typename Backend>
void CopyDlTensor(void *out_data, DLMTensorPtr &dlm_tensor_ptr, cudaStream_t stream = 0) {
  auto &dl_tensor = dlm_tensor_ptr->dl_tensor;
  auto item_size = dl_tensor.dtype.bits / 8;
  if (dl_tensor.strides) {
    std::vector<Index> strides(dl_tensor.ndim);
    for (Index i = 0; i < dl_tensor.ndim; ++i) strides[i] = dl_tensor.strides[i] * item_size;
    CopyWithStride<Backend>(out_data, dl_tensor.data, strides.data(),
                            dl_tensor.shape, dl_tensor.ndim, item_size, stream);
  } else {
    CopyWithStride<Backend>(out_data, dl_tensor.data, nullptr,
                            dl_tensor.shape, dl_tensor.ndim, item_size, stream);
  }
}

template <typename Backend>
py::list PrepareDLTensorInputs(workspace_t<Backend> &ws);

template <typename Backend>
py::list PrepareDLTensorInputsPerSample(workspace_t<Backend> &ws);

template <typename Workspace, typename Output>
void CopyOutputData(Output& output, std::vector<DLMTensorPtr> &dl_tensors,
                    int batch_size, Workspace &workspace);

template <typename Backend>
void PrepareOutputs(workspace_t<Backend> &ws, const py::object &output_o, int batch_size) {
  py::tuple return_tuple = (py::tuple::check_(output_o)) ? output_o : py::make_tuple(output_o);
  for (Index idx = 0; idx < ws.NumOutput(); ++idx) {
    py::list dl_list = py::cast<py::list>(return_tuple[idx]);
    auto dl_tensors = CastToDLTensorList<Backend>(dl_list, batch_size, idx);
    if (dl_tensors.empty()) continue;
    auto &tlist = ws.template OutputRef<Backend>(idx);
    tlist.set_type(TypeTable::GetTypeInfo(DLToDALIType(dl_tensors[0]->dl_tensor.dtype)));
    tlist.Resize(GetDLTensorListShape(dl_tensors));
    CopyOutputData(tlist, dl_tensors, batch_size, ws);
  }
}

template <typename Backend>
void PrepareOutputsPerSample(workspace_t<Backend> &ws, const py::object &output_o, int batch_size) {
  py::list output = output_o;
  py::tuple output_tuple(ws.NumOutput());
  std::vector<py::list> output_lists(ws.NumOutput());
  for (py::handle sample_out : output) {
    if (py::tuple::check_(sample_out)) {
      auto out = py::reinterpret_borrow<py::tuple>(sample_out);
      for (Index idx = 0; idx < ws.NumOutput(); ++idx) {
        output_lists[idx].append(out[idx]);
      }
    } else {
      output_lists[0].append(sample_out);
    }
  }
  for (Index idx = 0; idx < ws.NumOutput(); ++idx) {
    output_tuple[idx] = std::move(output_lists[idx]);
  }
  PrepareOutputs<Backend>(ws, output_tuple, batch_size);
}

template <typename Backend>
class StreamSynchronizer;

template <>
class StreamSynchronizer<GPUBackend> {
 public:
  StreamSynchronizer(DeviceWorkspace &ws, bool synchronize): previous_(GetCurrentStream()) {
    SetCurrentStream(ws.stream());
    if (synchronize) CUDA_CALL(cudaStreamSynchronize(ws.stream()));
  }

  ~StreamSynchronizer() {
    SetCurrentStream(previous_);
  }
 private:
  cudaStream_t previous_;
};

template <>
class StreamSynchronizer<CPUBackend> {
 public:
  StreamSynchronizer(HostWorkspace &ws, bool synchronize) {}
};

}  // namespace detail


template <typename Backend>
class DLTensorPythonFunctionImpl : public Operator<Backend> {
 public:
  inline explicit DLTensorPythonFunctionImpl(const OpSpec &spec)
      : Operator<Backend>(spec)
      , python_function(py::reinterpret_borrow<py::object>(
          reinterpret_cast<PyObject*>(spec.GetArgument<int64_t>("function_id")))) {
    synchronize_stream_ = spec.GetArgument<bool>("synchronize_stream");
    batch_processing = spec.GetArgument<bool>("batch_processing");
    size_t num_outputs = spec.GetArgument<int>("num_outputs");
    bool listed_layouts = spec.TryGetRepeatedArgument(output_layouts_, "output_layouts");
    if (!listed_layouts && spec.HasArgument("output_layouts")) {
      auto layout = spec.GetArgument<TensorLayout>("output_layouts");
      output_layouts_ = std::vector<TensorLayout>(num_outputs, layout);
    }
    DALI_ENFORCE(output_layouts_.size() <= num_outputs,
                 make_string("The length of the ``output_layouts`` (=", output_layouts_.size(),
                             ") is greater than the number of outputs (=", num_outputs, ")."));
  }

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    return false;
  }

  void RunImpl(workspace_t<Backend> &ws) override {
    SetOutputLayouts(ws);
    std::lock_guard<std::mutex> operator_guard(operator_lock);
    py::gil_scoped_acquire interpreter_guard{};
    py::object output_o = py::none();
    auto curr_batch_size = GetCurrBatchSize(ws);
    try {
      detail::StreamSynchronizer<Backend> sync(ws, synchronize_stream_);
      if (batch_processing) {
        auto input = detail::PrepareDLTensorInputs<Backend>(ws);
        output_o = python_function(*input);
      } else {
        auto inputs = detail::PrepareDLTensorInputsPerSample<Backend>(ws);
        py::list out_batch;
        if (inputs.size() > 0) {
          for (auto &input_tuple : inputs) {
            py::object output = python_function(*input_tuple);
            if (!output.is_none()) out_batch.append(output);
          }
        } else {
          for (int s = 0; s < curr_batch_size; ++s) {
            py::object output = python_function();
            if (!output.is_none()) out_batch.append(output);
          }
        }
        if (out_batch.size() != 0) output_o = out_batch;
      }
    } catch(const py::error_already_set &e) {
      throw std::runtime_error(to_string("DLTensorPythonFunction error: ") + to_string(e.what()));
    }
    if (!output_o.is_none()) {
      if (batch_processing) {
        detail::PrepareOutputs<Backend>(ws, output_o, curr_batch_size);
      } else {
        detail::PrepareOutputsPerSample<Backend>(ws, output_o, curr_batch_size);
      }
    } else {
      DALI_ENFORCE(ws.NumOutput() == 0, "Python function returned 0 outputs and "
          + std::to_string(ws.NumOutput()) + " were expected.");
    }
  };

  void SetOutputLayouts(workspace_t<Backend> &ws) {
    Index output_idx = 0;
    for (auto layout : output_layouts_) {
      auto &output = ws.template OutputRef<Backend>(output_idx);
      output.SetLayout(layout);
      ++output_idx;
    }
  }

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;

  py::object python_function;
  bool synchronize_stream_;
  bool batch_processing;
  std::vector<TensorLayout> output_layouts_;

 private:
  int GetCurrBatchSize(workspace_t<Backend> &ws) {
    if (ws.NumInput() > 0) {
      auto curr_batch_size = ws.GetInputBatchSize(0);
      for (int i = 1; i < ws.NumInput(); i++) {
        DALI_ENFORCE(ws.GetInputBatchSize(i) == curr_batch_size,
                     make_string("Every input shall have the same batch size. Found inconsistent "
                                 "batch sizes (val@idx): ",
                                 ws.GetInputBatchSize(i), "@", i, " vs ", curr_batch_size, "@0."));
      }
      return curr_batch_size;
    } else {
      auto curr_batch_size = ws.GetRequestedBatchSize(0);
      for (int i = 1; i < ws.NumOutput(); i++) {
        DALI_ENFORCE(
            ws.GetRequestedBatchSize(i) == curr_batch_size,
            make_string("This operator assumes, that requested batch size is the same for every "
                        "output. Found inconsistent batch sizes (val@idx): ",
                        ws.GetRequestedBatchSize(i), "@", i, " vs ", curr_batch_size, "@0."));
      }
      return curr_batch_size;
    }
  }
};

}  // namespace dali

#endif  // DALI_OPERATORS_PYTHON_FUNCTION_DLTENSOR_FUNCTION_H_
