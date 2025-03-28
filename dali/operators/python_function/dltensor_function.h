// Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <string>
#include <utility>
#include <vector>
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
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
      DALI_ENFORCE(ToDALIType(result[i]->dl_tensor.dtype) == ToDALIType(dtype),
                   "Output DLPack tensor list should have consistent data type.");
      DALI_ENFORCE(result[i]->dl_tensor.ndim == ndim,
                   "All samples in the batch should have the same number of dimensions.");
    }
  }
  return result;
}

TensorListShape<> GetDLTensorListShape(const std::vector<DLMTensorPtr> &dl_tensors);

template <typename Backend>
py::list PrepareDLTensorInputs(Workspace &ws);

template <typename Backend>
py::list PrepareDLTensorInputsPerSample(Workspace &ws);

template <typename Workspace, typename Output>
void CopyOutputData(Output& output, std::vector<DLMTensorPtr> &dl_tensors,
                    Workspace &workspace);

template <typename Backend>
void PrepareOutputs(Workspace &ws, const py::object &output_o, int batch_size) {
  py::tuple return_tuple = (py::tuple::check_(output_o)) ? output_o : py::make_tuple(output_o);
  for (Index idx = 0; idx < ws.NumOutput(); ++idx) {
    py::list dl_list = py::cast<py::list>(return_tuple[idx]);
    auto dl_tensors = CastToDLTensorList<Backend>(dl_list, batch_size, idx);
    if (dl_tensors.empty()) continue;
    auto &tlist = ws.Output<Backend>(idx);
    tlist.Resize(GetDLTensorListShape(dl_tensors), ToDALIType(dl_tensors[0]->dl_tensor.dtype));
    CopyOutputData(tlist, dl_tensors, ws);
  }
}

template <typename Backend>
void PrepareOutputsPerSample(Workspace &ws, const py::object &output_o, int batch_size) {
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
  StreamSynchronizer(Workspace &ws, bool synchronize): previous_(GetCurrentStream()) {
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
  StreamSynchronizer(Workspace &ws, bool synchronize) {}
};

}  // namespace detail

// NOTE: This operator isn't really stateless - we just ignore
//       the state of the underlying python function and allow
//       it to be checkpointed.
template <typename Backend>
class DLTensorPythonFunctionImpl : public StatelessOperator<Backend> {
 public:
  inline explicit DLTensorPythonFunctionImpl(const OpSpec &spec)
      : StatelessOperator<Backend>(spec)
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
  bool HasContiguousOutputs() const override {
    return false;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    return false;
  }

  void RunImpl(Workspace &ws) override {
    SetOutputLayouts(ws);
    std::lock_guard<std::mutex> operator_guard(operator_lock);
    py::gil_scoped_acquire interpreter_guard{};
    output_o_ = py::none();
    auto curr_batch_size = GetCurrBatchSize(ws);
    try {
      detail::StreamSynchronizer<Backend> sync(ws, synchronize_stream_);
      if (batch_processing) {
        input_o_ = detail::PrepareDLTensorInputs<Backend>(ws);
        output_o_ = python_function(*input_o_);
      } else {
        input_o_ = detail::PrepareDLTensorInputsPerSample<Backend>(ws);
        py::list out_batch;
        if (input_o_.size() > 0) {
          for (auto &input_tuple : input_o_) {
            py::object output = python_function(*input_tuple);
            if (!output.is_none()) out_batch.append(output);
          }
        } else {
          for (int s = 0; s < curr_batch_size; ++s) {
            py::object output = python_function();
            if (!output.is_none()) out_batch.append(output);
          }
        }
        if (out_batch.size() != 0) output_o_ = out_batch;
      }
    } catch(const py::error_already_set &e) {
      throw std::runtime_error(to_string("DLTensorPythonFunction error: ") + to_string(e.what()));
    }
    if (!output_o_.is_none()) {
      if (batch_processing) {
        detail::PrepareOutputs<Backend>(ws, output_o_, curr_batch_size);
      } else {
        detail::PrepareOutputsPerSample<Backend>(ws, output_o_, curr_batch_size);
      }
    } else {
      DALI_ENFORCE(ws.NumOutput() == 0, "Python function returned 0 outputs and "
          + std::to_string(ws.NumOutput()) + " were expected.");
    }
  }

  void SetOutputLayouts(Workspace &ws) {
    Index output_idx = 0;
    for (const auto &layout : output_layouts_) {
      auto &output = ws.Output<Backend>(output_idx);
      output.SetLayout(layout);
      ++output_idx;
    }
  }

  USE_OPERATOR_MEMBERS();
  using StatelessOperator<Backend>::RunImpl;

  py::object python_function;
  py::object output_o_;
  py::list input_o_;
  bool synchronize_stream_;
  bool batch_processing;
  std::vector<TensorLayout> output_layouts_;

 private:
  int GetCurrBatchSize(Workspace &ws) {
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

  ~DLTensorPythonFunctionImpl() {
    auto interpreter_lock = py::gil_scoped_acquire();
    python_function.dec_ref();
    python_function.release();
    output_o_.dec_ref();
    output_o_.release();
    input_o_.dec_ref();
    input_o_.release();
  }
};

}  // namespace dali

#endif  // DALI_OPERATORS_PYTHON_FUNCTION_DLTENSOR_FUNCTION_H_
