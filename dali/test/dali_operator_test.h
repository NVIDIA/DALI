// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_TEST_DALI_OPERATOR_TEST_H_
#define DALI_TEST_DALI_OPERATOR_TEST_H_

#include <dali/pipeline/pipeline.h>
#include <dali/test/graph_descr.h>
#include <dali/test/tensor_list_wrapper.h>
#include <dali/test/argument_key.h>
#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <map>

namespace dali {
namespace testing {

namespace detail {

template<typename Backend>
std::string BackendStringName() {
  DALI_ENFORCE(false, "Backend not supported. You may want to write your own specialization");
}


template<>
inline std::string BackendStringName<CPUBackend>() {
  return "cpu";
}


template<>
inline std::string BackendStringName<GPUBackend>() {
  return "gpu";
}

}  // namespace detail

using Arguments = std::map<ArgumentKey, int>;  // TODO(mszolucha) some generalization boost::any?

class DaliOperatorTest : public ::testing::Test, public ::testing::WithParamInterface<Arguments> {
 public:
  using Verify = std::function<void(TensorListWrapper /* single input */,
                                    TensorListWrapper /* single output */, Arguments)>;

 protected:
  template<typename InputBackend = CPUBackend, typename OutputBackend = CPUBackend>
  void RunTest(const TensorListWrapper &input, const Arguments &operator_arguments,
               const Verify &verify) {
    InitPipeline(batch_size_, num_threads_);
    if (input) {
      AddInputToPipeline<InputBackend>(pipeline_.get(), input);
    }
    auto outputs = RunTestImpl<InputBackend, OutputBackend>(operator_arguments, verify,
                                                            input ? true : false);
    assert(outputs.size() == 1); // one input, one output
    verify(input, outputs[0], operator_arguments);
  }


  template<typename InputBackend = CPUBackend, typename OutputBackend = CPUBackend>
  void RunTest(const std::vector<TensorListWrapper> &inputs, const Arguments &operator_arguments,
               const std::vector<Verify> &verify) {
    //TODO(mszolucha) implement
  }


 private:
  virtual GraphDescr GenerateOperatorsGraph() const noexcept = 0;


  void SetUp() final {
  }


  void TearDown() final {
  }


  template<typename InputBackend, typename OutputBackend>
  std::vector<TensorListWrapper>
  RunTestImpl(const Arguments &operator_arguments, const Verify &verify, bool has_inputs) {
    const auto op_spec = CreateOpSpec<InputBackend, OutputBackend>(
            GenerateOperatorsGraph().get_op_name(), operator_arguments, has_inputs);
    AddOperatorToPipeline(pipeline_.get(), op_spec);
    BuildPipeline(pipeline_.get(), op_spec);
    RunPipeline(pipeline_.get());
    return GetOutputsFromPipeline<OutputBackend>(pipeline_.get());
  }


  void InitPipeline(size_t batch_size, size_t num_threads) {
    if (!pipeline_) {
      pipeline_.reset(new Pipeline(static_cast<int>(batch_size), static_cast<int>(num_threads), 0));
    }
  }


  template<typename Backend>
  void AddInputToPipeline(Pipeline *pipeline, const TensorListWrapper &input) {
    const std::string input_name = "input";
    pipeline->AddExternalInput(input_name);
    auto tl = input.get<Backend>();
    pipeline->SetExternalInput(input_name, *input.get<Backend>());
  }


  void AddOperatorToPipeline(Pipeline *pipeline, const OpSpec &op_spec) {
    pipeline->AddOperator(op_spec);
  }


  DeviceWorkspace CreateWorkspace() const {
    DeviceWorkspace ws;
    return ws;
  }


  void RunPipeline(Pipeline *pipeline) {
    pipeline->RunCPU();
    pipeline->RunGPU();
  }


  template<typename Backend>
  std::vector<TensorListWrapper> GetOutputsFromPipeline(Pipeline *pipeline) {
    std::vector<TensorListWrapper> ret;
    auto workspace = CreateWorkspace();
    pipeline->Outputs(&workspace);
    for (int output_idx = 0; output_idx < workspace.NumOutput(); output_idx++) {
      ret.emplace_back(workspace.template Output<Backend>(output_idx));
    }
    return ret;
  }


  void BuildPipeline(Pipeline *pipeline, const OpSpec &spec) {
    std::vector<std::pair<string, string>> vecoutputs_;
    for (int i = 0; i < spec.NumOutput(); ++i) {
      vecoutputs_.emplace_back(spec.OutputName(i), spec.OutputDevice(i));
    }
    pipeline->Build(vecoutputs_);
  }


  // TODO(mszolucha): graph of operators
  template<typename InputBackend, typename OutputBackend>
  OpSpec CreateOpSpec(const std::string &operator_name, Arguments operator_arguments,
                      bool has_input) const {
    OpSpec opspec = OpSpec(operator_name);
    for (const auto &arg : operator_arguments) {
      opspec.AddArg(arg.first.arg_name(), arg.second);
    }
    if (has_input) {
      opspec.AddInput("input", detail::BackendStringName<InputBackend>());
    }
    opspec.AddOutput("output", detail::BackendStringName<OutputBackend>());
    return opspec;
  }


  std::unique_ptr<Pipeline> pipeline_;
  size_t batch_size_ = 1;
  const size_t num_threads_ = 1;
};

}  // namespace testing

}  // namespace dali

#endif  // DALI_TEST_DALI_OPERATOR_TEST_H_
