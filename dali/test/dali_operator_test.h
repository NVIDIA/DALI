// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <utility>
#include "dali/test/operator_argument.h"
#include "dali/pipeline/pipeline.h"
#include "dali/test/graph_descr.h"
#include "dali/test/tensor_list_wrapper.h"
#include "dali/test/argument_key.h"


namespace dali {
namespace testing {

using Arguments = std::map<ArgumentKey, TestOpArg>;

namespace detail {

template<typename Backend>
std::string BackendStringName() {
  FAIL() << "Backend not supported. You may want to write your own specialization", std::string();
}


template<>
inline std::string BackendStringName<CPUBackend>() {
  return "cpu";
}


template<>
inline std::string BackendStringName<GPUBackend>() {
  return "gpu";
}

// TODO(klecki): mixed, inputs on different devices
inline std::string GetOpDevice(const Arguments &op_args) {
  auto op_device_it = op_args.find(ArgumentKey("device"));
  if (op_device_it != op_args.end()) {
    return op_device_it->second.GetValue<std::string>();
  } else {
    return "cpu";
  }
}

}  // namespace detail

// JSON-like output
inline std::ostream& operator<<(std::ostream& os, const Arguments& args) {
  std::string separator("");
  std::string indent("");
  if (args.size() == 0) {
    os << "{";
  } else if (args.size() == 1) {
    os << "{ ";
  } else {
    os << "{\n";
    separator = ",\n";
    indent = "    ";
  }

  for (const auto arg : args) {
    os << indent << "\"" << arg.first << "\" : " << arg.second << separator;
  }

  if (args.size() == 0) {
    os << "}";
  } else if (args.size() == 1) {
    os << " }";
  } else {
    os << "}\n";
  }

  return os;
}

// Force GTest to write our way
inline void PrintTo(const Arguments& args, std::ostream* os) {
  *os << args;
}

inline std::unique_ptr<Pipeline> CreatePipeline(size_t batch_size, size_t num_threads) {
  return std::unique_ptr<Pipeline>(
          new Pipeline(static_cast<int>(batch_size), static_cast<int>(num_threads), 0));
}


inline void AddInputToPipeline(Pipeline &pipeline, const TensorListWrapper &input) {
  ASSERT_TRUE(input && input.has_cpu()) << "External input works only for CPUBackend";
  const std::string input_name = "input";
  pipeline.AddExternalInput(input_name);
  pipeline.SetExternalInput(input_name, *input.get<CPUBackend>());
}


inline void AddOperatorToPipeline(Pipeline &pipeline, const OpSpec &op_spec) {
  pipeline.AddOperator(op_spec);
}


inline DeviceWorkspace CreateWorkspace() {
  DeviceWorkspace ws;
  return ws;
}


inline void RunPipeline(Pipeline &pipeline) {
  pipeline.RunCPU();
  pipeline.RunGPU();
}

inline std::vector<TensorListWrapper>
GetOutputsFromPipeline(Pipeline &pipeline, const std::string &output_backend) {
  std::vector<TensorListWrapper> ret;
  auto workspace = CreateWorkspace();
  pipeline.Outputs(&workspace);
  for (int output_idx = 0; output_idx < workspace.NumOutput(); output_idx++) {
    if (workspace.OutputIsType<CPUBackend>(output_idx)) {
      ret.emplace_back(&workspace.template Output<CPUBackend>(output_idx));
    } else {
      ret.emplace_back(&workspace.template Output<GPUBackend>(output_idx));
    }
  }
  return ret;
}


inline void BuildPipeline(Pipeline &pipeline, const OpSpec &spec) {
  std::vector<std::pair<std::string, std::string>> vecoutputs_;
  for (int i = 0; i < spec.NumOutput(); ++i) {
    vecoutputs_.emplace_back(spec.OutputName(i), spec.OutputDevice(i));
  }
  pipeline.Build(vecoutputs_);
}


// TODO(mszolucha): graph of operators
inline OpSpec
CreateOpSpec(const std::string &operator_name, Arguments operator_arguments, bool has_input,
             const std::string &input_backend, const std::string &output_backend) {
  ASSERT_TRUE(input_backend == "cpu" || input_backend == "gpu"), OpSpec();
  ASSERT_TRUE(output_backend == "cpu" || output_backend == "gpu"), OpSpec();

  OpSpec opspec = OpSpec(operator_name);
  for (auto &arg : operator_arguments) {
    arg.second.SetArg(arg.first.arg_name(), opspec, nullptr);
  }
  if (has_input) {
    opspec.AddInput("input", input_backend);
  }
  opspec.AddOutput("output", output_backend);
  return opspec;
}


/**
 * Defines and conducts test for given graph of operators.
 * Note: whenever "graph of operators" is used, in particular it can mean,
 *       that the graph consists of one operator.
 *
 * In order to set up a test, please define custom subclass of this class.
 *
 * Along with custom class, a verification function should be defined (Verify, VerifySingleIo)
 *
 * GenerateOperatorGraph is a template method, where you can define,
 * what precisely should be tested.
 *
 * Lastly, to actually run a test, please call RunTest within GTest's testing macro
 */
class DaliOperatorTest : public ::testing::Test, public ::testing::WithParamInterface<Arguments> {
 public:
  DaliOperatorTest() = default;


  /**
   * Type of the function used for verification, whether graph test passed or failed.
   * Within this function, please use GTest's ASSERTs and EXPECTs, for test validation.
   *
   * First argument of function contains all inputs, that have been provided;
   * Second argument are all outputs from pipeline;
   * Third argument contains Arguments, with which the pipeline was called
   *
   * This is some sort or template method pattern.
   */
  using Verify = std::function<void(const std::vector<TensorListWrapper> & /* inputs */,
                                    const std::vector<TensorListWrapper> & /* outputs */,
                                    const Arguments &)>;


  using VerifySingleIo = std::function<void(const TensorListWrapper & /* single input */,
                                            const TensorListWrapper & /* single output */,
                                            const Arguments &)>;


 protected:
  /**
   * Runs defined test.
   *
   * Call this method within GTest's testing macro (TEST_F, TEST_P, etc...).
   * The method will set up a testing pipeline,
   * push data through it and call output verification routines.
   * Every RunTest call will set up its own pipeline.
   *
   * @param inputs all inputs to the pipeline
   * @param outputs placeholder for outputs from pipeline
   * @param operator_arguments arguments, with which the pipeline will be called
   * @param verify function, that will be used for test verification
   */
  void RunTest(const std::vector<TensorListWrapper> &inputs,
               std::vector<TensorListWrapper> &outputs, const Arguments &operator_arguments,
               const Verify &verify) {
    // TODO(mszolucha) implement
  }

  /**
   * Convenient overload, for specific, single-input/single-output graph
   */
  void RunTest(const TensorListWrapper &input, TensorListWrapper &output,
               const Arguments &operator_arguments, const VerifySingleIo &verify) {
    std::string op_backend = detail::GetOpDevice(operator_arguments);
    const auto batch_size = input.has_cpu() ? input.cpu().ntensor() : input.gpu().ntensor();
    ASSERT_GT(batch_size, 0) << "Looks like there ain't no tensors in input";
    auto pipeline = CreatePipeline(batch_size, num_threads_);
    if (input) {
      AddInputToPipeline(*pipeline, input);
    }
    auto outputs = RunTestImpl(*pipeline, operator_arguments, input ? true : false,
                                              op_backend, op_backend);
    verify(input, outputs[0], operator_arguments);
    output = outputs[0];
  }


 private:
  /**
   * Template method. This function serves as a place of definition of
   * operator graph, that will be tested within this fixture.
   * @return tested operator graph
   */
  virtual GraphDescr GenerateOperatorGraph() const = 0;


  void SetUp() final {
  }


  void TearDown() final {
  }

  std::vector<TensorListWrapper> RunTestImpl(Pipeline &pipeline,
                                             const Arguments &operator_arguments, bool has_inputs,
                                             const std::string &input_backend,
                                             const std::string &output_backend) {
    const auto op_spec = CreateOpSpec(GenerateOperatorGraph().get_op_name(), operator_arguments,
                                      has_inputs, input_backend, output_backend);
    AddOperatorToPipeline(pipeline, op_spec);
    BuildPipeline(pipeline, op_spec);
    RunPipeline(pipeline);
    return GetOutputsFromPipeline(pipeline, output_backend);
  }

  size_t num_threads_ = 1;
};

}  // namespace testing

}  // namespace dali

#endif  // DALI_TEST_DALI_OPERATOR_TEST_H_
