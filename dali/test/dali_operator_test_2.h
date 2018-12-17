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

#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <dali/pipeline/pipeline.h>
#include <dali/test/op_graph.h>

namespace dali {

namespace testing {

namespace detail {

/**
 * Return std::string name of given Backend
 */
template<typename Backend>
std::string BackendStringName();

} // namespace detail

using Arguments = std::map<std::string, double>; // TODO: some generalization. boost::any?


struct TensorListWrapper {};


class DaliOperatorTest : public ::testing::Test, public ::testing::WithParamInterface<Arguments> {
public:

  template<typename Backend>
  using Verify = std::function<void(TensorListWrapper, TensorListWrapper, Arguments)>;

protected:

  template<typename InputBackend, typename OutputBackend>
  void RunTest(const TensorListWrapper &inputs, const TensorListWrapper &outputs, Arguments operator_arguments,
               Verify<OutputBackend> verify) {
    InitPipeline(batch_size_, num_threads_);
    const auto op_spec = CreateOpSpec<InputBackend, OutputBackend>(operator_graph_->get_name(), operator_arguments);
    if (has_input_) {
      AddInputsToPipeline<InputBackend>(pipeline_.get(), inputs);
    }
//    AddOperatorToPipeline(pipeline_.get(), op_spec);
//    BuildPipeline(pipeline_.get(), op_spec);
//    RunPipeline(pipeline_.get());
//    auto output_batch = GetOutputsFromPipeline(pipeline_.get());
//    DALI_ENFORCE(output_batch.size() == anticipated_outputs.size(), "Sizes of outputs don't match");
//    for (size_t i = 0; i < anticipated_outputs.size(); i++) {
//      EXPECT_TRUE(Verify(output_batch[i], anticipated_outputs[i]))
//                    << "Verification fails for input_idx=" + std::to_string(i);
//    }
  }


private:
  virtual std::unique_ptr<OpDag> GenerateOperatorsGraph() const noexcept = 0;


  void SetUp() final {
//    const auto inp = SetInputs();
//    const auto num_inputs = inp.size();
//    assert(num_inputs >= 0);
//    has_input_ = num_inputs != 0;
//    if (has_input_) {
//      const auto input_pair = ReshapeInputType(inp);
//      input_batch_ = input_pair.first;
//      batch_size_ = input_batch_.size();
//      input_shape_ = input_pair.second[0];
//    }
//    operator_name_ = SetOperator();
  }


  void TearDown() final {}


  void InitPipeline(size_t batch_size, size_t num_threads) {
    if (!pipeline_.get()) {
      pipeline_.reset(new Pipeline(static_cast<int>(batch_size), static_cast<int>(num_threads), 0));
    }
  }


//  /**
//   * Converting vector of pairs to pair of vectors
//   */
//  std::pair<std::vector<Input>, std::vector<Shape>>
//  ReshapeInputType(const std::vector<std::pair<Input, Shape>> &input) const {
//    std::vector<Input> in(input.size());
//    std::vector<Shape> shape(input.size());
//    auto extract_input = [](const std::pair<Input, Shape> &input) -> Input { return input.first; };
//    auto extract_shape = [](const std::pair<Input, Shape> &input) -> Shape { return input.second; };
//    std::transform(input.begin(), input.end(), in.begin(), extract_input);
//    std::transform(input.begin(), input.end(), shape.begin(), extract_shape);
//    return std::make_pair(in, shape);
//  }


  template<typename Backend>
  void AddInputsToPipeline(Pipeline *pipeline, const Inputs<Backend> &input_batch) {
    const std::string input_name = "input";
    pipeline->AddExternalInput(input_name);
    pipeline->SetExternalInput(input_name, *input_batch);
  }


//  void AddOperatorToPipeline(Pipeline *pipeline, const OpSpec &op_spec) {
//    pipeline->AddOperator(op_spec);
//  }


//  DeviceWorkspace CreateWorkspace() const {
//    DeviceWorkspace ws;
//    return ws;
//  }


//  void RunPipeline(Pipeline *pipeline) {
//    pipeline->RunCPU();
//    pipeline->RunGPU();
//  }


//  std::vector<Output> GetOutputsFromPipeline(Pipeline *pipeline) {
//    auto workspace = CreateWorkspace();
//    pipeline->Outputs(&workspace);
//    auto tl = workspace.template Output<CPUBackend>(0);
//    return FromTensorList(*tl);
//  }


//  void BuildPipeline(Pipeline *pipeline, const OpSpec &spec) {
//    std::vector<std::pair<string, string>> vecoutputs_;
//    for (int i = 0; i < spec.NumOutput(); ++i) {
//      vecoutputs_.emplace_back(spec.OutputName(i), spec.OutputDevice(i));
//    }
//    pipeline->Build(vecoutputs_);
//  }


  template<typename InputBackend, typename OutputBackend>
  OpSpec CreateOpSpec(const std::string &operator_name, Arguments operator_arguments) const {
    OpSpec opspec = OpSpec(operator_name);
    for (const auto &arg : operator_arguments) {
      opspec.AddArg(arg.first, arg.second);
    }
    if (has_input_) {
      opspec.AddInput("input", detail::BackendStringName<InputBackend>());
    }
    opspec.AddOutput("output", detail::BackendStringName<OutputBackend>());
    return opspec;
  }


//  std::vector<Input> input_batch_;
//  Shape input_shape_;
//  std::string operator_name_;
  std::unique_ptr<Pipeline> pipeline_;
  size_t batch_size_ = 1;
  const size_t num_threads_ = 1;
  bool has_input_ = false;
  std::unique_ptr<OpDag> operator_graph_;

};

namespace detail {

template<typename Backend>
std::string BackendStringName() {
  DALI_FAIL("Selected Backend is not supported");
}


template<>
inline std::string BackendStringName<CPUBackend>() {
  return "cpu";
}


template<>
inline std::string BackendStringName<GPUBackend>() {
  return "gpu";
}


} // namespace detail

}  // namespace testing

}  // namespace dali

#endif // DALI_TEST_DALI_OPERATOR_TEST_H_

