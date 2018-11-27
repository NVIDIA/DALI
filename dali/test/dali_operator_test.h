// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
#include <utility>
#include <map>
#include <memory>
#include <algorithm>
#include <string>
#include "dali/pipeline/pipeline.h"
#include "dali/pipeline/data/backend.h"
#include "dali/test/datatype_conversions.h"

namespace dali {

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


}  // namespace detail

/**
 * Class for creating a test for pipeline, which contains a chain of operators
 *
 * In order to set up a test, create a custom test class, which will extend DaliOperatorTest.
 * There you have 4 things to define:
 * 1. Set of inputs
 * 2. Sequence of tested operators
 * 3. Method to verify anticipated_output vs output from operator execution
 * 4. Input and Output types (see below)
 * For details see corresponding functions' definitions.
 *
 * This will create GTest fixture for unit tests.
 *
 * To actually run the test, define `TEST_F` macro. There, a `DaliOperatorTest::RunTest(...)`
 * function shall be called.
 *
 * The operator test idea, regarding GTest workflow is as follows:
 * 1. TestSuite (==TestFixture) -> a test for given chain of ops
 *                                 (e.g. [Crop], [Resize -> Crop], [Crop->Resize], etc.)
 * 2. TestCase  (==TEST_F)      -> a test, for given set of OperatorArguments, that is
 *                                 performed for every input in provided input set.
 *
 * @tparam Input type of a single input to sequence of operators, e.g.
 *               std::array<float, 4>  -> for bounding box
 *               std::vector<float>    -> also can be applied for bounding box
 *               std::vector<float>    -> it can be an image
 *               float*                -> this also can be an image
 * @tparam Output type of a single output from sequence of operators
 */
template<typename Input, typename Output>
class DaliOperatorTest : public ::testing::Test {
 public:
  using Shape = std::vector<size_t>;
  using Arguments =  std::map<std::string, int>;  // TODO(mszolucha): Some generalization (boost::any?) NOLINT

  template<typename Backend>
  void RunTest(Arguments operator_arguments, std::vector<Output> anticipated_outputs) {
    InitPipeline(batch_size_, num_threads_);
    const auto op_spec = CreateOpSpec<Backend>(operator_name_, operator_arguments);
    if (has_input_) {
      AddInputsToPipeline<Backend>(pipeline_.get(), input_batch_, input_shape_);
    }
    AddOperatorToPipeline(pipeline_.get(), op_spec);
    BuildPipeline(pipeline_.get(), op_spec);
    RunPipeline(pipeline_.get());
    auto output_batch = GetOutputsFromPipeline(pipeline_.get());
    DALI_ENFORCE(output_batch.size() == anticipated_outputs.size(), "Sizes of outputs don't match");
    for (size_t i = 0; i < anticipated_outputs.size(); i++) {
      EXPECT_TRUE(Verify(output_batch[i], anticipated_outputs[i]))
                    << "Verification fails for input_idx=" + std::to_string(i);
    }
  }


 private:
  virtual std::vector<std::pair<Input, Shape>> SetInputs() const = 0;

  virtual std::string SetOperator() const = 0;  // TODO(mszolucha): Enable chained op (std::vector)

  virtual bool Verify(Output output, Output anticipated_output) const = 0;


  void SetUp() final {
    const auto inp = SetInputs();
    const auto num_inputs = inp.size();
    assert(num_inputs >= 0);
    has_input_ = num_inputs != 0;
    if (has_input_) {
      const auto input_pair = ReshapeInputType(inp);
      input_batch_ = input_pair.first;
      batch_size_ = input_batch_.size();
      input_shape_ = input_pair.second[0];
    }
    operator_name_ = SetOperator();
  }


  void TearDown() final {
  }


  void InitPipeline(size_t batch_size, size_t num_threads) {
    if (!pipeline_.get()) {
      pipeline_.reset(new Pipeline(static_cast<int>(batch_size), static_cast<int>(num_threads), 0));
    }
  }


  /**
   * Converting vector of pairs to pair of vectors
   */
  std::pair<std::vector<Input>, std::vector<Shape>>
  ReshapeInputType(const std::vector<std::pair<Input, Shape>> &input) const {
    std::vector<Input> in(input.size());
    std::vector<Shape> shape(input.size());
    auto extract_input = [](const std::pair<Input, Shape> &input) -> Input { return input.first; };
    auto extract_shape = [](const std::pair<Input, Shape> &input) -> Shape { return input.second; };
    std::transform(input.begin(), input.end(), in.begin(), extract_input);
    std::transform(input.begin(), input.end(), shape.begin(), extract_shape);
    return std::make_pair(in, shape);
  }


  template<typename Backend>
  void AddInputsToPipeline(Pipeline *pipeline, const std::vector<Input> &input_batch,
                           Shape input_shape) {
    const std::string input_name = "input";
    pipeline->AddExternalInput(input_name);
    auto tensor_list = ToTensorList<Backend>(input_batch, input_shape);
    pipeline->SetExternalInput(input_name, *tensor_list);
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


  std::vector<Output> GetOutputsFromPipeline(Pipeline *pipeline) {
    auto workspace = CreateWorkspace();
    pipeline->Outputs(&workspace);
    auto tl = workspace.template Output<CPUBackend>(0);
    return FromTensorList(*tl);
  }


  void BuildPipeline(Pipeline *pipeline, const OpSpec &spec) {
    std::vector<std::pair<string, string>> vecoutputs_;
    for (int i = 0; i < spec.NumOutput(); ++i) {
      vecoutputs_.emplace_back(spec.OutputName(i), spec.OutputDevice(i));
    }
    pipeline->Build(vecoutputs_);
  }


  template<typename Backend>
  OpSpec CreateOpSpec(const std::string &operator_name, Arguments operator_arguments) const {
    OpSpec opspec = OpSpec(operator_name);
    for (const auto &arg : operator_arguments) {
      opspec.AddArg(arg.first, arg.second);
    }
    if (has_input_) {
      opspec.AddInput("input", detail::BackendStringName<Backend>());
    }
    opspec.AddOutput("output", detail::BackendStringName<Backend>());
    return opspec;
  }


  std::vector<Input> input_batch_;
  Shape input_shape_;
  std::string operator_name_;
  std::unique_ptr<Pipeline> pipeline_;
  size_t batch_size_ = 1;
  const size_t num_threads_ = 1;
  bool has_input_ = false;
};

}  // namespace dali

#endif  // DALI_TEST_DALI_OPERATOR_TEST_H_
