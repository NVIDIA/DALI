// Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include <gtest/gtest.h>
#include <exception>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/test/dali_operator_test.h"

#include "dali/pipeline/pipeline.h"
#include "dali/test/test_tensors.h"


namespace dali {
namespace test {

class SplitMergeTest : public ::testing::Test {
 public:
  /**
   * @brief Generate input tensor that will be split.
   * This version uses Tensors that keep their sample_idx and batch size internally
   *
   */
  virtual TensorList<CPUBackend> GetInput(int iter_idx) {
    return GetInputImpl(iter_idx, false);
  }

  /**
   * @brief Customization point for the split used in succeeding iterations,
   * We use a functor to generate the `predicate` input based on sample index.
   */
  virtual std::vector<std::function<int(int)>> GetSplitGenerators() {
    static std::vector<std::function<int(int)>> split_generators = {
        [](int idx) { return idx % 2; },  // interleaved 1-by-1
        [](int idx) { return 0; },        // all false, to the right
        [](int idx) { return 1; },        // all true, to the left
        [](int idx) { return idx < 4; },  // uneven split
    };
    return split_generators;
  }

  std::function<int(int)> GetSplitGenerator(int iter_idx) {
    return GetSplitGenerators()[iter_idx];
  }

  int GetIterCount() {
    return static_cast<int>(GetSplitGenerators().size());
  }

  /**
   * @brief Generate the predicate to be used in this iteration
   */
  virtual TensorList<CPUBackend> GetPredicate(int iter_idx) {
    TensorList<CPUBackend> predicate;
    predicate.set_pinned(false);
    predicate.set_order(AccessOrder::host());

    predicate.Resize(uniform_list_shape(kBatchSize, TensorShape<0>{}), DALI_BOOL);
    auto split_gen = GetSplitGenerator(iter_idx);
    for (int i = 0; i < kBatchSize; i++) {
      *predicate.mutable_tensor<bool>(i) = split_gen(i);
    }
    return predicate;
  }

  /**
   * @brief Validate the outputs of the pipeline against the input
   */
  template <typename Backend>
  void Validate(int iter_idx, int pipe_output_idx, const Workspace &ws,
                const TensorList<CPUBackend> &input) {
    TensorList<CPUBackend> output;
    output.set_pinned(false);
    output.set_order(AccessOrder::host());
    output.Copy(ws.Output<Backend>(pipe_output_idx));
    EXPECT_EQ(output.shape(), input.shape());

    for (int i = 0; i < input.shape().num_samples(); i++) {
      for (int elem = 0; elem < input.shape()[i].num_elements(); elem++) {
        EXPECT_EQ(output.tensor<int32_t>(i)[elem], input.tensor<int32_t>(i)[elem]);
      }
    }
  }

  /**
   * @brief Check directly within the node if input or output data is pinned, observing via outputs
   * just looks at the last MakeContiguous node
   */
  void ValidateSplitPinned(Pipeline &pipe, const std::string &node_name, bool in_pinned_expected,
                           bool out_0_pinned_expected, bool out_1_pinned_expected) {
    auto op = pipe.GetOperator(node_name);
    auto in_pinned = op->GetDiagnostic<bool>("input_pinned");
    EXPECT_EQ(in_pinned, in_pinned_expected) << "[Split]: " << node_name;
    auto out_0_pinned = op->GetDiagnostic<bool>("output_0_pinned");
    EXPECT_EQ(out_0_pinned, out_0_pinned_expected) << "[Split]: " << node_name;
    auto out_1_pinned = op->GetDiagnostic<bool>("output_1_pinned");
    EXPECT_EQ(out_1_pinned, out_1_pinned_expected) << "[Split]: " << node_name;
  }


  /**
   * @brief Check directly within the node if input or output data is pinned, observing via outputs
   * just looks at the last MakeContiguous node
   */
  void ValidateMergePinned(Pipeline &pipe, const std::string &node_name, bool in_0_pinned_expected,
                           bool in_1_pinned_expected, bool out_pinned_expected) {
    auto op = pipe.GetOperator(node_name);
    auto in_0_pinned = op->GetDiagnostic<bool>("input_0_pinned");
    EXPECT_EQ(in_0_pinned, in_0_pinned_expected) << "[Merge]: " << node_name;
    auto in_1_pinned = op->GetDiagnostic<bool>("input_1_pinned");
    EXPECT_EQ(in_1_pinned, in_1_pinned_expected) << "[Merge]: " << node_name;
    auto out_pinned = op->GetDiagnostic<bool>("output_pinned");
    EXPECT_EQ(out_pinned, out_pinned_expected) << "[Merge]: " << node_name;
  }

  void AddExternalInput(Pipeline &pipe, const string &input_name = "input") {
    pipe.AddOperator(OpSpec("ExternalSource")
                         .AddArg("device", "cpu")
                         .AddArg("name", input_name)
                         .AddOutput(input_name, StorageDevice::CPU),
                     input_name);
  }

  /**
   * @brief Boilerplate code for defining inputs to the graph (input and pred nodes).
   */
  void AddExternalInputs(Pipeline &pipe) {
    AddExternalInput(pipe, "input");
    AddExternalInput(pipe, "pred");
  }

  void AddSplit(Pipeline &pipe, const std::string &name, const std::string &dev,
                const std::string &input, const std::string &predicate,
                const std::string &true_output, const std::string &false_output) {
    auto storage_dev = ParseStorageDevice(dev);
    pipe.AddOperator(OpSpec("_conditional__Split")
                         .AddArg("device", dev)
                         .AddInput(input, storage_dev)
                         .AddArgumentInput("predicate", predicate)
                         .AddOutput(true_output, storage_dev)
                         .AddOutput(false_output, storage_dev),
                     name);
  }

  void AddMerge(Pipeline &pipe, const std::string &name, const std::string &dev,
                const std::string &true_input, const std::string &false_input,
                const std::string &predicate, const std::string &output) {
    auto storage_dev = ParseStorageDevice(dev);
    pipe.AddOperator(OpSpec("_conditional__Merge")
                         .AddArg("device", dev)
                         .AddInput(true_input, storage_dev)
                         .AddInput(false_input, storage_dev)
                         .AddArgumentInput("predicate", predicate)
                         .AddOutput(output, storage_dev),
                     name);
  }

  /**
   * @brief Do the boilerplate part of the test loop, feed the input and predicate external source,
   * and run the pipeline. Returns the {input, predicate} pair for later use in test.
   * @return {input, predicate} for use in the test body.
   */
  std::tuple<TensorList<CPUBackend>, TensorList<CPUBackend>> FeedAndRun(Pipeline &pipe,
                                                                        int iter_idx) {
    auto input = GetInput(iter_idx);
    auto predicate = GetPredicate(iter_idx);
    pipe.SetExternalInput("input", input);
    pipe.SetExternalInput("pred", predicate);

    pipe.Run();
    return {std::move(input), std::move(predicate)};
  }

  static constexpr int kBatchSize = 10;

 protected:
  TensorList<CPUBackend> GetInputImpl(int iter_idx, bool pinned = false) {
    auto shape = uniform_list_shape(kBatchSize, {1, 1, 3});
    TensorList<CPUBackend> input;
    input.set_pinned(pinned);
    input.set_order(AccessOrder::host());
    input.Resize(shape, DALI_INT32);
    for (int i = 0; i < shape.num_samples(); i++) {
      for (int elem = 0; elem < shape[i].num_elements(); elem++) {
        input.mutable_tensor<int32_t>(i)[elem] = iter_idx * kBatchSize + i;
      }
    }
    return input;
  }
};

template <typename T>
class SplitMergeTyped : public SplitMergeTest {};

typedef ::testing::Types<CPUBackend, GPUBackend> Backends;

TYPED_TEST_SUITE(SplitMergeTyped, Backends);

TEST_F(SplitMergeTest, SplitCpuMergeGpu) {
  Pipeline pipe(kBatchSize, 4, 0);
  AddExternalInputs(pipe);

  AddSplit(pipe, "split", "cpu", "input", "pred", "split_0", "split_1");
  AddMerge(pipe, "merge", "gpu", "split_0", "split_1", "pred", "merge");

  vector<std::pair<string, string>> outputs = {{"merge", "gpu"}};
  pipe.Build(outputs);

  for (int iter_idx = 0; iter_idx < GetIterCount(); iter_idx++) {
    auto [input, predicate] = FeedAndRun(pipe, iter_idx);

    Workspace ws;
    pipe.Outputs(&ws);

    Validate<GPUBackend>(iter_idx, 0, ws, input);
  }
}

/**
 * @brief Trigger pinning one branch in split, and see if both are pinned.
 */
TEST_F(SplitMergeTest, PinnedInside) {
  Pipeline pipe(kBatchSize, 4, 0);
  AddExternalInputs(pipe);

  // we can see impact of pinning
  pipe.AddOperator(OpSpec("Copy").AddInput("input", StorageDevice::CPU)
                                 .AddOutput("input_copy", StorageDevice::CPU),
                   "input_copy");

  AddSplit(pipe, "split", "cpu", "input_copy", "pred", "split_0", "split_1");

  // copy it, so we don't pin split_0 due to passing it to GPU, but to check it is required
  // to be pinned for consistency reasons
  pipe.AddOperator(OpSpec("Copy").AddInput("split_0", StorageDevice::CPU)
                                 .AddOutput("split_0_copy", StorageDevice::CPU),
                   "split_0_copy");

  // this should be made pinned, thus making the input_copy pinned.
  pipe.AddOperator(OpSpec("MakeContiguous")
                       .AddArg("device", "mixed")
                       .AddInput("split_1", StorageDevice::CPU)
                       .AddOutput("split_1_contiguous", StorageDevice::GPU),
                   "make_contiguous");

  // as the split_1 is made pinned, split_0 also should be pinned due to coming together into merge
  AddMerge(pipe, "merge_cpu", "cpu", "split_0", "split_1", "pred", "merge_cpu");

  // consume the data transferred to GPU
  AddMerge(pipe, "merge_gpu", "gpu", "split_0", "split_1", "pred", "merge_gpu");

  vector<std::pair<string, string>> outputs = {{"merge_cpu", "cpu"},
                                               {"merge_gpu", "gpu"},
                                               {"split_0", "cpu"},
                                               {"split_1", "cpu"},
                                               {"input_copy", "cpu"}};
  pipe.Build(outputs);

  for (int iter_idx = 0; iter_idx < GetIterCount(); iter_idx++) {
    auto [input, predicate] = FeedAndRun(pipe, iter_idx);

    Workspace ws;
    pipe.Outputs(&ws);

    Validate<CPUBackend>(iter_idx, 0, ws, input);
    Validate<GPUBackend>(iter_idx, 1, ws, input);

    ValidateMergePinned(pipe, "merge_cpu", true, true, true);
    ValidateSplitPinned(pipe, "split", true, true, true);
  }
}

TEST_F(SplitMergeTest, PinnedThroughMerge) {
  Pipeline pipe(kBatchSize, 4, 0);
  AddExternalInputs(pipe);

  // we can see impact of pinning
  pipe.AddOperator(OpSpec("Copy").AddInput("input", StorageDevice::CPU)
                                 .AddOutput("input_copy", StorageDevice::CPU),
                   "input_copy");

  AddSplit(pipe, "split", "cpu", "input_copy", "pred", "split_0", "split_1");

  // We will cause the output to be pinned, so the input to split should also be pinned
  AddMerge(pipe, "merge_cpu", "cpu", "split_0", "split_1", "pred", "merge_cpu");

  // consume the data transferred to GPU
  pipe.AddOperator(OpSpec("MakeContiguous")
                       .AddArg("device", "mixed")
                       .AddInput("merge_cpu", StorageDevice::CPU)
                       .AddOutput("merge_gpu", StorageDevice::GPU),
                   "merge_gpu");

  vector<std::pair<string, string>> outputs = {
      {"merge_cpu", "cpu"}, {"input_copy", "cpu"}, {"merge_gpu", "gpu"}};
  pipe.Build(outputs);

  for (int iter_idx = 0; iter_idx < GetIterCount(); iter_idx++) {
    auto [input, predicate] = FeedAndRun(pipe, iter_idx);

    Workspace ws;
    pipe.Outputs(&ws);

    Validate<CPUBackend>(iter_idx, 0, ws, input);
    Validate<GPUBackend>(iter_idx, 2, ws, input);

    ValidateSplitPinned(pipe, "split", true, true, true);
    ValidateMergePinned(pipe, "merge_cpu", true, true, true);
  }
}


/**
 * @brief Split and Merge in the same stage.
 */
TYPED_TEST(SplitMergeTyped, SimpleCase) {
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  auto backend = testing::detail::BackendStringName<TypeParam>();

  Pipeline pipe(this->kBatchSize, 4, 0);
  this->AddExternalInputs(pipe);

  this->AddSplit(pipe, "split", backend, "input", "pred", "split_0", "split_1");

  auto storage_dev = ParseStorageDevice(backend);
  pipe.AddOperator(OpSpec("Copy")
                       .AddArg("device", backend)
                       .AddInput("split_0", storage_dev)
                       .AddOutput("split_0_copy", storage_dev),
                   "copy_0");

  pipe.AddOperator(OpSpec("Copy")
                       .AddArg("device", backend)
                       .AddInput("split_1", storage_dev)
                       .AddOutput("split_1_copy", storage_dev),
                   "copy_1");

  this->AddMerge(pipe, "merge", backend, "split_0_copy", "split_1_copy", "pred", "merge");

  vector<std::pair<string, string>> outputs = {{"merge", backend}};
  pipe.Build(outputs);

  for (int iter_idx = 0; iter_idx < this->GetIterCount(); iter_idx++) {
    auto [input, predicate] = this->FeedAndRun(pipe, iter_idx);

    Workspace ws;
    pipe.Outputs(&ws);

    this->template Validate<TypeParam>(iter_idx, 0, ws, input);
    if (!is_device) {
      this->ValidateSplitPinned(pipe, "split", false, false, false);
      this->ValidateMergePinned(pipe, "merge", false, false, false);
    }
  }
}

// Negative tests
TYPED_TEST(SplitMergeTyped, ReturnSplit) {
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  auto backend = testing::detail::BackendStringName<TypeParam>();

  Pipeline pipe(this->kBatchSize, 4, 0);
  this->AddExternalInputs(pipe);

  this->AddSplit(pipe, "split", backend, "input", "pred", "split_0", "split_1");

  vector<std::pair<string, string>> outputs = {{"split_0", backend}, {"split_1", backend}};
  pipe.Build(outputs);

  for (int iter_idx = 0; iter_idx < this->GetIterCount(); iter_idx++) {
    auto [input, predicate] = this->FeedAndRun(pipe, iter_idx);

    Workspace ws;
    pipe.Outputs(&ws);
    TensorList<CPUBackend> output[2];
    for (int i = 0; i < 2; i++) {
      output[i].set_pinned(false);
      output[i].Copy(ws.Output<TypeParam>(i));
    }
    int idxs[2] = {0, 0};
    for (int i = 0; i < this->kBatchSize; i++) {
      // the indexing is reversed, truthy values go to (0) output, falsy to (1)
      int which = !*predicate.template tensor<bool>(i);
      int output_sample_idx = idxs[which];
      idxs[which]++;
      EXPECT_LT(output_sample_idx, output[which].shape().num_samples());
      EXPECT_EQ(output[which][output_sample_idx].shape(), input[i].shape());
      for (int elem = 0; elem < input[i].shape().num_elements(); elem++) {
        EXPECT_EQ((output[which].template tensor<int32_t>(output_sample_idx)[elem]),
                  (input.template tensor<int32_t>(i)[elem]));
      }
    }
  }
}

class SplitMergeNegativeTest : public SplitMergeTest {
  // Unbiased splits only
  std::vector<std::function<int(int)>> GetSplitGenerators() override {
    static std::vector<std::function<int(int)>> split_generators = {
        [](int idx) { return idx < 3; },  // uneven split
    };
    return split_generators;
  }
};

TEST_F(SplitMergeNegativeTest, MismatchedMerge) {
  Pipeline pipe(kBatchSize, 4, 0);
  AddExternalInputs(pipe);

  AddSplit(pipe, "split", "cpu", "input", "pred", "split_0", "split_1");

  // Try to merge two bigger parts
  AddMerge(pipe, "merge", "cpu", "split_0", "split_0", "pred", "merge");

  vector<std::pair<string, string>> outputs = {{"merge", "cpu"}};
  pipe.Build(outputs);

  for (int iter_idx = 0; iter_idx < GetIterCount(); iter_idx++) {
    auto input = GetInput(iter_idx);
    auto predicate = GetPredicate(iter_idx);
    pipe.SetExternalInput("input", input);
    pipe.SetExternalInput("pred", predicate);

    try {
      pipe.Run();
      Workspace ws;
      pipe.Outputs(&ws);
      FAIL() << "Exception was expected but was not thrown.";
    } catch (std::exception &e) {
      static const char expected[] = "Merge description must cover whole input, got ";
      EXPECT_NE(std::string(e.what()).rfind(expected), std::string::npos)
          << expected << "\n====\nvs\n====\n"
          << e.what();
    } catch (...) { FAIL() << "Unexpected exception."; }
  }
}

TEST_F(SplitMergeNegativeTest, MismatchedSplit) {
  Pipeline pipe(kBatchSize, 4, 0);
  AddExternalInputs(pipe);

  // split the predicates
  AddSplit(pipe, "split_pred", "cpu", "pred", "pred", "pred_left", "pred_right");

  // split the input
  AddSplit(pipe, "split_input", "cpu", "input", "pred", "input_left", "input_right");

  // try to split smaller input with bigger predicate
  AddSplit(pipe, "split", "cpu", "input_left", "pred_right", "split_0", "split_1");

  vector<std::pair<string, string>> outputs = {{"split_0", "cpu"}, {"split_1", "cpu"}};
  pipe.Build(outputs);

  for (int iter_idx = 0; iter_idx < GetIterCount(); iter_idx++) {
    auto input = GetInput(iter_idx);
    auto predicate = GetPredicate(iter_idx);
    pipe.SetExternalInput("input", input);
    pipe.SetExternalInput("pred", predicate);

    try {
      pipe.Run();
      Workspace ws;
      pipe.Outputs(&ws);
      FAIL() << "Exception was expected but was not thrown.";
    } catch (std::exception &e) {
      static const char expected[] = "Split description must cover whole input, got ";
      EXPECT_NE(std::string(e.what()).rfind(expected), std::string::npos)
          << expected << "\n====\nvs\n====\n"
          << e.what();
    } catch (...) { FAIL() << "Unexpected exception."; }
  }
}

TEST_F(SplitMergeNegativeTest, MismatchedTypes) {
  Pipeline pipe(kBatchSize, 4, 0);
  AddExternalInput(pipe, "input_i32");
  AddExternalInput(pipe, "input_f32");
  AddExternalInput(pipe, "pred");

  AddSplit(pipe, "split_input_i32", "cpu", "input_i32", "pred", "split_i32_0", "split_i32_1");
  AddSplit(pipe, "split_input_f32", "cpu", "input_f32", "pred", "split_f32_0", "split_f32_1");

  AddMerge(pipe, "merge", "cpu", "split_i32_0", "split_f32_1", "pred", "merge");

  vector<std::pair<string, string>> outputs = {{"merge", "cpu"}};
  pipe.Build(outputs);

  for (int iter_idx = 0; iter_idx < GetIterCount(); iter_idx++) {
    auto input_i32 = GetInput(iter_idx);
    auto input_f32 = GetInput(iter_idx);
    // just change the type for, we care only about metadata
    input_f32.Resize(input_f32.shape(), DALI_FLOAT);
    auto predicate = GetPredicate(iter_idx);
    pipe.SetExternalInput("input_i32", input_i32);
    pipe.SetExternalInput("input_f32", input_f32);
    pipe.SetExternalInput("pred", predicate);

    try {
      pipe.Run();
      Workspace ws;
      pipe.Outputs(&ws);
      FAIL() << "Exception was expected but was not thrown.";
    } catch (std::exception &e) {
      static const char expected[] = "Found distinct types: int32 and float.";
      EXPECT_NE(std::string(e.what()).rfind(expected), std::string::npos)
          << expected << "\n====\nvs\n====\n"
          << e.what();
    } catch (...) { FAIL() << "Unexpected exception."; }
  }
}

class SplitMergePinnedInputsTest : public SplitMergeTest {
 public:
  TensorList<CPUBackend> GetPinnedInput(int iter_idx) {
    return GetInputImpl(iter_idx, true);
  }
};

TEST_F(SplitMergePinnedInputsTest, Mixes) {
  Pipeline pipe(kBatchSize, 4, 0);
  AddExternalInputs(pipe);

  pipe.AddOperator(OpSpec("ExternalSource")
                       .AddArg("device", "cpu")
                       .AddArg("name", "input")
                       .AddOutput("pinned_input", StorageDevice::CPU),
                   "pinned_input");

  AddSplit(pipe, "split", "cpu", "input", "pred", "split_0", "split_1");
  AddSplit(pipe, "split_pinned", "cpu", "pinned_input", "pred", "split_pinned_0", "split_pinned_1");

  AddMerge(pipe, "merge_nn", "cpu", "split_0", "split_1", "pred", "merge_nn");
  AddMerge(pipe, "merge_pp", "cpu", "split_pinned_0", "split_pinned_1", "pred", "merge_pp");
  AddMerge(pipe, "merge_pn", "cpu", "split_pinned_0", "split_1", "pred", "merge_pn");
  AddMerge(pipe, "merge_np", "cpu", "split_0", "split_pinned_1", "pred", "merge_np");

  vector<std::pair<string, string>> outputs = {
      {"merge_nn", "cpu"}, {"merge_pp", "cpu"}, {"merge_pn", "cpu"}, {"merge_np", "cpu"}};
  pipe.Build(outputs);

  for (int iter_idx = 0; iter_idx < GetIterCount(); iter_idx++) {
    auto input = GetInput(iter_idx);
    auto pinned_input = GetPinnedInput(iter_idx);
    auto predicate = GetPredicate(iter_idx);
    pipe.SetExternalInput("input", input);
    pipe.SetExternalInput("pinned_input", pinned_input);
    pipe.SetExternalInput("pred", predicate);

    pipe.Run();
    Workspace ws;
    pipe.Outputs(&ws);

    Validate<CPUBackend>(iter_idx, 0, ws, input);
    Validate<CPUBackend>(iter_idx, 1, ws, input);
    Validate<CPUBackend>(iter_idx, 2, ws, input);
    Validate<CPUBackend>(iter_idx, 3, ws, input);

    ValidateSplitPinned(pipe, "split", false, false, false);
    ValidateSplitPinned(pipe, "split_pinned", true, true, true);
    ValidateMergePinned(pipe, "merge_nn", false, false, false);
    ValidateMergePinned(pipe, "merge_pp", true, true, true);
    ValidateMergePinned(pipe, "merge_pn", true, false, true);
    ValidateMergePinned(pipe, "merge_np", false, true, true);
  }
}

}  // namespace test
}  // namespace dali
