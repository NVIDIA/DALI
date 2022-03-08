// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <unordered_map>
#include <gtest/gtest.h>
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/pipeline/operator/callable_operator.h"
#include "dali/pipeline/pipeline.h"

namespace dali {

void cropOperatorCallableTest() {
  OpSpec spec1("readers__File");
  spec1.AddArg("max_batch_size", 8);
  spec1.AddArg("device_id", 0);
  spec1.AddArg("file_root", "/home/ksztenderski/DALI_extra/db/single/jpeg");
  spec1.AddArg("shard_id", 0);
  spec1.AddArg("num_shards", 2);

  CallableOperator<CPUBackend> reader(spec1);

  std::vector<std::shared_ptr<TensorList<CPUBackend>>> inputs1{};
  std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> kwargs1;

  auto reader_out = reader.Run<CPUBackend, CPUBackend>(inputs1, kwargs1);

  OpSpec spec2("decoders__Image");
  spec2.AddArg("max_batch_size", 8);
  spec2.AddArg("device_id", 0);
  spec2.AddArg("output_type", 0);

  CallableOperator<CPUBackend> decoder(spec2);

  std::vector<std::shared_ptr<TensorList<CPUBackend>>> inputs2{reader_out[0]};
  std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> kwargs2;

  auto decoder_out = decoder.Run<CPUBackend, CPUBackend>(inputs2, kwargs2);
}

void test_op() {
  OpSpec spec("random__CoinFlip");
  spec.AddArg("max_batch_size", 8);
  spec.AddArg("device_id", 0);
  spec.AddArg("probability", 0.5f);

  CallableOperator<CPUBackend> op(spec);
  std::vector<std::shared_ptr<TensorList<CPUBackend>>> inputs{};
  std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> kwargs;

  auto out = op.Run<CPUBackend, CPUBackend>(inputs, kwargs);
}

void test_pipeline() {
  auto pipe_ptr = std::make_unique<Pipeline>(8, 1, 0, 43, true, 2, false);
  auto &pipe = *pipe_ptr;
  pipe.AddOperator(
      OpSpec("random__CoinFlip").AddArg("probability", 0.5f).AddOutput("outputs", "cpu"));

  std::vector<std::pair<std::string, std::string>> outputs = {{"outputs", "cpu"}};

  pipe.SetOutputNames(outputs);
  pipe.Build();
  pipe.RunCPU();
}

TEST(CallableOperatorsTest, TestOperator) {
  test_op();
}

}  // namespace dali
