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
#include "dali/pipeline/operator/direct_operator.h"
#include "dali/pipeline/pipeline.h"
#include "dali/pipeline/pipeline_debug.h"

namespace dali {

void cropOperatorCallableTest() {
  OpSpec spec1("readers__File");
  spec1.AddArg("max_batch_size", 8);
  spec1.AddArg("device_id", 0);
  spec1.AddArg("file_root", "/home/ksztenderski/DALI_extra/db/single/jpeg");
  spec1.AddArg("shard_id", 0);
  spec1.AddArg("num_shards", 2);

  DirectOperator<CPUBackend> reader(spec1);

  std::vector<std::shared_ptr<TensorList<CPUBackend>>> inputs1{};
  std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> kwargs1;

  auto reader_out = reader.Run<CPUBackend, CPUBackend>(inputs1, kwargs1);

  OpSpec spec2("decoders__Image");
  spec2.AddArg("max_batch_size", 8);
  spec2.AddArg("device_id", 0);
  spec2.AddArg("output_type", 0);

  DirectOperator<CPUBackend> decoder(spec2);

  std::vector<std::shared_ptr<TensorList<CPUBackend>>> inputs2{reader_out[0]};
  std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> kwargs2;

  auto decoder_out = decoder.Run<CPUBackend, CPUBackend>(inputs2, kwargs2);
}

void test_op() {
  OpSpec spec("random__CoinFlip");
  spec.AddArg("max_batch_size", 8);
  spec.AddArg("device_id", 0);
  spec.AddArg("probability", 0.5f);

  DirectOperator<CPUBackend> op(spec);
  std::vector<std::shared_ptr<TensorList<CPUBackend>>> inputs{};
  std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> kwargs;

  auto out = op.Run<CPUBackend, CPUBackend>(inputs, kwargs);
}

void test_standerd_pipeline() {
  auto pipe_ptr = std::make_unique<Pipeline>(8, 1, 0, 43, true, 2, false);
  auto &pipe = *pipe_ptr;
  pipe.AddOperator(
      OpSpec("random__CoinFlip").AddArg("probability", 0.5f).AddOutput("outputs", "cpu"));

  std::vector<std::pair<std::string, std::string>> outputs = {{"outputs", "cpu"}};

  pipe.SetOutputNames(outputs);
  pipe.Build();
  pipe.RunCPU();
}

/*
@pipeline_def(batch_size=8, num_threads=3, device_id=0, seed=47, debug=True)
def rn50_pipeline_base():
    rng = fn.random.coin_flip(probability=0.5)
    jpegs, labels = fn.readers.file(file_root=file_root, shard_id=0, num_shards=2)
    images = fn.decoders.image(jpegs, device='mixed')
    resized_images = fn.random_resized_crop(images, size=(224, 224), seed=27, device='gpu')
    out_type = types.FLOAT16
    output = fn.crop_mirror_normalize(resized_images, mirror=rng, device='gpu', dtype=out_type, crop=(
        224, 224), mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    return images, output
*/

void test_debug_pipeline() {
  OpSpec spec1("random__CoinFlip");
  spec1.AddArg("probability", 0.5f);

  OpSpec spec2("readers__File");
  spec2.AddArg("file_root", "/home/ksztenderski/DALI_extra/db/single/jpeg");
  spec2.AddArg("shard_id", 0);
  spec2.AddArg("num_shards", 2);

  OpSpec spec3("decoders__Image");
  spec3.AddArg("device", "mixed");

  OpSpec spec4("RandomResizedCrop");
  spec4.AddArg("device", "gpu");
  spec4.AddArg("size", std::vector<int>{224, 224});
  spec4.AddArg("seed", 27);

  std::vector<std::shared_ptr<TensorList<CPUBackend>>> empty_inputs{};
  std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> empty_kwargs;

  PipelineDebug pipe(8, 3, 0);
  pipe.AddOperator(spec1, 0);
  pipe.AddOperator(spec2, 1);
  pipe.AddOperator(spec3, 2);
  pipe.AddOperator(spec4, 3);

  auto rng = pipe.RunOperator<CPUBackend, CPUBackend>(0, empty_inputs, empty_kwargs);
  auto jpegs_labels = pipe.RunOperator<CPUBackend, CPUBackend>(1, empty_inputs, empty_kwargs);
  std::vector<std::shared_ptr<TensorList<CPUBackend>>> jpegs{jpegs_labels[0]};
  auto images = pipe.RunOperator<CPUBackend, GPUBackend>(2, jpegs, empty_kwargs);
  auto resized_images = pipe.RunOperator<GPUBackend, GPUBackend>(3, images, empty_kwargs);
}

TEST(DirectOperatorsTest, TestOperator) {
  test_debug_pipeline();
}

}  // namespace dali
