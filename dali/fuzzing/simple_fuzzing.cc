// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/core/common.h"
#include "dali/operators.h"
#include "dali/pipeline/data/allocator.h"
#include "dali/pipeline/init.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/pipeline.h"
#include "dali/util/image.h"
#include "dali/test/dali_test_config.h"

using namespace std;
using namespace dali;

namespace dali {
const string jpeg_folder = make_string(
  testing::dali_extra_path(),
  "/db/fuzzing/");

class DecoderHarness {
 public:
  DecoderHarness(string &path, int batch_size=4, int num_threads=4, int device_id=0) :
      batch_size_(batch_size),
      pipeline_(batch_size, num_threads, device_id) {
    jpeg_names_ = ImageList(jpeg_folder, {".jpg"}, batch_size_ - 1);
    jpeg_names_.push_back(path);
    LoadImages(jpeg_names_, &jpegs_);

    MakeBatch();
    SetupPipeline();
  }

  void MakeBatch() {
    TensorListShape<> shape(batch_size_, 1);
    for (int i = 0; i < batch_size_; ++i) {
      shape.set_tensor_shape(i, {jpegs_.sizes_[i]});
    }

    input_data_.template mutable_data<uint8>();
    input_data_.Resize(shape);

    for (int i = 0; i < batch_size_; ++i) {
      std::memcpy(
        input_data_.template mutable_tensor<uint8>(i),
        jpegs_.data_[i],
        jpegs_.sizes_[i]);
      input_data_.SetSourceInfo(i, jpeg_names_[i] + "_" + std::to_string(i));
    }
  }

  void SetupPipeline() {
    pipeline_.AddExternalInput("jpegs");
    pipeline_.SetExternalInput("jpegs", input_data_);

    pipeline_.AddOperator(
      OpSpec("ImageDecoder")
        .AddArg("device", "mixed")
        .AddArg("output_type", DALI_RGB)
        .AddInput("jpegs", "cpu")
        .AddOutput("images", "gpu"));

    pipeline_.Build({{"images", "gpu"}});
  }

  void Run() {
    DeviceWorkspace ws;
    pipeline_.RunCPU();
    pipeline_.RunGPU();
    pipeline_.Outputs(&ws);
  }

 private:
  int batch_size_;
  Pipeline pipeline_;
  TensorList<CPUBackend> input_data_;

  vector<string> jpeg_names_;
  ImgSetDescr jpegs_;
};

}  // namespace dali

int main(int argc, char *argv[]) {
  // Parse and validate command line arg
  string path(argv[1]);

  // Init DALI
  dali::InitOperatorsLib();
  dali::DALIInit(
    dali::OpSpec("CPUAllocator"),
    dali::OpSpec("PinnedCPUAllocator"),
    dali::OpSpec("GPUAllocator"));

  // Build and run harness
  DecoderHarness harness(path);
  harness.Run();

  return 0;
}
