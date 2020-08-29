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

#ifndef DALI_FUZZING_DALI_HARNESS_H_
#define DALI_FUZZING_DALI_HARNESS_H_

#include <string>
#include <vector>
#include <utility>

#include "dali/core/common.h"
#include "dali/pipeline/pipeline.h"
#include "dali/util/image.h"
#include "dali/test/dali_test_config.h"

namespace dali {
class FileListHarness {
 public:
  FileListHarness(
    string &path,
    string file_extension = ".jpg",
    int device_id = 0) :
      device_id_(device_id) {
    const string jpeg_folder = make_string(
      testing::dali_extra_path(),
      "/db/fuzzing/");
    image_names_ = ImageList(jpeg_folder, {file_extension});
    image_names_.push_back(path);
    batch_size_ = image_names_.size();
    LoadImages(image_names_, &images_);
    MakeBatch();
  }

  void MakeBatch() {
    TensorListShape<> shape(batch_size_, 1);
    for (int i = 0; i < batch_size_; ++i) {
      shape.set_tensor_shape(i, {images_.sizes_[i]});
    }

    input_data_.template mutable_data<uint8>();
    input_data_.Resize(shape);

    for (int i = 0; i < batch_size_; ++i) {
      std::memcpy(
        input_data_.template mutable_tensor<uint8>(i),
        images_.data_[i],
        images_.sizes_[i]);
      input_data_.SetSourceInfo(i, image_names_[i] + "_" + std::to_string(i));
    }
  }

  virtual void SetupPipeline(Pipeline &pipeline) = 0;

  void Run() {
    Pipeline pipeline(batch_size_, 4, device_id_);
    SetupPipeline(pipeline);

    DeviceWorkspace ws;
    pipeline.RunCPU();
    pipeline.RunGPU();
    pipeline.Outputs(&ws);
  }

  virtual ~FileListHarness() {}

 protected:
  int batch_size_;
  int device_id_;
  TensorList<CPUBackend> input_data_;
  vector<string> image_names_;
  ImgSetDescr images_;
};

class DecoderHarness : public FileListHarness {
 public:
  explicit DecoderHarness(string &path, int device_id = 0) :
    FileListHarness(path, ".jpg",  device_id) { }

  void SetupPipeline(Pipeline &pipeline) override {
    pipeline.AddExternalInput("raw_images");
    pipeline.SetExternalInput("raw_images", input_data_);

    pipeline.AddOperator(
      OpSpec("ImageDecoder")
        .AddArg("device", "mixed")
        .AddArg("output_type", DALI_RGB)
        .AddInput("raw_images", "cpu")
        .AddOutput("images", "gpu"));
    pipeline.Build({{"images", "gpu"}});
  }
};

class ResNetHarness : public FileListHarness {
 public:
  explicit ResNetHarness(string &path, int device_id = 0) :
    FileListHarness(path, ".bmp",  device_id) { }

  void SetupPipeline(Pipeline &pipeline) override {
    pipeline.AddExternalInput("raw_images");
    pipeline.SetExternalInput("raw_images", input_data_);

    pipeline.AddOperator(
      OpSpec("ImageDecoder")
        .AddArg("device", "cpu")
        .AddArg("output_type", DALI_RGB)
        .AddInput("raw_images", "cpu")
        .AddOutput("images", "cpu"));

    // Add uniform RNG
    pipeline.AddOperator(
        OpSpec("Uniform")
        .AddArg("device", "cpu")
        .AddArg("range", vector<float>{0, 1})
        .AddOutput("uniform1", "cpu"));

    pipeline.AddOperator(
        OpSpec("Uniform")
        .AddArg("device", "cpu")
        .AddArg("range", vector<float>{0, 1})
        .AddOutput("uniform2", "cpu"));

    pipeline.AddOperator(
        OpSpec("Uniform")
        .AddArg("device", "cpu")
        .AddArg("range", vector<float>{256, 480})
        .AddOutput("resize", "cpu"));

    // Add coin flip RNG for mirror mask
    pipeline.AddOperator(
        OpSpec("CoinFlip")
        .AddArg("device", "cpu")
        .AddArg("probability", 0.5f)
        .AddOutput("mirror", "cpu"));

    std::string resize_op = "FastResizeCropMirror";
    // Add a resize+crop+mirror op
    pipeline.AddOperator(
        OpSpec(resize_op)
        .AddArg("device", "cpu")
        .AddArg("crop", vector<float>{224, 224})
        .AddInput("images", "cpu")
        .AddArgumentInput("mirror", "mirror")
        .AddArgumentInput("crop_pos_x", "uniform1")
        .AddArgumentInput("crop_pos_y", "uniform2")
        .AddArgumentInput("resize_shorter", "resize")
        .AddOutput("resized", "cpu"));

    pipeline.AddOperator(
        OpSpec("CropMirrorNormalize")
        .AddArg("device", "cpu")
        .AddArg("dtype", DALI_FLOAT16)
        .AddArg("mean", vector<float>{128, 128, 128})
        .AddArg("std", vector<float>{1, 1, 1})
        .AddInput("resized", "cpu")
        .AddOutput("final_batch", "cpu"));

    // Build and run the pipeline
    pipeline.Build({{"final_batch", "gpu"}});
  }
};

}  // namespace dali

#endif  // DALI_FUZZING_DALI_HARNESS_H_
