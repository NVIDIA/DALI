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

#ifndef DALI_TEST_DALI_TEST_SINGLE_OP_H_
#define DALI_TEST_DALI_TEST_SINGLE_OP_H_

#include "dali/test/dali_test.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/pipeline/pipeline.h"

namespace dali {

namespace images {

const vector<string> jpeg_test_images = {
  image_folder + "/420.jpg",
  image_folder + "/422.jpg",
  image_folder + "/440.jpg",
  image_folder + "/444.jpg",
  image_folder + "/gray.jpg",
  image_folder + "/411.jpg",
  image_folder + "/411-non-multiple-4-width.jpg",
  image_folder + "/420-odd-height.jpg",
  image_folder + "/420-odd-width.jpg",
  image_folder + "/420-odd-both.jpg",
  image_folder + "/422-odd-width.jpg"
};

const vector<string> png_test_images = {
  image_folder + "/png/000000000139.png",
  image_folder + "/png/000000000285.png",
  image_folder + "/png/000000000632.png",
  image_folder + "/png/000000000724.png",
  image_folder + "/png/000000000776.png",
  image_folder + "/png/000000000785.png",
  image_folder + "/png/000000000802.png",
  image_folder + "/png/000000000872.png",
  image_folder + "/png/000000000885.png",
  image_folder + "/png/000000001000.png",
  image_folder + "/png/000000001268.png"
};

}  // namespace images

// Define a virtual base class for single operator tests,
// where we want to add a single operator to a pipeline,
// run the pipe using known data, and compare the result to
// a reference solution.
//
// Implementaions must define:
//  OpSpec AddOp() - definition of the operator
//  void SetInputs() - define all external inputs to the graph
//  void GetOutputs() - retrieve all testable outputs from the graph
//  bool Compare() - Compare against a (supplied) reference implementation
class DALISingleOpTest : public DALITest {
 public:
  inline void SetUp() override {
    DALITest::SetUp();
    jpegs_.clear();
    jpeg_sizes_.clear();

    // encoded in jpegs_
    LoadJPEGS(images::jpeg_test_images, &jpegs_, &jpeg_sizes_);
    LoadImages(images::png_test_images, &png_, &png_sizes_);

    // decoded in images_
    DecodeImages(DALI_RGB, jpegs_, jpeg_sizes_, &jpeg_decoded_, &jpeg_dims_);
    DecodeImages(DALI_RGB, png_, png_sizes_, &png_decoded_, &png_dims_);

    // Set the pipeline batch size
    batch_size_ = 32;
  }
  inline void TearDown() override {
    DALITest::TearDown();
  }

  inline void SetBatchSize(int b) {
    batch_size_ = b;
  }

  inline void SetNumThreads(int t) {
    num_threads_ = t;
  }

  inline void SetEps(double e) {
    eps_ = e;
  }

  void AddSingleOp(const OpSpec& spec) {
    spec_ = spec;
    InitPipeline();
    // generate the output mapping for this op
    for (int i = 0; i < spec.NumOutput(); ++i) {
      auto output_name = spec.OutputName(i);
      auto output_device = spec.OutputDevice(i);

      outputs_.push_back(std::make_pair(output_name, output_device));
    }

    pipeline_->AddOperator(spec);

    pipeline_->Build(outputs_);
  }

  void SetExternalInputs(const vector<std::pair<string, TensorList<CPUBackend>*>> &inputs) {
    InitPipeline();
    inputs_ = inputs;
    for (auto& input : inputs) {
      input_data_.push_back(input.second);
      pipeline_->AddExternalInput(input.first);
      pipeline_->SetExternalInput(input.first, *input.second);
    }
  }

  void RunOperator(DeviceWorkspace *ws) {
    pipeline_->RunCPU();
    pipeline_->RunGPU();

    pipeline_->Outputs(ws);
  }

  virtual
  vector<TensorList<CPUBackend>*>
  Reference(const vector<TensorList<CPUBackend>*> &inputs,
            DeviceWorkspace *ws) = 0;

  /**
   * Check the desireed calculated answers in ws (given by user-provided indices)
   * against a supplied reference implementation.
   */
  void CheckAnswers(DeviceWorkspace *ws,
                    const vector<int>& output_indices) {
    // outputs_ contains map of idx -> (name, device)
    vector<TensorList<CPUBackend>*> res = Reference(input_data_, ws);

    // get outputs from pipeline, copy to host if necessary
    for (size_t i = 0; i < output_indices.size(); ++i) {
      auto output_device = outputs_[i].second;

      if (output_device == "gpu") {
        // output on GPU
        auto calc_output = ws->Output<GPUBackend>(output_indices[i]);

        // copy to host
        TensorList<CPUBackend> calc_host;
        calc_host.Copy(*calc_output, 0);

        auto *ref_output = res[i];

        // check calculated vs. reference answers
        CheckTensorLists(&calc_host, ref_output);
      } else {
        auto calc_output = ws->Output<CPUBackend>(output_indices[i]);
        auto *ref_output = res.at(i);

        // check calculated vs. reference answers
        CheckTensorLists(calc_output, ref_output);
      }
    }
  }

  /**
   * Provide some encoded data
   * TODO(slayton): Add different encodings
   */
  void EncodedJPEGData(TensorList<CPUBackend>* t, int n) {
    DALITest::MakeEncodedBatch(t, n, jpegs_, jpeg_sizes_);
  }

  void EncodedPNGData(TensorList<CPUBackend>* t, int n) {
    DALITest::MakeEncodedBatch(t, n, png_, png_sizes_);
  }

  /**
   * Provide decoded (i.e. decoded JPEG) data
   */
  void DecodedData(TensorList<CPUBackend>* t, int n,
                   DALIImageType type = DALI_RGB) {
    DALITest::MakeImageBatch(n, t, type);
  }

 private:
  // use a Get mean, std-dev of difference
  template <typename T>
  void CheckBuffers(int N, const T *a, const T *b) {
    vector<double> diff(N);

    double mean = 0, std;
    for (int i = 0; i < N; ++i) {
      diff[i] = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    }
    MeanStdDev<double>(diff, &mean, &std);

    ASSERT_LE(fabs(mean), eps_);
  }

  void CheckTensorLists(const TensorList<CPUBackend> *t1,
                        const TensorList<CPUBackend> *t2) {
    ASSERT_TRUE(t1);
    ASSERT_TRUE(t2);
    ASSERT_EQ(t1->ntensor(), t2->ntensor());

    ASSERT_EQ(t1->size(), t2->size());

    if (IsType<float>(t1->type())) {
      CheckBuffers<float>(t1->size(),
                          t1->data<float>(),
                          t2->data<float>());
    } else if (IsType<unsigned char>(t1->type())) {
      CheckBuffers<unsigned char>(t1->size(),
                                  t1->data<unsigned char>(),
                                  t2->data<unsigned char>());
    }
  }

  void InitPipeline() {
    if (!pipeline_.get()) {
      pipeline_.reset(new Pipeline(batch_size_, num_threads_, 0));
    }
  }
  vector<std::pair<string, TensorList<CPUBackend>*>> inputs_;
  vector<TensorList<CPUBackend>*> input_data_;
  vector<std::pair<string, string>> outputs_;
  shared_ptr<Pipeline> pipeline_;

  vector<uint8*> png_;
  vector<int> png_sizes_;

  vector<uint8*> jpeg_decoded_, png_decoded_;
  vector<DimPair> jpeg_dims_, png_dims_;


 protected:
  int batch_size_ = 32;
  int num_threads_ = 2;
  double eps_ = 1e-4;

  // keep a copy of the creation OpSpec for reference
  OpSpec spec_;
};

#define USING_DALI_SINGLE_OP_TEST() \
  using DALISingleOpTest::AddSingleOp; \
  using DALISingleOpTest::SetExternalInputs; \
  using DALISingleOpTest::EncodedJPEGData; \
  using DALISingleOpTest::DecodedData;

}  // namespace dali

#endif  // DALI_TEST_DALI_TEST_SINGLE_OP_H_
