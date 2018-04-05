// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_TEST_NDLL_TEST_SINGLE_OP_H_
#define NDLL_TEST_NDLL_TEST_SINGLE_OP_H_

#include "ndll/test/ndll_test.h"

#include <gtest/gtest.h>

#include "ndll/pipeline/pipeline.h"

namespace ndll {

namespace {

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

}

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
class NDLLSingleOpTest : public NDLLTest {
 public:
  inline void SetUp() override {
    NDLLTest::SetUp();

    // encoded in jpegs_
    LoadJPEGS(jpeg_test_images, &jpegs_, &jpeg_sizes_);
    // decoded in images_
    DecodeJPEGS(NDLL_RGB);

    InitPipeline();
  }
  inline void TearDown() override {
    NDLLTest::TearDown();
  }

  void AddSingleOp(const OpSpec& spec) {
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
  Reference(const vector<TensorList<CPUBackend>*> &inputs) = 0;

  /**
   * Check the desireed calculated answers in ws (given by user-provided indices)
   * against a supplied reference implementation.
   */
  void CheckAnswers(DeviceWorkspace *ws, const vector<int>& output_indices = {}) {
    // outputs_ contains map of idx -> (name, device)
    vector<TensorList<CPUBackend>*> res = Reference(input_data_);

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
  void EncodedData(TensorList<CPUBackend>* t) {
    NDLLTest::MakeJPEGBatch(t, 32);
  }

  /**
   * Provide decoded (i.e. decoded JPEG) data
   */
  void DecodedData(TensorList<CPUBackend>* t) {
    NDLLTest::MakeImageBatch(32, t);
  }

 private:

  // use a Get mean, std-dev of difference
  template <typename T>
  void CheckBuffers(int N, const T *a, const T *b, double eps=1e-4) {
    double diff_sum = 0;

    vector<double> diff(N);

    double div_eps = 1e-7;
    double mean, std;
    for (int i = 0; i < N; ++i) {
      diff[i] = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    }
    MeanStdDev<double>(diff, &mean, &std);

    ASSERT_LE(abs(mean), eps);
  }

  void CheckTensorLists(const TensorList<CPUBackend> *t1,
                        const TensorList<CPUBackend> *t2) {
    ASSERT_TRUE(t1);
    ASSERT_TRUE(t2);
    ASSERT_EQ(t1->ntensor(), t2->ntensor());

    ASSERT_EQ(t1->size(), t2->size());

    if (IsType<float>(t1->type())) {
      CheckBuffers<float>(t1->ntensor(),
                          t1->data<float>(),
                          t2->data<float>());
    } else if (IsType<unsigned char>(t1->type())) {
      CheckBuffers<unsigned char>(t1->ntensor(),
                                  t1->data<unsigned char>(),
                                  t2->data<unsigned char>());
    }
  }

  void InitPipeline() {
    if (!pipeline_.get()) {
      pipeline_.reset(new Pipeline(32, 1, 0));
    }
  }
  vector<std::pair<string, TensorList<CPUBackend>*>> inputs_;
  vector<TensorList<CPUBackend>*> input_data_;
  vector<std::pair<string, string>> outputs_;
  shared_ptr<Pipeline> pipeline_;
};

#define USING_NDLL_SINGLE_OP_TEST() \
  using NDLLSingleOpTest::AddSingleOp; \
  using NDLLSingleOpTest::SetExternalInputs; \
  using NDLLSingleOpTest::EncodedData; \
  using NDLLSingleOpTest::DecodedData;

}  // namespace ndll

#endif  // NDLL_TEST_NDLL_TEST_SINGLE_OP_H_
