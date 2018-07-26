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

#define MAKE_IMG_OUTPUT    0      // Make the output of compared (obtained and referenced) images
#if MAKE_IMG_OUTPUT
  #define PIXEL_STAT_FILE "pixelStatFile"  // Output of statistics for compared sets of images
#endif

namespace images {

const vector<string> jpeg_test_images = {
  image_folder + "/420.jpg",
  image_folder + "/422.jpg",
  image_folder + "/440.jpg",
  image_folder + "/444.jpg",
//  image_folder + "/gray.jpg",   // Batched decoding has a bug in nvJPEG when both grayscale
                                  //  and color images are decoded in the same batch
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

typedef enum {            // Checking:
  t_checkDefault    = 0,  //    combined vectors (all images, all colors)
  t_checkColorComp  = 1,  //    colors separately
  t_checkElements   = 2,  //    images separately
  t_checkAll        = 4,  //    everything (no assertion after first fail)
  t_checkNoAssert   = 8   //    no assertion even when test failed
} t_checkType;

typedef enum {
  t_undefinedImgType,
  t_jpegImgType,
  t_pngImgType,
} t_imgType;

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

  inline void SetTestCheckType(uint32_t type) {
    testCheckType_ = type;
  }

  inline bool TestCheckType(uint32_t type) const {
    return testCheckType_ & type;
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
#if MAKE_IMG_OUTPUT
        WriteHWCBatch<CPUBackend>(calc_host, "img");
        WriteHWCBatch<CPUBackend>(*ref_output, "ref");
#endif
        // check calculated vs. reference answers
        CheckTensorLists(&calc_host, ref_output);
      } else {
        auto calc_output = ws->Output<CPUBackend>(output_indices[i]);
        auto *ref_output = res.at(i);
#if MAKE_IMG_OUTPUT
        WriteHWCBatch<CPUBackend>(*calc_output, "img");
        WriteHWCBatch<CPUBackend>(*ref_output, "ref");
#endif
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
  // use a Get mean, std-dev of difference separately for each color component

  template <typename T>
  int CheckBuffers(int N, const T *a, const T *b, bool checkAll, double *pDiff = NULL) {
    const int jMax = TestCheckType(t_checkColorComp)?  c_ : 1;
    const int len = N / jMax;
    double mean = 0, std;
    vector<double> diff(len);
#ifndef PIXEL_STAT_FILE
    for (int j = 0; j < jMax; ++j) {
      for (int i = j; i < N; i += jMax)
        diff[i / jMax] = abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));

      MeanStdDev<double>(diff, &mean, &std);
      if (checkAll) {
        const auto diff = fabs(mean) - eps_;
        if (diff <= 0)
          continue;

        if (pDiff)
          *pDiff = diff;

        return j;
      }

      ASSERT_LE(fabs(mean), eps_), -1;
    }
#else
    static int fff;
    FILE *file = fopen(PIXEL_STAT_FILE".txt", "a");
    if (!fff++)
      fprintf(file, "Buffer Length: %7d (for each color component)\n", len);

    for (int j = 0; j < c_; ++j) {
      int pos = 0, neg = 0;
      for (int i = j; i < N; i += c_) {
        diff[i / c_] = abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (a[i] > b[i])
          pos++;
        else if (a[i] < b[i])
          neg++;
      }

      MeanStdDev<double>(diff, &mean, &std);
      fprintf(file, "      %1d    %8.2f     %8.2f       %7d      %7d      %7d\n",
              j, mean, std, len - pos - neg, pos, neg);
    }

    fclose(file);
#endif

    return -1;
  }

  void ReportTestFailure(double diff, int colorIdx, int idx = -1,
                         const vector<Index> *pShape = NULL) {
    if (TestCheckType(t_checkNoAssert))
      cout << "Test warning:";
    else
      cout << "Test failed:";

    if (TestCheckType(t_checkColorComp))
      cout << " color # " << colorIdx;

    if (idx >= 0)
      cout << " element # " << idx;

    if (pShape)
      cout << " (h, w) = (" << (*pShape)[0] << ", " << (*pShape)[1] << ")";

    cout << " fabs(mean) = " << diff + eps_ << " and it was expected to be <= " << eps_ << endl;
  }

  void CheckTensorLists(const TensorList<CPUBackend> *t1,
                        const TensorList<CPUBackend> *t2) {
    ASSERT_TRUE(t1);
    ASSERT_TRUE(t2);
    ASSERT_EQ(t1->ntensor(), t2->ntensor());

    ASSERT_EQ(t1->size(), t2->size());

    const bool floatType = IsType<float>(t1->type());
    if (!floatType && !IsType<unsigned char>(t1->type()))
      return;   // For now we check buffers only for "float" and "uchar"

    int failNumb = 0, colorIdx = 0;
    const bool checkAll = TestCheckType(t_checkAll);
    double diff;
    if (TestCheckType(t_checkElements)) {
      // The the results are checked for each element separately
      for (int i = 0; i < t1->ntensor(); ++i) {
        const auto shape1 = t1->tensor_shape(i);
        const auto shape2 = t2->tensor_shape(i);
        ASSERT_EQ(shape1.size(), 3);
        ASSERT_EQ(shape2.size(), 3);
        for (auto j = shape1.size(); j--;) {
          ASSERT_EQ(shape1[j], shape2[j]);
        }

        const int lenBuffer = shape1[0] * shape1[1] * shape1[2];

        if (floatType) {
          colorIdx = CheckBuffers<float>(lenBuffer,
                          (*t1).template tensor<float>(i),
                          (*t2).template tensor<float>(i), true, &diff);
        } else {
          colorIdx = CheckBuffers<unsigned char>(lenBuffer,
                          (*t1).template tensor<uint8>(i),
                          (*t2).template tensor<uint8>(i), true, &diff);
        }

        if (colorIdx >= 0) {
          // Test failed for colorIdx
          ReportTestFailure(diff, colorIdx, i, &shape1);
          failNumb++;
          if (!checkAll)
            break;
        }
      }
    } else {
      if (floatType) {
        colorIdx = CheckBuffers<float>(t1->size(),
                            t1->data<float>(),
                            t2->data<float>(), true, &diff);
      } else {
        colorIdx = CheckBuffers<unsigned char>(t1->size(),
                                    t1->data<unsigned char>(),
                                    t2->data<unsigned char>(), checkAll, &diff);
      }
      if (colorIdx >= 0 && checkAll)
        ReportTestFailure(diff, colorIdx);
    }

    if (!TestCheckType(t_checkNoAssert)) {
      ASSERT_EQ(failNumb, 0);
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
  uint32_t testCheckType_ = t_checkDefault;

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
