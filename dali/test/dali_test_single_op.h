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
                                           // Use "" to make the output in stdout
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

typedef enum {
  t_loadJPEGs   = 1,
  t_decodeJPEGs = 2,
  t_loadPNGs    = 4,
  t_decodePNGs  = 8
} t_loadingFlags;

typedef enum {
  t_intParam,
  t_floatParam,
  t_stringParam,
  t_floatVector
} t_paramType;

typedef struct  {
  const char *m_Name;
  const char *m_val;
  t_paramType type;
} OpArg;

class opDescr {
 public:
  explicit opDescr(const char *name, double eps = 0.0, const vector<OpArg> *argPntr = NULL) :
                   opName(name), epsVal(eps), args(argPntr) {}
  const char *opName;
  double epsVal;
  const vector<OpArg> *args;
};

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
template <typename ImgType>
class DALISingleOpTest : public DALITest {
 public:
  inline void SetUp() override {
    DALITest::SetUp();
    c_ = (IsColor(img_type_) ? 3 : 1);
    jpegs_.clear();

    const auto flags = GetImageLoadingFlags();

    if (flags & t_loadJPEGs) {
      LoadJPEGS(images::jpeg_test_images, &jpegs_);
      if (flags & t_decodeJPEGs)
        DecodeImages(DALI_RGB, jpegs_, &jpeg_decoded_, &jpeg_dims_);
    }

    if (flags & t_loadPNGs) {
      LoadImages(images::png_test_images, &png_);

      if (flags & t_decodePNGs)
        DecodeImages(DALI_RGB, png_, &png_decoded_, &png_dims_);
    }

    // Set the pipeline batch size
    SetBatchSize(32);
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

  virtual uint8 GetTestCheckType() const {
    return t_checkDefault;
  }

  void AddOperatorWithOutput(const OpSpec& spec) {
    // generate the output mapping for this op
    for (int i = 0; i < spec.NumOutput(); ++i)
      outputs_.push_back(std::make_pair(spec.OutputName(i), spec.OutputDevice(i)));

    pipeline_->AddOperator(spec);
  }

  void AddOperatorWithOutput(const opDescr &descr, const string &pDevice = "cpu",
                             const string &pInput = "input", const string &pOutput = "outputCPU") {
    OpSpec spec(descr.opName);
    AddOperatorWithOutput(AddArguments(&spec, descr.args)
                            .AddInput(pInput, pDevice)
                            .AddOutput(pOutput, pDevice));
  }

  void AddSingleOp(const OpSpec& spec) {
    spec_ = spec;
    InitPipeline();
    AddOperatorWithOutput(spec);
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
    SetTestCheckType(GetTestCheckType());
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

      delete res[i];
    }
  }

  /**
   * Provide some encoded data
   * TODO(slayton): Add different encodings
   */
  void EncodedJPEGData(TensorList<CPUBackend>* t) {
    DALITest::MakeEncodedBatch(t, batch_size_, jpegs_);
  }

  void EncodedPNGData(TensorList<CPUBackend>* t) {
    DALITest::MakeEncodedBatch(t, batch_size_, png_);
  }

  /**
   * Provide decoded (i.e. decoded JPEG) data
   */
  void DecodedData(TensorList<CPUBackend>* t, int n,
                   DALIImageType type = DALI_RGB) {
    DALITest::MakeImageBatch(n, t, type);
  }

 protected:
  inline shared_ptr<Pipeline> GetPipeline() const {
    return pipeline_;
  }

  virtual uint32_t GetImageLoadingFlags() const   {
    return t_loadJPEGs;   // Only loading of JPEG files
  }

  const OpSpec &GetOperationSpec() const          {
    return spec_;
  }

  DALIImageType ImageType() const                 {
    return img_type_;
  }

  void TstBody(const string &pName, const string &pDevice = "gpu", double eps = 2e-1) {
    OpSpec operation = DefaultSchema(pName, pDevice);
    TstBody(operation, eps);
  }

  void TstBody(const OpSpec &operation, double eps = 2e-1, bool flag = true) {
    TensorList<CPUBackend> data;
    DecodedData(&data, this->batch_size_, this->img_type_);
    if (flag)
      SetExternalInputs({std::make_pair("input", &data)});

    RunOperator(operation, eps);
  }

  virtual OpSpec DefaultSchema(const string &pName, const string &pDevice = "gpu") const {
    return OpSpec(pName)
      .AddArg("device", pDevice)
      .AddArg("output_type", this->img_type_)
      .AddInput("input", pDevice)
      .AddOutput("output", pDevice);
  }

  OpSpec AddArguments(OpSpec *spec, const vector<OpArg> *args) const {
    if (!args || args->empty())
      return *spec;

    for (auto param : *args) {
      auto val = param.m_val;
      auto name = param.m_Name;
      switch (param.type) {
        case t_intParam:
          spec->AddArg(name, atoi(val));
          break;
        case t_floatParam:
          spec->AddArg(name, strtof(val, NULL));
          break;
        case t_stringParam:
          spec->AddArg(name, val);
          break;
        case t_floatVector: {
          const auto len = strlen(val);
          vector<float> vect;
          char *pEnd, *pTmp = new char[len+1];
          memcpy(pEnd = pTmp, val, len);
          pEnd[len] = '\0';
          while (pEnd[0]) {
            if (pEnd[0] == ',')
              pEnd++;

            vect.push_back(strtof(pEnd, &pEnd));
          }

          delete [] pTmp;
          spec->AddArg(name, vect);
        }
      }
    }

    return *spec;
  }

  void RunOperator(const opDescr &descr) {
    OpSpec spec(DefaultSchema(descr.opName));
    RunOperator(AddArguments(&spec, descr.args), descr.epsVal);
  }

  void RunOperator(const OpSpec& spec, double eps, DeviceWorkspace *pWS = NULL) {
    AddSingleOp(spec);

    DeviceWorkspace ws;
    if (!pWS)
      pWS = &ws;

    RunOperator(pWS);

    SetEps(eps);
    CheckAnswers(pWS, {0});
  }

  template <typename T>
  vector<TensorList<CPUBackend> *>CopyToHost(const TensorList<T> &calcOutput) {
    // copy to host
    vector<TensorList<CPUBackend> *> outputs(1);
    outputs[0] = new TensorList<CPUBackend>();
    outputs[0]->Copy(calcOutput, 0);
    return outputs;
  }

  template <typename T>
  int CheckBuffers(int N, const T *a, const T *b, bool checkAll, double *pMean = NULL) const {
    // use a Get mean, std-dev of difference separately for each color component
    const int jMax = TestCheckType(t_checkColorComp)?  c_ : 1;
    const int len = N / jMax;
    double mean = 0, std = 0;
    vector<double> diff(len);
    int retVal = -1;
#ifndef PIXEL_STAT_FILE
    for (int j = 0; j < jMax; ++j) {
      for (int i = j; i < N; i += jMax)
        diff[i / jMax] = abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));

      MeanStdDevColorNorm<double>(diff, &mean, &std);
      if (checkAll) {
        const auto diff = mean - eps_;
        if (diff <= 0)
          continue;

        if (pMean)
          *pMean = mean;

        return j;
      }

      ASSERT_LE(mean, eps_), -1;
    }
#else
    static int fff = 0;
    FILE *file = NULL;
    if (!fff) {
      // Header of the pixel statistic table
      const char *pHeader =
        "\nImgID: ClrID:     Mean:        Std:      SameValue:     Bigger:         Less:";

      if (strlen(PIXEL_STAT_FILE)) {
        file = fopen(PIXEL_STAT_FILE".txt", fff ? "a" : "w");
        fprintf(file, "%s", pHeader);
      } else {
        cout << pHeader;
      }
    }

    char buffer[256];
    snprintf(buffer, sizeof(buffer), "%s%3d:", (fff % 32? "" : "\n"), fff);
    fff++;

    // Image number
    if (file)
      fprintf(file, "%s", buffer);
    else
      cout << buffer;

    for (int j = 0; j < c_; ++j) {
      int pos = 0, neg = 0;
      for (int i = j; i < N; i += c_) {
        diff[i / c_] = abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (a[i] > b[i])
          pos++;
        else if (a[i] < b[i])
          neg++;
      }

      MeanStdDevColorNorm<double>(diff, &mean, &std);
      snprintf(buffer, sizeof(buffer),
               "%s     %1d    %8.2f     %8.2f       %7d      %7d      %7d\n",
               j? "    " : "", j, mean, std, len - pos - neg, pos, neg);

      if (file)
        fprintf(file, "%s", buffer);
      else
        cout << buffer;

      if (mean > eps_) {
        if (retVal < 0) {
          retVal = j;       // First violation of the boundary found
          if (pMean)        // Keep the color index as a return value
            *pMean = mean;
        } else {
          if (pMean && *pMean < mean) {
            *pMean = mean;  // More strong violation of the boundary found
            retVal = j;     // Change the color index as a return value
          }
        }
      }
    }

    if (file)
      fclose(file);
#endif

    return retVal;
  }

  void ReportTestFailure(double mean, int colorIdx, int idx = -1,
                         const vector<Index> *pShape = NULL) const {
    if (TestCheckType(t_checkNoAssert))
      cout << "\nTest warning:";
    else
      cout << "\nTest failed:";

    if (TestCheckType(t_checkColorComp))
      cout << " color # " << colorIdx;

    if (idx >= 0)
      cout << " element # " << idx;

    if (pShape)
      cout << " (h, w) = (" << (*pShape)[0] << ", " << (*pShape)[1] << ")";

    cout << " mean = " << mean << " and it was expected to be <= " << eps_ << endl;
  }

  void CheckTensorLists(const TensorList<CPUBackend> *t1,
                        const TensorList<CPUBackend> *t2) const {
    ASSERT_TRUE(t1);
    ASSERT_TRUE(t2);
    ASSERT_EQ(t1->ntensor(), t2->ntensor());

    ASSERT_EQ(t1->size(), t2->size());

    const bool floatType = IsType<float>(t1->type());
    if (!floatType && !IsType<unsigned char>(t1->type()))
      return;   // For now we check buffers only for "float" and "uchar"

    int failNumb = 0, colorIdx = 0;
    const bool checkAll = TestCheckType(t_checkAll);
    double mean;
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
                          (*t2).template tensor<float>(i), checkAll, &mean);
        } else {
          colorIdx = CheckBuffers<unsigned char>(lenBuffer,
                          (*t1).template tensor<uint8>(i),
                          (*t2).template tensor<uint8>(i), checkAll, &mean);
        }

        if (colorIdx >= 0) {
          // Test failed for colorIdx
          ReportTestFailure(mean, colorIdx, i, &shape1);
          failNumb++;
          if (!checkAll)
            break;
        }
      }
    } else {
      if (floatType) {
        colorIdx = CheckBuffers<float>(t1->size(),
                          t1->data<float>(),
                          t2->data<float>(), checkAll, &mean);
      } else {
        colorIdx = CheckBuffers<unsigned char>(t1->size(),
                          t1->data<unsigned char>(),
                          t2->data<unsigned char>(), checkAll, &mean);
      }
      if (colorIdx >= 0)
        ReportTestFailure(mean, colorIdx);
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

  ImgSetDescr png_;

  vector<uint8*> jpeg_decoded_, png_decoded_;
  vector<DimPair> jpeg_dims_, png_dims_;


 protected:
  int batch_size_ = 32;
  int num_threads_ = 2;
  double eps_ = 1e-4;
  uint32_t testCheckType_ = t_checkDefault;
  const DALIImageType img_type_ = ImgType::type;

  // keep a copy of the creation OpSpec for reference
  OpSpec spec_;
};

}  // namespace dali

#endif  // DALI_TEST_DALI_TEST_SINGLE_OP_H_
