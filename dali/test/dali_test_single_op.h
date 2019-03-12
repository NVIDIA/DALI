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

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "dali/test/dali_test.h"
#include "dali/pipeline/pipeline.h"

namespace dali {

#define MAKE_IMG_OUTPUT    0      // Make the output of compared (obtained and referenced) images
#if MAKE_IMG_OUTPUT
  #define PIXEL_STAT_FILE "pixelStatFile"  // Output of statistics for compared sets of images
                                           // Use "" to make the output in stdout
#endif

#define SAVE_TMP_IMAGES 0

namespace images {

// TODO(janton): DALI-582 Using this order, breaks some tests
// ImageList(image_folder, {".jpg"})
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

const vector<string> png_test_images = ImageList(image_folder + "/png", {".png"});
const vector<string> tiff_test_images = ImageList(image_folder + "/tiff", {".tiff", ".tif"});

}  // namespace images

typedef enum {            // Checking:
  t_checkDefault    = 0,  //    combined vectors (all images, all colors)
  t_checkColorComp  = 1,  //    colors separately
  t_checkElements   = 2,  //    images separately
  t_checkAll        = 4,  //    everything (no assertion after first fail)
  t_checkNoAssert   = 8,  //    no assertion even when test failed
  t_checkBestMatch  = 16  //    best match of two images with possible left/right & up/down shifts
} t_checkType;

typedef enum {
  t_undefinedImgType,
  t_jpegImgType,
  t_pngImgType,
  t_tiffImgType,
} t_imgType;

typedef enum {
  t_loadJPEGs   = 1,
  t_decodeJPEGs = 2,
  t_loadPNGs    = 4,
  t_decodePNGs  = 8,
  t_loadTiffs   = 16,
  t_decodeTiffs = 32,
} t_loadingFlags;

typedef struct {
  const char *m_Name;
  std::string m_val;
  const DALIDataType type;
} OpArg;

class opDescr {
 public:
  explicit opDescr(const char *name, double eps = 0.0, bool addImgType = false,
                   const vector<OpArg> *argPntr = nullptr) :
                   opName(name), opAddImgType(addImgType), epsVal(eps), args(argPntr) {}
  const char *opName;         // the name of the operator
  const bool opAddImgType;    // the image_type argument needs to be added to the list of
                              // the operator's arguments
  const double epsVal;
  const vector<OpArg> *args;
};

template <typename T>
void StringToVector(const char *name, const char *val, OpSpec *spec, DALIDataType dataType) {
  const auto len = strlen(val);
  vector<T> vect;
  char *pEnd, *pTmp = new char[len + 1];
  memcpy(pEnd = pTmp, val, len);
  pEnd[len] = '\0';
  T value;
  while (pEnd[0]) {
    if (pEnd[0] == ',')
      pEnd++;

    switch (dataType) {
      case DALI_FLOAT_VEC:  value = strtof(pEnd, &pEnd);
                            break;
      case DALI_INT_VEC:    value = strtol(pEnd, &pEnd, 10);
                            break;
      default:  DALI_FAIL("Unknown type of vector \"" + std::string(val) + "\" "
                          "used for \"" + std::string(name) + "\"");
    }

    vect.push_back(value);
  }

  delete [] pTmp;
  spec->AddArg(name, vect);
}


/**
 * Virtual base class for single operator tests.
 * 1. Adds single operator to pipeline
 * 2. Runs the pipe using known data
 *    (specified in class, that extends DALISingleOpTest)
 * 3. Compares result to reference solution (also specified in subclass)
 *
 * Pipeline does the following:
 * 1. Sets input data
 * 2. Runs operator (specified in RunOperator) on CPU & GPU
 * 3. Returns output data
 *
 * Example usage is to overload Reference(...) function in subclass,
 * which has access to input data. The function should return batch of
 * reference data, calculated for given input data. Following, define
 * a TYPED_TEST case, where you set test conditions (at least SetExternalInputs
 * to set up input data) and run operator, using one of RunOperator overloads.
 * @tparam ImgType @see DALIImageType
 */
template<typename ImgType, typename OutputImgType = ImgType>
class DALISingleOpTest : public DALITest {
 public:
  inline void SetUp() override {
    DALITest::SetUp();
    c_ = (IsColor(OutputImageType()) ? 3 : 1);
    jpegs_.clear();

    const auto flags = GetImageLoadingFlags();

    if (flags & t_loadJPEGs) {
      LoadImages(images::jpeg_test_images, &jpegs_);
      if (flags & t_decodeJPEGs)
        DecodeImages(img_type_, jpegs_, &jpeg_decoded_, &jpeg_dims_);
    }

    if (flags & t_loadPNGs) {
      LoadImages(images::png_test_images, &png_);

      if (flags & t_decodePNGs)
        DecodeImages(img_type_, png_, &png_decoded_, &png_dims_);
    }

    if (flags & t_loadTiffs) {
      LoadImages(images::tiff_test_images, &tiff_);

      if (flags & t_decodeTiffs) {
        DecodeImages(img_type_, tiff_, &tiff_decoded_, &tiff_dims_);
      }
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

  virtual uint32_t GetTestCheckType() const {
    return t_checkDefault;
  }

  void AddOperatorWithOutput(const OpSpec& spec) {
    // generate the output mapping for this op
    for (int i = 0; i < spec.NumOutput(); ++i)
      outputs_.push_back(std::make_pair(spec.OutputName(i), spec.OutputDevice(i)));

    pipeline_->AddOperator(spec, spec.name());
  }

  virtual void AddDefaultArgs(OpSpec& spec) {
  }

  void AddOperatorWithOutput(const opDescr &descr, const string &pDevice = "cpu",
                             const string &pInput = "input", const string &pOutput = "outputCPU") {
    OpSpec spec(descr.opName);
    if (descr.opAddImgType)
      spec = spec.AddArg("image_type", ImageType());
    AddDefaultArgs(spec);

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
   * Check the desired calculated answers in ws (given by user-provided indices)
   * against a supplied reference implementation.
   */
  void CheckAnswers(DeviceWorkspace *ws,
                    const vector<int>& output_indices) {
    // outputs_ contains map of idx -> (name, device)
    vector<TensorList<CPUBackend>*> res = Reference(input_data_, ws);

    std::unique_ptr<TensorList<CPUBackend>> calc_output(new TensorList<CPUBackend>());
    // get outputs from pipeline, copy to host if necessary
    for (size_t i = 0; i < output_indices.size(); ++i) {
      auto output_device = outputs_[i].second;

      auto idx = output_indices[i];
      if (output_device == "gpu") {
        // copy to host
        calc_output->Copy(ws->Output<GPUBackend>(idx), nullptr);
      } else {
        calc_output->Copy(ws->Output<CPUBackend>(idx), nullptr);
      }

      auto ref_output = res[i];
      calc_output->SetLayout(ref_output->GetLayout());

      // check calculated vs. reference answers
      CheckTensorLists(calc_output.get(), ref_output);
    }

    for (auto *&ref_output : res) {
      delete ref_output;
      ref_output = nullptr;
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

  void EncodedTiffData(TensorList<CPUBackend>* t) {
    DALITest::MakeEncodedBatch(t, batch_size_, tiff_);
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

  DALIImageType OutputImageType() const                 {
    return output_img_type_;
  }

  void TstBody(const string &pName, const string &pDevice = "gpu", double eps = 2e-1) {
    OpSpec operation = DefaultSchema(pName, pDevice);
    TstBody(operation, eps);
  }

  void TstBody(const OpSpec &operation, double eps = 2e-1, bool flag = true) {
    TensorList<CPUBackend> data;
    DecodedData(&data, this->batch_size_, this->ImageType());
    if (flag)
      SetExternalInputs({std::make_pair("input", &data)});

    RunOperator(operation, eps);
  }

  virtual OpSpec DefaultSchema(const string &pName, const string &pDevice = "gpu") const {
    return OpSpec(pName)
      .AddArg("device", pDevice)
      .AddArg("image_type", this->ImageType())
      .AddArg("output_type", this->ImageType())
      .AddInput("input", pDevice)
      .AddOutput("output", pDevice);
  }

  OpSpec AddArguments(OpSpec *spec, const vector<OpArg> *args) const {
    if (!args || args->empty())
      return *spec;

    for (auto param : *args) {
      const auto &val = param.m_val;
      const auto &name = param.m_Name;
      switch (param.type) {
        case DALI_INT32:
          spec->AddArg(name, strtol(val.c_str(), nullptr, 10));
          break;
        case DALI_FLOAT:
          spec->AddArg(name, strtof(val.c_str(), nullptr));
          break;
        case DALI_STRING:
          spec->AddArg(name, val);
          break;
        case DALI_BOOL: {
          bool b;
          std::istringstream(val) >> std::nouppercase >> std::boolalpha >> b;
          spec->AddArg(name, b);
          break;
        }
        case DALI_FLOAT_VEC:
          StringToVector<float>(name, val.c_str(), spec, param.type);
          break;
        case DALI_INT_VEC:
          StringToVector<int>(name, val.c_str(), spec, param.type);
          break;
        default: DALI_FAIL("Unknown type of parameters \"" + std::string(val) + "\" "
                           "used for \"" + std::string(name) + "\"");
      }
    }

    return *spec;
  }

  void RunOperator(const opDescr &descr) {
    OpSpec spec(DefaultSchema(descr.opName));
    if (descr.opAddImgType && !spec.HasArgument("image_type"))
      spec = spec.AddArg("image_type", ImageType());
    AddDefaultArgs(spec);

    RunOperator(AddArguments(&spec, descr.args), descr.epsVal);
  }

  void RunOperator(const OpSpec& spec, double eps, DeviceWorkspace *pWS = nullptr) {
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
  std::unique_ptr<TensorList<CPUBackend>> CopyTensorListToHost(const TensorList<T> &calcOutput) {
    std::unique_ptr<TensorList<CPUBackend>> output(new TensorList<CPUBackend>());
    output->Copy(calcOutput, 0);

    return output;
  }

  template <typename T>
  int CheckBuffers(int lenRaster, const T *img1, const T *img2, bool checkAll,
                   double *pMean = nullptr, vector<Index> shape = {}) const {
#if SAVE_TMP_IMAGES
  if (!shape.empty()) {
    static int i = 0;
    int H = shape[0];
    int W = shape[1];
    int C = shape[2];
    const int input_channel_flag = GetOpenCvChannelType(C);
    const cv::Mat cv_img1 = CreateMatFromPtr(H, W, input_channel_flag, img1);
    const cv::Mat cv_img2 = CreateMatFromPtr(H, W, input_channel_flag, img2);
    std::cout << "Saving images #" << std::to_string(i) << std::endl;
    cv::imwrite("tmp_img_" + std::to_string(i) + "_A.png", cv_img1);
    cv::imwrite("tmp_img_" + std::to_string(i) + "_B.png", cv_img2);
    i++;
  }
#endif

#ifdef PIXEL_STAT_FILE
    static int imgNumb;
    FILE *file = strlen(PIXEL_STAT_FILE)? fopen(PIXEL_STAT_FILE".txt", imgNumb? "a" : "w") : NULL;
    if (!imgNumb % 32) {
      // Header of the pixel statistic table
      const char *pHeader =
        "\nImgID: ClrID:     Mean:        Std:      SameValue:     Bigger:         Less:\n";

      if (file)
        fprintf(file, "%s", pHeader);
      else
        cout << pHeader;
    }

    char buffer[256];
    snprintf(buffer, sizeof(buffer), "%3d:", imgNumb);

    // Image number
    if (file)
      fprintf(file, "%s", buffer);
    else
      cout << buffer;

    bool firstLine = true;
#endif

    // use a Get mean, std-dev of difference separately for each color component
    const int jMax = TestCheckType(t_checkColorComp)?  c_ : 1;

    const int length = lenRaster / jMax;

    const bool checkBest = TestCheckType(t_checkBestMatch);

    int shiftHorFrom, shiftHorTo, shiftVertFrom, shiftVertTo;
    shiftHorFrom = shiftVertFrom = -checkBest;
    shiftHorTo = shiftVertTo = checkBest;

    const auto H = checkBest ? shape[0] : 0;
    const auto W = checkBest ? shape[1] : 0;
    int retValBest = -1;
    double bestMean = 100.;

    for (int shiftVert = shiftVertFrom; shiftVert <= shiftVertTo; ++shiftVert) {
      for (int shiftHor = shiftHorFrom; shiftHor <= shiftHorTo; ++shiftHor) {
        // Calculate the length of the vector to store the delta of images
        int len = length;
        const T *a = img1;
        const T *b = img2;
        auto hMax = H;
        auto wMax = W;
        int N = lenRaster;
        if (shiftVert) {
          const auto lengthReduction = W * c_;
          len -= lengthReduction / jMax;
          N -= lengthReduction;
          if (shiftVert > 0)
            a += lengthReduction;
          else
            b += lengthReduction;

          hMax--;
        }

        if (shiftHor) {
          len -= (H - (shiftVert? 1 : 0)) * c_ / jMax;
          if (shiftHor > 0)
            a++;
          else
            b++;

          wMax--;
        }

        vector<double> diff(len);
        double mean = 0, std = 0;
        int retVal = -1;
        double worstMean = -1.;
#ifndef PIXEL_STAT_FILE
        for (int j = 0; j < jMax; ++j) {    // loop over the colors
          if (shiftHor == 0) {
            for (int i = j; i < N; i += jMax)
              diff[i / jMax] = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));

            ASSERT_EQ(N/jMax, len), -1;
          } else {
            int i = 0;
            for (int y = 0; y < hMax; ++y) {
              for (int x = 0; x < wMax; ++x) {
                const int idx = (W * y + x) * c_;
                diff[i++] = std::abs(static_cast<double>(a[idx]) - static_cast<double>(b[idx]));
              }
            }

            a++;    // to next color component
            b++;
            ASSERT_EQ(i, len), -1;
          }

          MeanStdDevColorNorm<double>(diff, &mean, &std);
          if (mean <= eps_)
            continue;

          if (checkAll) {
            if (worstMean < mean) {
              worstMean = mean;  // More strong violation of the boundary found
              retVal = j;        // Change the color index as a return value
            }
            continue;
          }

          if (!checkBest) {
            ASSERT_LE(mean, eps_), -1;
          }
        }
#else

        for (int j = 0; j < jMax; ++j) {
          int pos = 0, neg = 0;
          if (shiftHor == 0) {
            for (int i = j; i < N; i += jMax) {
              diff[i / jMax] = abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
              if (a[i] > b[i])
                pos++;
              else if (a[i] < b[i])
                neg++;
            }
            ASSERT_EQ(N/jMax, len), -1;
          } else {
            int i = 0;
            for (int y = 0; y < hMax; ++y) {
              for (int x = 0; x < wMax; ++x) {
                const int idx = (W * y + x) * c_;
                diff[i++] = abs(static_cast<double>(a[idx]) - static_cast<double>(b[idx]));
                if (a[i] > b[i])
                  pos++;
                else if (a[i] < b[i])
                  neg++;
              }
            }

            a++;    // to next color component
            b++;
            ASSERT_EQ(i, len), -1;
          }

          MeanStdDevColorNorm<double>(diff, &mean, &std);
          snprintf(buffer, sizeof(buffer),
                   "%s     %1d    %8.2f     %8.2f       %7d      %7d      %7d\n",
                   firstLine? "" : "    ", j, mean, std, len - pos - neg, pos, neg);

          firstLine = false;
          if (file)
            fprintf(file, "%s", buffer);
          else
            cout << buffer;

          if (mean <= eps_)
            continue;

          if (worstMean < mean) {
            worstMean = mean;  // More strong violation of the boundary found
            retVal = j;        // Change the color index as a return value
          }
        }


#endif
        if (bestMean > worstMean) {
          bestMean = worstMean;
          retValBest = retVal;
        }
      }
    }

#ifdef PIXEL_STAT_FILE
    imgNumb++;    // change image number
    if (file)
      fclose(file);
#endif

    if (bestMean <= eps_)
      return -1;

    if (!checkAll) {
      ASSERT_LE(bestMean, eps_), -1;
    }

    if (pMean)
        *pMean = bestMean;

    return retValBest;
  }

  void ReportTestFailure(double mean, int colorIdx, int idx = -1,
                         vector<Index> shape = {}) const {
    if (TestCheckType(t_checkNoAssert))
      cout << "\nTest warning:";
    else
      cout << "\nTest failed:";

    if (TestCheckType(t_checkColorComp))
      cout << " color # " << colorIdx;

    if (idx >= 0)
      cout << " element # " << idx;

    if (!shape.empty())
      cout << " (h, w) = (" << shape[0] << ", " << shape[1] << ")";

    cout << " mean = " << mean << " and it was expected to be <= " << eps_ << endl;
  }

  void CheckTensorLists(const TensorList<CPUBackend> *t1,
                        const TensorList<CPUBackend> *t2) const {
#if MAKE_IMG_OUTPUT
    WriteBatch(*t1, "img");
    WriteBatch(*t2, "ref");
#endif

    ASSERT_TRUE(t1);
    ASSERT_TRUE(t2);
    ASSERT_EQ(t1->ntensor(), t2->ntensor());
    ASSERT_EQ(t1->size(), t2->size());

    const bool floatType = IsType<float>(t1->type());
    if (!floatType && !IsType<unsigned char>(t1->type()))
      return;   // For now we check buffers only for "float" and "uchar"

    int failNumb = 0, colorIdx = 0;
    const bool checkAll = TestCheckType(t_checkAll);
    const bool checkElements = TestCheckType(t_checkElements);
    double mean;
    if (checkElements) {
      // The the results are checked for each element separately
      for (size_t i = 0; i < t1->ntensor(); ++i) {
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
                          (*t1).template tensor<float>(i), (*t2).template tensor<float>(i),
                          checkAll, &mean, shape1);
        } else {
          colorIdx = CheckBuffers<unsigned char>(lenBuffer,
                          (*t1).template tensor<uint8>(i), (*t2).template tensor<uint8>(i),
                          checkAll, &mean, shape1);
        }

        if (colorIdx >= 0) {
          // Test failed for colorIdx
          ReportTestFailure(mean, colorIdx, i, shape1);
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

  vector<uint8*> jpeg_decoded_, png_decoded_, tiff_decoded_;
  vector<DimPair> jpeg_dims_, png_dims_, tiff_dims_;


 protected:
  int batch_size_ = 32;
  int num_threads_ = 2;
  double eps_ = 1e-4;
  uint32_t testCheckType_ = t_checkDefault;
  const DALIImageType img_type_ = ImgType::type;
  const DALIImageType output_img_type_ = OutputImgType::type;

  // keep a copy of the creation OpSpec for reference
  OpSpec spec_;
};

}  // namespace dali

#endif  // DALI_TEST_DALI_TEST_SINGLE_OP_H_
