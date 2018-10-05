// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef DALI_TEST_DALI_TEST_MATCHING_H_
#define DALI_TEST_DALI_TEST_MATCHING_H_

#include "dali/test/dali_test_single_op.h"
#include <utility>
#include <vector>
#include <string>
#include <memory>

namespace dali {

typedef struct {
  const char *opName;
  OpArg opArg;
  double epsVal;
} singleParamOpDescr;

template <typename ImgType>
class GenericMatchingTest : public DALISingleOpTest<ImgType> {
 protected:
  void RunTest(const opDescr &descr) {
    const int batch_size = this->Imgs(t_jpegImgType).nImages();
    this->SetBatchSize(batch_size);
    this->SetNumThreads(1);

    TensorList<CPUBackend> data;
    this->MakeJPEGBatch(&data, batch_size);
    this->SetExternalInputs({{"jpegs", &data}});

    string input = "input";
    auto pipe = this->GetPipeline();
    // Decode the images
    pipe->AddOperator(
      OpSpec("HostDecoder")
        .AddArg("output_type", this->ImageType())
        .AddInput("jpegs", "cpu")
        .AddOutput(input, "cpu"));

    if (CroppingNeeded()) {
      AddCropping(pipe, &input);
    }

    if (MirroringNeeded()) {
      // When mirroring is used
      pipe->AddOperator(
        OpSpec("CoinFlip")
          .AddArg("device", "support")
          .AddArg("probability", 0.5f)
          .AddOutput("mirror", "cpu"));
    }

    // Launching the same transformation on CPU (outputIdx 0) and GPU (outputIdx 1)
    this->AddOperatorWithOutput(descr, "cpu", input);
    this->RunOperator(descr);
  }

  vector<TensorList<CPUBackend>*>
  Reference(const vector<TensorList<CPUBackend>*> &inputs, DeviceWorkspace *ws) override {
    if (OpType() == DALI_GPU)
      return this->CopyToHost(*ws->Output<GPUBackend>(1));
    else
      return this->CopyToHost(*ws->Output<CPUBackend>(1));
  }

  uint32_t GetTestCheckType() const override {
    return t_checkColorComp + t_checkElements;  // + t_checkAll + t_checkNoAssert;
  }

  void RunTest(const singleParamOpDescr &paramOp, bool addImgType = false) {
    vector<OpArg> args;
    args.push_back(paramOp.opArg);
    this->AddParameters(&args);
    opDescr finalDesc(paramOp.opName, paramOp.epsVal, addImgType, &args);
    RunTest(finalDesc);
  }

  void RunTest(const char *opName, const OpArg params[] = nullptr,
                int nParam = 0, bool addImgType = false, double eps = 0.001) {
    if (params && nParam > 0) {
      vector<OpArg> args(params, params + nParam);
      this->AddParameters(&args);
      RunTest(opDescr(opName, eps, addImgType, &args));
    } else {
      RunTest(opDescr(opName, eps, addImgType, nullptr));
    }
  }

  inline DALIOpType OpType() const                { return m_nOpType; }
  inline void setOpType(DALIOpType opType)        { m_nOpType = opType; }
  virtual bool MirroringNeeded() const            { return false; }
  virtual bool CroppingNeeded() const             { return false; }
  virtual const vector<int> *GetCrop() const      { return nullptr; }
  virtual float ResizeValue(int idx) const        { return 256.f; }

  bool AddArgumentInput(int idxParam, OpSpec *spec) override {
    // This method should be overwritten when
    //   a) Mirroring is NOT used for the test OR
    //   b) more argument inputs are used

    if (MirroringNeeded())
      spec->AddArgumentInput("mirror", "mirror");

    return false;
  }

 private:
  void AddCropping(shared_ptr<dali::Pipeline> pipe, string *pInput) const {
        pipe->AddOperator(
                 OpSpec("Uniform")
                   .AddArg("device", "support")
                   .AddArg("range", vector<float>{0, 1})
                   .AddOutput("uniform1", "cpu"));

        pipe->AddOperator(
                 OpSpec("Uniform")
                   .AddArg("device", "support")
                   .AddArg("range", vector<float>{0, 1})
                   .AddOutput("uniform2", "cpu"));

        // Add a resize+crop+mirror op
        string output = this->GetOutputOfCroppingOperator();
        pipe->AddOperator(
                 OpSpec("ResizeCropMirror")
                   .AddArg("device", "cpu")
                   .AddArg("resize_x", ResizeValue(1))
                   .AddArg("resize_y", ResizeValue(0))
                   .AddArg("crop", GetCrop()? *GetCrop() : vector<int>{224, 224})
                   .AddInput(*pInput, "cpu")
                   .AddArgumentInput("crop_pos_x", "uniform1")
                   .AddArgumentInput("crop_pos_y", "uniform2")
                   .AddOutput(output, "cpu"));

        // Changing the input of next operator in pipeline
        *pInput = output;
  }

  DALIOpType m_nOpType = DALI_GPU;
};


template <typename ImgType>
class NormalizePermuteMatch : public GenericMatchingTest<ImgType> {
 protected:
  bool AddParam(int idxParam, vector<OpArg> *argc) override {
    // Depending the color parameters of the images, we will add vector of size 1 or 3
    auto idx = this->c_ == 1? 0 : 1;
    switch (idxParam) {
      case 0: argc->push_back({"mean", meanValues_[idx], DALI_FLOAT_VEC});
      return true;
      case 1: argc->push_back({"std", stdValues_[idx], DALI_FLOAT_VEC});
    }

    return false;  // No more parameters will be added for that test
  }

  inline void setMeanValues(const char **values)  { meanValues_ = values; }
  inline void setStdValues(const char **values)   { stdValues_ = values; }

 private:
  const char **meanValues_ = nullptr;
  const char **stdValues_  = nullptr;
};

#define CONFORMITY_TEST(testGroupName, testName, assignOperators, ...)                \
    TYPED_TEST(testGroupName, testName) {                                             \
      assignOperators;                                                                \
      const OpArg params[] = __VA_ARGS__;                                             \
      this->RunTest(opName, params, sizeof(params)/sizeof(params[0]), addImageType);  \
    }

#define CONFORMITY_NORMALIZE_TEST(testGroupName, testName, assignOperators,           \
            meanValues, stdValues, ...)                                               \
            CONFORMITY_TEST(testGroupName, testName,                                  \
                assignOperators;                                                      \
                this->setMeanValues(meanValues);                                      \
                this->setStdValues(stdValues), __VA_ARGS__)

// Macros to create Normalize tests with statically defined meanValues/stdValues
#define CONFORMITY_NORMALIZE_TEST_DEF(testGroupName, testName, assignOperators, ...)  \
            CONFORMITY_NORMALIZE_TEST(testGroupName, testName, assignOperators,       \
                        meanValues, stdValues, __VA_ARGS__)
}  // namespace dali

#endif  // DALI_TEST_DALI_TEST_MATCHING_H_
