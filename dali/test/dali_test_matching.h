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

    shared_ptr<dali::Pipeline> pipe = this->GetPipeline();
    // Decode the images
    pipe->AddOperator(
      OpSpec("HostDecoder")
        .AddArg("output_type", this->ImageType())
        .AddInput("jpegs", "cpu")
        .AddOutput("input", "cpu"));

    if (MirroringNeeded()) {
      // When mirroring is used
      pipe->AddOperator(
        OpSpec("CoinFlip")
          .AddArg("device", "support")
          .AddArg("probability", 0.5f)
          .AddOutput("mirror", "cpu"));
    }

    // Launching the same transformation on CPU (outputIdx 0) and GPU (outputIdx 1)
    this->AddOperatorWithOutput(descr);
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

  bool AddArgumentInput(int idxParam, OpSpec *spec) override {
    // This method should be overwritten when
    //   a) Mirroring is NOT used for the test OR
    //   b) more argument inputs are used

    if (MirroringNeeded())
      spec->AddArgumentInput("mirror", "mirror");

    return false;
  }

 private:
  DALIOpType m_nOpType = DALI_GPU;
};

}  // namespace dali

#endif  // DALI_TEST_DALI_TEST_MATCHING_H_
