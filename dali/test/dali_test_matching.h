// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef DALI_TEST_DALI_TEST_MATCHING_H_
#define DALI_TEST_DALI_TEST_MATCHING_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/common.h"
#include "dali/test/dali_test_single_op.h"

namespace dali {

typedef struct {
  const char *opName;
  OpArg opArg;
  double epsVal;
} singleParamOpDescr;

template <typename ImgType, typename OutputImgType = ImgType>
class GenericMatchingTest : public DALISingleOpTest<ImgType, OutputImgType> {
 protected:
  virtual void RunTestImpl(const opDescr &descr) {
    const int batch_size = this->jpegs_.nImages();
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
        .AddOutput("input", "cpu"), "HostDecoder");

    // Launching the same transformation on CPU (outputIdx 0) and GPU (outputIdx 1)
    this->AddOperatorWithOutput(descr);
    this->RunOperator(descr);
  }

  vector<TensorList<CPUBackend>*>
  Reference(const vector<TensorList<CPUBackend>*> &inputs, DeviceWorkspace *ws) override {
    if (GetOpType() == OpType::GPU)
      return this->CopyToHost(ws->Output<GPUBackend>(1));
    else
      return this->CopyToHost(ws->Output<CPUBackend>(1));
  }

  uint32_t GetTestCheckType() const override {
    return t_checkColorComp + t_checkElements;  // + t_checkAll + t_checkNoAssert;
  }

  void RunTest(const singleParamOpDescr &paramOp, bool addImgType = false) {
    vector<OpArg> args;
    args.push_back(paramOp.opArg);
    opDescr finalDesc(paramOp.opName, paramOp.epsVal, addImgType, &args);
    RunTestImpl(finalDesc);
  }

  void RunTest(const char *opName, const OpArg params[] = nullptr,
                int nParam = 0, bool addImgType = false, double eps = 0.001) {
    if (params && nParam > 0) {
      vector<OpArg> args(params, params + nParam);
      RunTestImpl(opDescr(opName, eps, addImgType, &args));
    } else {
      RunTestImpl(opDescr(opName, eps, addImgType, nullptr));
    }
  }

  inline OpType GetOpType() const                { return op_type_; }
  inline void SetOpType(OpType opType)        { op_type_ = opType; }


  OpType op_type_ = OpType::GPU;
};

}  // namespace dali

#endif  // DALI_TEST_DALI_TEST_MATCHING_H_
