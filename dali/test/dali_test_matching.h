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
  const char *paramName;
  const char *paramVal;
  double epsVal;
} singleParamOpDescr;

template <typename ImgType>
class GenericMatchingTest : public DALISingleOpTest<ImgType> {
 protected:
  void RunTest(const opDescr &descr) {
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
        .AddArg("output_type", this->img_type_)
        .AddInput("jpegs", "cpu")
        .AddOutput("input", "cpu"));

    // Launching the same transformation on CPU (outputIdx 0) and GPU (outputIdx 1)
    this->AddOperatorWithOutput(descr);
    this->RunOperator(descr);
  }

  virtual vector<TensorList<CPUBackend>*>
  Reference(const vector<TensorList<CPUBackend>*> &inputs, DeviceWorkspace *ws) {
    return this->CopyToHost(*ws->Output<GPUBackend>(1));
  }

  uint8 GetTestCheckType() const  override {
    return t_checkColorComp + t_checkElements;  // + t_checkAll + t_checkNoAssert;
  }

  void RunTest(const singleParamOpDescr &paramOp) {
    OpArg arg = {paramOp.paramName, paramOp.paramVal, t_floatParam};
    vector<OpArg> args;
    args.push_back(arg);
    opDescr aaa(paramOp.opName, paramOp.epsVal, &args);
    RunTest(aaa);
  }

  void RunTest(const char *opName, const OpArg params[] = NULL,
                int nParam = 0, double eps = 0.001) {
    if (params && nParam > 0) {
      vector<OpArg> args(params, params + nParam);
      RunTest(opDescr(opName, eps, &args));
    } else {
      RunTest(opDescr(opName, eps, NULL));
    }
  }
};

}  // namespace dali

#endif  // DALI_TEST_DALI_TEST_MATCHING_H_
