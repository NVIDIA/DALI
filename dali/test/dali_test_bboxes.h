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
class GenericBBoxesTest : public DALISingleOpTest<ImgType> {
 protected:
  void RunTestGPU(const opDescr &descr) {
    const int batch_size = this->jpegs_.nImages();
    this->SetBatchSize(batch_size);
    this->SetNumThreads(1);

    TensorList<CPUBackend> encoded_data;
    this->EncodedJPEGData(&encoded_data);
    TensorList<CPUBackend> boxes;
    this->MakeBBoxesBatch(&boxes, batch_size);
    this->SetExternalInputs({{"encoded", &encoded_data}, {"boxes", &boxes}});

    shared_ptr<dali::Pipeline> pipe = this->GetPipeline();

    // Decode the images
    pipe->AddOperator(
    OpSpec("nvJPEGDecoder")
      .AddArg("device", "mixed")
      .AddArg("output_type", this->img_type_)
      .AddArg("use_batched_decode", true)
      .AddInput("encoded", "cpu")
      .AddOutput("decoded", "gpu"));

    OpSpec spec(descr.opName);
    if (descr.opAddImgType) spec = spec.AddArg("image_type", this->ImageType());

    this->AddOperatorWithOutput(this->AddArguments(&spec, descr.args)
                                    .AddArg("device", "gpu")
                                    .AddInput("decoded", "gpu")
                                    .AddInput("boxes", "gpu")
                                    .AddOutput("output", "gpu")
                                    .AddOutput("output1", "gpu")
                                    .AddOutput("output2", "gpu"));

    this->SetTestCheckType(this->GetTestCheckType());
    pipe->Build(DALISingleOpTest<ImgType>::outputs_);
    pipe->RunCPU();
    pipe->RunGPU();

    DeviceWorkspace ws;
    pipe->Outputs(&ws);
  }

  void RunTest(const opDescr &descr) {
    const int batch_size = this->jpegs_.nImages();
    this->SetBatchSize(batch_size);
    this->SetNumThreads(1);

    TensorList<CPUBackend> data;
    this->MakeJPEGBatch(&data, batch_size);
    TensorList<CPUBackend> boxes;
    this->MakeBBoxesBatch(&boxes, batch_size);
    this->SetExternalInputs({{"jpegs", &data}, {"boxes", &boxes}});

    shared_ptr<dali::Pipeline> pipe = this->GetPipeline();
    // Decode the images
    pipe->AddOperator(
      OpSpec("HostDecoder")
        .AddArg("output_type", this->ImageType())
        .AddInput("jpegs", "cpu")
        .AddOutput("input", "cpu"));

    OpSpec spec(descr.opName);
    if (descr.opAddImgType)
      spec = spec.AddArg("image_type", this->ImageType());

    this->AddOperatorWithOutput(this->AddArguments(&spec, descr.args)
                                    .AddInput("input", "cpu")
                                    .AddInput("boxes", "cpu")
                                    .AddOutput("output", "cpu")
                                    .AddOutput("output1", "cpu")
                                    .AddOutput("output2", "cpu"));

    this->SetTestCheckType(this->GetTestCheckType());
    pipe->Build(DALISingleOpTest<ImgType>::outputs_);
    pipe->RunCPU();
    pipe->RunGPU();

    DeviceWorkspace ws;
    pipe->Outputs(&ws);
  }
  vector<TensorList<CPUBackend>*>
  Reference(const vector<TensorList<CPUBackend>*> &inputs, DeviceWorkspace *ws) override {
    auto from = ws->Output<GPUBackend>(1);
    auto reference = this->CopyToHost(*from);
    reference[0]->SetLayout(from->GetLayout());
    return reference;
  }

  uint32_t GetTestCheckType() const  override {
    return t_checkColorComp + t_checkElements;  // + t_checkAll + t_checkNoAssert;
  }

  void RunTest(const singleParamOpDescr &paramOp, bool addImgType = false) {
    vector<OpArg> args;
    args.push_back(paramOp.opArg);
    opDescr finalDesc(paramOp.opName, paramOp.epsVal, addImgType, &args);
    RunTest(finalDesc);
  }

  void RunTestGPU(const singleParamOpDescr &paramOp, bool addImgType = false) {
    vector<OpArg> args;
    args.push_back(paramOp.opArg);
    opDescr finalDesc(paramOp.opName, paramOp.epsVal, addImgType, &args);
    RunTestGPU(finalDesc);
  }
};

}  // namespace dali

#endif  // DALI_TEST_DALI_TEST_MATCHING_H_
