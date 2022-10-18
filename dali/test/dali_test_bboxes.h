// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#ifndef DALI_TEST_DALI_TEST_BBOXES_H_
#define DALI_TEST_DALI_TEST_BBOXES_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/test/dali_test_single_op.h"

namespace dali {

struct SingleParamOpDescr {
  SingleParamOpDescr() = default;
  SingleParamOpDescr(const char *name, OpArg &&arg, double eps = 0)  // NOLINT
  : opName(name), opArg(std::move(arg)), epsVal(eps) {}
  const char *opName = nullptr;
  OpArg opArg;
  double epsVal = 0;
};

template <typename ImgType>
class GenericBBoxesTest : public DALISingleOpTest<ImgType> {
 protected:
  void RunBBoxesCPU(const opDescr &descr, bool ltrb) {
    const int batch_size = this->jpegs_.nImages();
    this->SetBatchSize(batch_size);
    this->SetNumThreads(1);

    TensorList<CPUBackend> boxes;
    TensorList<CPUBackend> labels;
    this->MakeBBoxesAndLabelsBatch(&boxes, &labels, batch_size, ltrb);
    this->AddExternalInputs({{"boxes", &boxes}, {"labels", &labels}});

    auto pipe = this->GetPipeline();

    OpSpec spec(descr.opName);
    if (descr.opAddImgType) spec = spec.AddArg("image_type", this->ImageType());

    this->AddOperatorWithOutput(this->AddArguments(&spec, descr.args)
                                    .AddInput("boxes", "cpu")
                                    .AddInput("labels", "cpu")
                                    .AddOutput("output", "cpu")
                                    .AddOutput("output1", "cpu")
                                    .AddOutput("output2", "cpu")
                                    .AddOutput("output3", "cpu"));

    this->SetTestCheckType(this->GetTestCheckType());
    pipe->Build(DALISingleOpTest<ImgType>::outputs_);
    this->FillExternalInputs();
    pipe->RunCPU();
    pipe->RunGPU();

    Workspace ws;
    pipe->Outputs(&ws);
  }

  std::vector<std::shared_ptr<TensorList<CPUBackend>>> RunSliceGPU(
      const vector<std::pair<string, TensorList<CPUBackend> *>> &inputs) {
    const int batch_size = this->jpegs_.nImages();
    this->SetBatchSize(batch_size);
    this->SetNumThreads(1);

    this->AddExternalInputs(inputs);

    auto pipe = this->GetPipeline();

    // Prospective crop
    pipe->AddOperator(OpSpec("RandomBBoxCrop")
                          .AddArg("device", "cpu")
                          .AddArg("image_type", this->ImageType())
                          .AddArg("bytes_per_sample_hint", vector<int>{ 8, 8, 256, 128 })
                          .AddInput("boxes", "cpu")
                          .AddInput("labels", "cpu")
                          .AddOutput("begin", "cpu")
                          .AddOutput("crop", "cpu")
                          .AddOutput("resized_boxes", "cpu")
                          .AddOutput("filtered_labels", "cpu"));

    // GPU slice
    pipe->AddOperator(OpSpec("Slice")
                          .AddArg("device", "gpu")
                          .AddInput("images", "gpu")
                          .AddInput("begin", "cpu")
                          .AddInput("crop", "cpu")
                          .AddOutput("cropped_images", "gpu"));

    this->SetTestCheckType(this->GetTestCheckType());
    pipe->Build({{"cropped_images", "gpu"}, {"resized_boxes", "gpu"}});
    this->FillExternalInputs();
    pipe->RunCPU();
    pipe->RunGPU();

    Workspace ws;
    pipe->Outputs(&ws);

    auto images_cpu = this->CopyToHost(ws.Output<GPUBackend>(0))[0];
    images_cpu->SetLayout(ws.Output<GPUBackend>(0).GetLayout());

    auto boxes_cpu = this->CopyToHost(ws.Output<GPUBackend>(1))[0];
    boxes_cpu->SetLayout(ws.Output<GPUBackend>(1).GetLayout());

    return {images_cpu, boxes_cpu};
  }

  std::vector<std::shared_ptr<TensorList<CPUBackend>>> RunSliceCPU(
      const vector<std::pair<string, TensorList<CPUBackend> *>> &inputs) {
    const int batch_size = this->jpegs_.nImages();
    this->SetBatchSize(batch_size);
    this->SetNumThreads(1);

    this->AddExternalInputs(inputs);

    auto pipe = this->GetPipeline();

    // Prospective crop
    pipe->AddOperator(OpSpec("RandomBBoxCrop")
                          .AddArg("device", "cpu")
                          .AddArg("image_type", this->ImageType())
                          .AddInput("boxes", "cpu")
                          .AddInput("labels", "cpu")
                          .AddOutput("begin", "cpu")
                          .AddOutput("crop", "cpu")
                          .AddOutput("resized_boxes", "cpu")
                          .AddOutput("filtered_labels", "cpu"));

    // GPU slice
    pipe->AddOperator(OpSpec("Slice")
                          .AddArg("device", "cpu")
                          .AddInput("images", "cpu")
                          .AddInput("begin", "cpu")
                          .AddInput("crop", "cpu")
                          .AddOutput("cropped_images", "cpu"));

    this->SetTestCheckType(this->GetTestCheckType());
    pipe->Build({{"cropped_images", "cpu"}, {"resized_boxes", "cpu"}});
    this->FillExternalInputs();
    pipe->RunCPU();
    pipe->RunGPU();

    Workspace ws;
    pipe->Outputs(&ws);

    std::vector<std::shared_ptr<TensorList<CPUBackend>>> ret;
    ret.push_back(std::make_shared<TensorList<CPUBackend>>());
    ret.push_back(std::make_shared<TensorList<CPUBackend>>());
    ret[0]->Copy(ws.Output<CPUBackend>(0));
    ret[1]->Copy(ws.Output<CPUBackend>(1));
    return ret;
  }

  vector<std::shared_ptr<TensorList<CPUBackend>>> Reference(
      const vector<TensorList<CPUBackend> *> &inputs,
      Workspace *ws) override {
    auto &from = ws->Output<GPUBackend>(1);
    auto reference = this->CopyToHost(from);
    reference[0]->SetLayout(from.GetLayout());
    return reference;
  }

  uint32_t GetTestCheckType() const override {
    return t_checkColorComp +
           t_checkElements;  // + t_checkAll + t_checkNoAssert;
  }

  void RunBBoxesCPU(const SingleParamOpDescr &paramOp, bool addImgType = false,
                    bool ltrb = true) {
    vector<OpArg> args;
    args.push_back(paramOp.opArg);
    opDescr finalDesc(paramOp.opName, paramOp.epsVal, addImgType, &args);
    RunBBoxesCPU(finalDesc, ltrb);
  }

  TensorList<CPUBackend> images_out;
  TensorList<CPUBackend> boxes_out;
};

}  // namespace dali

#endif  // DALI_TEST_DALI_TEST_BBOXES_H_
