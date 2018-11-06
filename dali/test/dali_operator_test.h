#ifndef DALI_DALI_OPERATOR_TEST_H
#define DALI_DALI_OPERATOR_TEST_H

#include <gtest/gtest.h>
#include "dali/pipeline/pipeline.h"
#include "dali/pipeline/data/backend.h"

namespace dali {

template<typename T>
struct is_tuple : std::false_type {
};

template<typename... T>
struct is_tuple<std::tuple<T...>> : std::true_type {
};

template<typename Inputs, typename Outputs>
class DaliOperatorTest : public ::testing::Test {
  //is_tuple
 public:
  using Arguments =  std::map<std::string, float>; // TODO boost::any

//    template<typename Backend>
  void RunTest(Arguments operator_arguments, Outputs anticipated_outputs) {
    //is base of Backend
    auto op_spec = CreateOpSpec(operator_name_, operator_arguments);
    AddOperatorToPipeline(op_spec);
    outputs_ = RunPipeline<CPUBackend>(op_spec);
    ASSERT_TRUE(Verify(outputs_, anticipated_outputs));
  }


 private:
  virtual Inputs SetInputs() const = 0;

  virtual std::string SetOperator() const = 0;

  virtual bool Verify(Outputs outputs, Outputs anticipated_outputs) const = 0;


  void SetUp() final {
    inputs_ = SetInputs();
    operator_name_ = SetOperator();
    InitPipeline();
    AddInputsToPipeline("input");
    CreateWorkspace();
  }


  void TearDown() final {

  }


  void InitPipeline() {
    if (!pipeline_.get()) {
      pipeline_.reset(new Pipeline(batch_size_, num_threads_, 0));
    }
  }


//  template<typename Backend>
  void AddInputsToPipeline(std::string input_name) {
    pipeline_->AddExternalInput(input_name);
    auto tensor_list = ToTensorList<CPUBackend>(inputs_);
    pipeline_->SetExternalInput(input_name, tensor_list);
  }


  void AddOperatorToPipeline(OpSpec op_spec) {
    pipeline_->AddOperator(op_spec);
  }


  DeviceWorkspace CreateWorkspace() {
    DeviceWorkspace ws;
    return ws;
  }


  /**
   * TODO
   * @tparam Backend
   * @param spec
   * @return
   */
  template<typename Backend>
  Outputs RunPipeline(OpSpec spec) {
    for (int i = 0; i < spec.NumOutput(); ++i)
      vecoutputs_.push_back(std::make_pair(spec.OutputName(i), spec.OutputDevice(i)));
    pipeline_->Build(vecoutputs_);
    pipeline_->RunCPU();
    pipeline_->RunGPU();
    pipeline_->Outputs(&workspace_);
    auto i = workspace_.Output<Backend>(0);
    auto ptr = i->template data<int>();
    return *ptr;
  }


  Outputs GetOutputsFromWorkspace(DeviceWorkspace ws) {

  }


//  template<typename Backend>
  OpSpec CreateOpSpec(std::string operator_name, Arguments operator_arguments) {
    OpSpec opspec = OpSpec(operator_name);
    for (const auto &arg : operator_arguments) {
      opspec.AddArg(arg.first, arg.second);
    }
    opspec.AddOutput("cf_out", "cpu");
    return opspec;
  }


  template<typename Backend>
  TensorList<Backend> ToTensorList(const Inputs &inputs) {
    //is_tuple
    TensorList<Backend> tensor_list;
    tensor_list.Resize({{1}});
    auto ptr = tensor_list.template mutable_tensor<Inputs>(0);
    ptr[0] = inputs;

    return tensor_list;
  };

  Inputs inputs_;
  Outputs outputs_;
  std::string operator_name_;
  std::shared_ptr<Pipeline> pipeline_;
  DeviceWorkspace workspace_;
  int batch_size_ = 32;
  int num_threads_ = 2;
  vector<std::pair<string, string>> vecoutputs_;
};

}  // namespace dali

#endif //DALI_DALI_OPERATOR_TEST_H
