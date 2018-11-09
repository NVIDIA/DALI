#ifndef DALI_DALI_OPERATOR_TEST_H
#define DALI_DALI_OPERATOR_TEST_H

#include <gtest/gtest.h>
#include "dali/pipeline/pipeline.h"
#include "dali/pipeline/data/backend.h"

namespace dali {

namespace util {

template<typename T>
struct is_tuple : std::false_type {
};

template<typename... T>
struct is_tuple<std::tuple<T...>> : std::true_type {
};


/// This is a helper function, because there's no `if constexpr` in C++11
template<typename Backend>
std::string BackendStringName() {
  DALI_FAIL("Selected Backend is not supported");
}


template<>
std::string BackendStringName<CPUBackend>() {
  return "cpu";
}


template<>
std::string BackendStringName<GPUBackend>() {
  return "gpu";
}

}  // namespace util

template<typename Inputs, typename Outputs>
class DaliOperatorTest : public ::testing::Test {
//    static_assert(util::is_tuple<Inputs>::value, "Inputs has to be either a tuple or a void");
//    static_assert(util::is_tuple<Outputs>::value, "Outputs has to be a tuple");

public:
  using Arguments =  std::map<std::string, float>; // TODO boost::any

  template<typename Backend>
  void RunTest(Arguments operator_arguments, Outputs anticipated_outputs) {
    auto op_spec = CreateOpSpec<Backend>(operator_name_, operator_arguments);
    AddOperatorToPipeline(*pipeline_, op_spec);
    AddInputsToPipeline<Backend>(*pipeline_, "input");
    outputs_ = RunPipeline<Backend>(*pipeline_, op_spec);
    ASSERT_TRUE(Verify(outputs_, anticipated_outputs));
  }


private:
  virtual Inputs SetInputs() const = 0;  // TODO std::vector

  virtual std::string SetOperator() const = 0; // TODO std::vector

  virtual bool Verify(Outputs outputs, Outputs anticipated_outputs) const = 0;


  void SetUp() final {
    inputs_ = SetInputs();
    operator_name_ = SetOperator();
    InitPipeline();
    CreateWorkspace();
  }


  void TearDown() final {

  }


  void InitPipeline() {
    if (!pipeline_.get()) {
      pipeline_.reset(new Pipeline(batch_size_, num_threads_, 0));
    }
  }


  template<typename Backend>
  void AddInputsToPipeline(Pipeline &pipeline, std::string input_name) {
    pipeline.AddExternalInput(input_name);
    auto tensor_list = ToTensorList<Backend>(inputs_);
    pipeline.SetExternalInput(input_name, tensor_list);
  }


  void AddOperatorToPipeline(Pipeline &pipeline, OpSpec op_spec) {
    pipeline.AddOperator(op_spec);
  }


//  template<typename Backend>
  Workspace<CPUBackend> CreateWorkspace() {
    Workspace<CPUBackend> ws;
    return ws;
  }


  /**
   * TODO
   * @tparam Backend
   * @param spec
   * @return
   */
  template<typename Backend>
  Outputs RunPipeline(Pipeline &pipeline, OpSpec spec) {
    vector<std::pair<string, string>> vecoutputs_;
    DeviceWorkspace workspace_;
    for (int i = 0; i < spec.NumOutput(); ++i)
      vecoutputs_.push_back(std::make_pair(spec.OutputName(i), spec.OutputDevice(i)));
    pipeline.Build(vecoutputs_);
    pipeline.RunCPU();
    pipeline.RunGPU();
    pipeline.Outputs(&workspace_);
    auto i = workspace_.Output<Backend>(0);
    auto ptr = i->template data<int>();
    return *ptr;
  }


  Outputs GetOutputsFromWorkspace(DeviceWorkspace ws) {

  }


  template<typename Backend>
  OpSpec CreateOpSpec(std::string operator_name, Arguments operator_arguments) {
    OpSpec opspec = OpSpec(operator_name);
    for (const auto &arg : operator_arguments) {
      opspec.AddArg(arg.first, arg.second);
    }
    opspec.AddOutput("output", util::BackendStringName<Backend>());
    return opspec;
  }


  template<typename Backend>
  TensorList<Backend> ToTensorList(const Inputs &inputs) {
    TensorList<Backend> tensor_list;
    tensor_list.Resize({{1}});
    auto ptr = tensor_list.template mutable_tensor<Inputs>(0);
    ptr[0] = inputs;

    return tensor_list;
  };

  Inputs inputs_;
  Outputs outputs_;
  std::string operator_name_;
  std::unique_ptr<Pipeline> pipeline_;
  const size_t batch_size_ = 32;
  const size_t num_threads_ = 2;
};

}  // namespace dali

#endif //DALI_DALI_OPERATOR_TEST_H
