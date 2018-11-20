#ifndef DALI_DALI_OPERATOR_TEST_H
#define DALI_DALI_OPERATOR_TEST_H

#include <gtest/gtest.h>
#include "dali/pipeline/pipeline.h"
#include "dali/pipeline/data/backend.h"
#include "datatype_convertions.h"

namespace dali {

namespace detail {

template<typename T>
struct is_tuple : std::false_type {
};

template<typename... T>
struct is_tuple<std::tuple<T...>> : std::true_type {
};


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

template<typename Input, typename Output>  // TODO: Inputs, Outputs
class DaliOperatorTest : public ::testing::Test {
//    static_assert(util::is_tuple<Inputs>::value, "Inputs has to be either a tuple or a void");
//    static_assert(util::is_tuple<Outputs>::value, "Outputs has to be a tuple");

public:
  using Shape = std::vector<size_t>;
  using Arguments =  std::map<std::string, float>; // TODO boost::any

  template<typename Backend>
  void RunTest(Arguments operator_arguments, Output anticipated_outputs) {
    auto op_spec = CreateOpSpec<Backend>(operator_name_, operator_arguments);
    AddInputsToPipeline<Backend>(*pipeline_, "input");
    AddOperatorToPipeline(*pipeline_, op_spec);
    outputs_ = RunPipeline<Backend>(*pipeline_, op_spec);
    ASSERT_TRUE(Verify(outputs_, anticipated_outputs));
  }


private:
  virtual std::pair<Input, Shape> SetInputs() const = 0;  // TODO std::vector

  virtual std::string SetOperator() const = 0; // TODO std::vector

  virtual bool Verify(Output outputs, Output anticipated_outputs) const = 0;


  void SetUp() final {
    auto inp = SetInputs();
    inputs_ = inp.first;
    input_shape_ = inp.second;
    operator_name_ = SetOperator();
    InitPipeline();
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
    auto tensor_list = ToTensorList<Backend>(inputs_, input_shape_);

    //tmp
    auto ptr = tensor_list.template tensor<float>(0);

    pipeline.SetExternalInput(input_name, tensor_list);
  }


  void AddOperatorToPipeline(Pipeline &pipeline, OpSpec op_spec) {
    pipeline.AddOperator(op_spec);
  }


//  template<typename Backend>
  DeviceWorkspace CreateWorkspace() const {
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
  Output RunPipeline(Pipeline &pipeline, OpSpec spec) {
    vector<std::pair<string, string>> vecoutputs_;
    for (int i = 0; i < spec.NumOutput(); ++i) {
      vecoutputs_.push_back(std::make_pair(spec.OutputName(i), spec.OutputDevice(i)));
    }
    pipeline.Build(vecoutputs_);
    pipeline.RunCPU();
    pipeline.RunGPU();
    auto workspace = CreateWorkspace();
//    DeviceWorkspace workspace;
    pipeline.Outputs(&workspace);
    GetOutputsFromWorkspace<Backend>(workspace);

//    return *ptr;
  }


  template<typename Backend>
  Output GetOutputsFromWorkspace(DeviceWorkspace ws) {
    auto i = ws.Output<Backend>(0);
    auto ptr = i->template data<int>();
  }


  template<typename Backend>
  OpSpec CreateOpSpec(std::string operator_name, Arguments operator_arguments) {
    OpSpec opspec = OpSpec(operator_name);
    for (const auto &arg : operator_arguments) {
      opspec.AddArg(arg.first, arg.second);
    }
    opspec.AddInput("input", detail::BackendStringName<Backend>());
    opspec.AddOutput("output", detail::BackendStringName<Backend>());
    return opspec;
  }


  Input inputs_;
  Output outputs_;
  Shape input_shape_;
  std::string operator_name_;
  std::unique_ptr<Pipeline> pipeline_;
  const size_t batch_size_ = 32;
  const size_t num_threads_ = 2;
};

}  // namespace dali

#endif //DALI_DALI_OPERATOR_TEST_H
