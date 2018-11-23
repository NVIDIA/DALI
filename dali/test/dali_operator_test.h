#ifndef DALI_DALI_OPERATOR_TEST_H
#define DALI_DALI_OPERATOR_TEST_H

#include <gtest/gtest.h>
#include "dali/pipeline/pipeline.h"
#include "dali/pipeline/data/backend.h"
#include "datatype_conversions.h"

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
inline std::string BackendStringName<CPUBackend>() {
  return "cpu";
}


template<>
inline std::string BackendStringName<GPUBackend>() {
  return "gpu";
}


template<typename T>
T GetOutputsFromWorkspace(DeviceWorkspace workspace) {
  DALI_FAIL("Provided type not supported yet. You may want to write your own specialization.");
}


template<>
inline std::vector<float> GetOutputsFromWorkspace(DeviceWorkspace workspace) {
  // TODO(mszolucha): support other backends
  auto tl = workspace.Output<CPUBackend>(0);
  auto ptr = tl->template data<float>();
  auto num = tl->size();
  return std::vector<float>{ptr, ptr + num};
}


template<>
inline int GetOutputsFromWorkspace(DeviceWorkspace workspace) {
  // TODO(mszolucha): support other backends
  auto i = workspace.Output<CPUBackend>(0);
  auto val = i->template data<int>();
  return *val;
}


}  // namespace util

template<typename Input, typename Output>
class DaliOperatorTest : public ::testing::Test {

 public:
  using Shape = std::vector<size_t>;
  using Arguments =  std::map<std::string, float>; // TODO(mszolucha): boost::any

  template<typename Backend>
  void RunTest(Arguments operator_arguments, Output anticipated_outputs) {
    InitPipeline();
    const auto op_spec = CreateOpSpec<Backend>(operator_name_, operator_arguments);
    if (has_input_) {
      AddInputsToPipeline<Backend>(*pipeline_);
    }
    AddOperatorToPipeline(*pipeline_, op_spec);
    BuildPipeline(*pipeline_, op_spec);
    RunPipeline(*pipeline_);
    outputs_ = GetOutputFromPipeline(*pipeline_);
    ASSERT_TRUE(Verify(outputs_, anticipated_outputs));
  }


 private:

  virtual std::vector<std::pair<Input, Shape>> SetInputs() const = 0;

  virtual std::string SetOperator() const = 0; // TODO(mszolucha): std::vector

  virtual bool Verify(Output output, Output anticipated_output) const = 0;


  void SetUp() final {
    const auto inp = SetInputs();
    const auto num_inputs = inp.size();
    assert(num_inputs >= 0);
    has_input_ = num_inputs != 0;
    if (has_input_) {
      inputs_ = inp[0].first;
      input_shape_ = inp[0].second;
    }
    operator_name_ = SetOperator();
  }


  void TearDown() final {

  }


  void InitPipeline() {
    if (!pipeline_.get()) {
      pipeline_.reset(new Pipeline(batch_size_, num_threads_, 0));
    }
  }


  template<typename Backend>
  void AddInputsToPipeline(Pipeline &pipeline) {
    const std::string input_name = "input";
    pipeline.AddExternalInput(input_name);
    auto tensor_list = ToTensorList<Backend>(inputs_, input_shape_);
    pipeline.SetExternalInput(input_name, *tensor_list);
  }


  void AddOperatorToPipeline(Pipeline &pipeline, OpSpec op_spec) {
    pipeline.AddOperator(op_spec);
  }


  DeviceWorkspace CreateWorkspace() const {
    DeviceWorkspace ws;
    return ws;
  }


  void RunPipeline(Pipeline &pipeline) {
    pipeline.RunCPU();
    pipeline.RunGPU();
  }


  Output GetOutputFromPipeline(Pipeline &pipeline) {
    auto workspace = CreateWorkspace();
    pipeline.Outputs(&workspace);
    return detail::GetOutputsFromWorkspace<Output>(workspace);
  }


  void BuildPipeline(Pipeline &pipeline, OpSpec spec) {
    std::vector<std::pair<string, string>> vecoutputs_;
    for (int i = 0; i < spec.NumOutput(); ++i) {
      vecoutputs_.emplace_back(spec.OutputName(i), spec.OutputDevice(i));
    }
    pipeline.Build(vecoutputs_);
  }


  template<typename Backend>
  OpSpec CreateOpSpec(std::string operator_name, Arguments operator_arguments) {
    OpSpec opspec = OpSpec(operator_name);
    for (const auto &arg : operator_arguments) {
      opspec.AddArg(arg.first, arg.second);
    }
    if (has_input_) {
      opspec.AddInput("input", detail::BackendStringName<Backend>());
    }
    opspec.AddOutput("output", detail::BackendStringName<Backend>());
    return opspec;
  }


  Input inputs_;
  Output outputs_;
  Shape input_shape_;
  std::string operator_name_;
  std::unique_ptr<Pipeline> pipeline_;
  const size_t batch_size_ = 1;
  const size_t num_threads_ = 1;
  bool has_input_;

};

}  // namespace dali

#endif //DALI_DALI_OPERATOR_TEST_H
