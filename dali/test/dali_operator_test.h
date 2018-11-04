#ifndef DALI_DALI_OPERATOR_TEST_H
#define DALI_DALI_OPERATOR_TEST_H

#include <gtest/gtest.h>
#include "dali/pipeline/pipeline.h"
#include "dali/pipeline/data/backend.h"

namespace dali {

template<typename T>
struct is_tuple;

template<typename Inputs, typename Outputs>
class DaliOperatorTest : public ::testing::Test {
    //is_tuple
public:
    using Arguments =  std::map<std::string, float>; // TODO custom type

//    template<typename Backend>
    void RunTest(Arguments operator_arguments) {
        //is base of Backend
//        auto op_spec = CreateOpSpec(operator_name_, operator_arguments);
//        AddOperatorToPipeline(op_spec);
//        RunPipeline(op_spec);
//        ASSERT_TRUE(Verify(outputs_, anticipated_outputs_)) << "Działka\n";
    }


private:
    virtual Inputs SetInputs() const = 0;

    virtual Outputs SetAnticipatedOutputs() const = 0;

    virtual std::string SetOperator() const = 0;

    virtual bool Verify(Outputs outputs, Outputs anticipated_outputs) const = 0;


    void SetUp() override final {
//        inputs_ = SetInputs();
//        anticipated_outputs_ = SetAnticipatedOutputs();
//        operator_name_ = SetOperator();
//        InitPipeline();
//        AddInputsToPipeline("input");
//        CreateWorkspace();
    }


    void TearDown() override final {

    }


    void InitPipeline() {
//        if (!pipeline_.get()) {
//            pipeline_.reset(new Pipeline(batch_size_, num_threads_, 0));
//        }
    }


//    void AddInputsToPipeline(std::string input_name) {
//        pipeline_->AddExternalInput(input_name);
//        auto tensor_list = ToTensorList<CPUBackend>(inputs_);
//        TensorList<CPUBackend> tensor_list;
//        pipeline_->SetExternalInput(input_name, tensor_list);
//    }


    void AddOperatorToPipeline(OpSpec op_spec) {
//        pipeline_->AddOperator(op_spec);
    }


//    DeviceWorkspace CreateWorkspace() {
//        DeviceWorkspace ws;
//        return ws;
//    }


    void RunPipeline(OpSpec spec) {
//        for (int i = 0; i < spec.NumOutput(); ++i)
//            vecoutputs_.push_back(std::make_pair(spec.OutputName(i), spec.OutputDevice(i)));
//        pipeline_->Build(vecoutputs_);
//        pipeline_->RunCPU();
//        pipeline_->RunGPU();
//        pipeline_->Outputs(&workspace_);
//        auto i = workspace_.Output<CPUBackend>(0);
//        auto ptr  = i->data<int>();
    }


    Outputs GetOutputsFromWorkspace(DeviceWorkspace ws) {

    }

    OpSpec CreateOpSpec(std::string operator_name, Arguments operator_arguments) {
//        return OpSpec(operator_name).AddArg("probability",1.f).AddOutput("cf_out","cpu");
    }


    template<typename Backend>
    TensorList<Backend> ToTensorList(const Inputs &inputs) {
        //is_tuple
//        return TensorList<Backend>();
    };

//    Inputs inputs_;
//    Outputs outputs_, anticipated_outputs_;
//    std::string operator_name_;
//    std::shared_ptr<Pipeline> pipeline_;
//    DeviceWorkspace workspace_;
//    int batch_size_ = 32;
//    int num_threads_ = 2;
//    vector<std::pair<string, string>> vecoutputs_;
};

}  // namespace dali

#endif //DALI_DALI_OPERATOR_TEST_H
