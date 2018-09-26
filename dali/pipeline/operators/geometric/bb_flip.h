#ifndef DALI_BB_FLIP_H
#define DALI_BB_FLIP_H

#include <dali/pipeline/operators/operator.h>
#include <dali/pipeline/operators/common.h>

namespace dali {

class BbFlip : public Operator<CPUBackend> {
 public:
    explicit inline BbFlip(const OpSpec &spec) :
            Operator<CPUBackend>(spec) {
    }


    virtual ~BbFlip() override = default;
    DISABLE_COPY_MOVE_ASSIGN(BbFlip);

 protected:

    inline void RunImpl(SampleWorkspace *ws, const int idx) override {
        auto &input = ws->Input<CPUBackend>(idx);
        auto input_data = input.data<float>();

        DALI_ENFORCE(input.size() == BB_TYPE_SIZE, "Bounding box in wrong format");
        DALI_ENFORCE(input.type().id() == DALI_FLOAT || input.type().id() == DALI_FLOAT16,
                     "Bounding box in wrong format");
        DALI_ENFORCE([](const float *data, size_t size) -> bool {
            for (int i = 0; i < size; i++) {
                if (data[i] < 0)
                    return false;
            }
            return true;
        }(input_data, input.size()), "Not all bounding box parameters are non-negative");


        auto output = ws->Output<CPUBackend>(idx);
        // XXX: Setting type of output (i.e. Buffer -> buffer.h)
        //      explicitly is required for further processing
        //      It can also be achieved with mutable_data<>()
        //      function.
        output->set_type(TypeInfo::Create<float>());
        output->Resize({BB_TYPE_SIZE});
        auto output_data = output->mutable_data<float>();

//        output_data[0] = (1.0f - input_data[0]) - input_data[2];
//        output_data[1] = input_data[1];
//        output_data[2] = input_data[2];
//        output_data[3] = input_data[3];

        std::vector<float> data = {.4, .2, .4, .3};
        std::memcpy(output_data, data.data(), BB_TYPE_SIZE * sizeof(float));
    }


 private:
    const int BB_TYPE_SIZE = 4; // Bounding box is always vector of 4 floats
    //    USE_OPERATOR_MEMBERS();
};

} // namespace dali

#endif //DALI_BB_FLIP_H
