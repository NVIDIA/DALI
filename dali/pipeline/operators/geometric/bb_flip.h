#ifndef DALI_BB_FLIP_H
#define DALI_BB_FLIP_H

#include <dali/pipeline/operators/operator.h>
#include <dali/pipeline/operators/common.h>

namespace dali {

class BbFlip : public Operator<CPUBackend> {
 public:
  explicit inline BbFlip(const OpSpec &spec) :
          Operator<CPUBackend>(spec),
          coordinates_type_wh_(spec.GetArgument<bool>("coordinates_type")) {
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
          if (data[i] < 0 || data[i] > 1.0)
            return false;
        }
        return true;
    }(input_data, input.size()), "Not all bounding box parameters are in [0.0, 1.0]");
    DALI_ENFORCE([](const float *data, size_t size, bool coors_type_wh) -> bool {
        if (!coors_type_wh) return true; // Assert not applicable for 2-point representation
        for (int i = 0; i < size; i += 4) {
          if (data[i] + data[i + 2] > 1.0 || data[i + 1] + data[i + 3] > 1.0)
            return false;
        }
        return true;
    }(input_data, input.size(), coordinates_type_wh_), "Incorrect width or height");


    auto output = ws->Output<CPUBackend>(idx);
    // XXX: Setting type of output (i.e. Buffer -> buffer.h)
    //      explicitly is required for further processing
    //      It can also be achieved with mutable_data<>()
    //      function.
    output->set_type(TypeInfo::Create<float>());
    output->Resize({BB_TYPE_SIZE});
    auto output_data = output->mutable_data<float>();


    auto x = input_data[0];
    auto w = coordinates_type_wh_ ? input_data[2] : input_data[2] - input_data[0];
    auto h = coordinates_type_wh_ ? input_data[3] : input_data[3] - input_data[1];

    output_data[0] = (1.0f - x) - w;
    output_data[1] = input_data[1];
    output_data[2] = coordinates_type_wh_ ? w : output_data[0] + w;
    output_data[3] = coordinates_type_wh_ ? h : output_data[1] + h;

  }


 private:
  const int BB_TYPE_SIZE = 4; // Bounding box is always vector of 4 floats
  bool coordinates_type_wh_; // TODO doc
};

} // namespace dali

#endif //DALI_BB_FLIP_H
