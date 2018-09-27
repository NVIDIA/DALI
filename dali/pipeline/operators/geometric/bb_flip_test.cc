#include "dali/test/dali_test_single_op.h"

namespace dali {

namespace {

const int BB_STRUCT_SIZE = 4;

/**
 * Roi can be represented in two ways:
 * 1. Upper-left corner, width, height
 *    (x1, y1,  w,  h)
 * 2. Upper-left and Lower-right corners
 *    (x1, y1, x2, y2)
 *
 * Both of them have coordinates in image coordinate system
 * (i.e. 0.0-1.0)
 */
struct Roi {
  float roi[BB_STRUCT_SIZE];
};

/**
 * Functor for calculating Roi hash
 */
struct RoiHash {
  std::size_t operator()(Roi const &roi) const noexcept {
    std::stringstream ss;
    ss << std::to_string(roi.roi[0]) << std::to_string(roi.roi[1])
       << std::to_string(roi.roi[2]) << std::to_string(roi.roi[3]);
    return std::hash<std::string>{}(ss.str());
  }
};


bool operator==(const Roi &lh, const Roi &rh) noexcept {
  return RoiHash{}(lh) == RoiHash{}(rh);
}


std::unordered_map<Roi, Roi, RoiHash> wh_rois = {
        {{.2,  .2,  .4, .3}, {.4,  .2,  .4, .3}},
        {{.0,  .0,  .5, .5}, {.5,  .0,  .5, .5}},
        {{.3,  .2,  .1, .1}, {.6,  .2,  .1, .1}},
        {{.0,  .0,  .2, .3}, {.8,  .0,  .2, .3}},
        {{.0,  .0,  .1, .1}, {.9,  .0,  .1, .1}},
        {{.5,  .5,  .1, .1}, {.0,  .5,  .1, .1}},
        {{.0,  .6,  .7, .7}, {.3,  .6,  .7, .7}},
        {{.6,  .2,  .3, .3}, {.1,  .2,  .3, .3}},
        {{.4,  .3,  .5, .5}, {.1,  .3,  .5, .5}},
        {{.25, .25, .5, .5}, {.25, .25, .5, .5}},
};

std::unordered_map<Roi, Roi, RoiHash> two_pt_rois = {
        {{.2,  .2,  .6,  .5},  {.4,  .2,  .8,  .5}},
        {{.0,  .0,  .5,  .5},  {.5,  .5,  1.,  1.}},
        {{.3,  .2,  .4,  .3},  {.6,  .2,  .7,  .3}},
        {{.0,  .0,  .2,  .3},  {.8,  .0,  1.,  .3}},
        {{.0,  .0,  .1,  .1},  {.9,  .0,  1.,  .1}},
        {{.5,  .5,  .6,  .6},  {.0,  .5,  .1,  .6}},
        {{.0,  .6,  .7,  .9},  {.3,  .6,  1.,  .9}},
        {{.6,  .2,  .3,  .3},  {.1,  .2,  .4,  .5}},
        {{.4,  .3,  .9,  .8},  {.1,  .3,  .6,  .8}},
        {{.25, .25, .75, .75}, {.25, .25, .75, .75}},

};

using RoiMap = std::unordered_map<Roi, Roi, RoiHash>;

} // namespace

template<typename ImageType>
class BbFlipTest : public DALISingleOpTest<ImageType> {
 protected:
  std::vector<TensorList<CPUBackend> *>
  Reference(const std::vector<TensorList<CPUBackend> *> &inputs, DeviceWorkspace *ws) override {

    std::vector<Tensor<CPUBackend>> out(inputs[0]->ntensor());
    out[0].Resize({4});
    auto *out_data = out[0].mutable_data<float>();

    auto roi = test_data_->begin()->second.roi;
    std::memcpy(out_data, roi, 4 * sizeof(float));

    vector<TensorList<CPUBackend> *> outputs(1);
    outputs[0] = new TensorList<CPUBackend>();
    outputs[0]->Copy(out, nullptr);

    return outputs;
  }


  template<typename Backend>
  void LoadBbData(TensorList<Backend> &tensor_list, const RoiMap *input_data,
                  int batch_size) noexcept {
    test_data_ = input_data;
    this->SetBatchSize(batch_size);
    tensor_list.set_type(TypeInfo::Create<float>());
    std::vector<std::vector<long int>> sz = {std::vector<long int>(batch_size, 4)};
    tensor_list.Resize({{4}});

    auto roi = test_data_->begin()->first.roi;

    auto ptr = tensor_list.template mutable_tensor<float>(0);
    auto buffer = tensor_list.template data<float>();
    std::memcpy(ptr, roi, BB_STRUCT_SIZE * sizeof(float));
  }


  const OpSpec DecodingOp(bool wh_coordinates_type) const noexcept {

    return OpSpec("BbFlip")
            .AddArg("coordinates_type", wh_coordinates_type)
            .AddInput("bb_input", "cpu")
            .AddOutput("bb_output", "cpu");
  }


 private:
  const RoiMap *test_data_;

};

typedef ::testing::Types<Gray> Types;
TYPED_TEST_CASE(BbFlipTest, Types);

TYPED_TEST(BbFlipTest, WidthHeightRepresentation) {

  TensorList<CPUBackend> bb_test_data;
  this->LoadBbData(bb_test_data, &wh_rois, 1);
  this->SetExternalInputs({std::make_pair("bb_input", &bb_test_data)});
  this->RunOperator(this->DecodingOp(true), .01);
}


TYPED_TEST(BbFlipTest, TwoPointRepresentation) {
  TensorList<CPUBackend> bb_test_data;
  this->LoadBbData(bb_test_data, &two_pt_rois, 1);
  this->SetExternalInputs({std::make_pair("bb_input", &bb_test_data)});
  this->RunOperator(this->DecodingOp(false), .01);
}

} // namespace dali
