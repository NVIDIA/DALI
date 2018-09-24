#include "dali/test/dali_test_single_op.h"

namespace dali {

template<typename ImageType>
class BbFlipTest : public DALISingleOpTest<ImageType> {
 protected:
    std::vector<TensorList<CPUBackend> *>
    Reference(const std::vector<TensorList<CPUBackend> *> &inputs, DeviceWorkspace *ws) override {
        const int c = this->GetNumColorComp();
        auto cv_type = (c == 3) ? CV_8UC3 : CV_8UC1;
        const TensorList<CPUBackend> &image_data = *inputs[0];

        for (int i = 0; i < image_data.ntensor(); ++i) {
            auto *data = image_data.tensor<unsigned char>(i);
            auto shape = image_data.tensor_shape(i);
            const int H = shape[0], W = shape[1];

            cv::Mat imput(H, W, cv_type, const_cast<unsigned char *>(data));

            cv::imshow("asd", imput);
            cv::waitKey(0);

        }
    }

    const OpSpec DecodingOp() const {
        return OpSpec("BbFlip")
                .AddArg("device","cpu")
                .AddArg("myarg", this->img_type_)
                .AddInput("myinput", "cpu")
                .AddOutput("myoutput", "cpu");
    }

};

typedef ::testing::Types<RGB/*, BGR, Gray*/> Types;
TYPED_TEST_CASE(BbFlipTest, Types);

TYPED_TEST(BbFlipTest, HorizontalFlipTest) {

    TensorList<CPUBackend> encoded_data;
    this->EncodedJPEGData(&encoded_data);
    this->SetExternalInputs({std::make_pair("myinput", &encoded_data)});
    this->RunOperator(this->DecodingOp(), .7);

//    this->TstBody(this->DefaultSchema().AddArg("myarg", 666));
//    this->TstBody(this->DefaultSchema("BbFlip", "cpu"));
//    this->TstBody(this->DefaultSchema()
//                          .AddArg("resize_shorter", 480.f)
//                          .AddArg("crop", vector<int>{224, 224}), 5e-6);
}

} // namespace dali
