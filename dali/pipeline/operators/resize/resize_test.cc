// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/test/dali_test_single_op.h"

namespace dali {

template <typename ImgType>
class ResizeTest : public DALISingleOpTest {
 public:
  USING_DALI_SINGLE_OP_TEST();

  vector<TensorList<CPUBackend>*>
  Reference(const vector<TensorList<CPUBackend>*> &inputs,
            DeviceWorkspace *ws) {
    // single input - encoded images
    // single output - decoded images
    vector<TensorList<CPUBackend>*> outputs(1);

    vector<Tensor<CPUBackend>> out(inputs[0]->ntensor());

    const TensorList<CPUBackend>& image_data = *inputs[0];
    const auto n = spec_.name();

    c_ = (IsColor(img_type_) ? 3 : 1);
    auto cv_type = (c_ == 3) ? CV_8UC3 : CV_8UC1;

    const int resize_a = spec_.GetArgument<int>("resize_a");
    const int resize_b = spec_.GetArgument<int>("resize_b");
    const bool random_resize = spec_.GetArgument<bool>("random_resize");
    const bool warp_resize = spec_.GetArgument<bool>("warp_resize");

    // Can't handle these right now
    if (!spec_.GetArgument<bool>("save_attrs")) {
      assert(random_resize == false);
    }

    int rsz_h, rsz_w;

    for (int i = 0; i < image_data.ntensor(); ++i) {
      auto *data = image_data.tensor<unsigned char>(i);
      auto shape = image_data.tensor_shape(i);
      const int H = shape[0], W = shape[1];

      cv::Mat input = cv::Mat(H, W, cv_type,
                              const_cast<unsigned char*>(data));

      // perform the resize
      cv::Mat rsz_img;

      // determine resize parameters
      if (spec_.GetArgument<bool>("save_attrs")) {
        const int *t = ws->Output<CPUBackend>(1)->tensor<int>(i);
        rsz_h = t[0];
        rsz_w = t[1];
      } else {
        if (warp_resize) {
          rsz_h = resize_a;
          rsz_w = resize_b;
        } else {
          if (H >= W) {
            rsz_w = resize_a;
            rsz_h = static_cast<int>(H * static_cast<float>(rsz_w) / W);
          } else {  // W > H
            rsz_h = resize_a;
            rsz_w = static_cast<int>(W * static_cast<float>(rsz_h) / H);
          }
        }
      }

      cv::resize(input, rsz_img, cv::Size(rsz_h, rsz_w), 0, 0, cv::INTER_LINEAR);

      out[i].Resize({rsz_img.rows, rsz_img.cols, c_});
      auto *out_data = out[i].mutable_data<unsigned char>();

      std::memcpy(out_data, rsz_img.ptr(), rsz_img.rows * rsz_img.cols * c_);
    }
    outputs[0] = new TensorList<CPUBackend>();
    outputs[0]->Copy(out, 0);
    return outputs;
  }

 protected:
  OpSpec DefaultSchema() {
    return OpSpec("Resize")
           .AddArg("device", "gpu")
           .AddInput("input", "gpu")
           .AddOutput("output", "gpu");
  }

  const DALIImageType img_type_ = ImgType::type;
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(ResizeTest, Types);

TYPED_TEST(ResizeTest, TestFixedResize) {
  TensorList<CPUBackend> data;
  this->DecodedData(&data, this->batch_size_, this->img_type_);
  this->SetExternalInputs({std::make_pair("input", &data)});

  this->AddSingleOp(this->DefaultSchema()
                    .AddArg("resize_a", 480)
                    .AddArg("resize_b", 480)
                    .AddArg("warp_resize", false)
                    .AddArg("mirror_prob", 0.f));

  DeviceWorkspace ws;
  this->RunOperator(&ws);

  // Note: There seem to be implementation differences between
  // NPP and openCV - put a relatively high error bound that will catch
  // obvious errors, but pass on slight differences.
  this->SetEps(2e-1);

  this->CheckAnswers(&ws, {0});
}

TYPED_TEST(ResizeTest, TestFixedResizeWarp) {
  TensorList<CPUBackend> data;
  this->DecodedData(&data, this->batch_size_, this->img_type_);
  this->SetExternalInputs({std::make_pair("input", &data)});

  this->AddSingleOp(this->DefaultSchema()
                    .AddArg("resize_a", 480)
                    .AddArg("resize_b", 480)
                    .AddArg("warp_resize", true)
                    .AddArg("mirror_prob", 0.f));

  DeviceWorkspace ws;
  this->RunOperator(&ws);

  // Note: There seem to be implementation differences between
  // NPP and openCV - put a relatively high error bound that will catch
  // obvious errors, but pass on slight differences.
  this->SetEps(2e-1);

  this->CheckAnswers(&ws, {0});
}

TYPED_TEST(ResizeTest, TestRandomResize) {
  TensorList<CPUBackend> data;
  this->DecodedData(&data, this->batch_size_, this->img_type_);
  this->SetExternalInputs({std::make_pair("input", &data)});

  this->AddSingleOp(this->DefaultSchema()
                    .AddArg("resize_a", 256)
                    .AddArg("resize_b", 480)
                    .AddArg("random_resize", true)
                    .AddArg("mirror_prob", 0.f)
                    .AddArg("save_attrs", true)
                    .AddArg("warp_resize", false)
                    .AddOutput("attrs", "cpu"));

  DeviceWorkspace ws;
  this->RunOperator(&ws);

  // Note: There seem to be implementation differences between
  // NPP and openCV - put a relatively high error bound that will catch
  // obvious errors, but pass on slight differences.
  this->SetEps(2e-1);

  this->CheckAnswers(&ws, {0});
}

TYPED_TEST(ResizeTest, TestRandomResizeWarp) {
  TensorList<CPUBackend> data;
  this->DecodedData(&data, this->batch_size_, this->img_type_);
  this->SetExternalInputs({std::make_pair("input", &data)});

  this->AddSingleOp(this->DefaultSchema()
                    .AddArg("resize_a", 256)
                    .AddArg("resize_b", 480)
                    .AddArg("random_resize", true)
                    .AddArg("mirror_prob", 0.f)
                    .AddArg("save_attrs", true)
                    .AddArg("warp_resize", true)
                    .AddOutput("attrs", "cpu"));

  DeviceWorkspace ws;
  this->RunOperator(&ws);

  // Note: There seem to be implementation differences between
  // NPP and openCV - put a relatively high error bound that will catch
  // obvious errors, but pass on slight differences.
  this->SetEps(2e-1);

  this->CheckAnswers(&ws, {0});
}

}  // namespace dali
