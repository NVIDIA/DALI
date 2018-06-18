// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dali/test/dali_test_single_op.h"

namespace dali {

template <typename ImgType>
class ResizeCropMirrorTest : public DALISingleOpTest {
 public:
  USING_DALI_SINGLE_OP_TEST();

  vector<TensorList<CPUBackend>*>
  Reference(const vector<TensorList<CPUBackend>*> &inputs,
            DeviceWorkspace *) {
    // single input - encoded images
    // single output - decoded images
    vector<TensorList<CPUBackend>*> outputs(1);

    vector<Tensor<CPUBackend>> out(inputs[0]->ntensor());

    const TensorList<CPUBackend>& image_data = *inputs[0];
    const auto n = spec_.name();

    c_ = (IsColor(img_type_) ? 3 : 1);
    auto cv_type = (c_ == 3) ? CV_8UC3 : CV_8UC1;

    int resize_a = spec_.GetArgument<float>("resize_x");
    int resize_b = spec_.GetArgument<float>("resize_y");
    bool warp_resize = true;
    if (resize_a == 0) {
      resize_a = spec_.GetArgument<float>("resize_shorter");
      warp_resize = false;
    }
    const vector<int> crop = spec_.GetRepeatedArgument<int>("crop");

    int rsz_h, rsz_w;
    int crop_x, crop_y;
    int crop_h = crop.at(0), crop_w = crop.at(1);
    int mirror = true;

    for (int i = 0; i < image_data.ntensor(); ++i) {
      auto *data = image_data.tensor<unsigned char>(i);
      auto shape = image_data.tensor_shape(i);
      const int H = shape[0], W = shape[1];

      cv::Mat input = cv::Mat(H, W, cv_type,
                              const_cast<unsigned char*>(data));

      // perform the resize
      cv::Mat rsz_img;

      // determine resize parameters
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

      crop_y = (rsz_h - crop_h) / 2;
      crop_x = (rsz_w - crop_w) / 2;

      cv::resize(input, rsz_img, cv::Size(rsz_w, rsz_h), 0, 0, cv::INTER_LINEAR);

      // Perform a crop
      cv::Mat crop_img(crop_h, crop_w, cv_type);
      int crop_offset = crop_y * rsz_w * c_ + crop_x * c_;
      uint8 *crop_ptr = rsz_img.ptr() + crop_offset;

      CUDA_CALL(cudaMemcpy2D(crop_img.ptr(), crop_w * c_,
                             crop_ptr, rsz_w * c_, crop_w * c_, crop_h,
                             cudaMemcpyHostToHost));


      // Random mirror
      cv::Mat mirror_img;
      if (mirror) {
        cv::flip(crop_img, mirror_img, 1);
      } else {
        mirror_img = crop_img;
      }

      out[i].Resize({mirror_img.rows, mirror_img.cols, c_});
      auto *out_data = out[i].mutable_data<unsigned char>();

      std::memcpy(out_data, mirror_img.ptr(), mirror_img.rows * mirror_img.cols * c_);
    }
    outputs[0] = new TensorList<CPUBackend>();
    outputs[0]->Copy(out, 0);
    return outputs;
  }

 protected:
  OpSpec DefaultSchema(bool fast_resize = false) {
    const char *op = (fast_resize) ? "FastResizeCropMirror"
                                   : "ResizeCropMirror";
    return OpSpec(op)
           .AddArg("device", "cpu")
           .AddArg("output_type", this->img_type_)
           .AddInput("input", "cpu")
           .AddOutput("output", "cpu");
  }

  const DALIImageType img_type_ = ImgType::type;
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(ResizeCropMirrorTest, Types);

TYPED_TEST(ResizeCropMirrorTest, TestFixedResizeAndCrop) {
  TensorList<CPUBackend> data;
  this->DecodedData(&data, this->batch_size_, this->img_type_);
  this->SetExternalInputs({std::make_pair("input", &data)});

  this->AddSingleOp(this->DefaultSchema()
                    .AddArg("resize_shorter", 480.f)
                    .AddArg("crop", vector<int>{224, 224}));

  DeviceWorkspace ws;
  this->RunOperator(&ws);

  // Note: lower accuracy due to TJPG and OCV implementations for BGR/RGB.
  // Difference is consistent, deterministic and goes away if I force OCV
  // instead of TJPG decoding.
  this->SetEps(5e-2);

  this->CheckAnswers(&ws, {0});
}

TYPED_TEST(ResizeCropMirrorTest, TestFixedResizeAndCropWarp) {
  TensorList<CPUBackend> data;
  this->DecodedData(&data, this->batch_size_, this->img_type_);
  this->SetExternalInputs({std::make_pair("input", &data)});

  this->AddSingleOp(this->DefaultSchema()
                    .AddArg("resize_x", 480.f)
                    .AddArg("resize_y", 480.f)
                    .AddArg("crop", vector<int>{224, 224}));

  DeviceWorkspace ws;
  this->RunOperator(&ws);

  // Note: lower accuracy due to TJPG and OCV implementations for BGR/RGB.
  // Difference is consistent, deterministic and goes away if I force OCV
  // instead of TJPG decoding.
  this->SetEps(5e-2);

  this->CheckAnswers(&ws, {0});
}

TYPED_TEST(ResizeCropMirrorTest, TestFixedFastResizeAndCrop) {
  TensorList<CPUBackend> data;
  this->DecodedData(&data, this->batch_size_, this->img_type_);
  this->SetExternalInputs({std::make_pair("input", &data)});

  this->AddSingleOp(this->DefaultSchema(true)
                    .AddArg("resize_shorter", 480.f)
                    .AddArg("crop", vector<int>{224, 224}));

  DeviceWorkspace ws;
  this->RunOperator(&ws);

  // Note: lower accuracy due to TJPG and OCV implementations for BGR/RGB.
  // Difference is consistent, deterministic and goes away if I force OCV
  // instead of TJPG decoding.
  this->SetEps(5e-2);

  this->CheckAnswers(&ws, {0});
}

TYPED_TEST(ResizeCropMirrorTest, TestFixedFastResizeAndCropWarp) {
  TensorList<CPUBackend> data;
  this->DecodedData(&data, this->batch_size_, this->img_type_);
  this->SetExternalInputs({std::make_pair("input", &data)});

  this->AddSingleOp(this->DefaultSchema(true)
                    .AddArg("resize_x", 480.f)
                    .AddArg("resize_y", 480.f)
                    .AddArg("crop", vector<int>{224, 224}));

  DeviceWorkspace ws;
  this->RunOperator(&ws);

  // Note: lower accuracy due to TJPG and OCV implementations for BGR/RGB.
  // Difference is consistent, deterministic and goes away if I force OCV
  // instead of TJPG decoding.
  this->SetEps(6e-2);

  this->CheckAnswers(&ws, {0});
}

}  // namespace dali
