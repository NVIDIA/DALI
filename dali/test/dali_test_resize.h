// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef DALI_TEST_DALI_TEST_RESIZE_H_
#define DALI_TEST_DALI_TEST_RESIZE_H_

#include "dali/test/dali_test_single_op.h"
#include <utility>
#include <vector>
#include <string>

namespace dali {

typedef enum {
  t_externSizes = 1,
  t_cropping    = 2,
  t_mirroring   = 4
} t_resizeOptions;

template <typename ImgType>
class GenericResizeTest : public DALISingleOpTest {
 public:
  USING_DALI_SINGLE_OP_TEST();

  void TstBody(const string &pName, const string &pDevice = "gpu", double eps = 2e-1) {
    OpSpec operation = DefaultSchema(pName, pDevice);
    TstBody(GetOperationSpec(operation), eps);
  }

  void TstBody(const OpSpec &operation, double eps = 2e-1) {
#ifdef PIXEL_STAT_FILE
    FILE *file = fopen(PIXEL_STAT_FILE".txt", "a");
    fprintf(file, "Eps = %6.4f\n", eps);
    fprintf(file, " Color#:       mean:        std:          eq.         pos.         neg.\n");
    fclose(file);
#endif
    TensorList<CPUBackend> data;
    this->DecodedData(&data, this->batch_size_, this->img_type_);
    this->SetExternalInputs({std::make_pair("input", &data)});

    this->AddSingleOp(operation);

    DeviceWorkspace ws;
    this->RunOperator(&ws);

    this->SetEps(eps);
    this->CheckAnswers(&ws, {0});
  }

  vector<TensorList<CPUBackend>*>
  Reference(const vector<TensorList<CPUBackend>*> &inputs, DeviceWorkspace *ws) {
    c_ = (IsColor(img_type_) ? 3 : 1);
    auto cv_type = (c_ == 3) ? CV_8UC3 : CV_8UC1;

    // single input - encoded images
    // single output - decoded images
    vector<Tensor<CPUBackend>> out(inputs[0]->ntensor());
    const TensorList<CPUBackend>& image_data = *inputs[0];

    const uint resizeOptions = getResizeOptions();

    int resize_a = 0, resize_b = 0;
    bool warp_resize = true;

    const bool useExternSizes = (resizeOptions & t_externSizes) &&
                                spec_.GetArgument<bool>("save_attrs");
    if (!useExternSizes) {
      if (resizeOptions & t_externSizes)
        assert(false);  // Can't handle these right now

      resize_a = spec_.GetArgument<float>("resize_x");
      warp_resize = resize_a != 0;
      if (warp_resize)
        resize_b = spec_.GetArgument<float>("resize_y");
      else
        resize_a = spec_.GetArgument<float>("resize_shorter");
    }

    int crop_h = 0, crop_w = 0;
    if (resizeOptions & t_cropping) {
      // Perform a crop
      const vector<int> crop = spec_.GetRepeatedArgument<int>("crop");
      crop_h = crop.at(0), crop_w = crop.at(1);
    }

    int rsz_h, rsz_w;
    for (int i = 0; i < image_data.ntensor(); ++i) {
      auto *data = image_data.tensor<unsigned char>(i);
      auto shape = image_data.tensor_shape(i);
      const int H = shape[0], W = shape[1];

      // determine resize parameters
      if (useExternSizes) {
        const int *t = ws->Output<CPUBackend>(1)->tensor<int>(i);
        rsz_h = t[0];
        rsz_w = t[1];
      } else {
        if (warp_resize) {
          rsz_w = resize_a;
          rsz_h = resize_b;
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

      cv::Mat input = cv::Mat(H, W, cv_type, const_cast<unsigned char*>(data));

      // perform the resize
      cv::Mat rsz_img;
      cv::resize(input, rsz_img, cv::Size(rsz_w, rsz_h), 0, 0, cv::INTER_LINEAR);

      cv::Mat crop_img;
      cv::Mat const *finalImg = &rsz_img;
      if (resizeOptions & t_cropping) {
        finalImg = &crop_img;

        const int crop_y = (rsz_h - crop_h) / 2;
        const int crop_x = (rsz_w - crop_w) / 2;

        crop_img.create(crop_h, crop_w, cv_type);
        const int crop_offset = (crop_y * rsz_w + crop_x) * c_;
        uint8 *crop_ptr = rsz_img.ptr() + crop_offset;

        CUDA_CALL(cudaMemcpy2D(crop_img.ptr(), crop_w * c_,
                               crop_ptr, rsz_w * c_, crop_w * c_, crop_h,
                               cudaMemcpyHostToHost));
      }

      // Random mirror
      cv::Mat mirror_img;
      if (resizeOptions & t_mirroring) {
        cv::flip(*finalImg, mirror_img, 1);
        finalImg = &mirror_img;
      }

      out[i].Resize({finalImg->rows, finalImg->cols, c_});
      auto *out_data = out[i].mutable_data<unsigned char>();

      std::memcpy(out_data, finalImg->ptr(), finalImg->rows * finalImg->cols * c_);
    }

    vector<TensorList<CPUBackend>*> outputs(1);
    outputs[0] = new TensorList<CPUBackend>();
    outputs[0]->Copy(out, 0);
    return outputs;
  }

 protected:
  virtual uint32_t getResizeOptions() const         { return t_cropping /*+ t_mirroring*/; }
  virtual OpSpec DefaultSchema(const string &pName, const string &pDevice = "gpu") const {
    return OpSpec(pName)
      .AddArg("device", pDevice)
      .AddArg("output_type", this->img_type_)
      .AddInput("input", pDevice)
      .AddOutput("output", pDevice);
  }

  virtual const OpSpec &GetOperationSpec(const OpSpec &op) const { return op; }

  const DALIImageType img_type_ = ImgType::type;
};

}  // namespace dali

#endif  // DALI_TEST_DALI_TEST_RESIZE_H_
