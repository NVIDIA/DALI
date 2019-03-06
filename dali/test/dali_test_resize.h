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
class GenericResizeTest : public DALISingleOpTest<ImgType> {
 public:
  vector<TensorList<CPUBackend>*>
  Reference(const vector<TensorList<CPUBackend>*> &inputs, DeviceWorkspace *ws) override {
    const int c = this->GetNumColorComp();
    auto cv_type = (c == 3) ? CV_8UC3 : CV_8UC1;

    // single input - encoded images
    // single output - decoded images
    vector<Tensor<CPUBackend>> out(inputs[0]->ntensor());
    const TensorList<CPUBackend>& image_data = *inputs[0];

    const uint resizeOptions = getResizeOptions();

    int resize_a = 0, resize_b = 0;
    bool warp_resize = true, resize_shorter = true;
    const OpSpec &spec = this->GetOperationSpec();
    const bool useExternSizes = (resizeOptions & t_externSizes) &&
                                spec.GetArgument<bool>("save_attrs");
    if (!useExternSizes) {
      if (resizeOptions & t_externSizes)
        assert(false);  // Can't handle these right now

      resize_a = spec.GetArgument<float>("resize_x");
      warp_resize = resize_a != 0;
      if (warp_resize) {
        resize_b = spec.GetArgument<float>("resize_y");
      } else {
        resize_a = spec.GetArgument<float>("resize_shorter");
        if (resize_a == 0) {
          resize_a = spec.GetArgument<float>("resize_longer");
          resize_shorter = false;
        }
      }
    }

    int crop_h = 0, crop_w = 0;
    if (resizeOptions & t_cropping) {
      // Perform a crop
      const vector<float> crop = spec.GetRepeatedArgument<float>("crop");
      crop_h = crop.at(0), crop_w = crop.at(1);
    }

    int rsz_h, rsz_w;
    for (size_t i = 0; i < image_data.ntensor(); ++i) {
      auto *data = image_data.tensor<unsigned char>(i);
      auto shape = image_data.tensor_shape(i);
      const int H = shape[0], W = shape[1];

      // determine resize parameters
      if (useExternSizes) {
        const auto *t = ws->Output<CPUBackend>(1).tensor<int>(i);
        rsz_h = t[0];
        rsz_w = t[1];
      } else {
        if (warp_resize) {
          rsz_w = resize_a;
          rsz_h = resize_b;
        } else {
          if (H >= W) {
            if (resize_shorter) {
              rsz_w = resize_a;
              rsz_h = static_cast<int>(H * static_cast<float>(rsz_w) / W);
            } else {
              rsz_h = resize_a;
              rsz_w = static_cast<int>(W * static_cast<float>(rsz_h) / H);
            }
          } else {  // W > H
            if (resize_shorter) {
              rsz_h = resize_a;
              rsz_w = static_cast<int>(W * static_cast<float>(rsz_h) / H);
            } else {
              rsz_w = resize_a;
              rsz_h = static_cast<int>(H * static_cast<float>(rsz_w) / W);
            }
          }
        }
      }

      cv::Mat input(H, W, cv_type, const_cast<unsigned char*>(data));

      // perform the resize
      cv::Mat rsz_img;

      if (getInterpType() == cv::INTER_NEAREST) {
        // NN resampling with correct pixel center (unlike OpenCV)
        rsz_img.create(rsz_h, rsz_w, input.type());
        size_t elem_sz = rsz_img.elemSize();

        float dy = 1.0f * input.rows / rsz_h;
        float dx = 1.0f * input.cols / rsz_w;

        float sy = 0.5f * dy;
        for (int y = 0; y < rsz_h; y++, sy += dy) {
          int srcy = floor(sy);
          for (int x = 0; x < rsz_w; x++) {
            // This is a bit lame - i.e. this reproduces exactly what we got in CPU ResampleNN.
            // Proper solution would be to test the output and see if the output pixel is coming
            // from a source pixel near the reference location - this would require rewriting the
            // test from scratch for NN case, as it cannot be tested with a straightforward
            // pixelwise comparison.
            float sx = (x + 0.5f) * dx;
            int srcx = floor(sx);
            auto *dst = rsz_img.ptr<uint8_t>(y, x);
            auto *src = input.ptr<uint8_t>(srcy, srcx);
            memcpy(dst, src, elem_sz);
          }
        }
      } else {
        cv::resize(input, rsz_img, cv::Size(rsz_w, rsz_h), 0, 0, getInterpType());
      }

      cv::Mat crop_img;
      cv::Mat const *finalImg = &rsz_img;
      if (resizeOptions & t_cropping) {
        finalImg = &crop_img;

        const int crop_y = (rsz_h - crop_h) / 2;
        const int crop_x = (rsz_w - crop_w) / 2;

        crop_img.create(crop_h, crop_w, cv_type);
        const int crop_offset = (crop_y * rsz_w + crop_x) * c;
        uint8 *crop_ptr = rsz_img.ptr() + crop_offset;

        CUDA_CALL(cudaMemcpy2D(crop_img.ptr(), crop_w * c,
                               crop_ptr, rsz_w * c, crop_w * c, crop_h,
                               cudaMemcpyHostToHost));
      }

      // Random mirror
      cv::Mat mirror_img;
      if (resizeOptions & t_mirroring) {
        cv::flip(*finalImg, mirror_img, 1);
        finalImg = &mirror_img;
      }

      out[i].Resize({finalImg->rows, finalImg->cols, c});
      auto *out_data = out[i].mutable_data<unsigned char>();

      std::memcpy(out_data, finalImg->ptr(), finalImg->rows * finalImg->cols * c);
    }

    vector<TensorList<CPUBackend>*> outputs(1);
    outputs[0] = new TensorList<CPUBackend>();
    outputs[0]->Copy(out, nullptr);
    return outputs;
  }

 protected:
  virtual int getInterpType() const                 { return cv::INTER_LINEAR; }
  virtual uint32_t getResizeOptions() const         { return t_cropping /*+ t_mirroring*/; }
  int CurrentCheckTypeID() const {
    return (this->GetTestCheckType() & t_checkElements) == t_checkElements? 1 : 0;
  }
  virtual double *testEpsValues() const             { return nullptr; }
  virtual double getEps(int testId) const {
    const int numCheckTypes = 2;
    return *(testEpsValues() + testId * numCheckTypes + this->CurrentCheckTypeID());
  }
};

}  // namespace dali

#endif  // DALI_TEST_DALI_TEST_RESIZE_H_
