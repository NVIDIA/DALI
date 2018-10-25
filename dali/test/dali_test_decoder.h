// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifndef DALI_TEST_DALI_TEST_DECODER_H_
#define DALI_TEST_DALI_TEST_DECODER_H_

#include <string>
#include <utility>
#include <vector>
#include "dali/test/dali_test_single_op.h"

namespace dali {

template <typename ImgType>
class GenericDecoderTest : public DALISingleOpTest<ImgType> {
 public:
  vector<TensorList<CPUBackend> *> Reference(const vector<TensorList<CPUBackend> *> &inputs,
                                             DeviceWorkspace *ws) {
    // single input - encoded images
    // single output - decoded images

    vector<Tensor<CPUBackend>> out(inputs[0]->ntensor());

    const TensorList<CPUBackend> &encoded_data = *inputs[0];

    const int c = this->GetNumColorComp();
    for (int i = 0; i < encoded_data.ntensor(); ++i) {
      auto *data = encoded_data.tensor<unsigned char>(i);
      auto data_size = Product(encoded_data.tensor_shape(i));

      this->DecodeImage(data, data_size, c, this->ImageType(), &out[i]);
    }

    vector<TensorList<CPUBackend> *> outputs(1);
    outputs[0] = new TensorList<CPUBackend>();
    outputs[0]->Copy(out, 0);
    return outputs;
  }

 protected:
  virtual const OpSpec DecodingOp() const { return OpSpec(); }

  void RunTestDecode(t_imgType imageType, float eps = 5e-2) {
    TensorList<CPUBackend> encoded_data;
    switch (imageType) {
      case t_jpegImgType:
        this->EncodedJPEGData(&encoded_data);
        break;
      case t_pngImgType:
        this->EncodedPNGData(&encoded_data);
        break;
      default: {
        char buff[32];
        snprintf(buff, sizeof(buff), "%d", imageType);
        DALI_FAIL("Image of type `" + string(buff) + "` cannot be decoded");
      }
    }

    this->SetExternalInputs({std::make_pair("encoded", &encoded_data)});
    this->RunOperator(DecodingOp(), eps);
  }

  void RunTestDecode(const ImgSetDescr &imgs, float eps = 5e-2) {
    this->SetEps(eps);
    for (size_t imgIdx = 0; imgIdx < imgs.nImages(); ++imgIdx) {
      Tensor<CPUBackend> t;
      DALI_CALL(DecodeJPEGHost(imgs.data_[imgIdx], imgs.sizes_[imgIdx], this->img_type_, &t));

#if DALI_DEBUG
      WriteHWCImage(t.data<uint8_t>(), t.dim(0), t.dim(1), t.dim(2),
                    std::to_string(imgIdx) + "-img");
#ifndef NDEBUG
      cout << imgIdx << ": " << imgs.sizes_[imgIdx] << "  dims: " << t.dim(1) << "x" << t.dim(0)
           << endl;
#endif
#endif
      this->VerifyDecode(t.data<uint8_t>(), t.dim(0), t.dim(1), imgs, imgIdx);
    }
  }

  void VerifyDecode(const uint8 *img, int h, int w, const ImgSetDescr &imgs, int img_id) const {
    // Compare w/ opencv result
    const auto imgData = imgs.data_[img_id];
    const auto imgSize = imgs.sizes_[img_id];
    ASSERT_TRUE(CheckIsJPEG(imgData, imgSize));

    Tensor<CPUBackend> out;
    const int c = this->GetNumColorComp();
    this->DecodeImage(imgData, imgSize, c, this->ImageType(), &out);
    this->CheckBuffers(h * w * c, out.mutable_data<uint8>(), img, false);
  }
};

}  // namespace dali
#endif  // DALI_TEST_DALI_TEST_DECODER_H_
