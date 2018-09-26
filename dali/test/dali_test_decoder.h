// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifndef DALI_TEST_DALI_TEST_DECODER_H_
#define DALI_TEST_DALI_TEST_DECODER_H_

#include "dali/test/dali_test_single_op.h"
#include <utility>
#include <vector>
#include <string>

namespace dali {

template <typename ImgType>
class GenericDecoderTest : public DALISingleOpTest<ImgType> {
 public:
  vector<TensorList<CPUBackend>*>
  Reference(const vector<TensorList<CPUBackend>*> &inputs,
            DeviceWorkspace *ws) {
    // single input - encoded images
    // single output - decoded images

    vector<Tensor<CPUBackend>> out(inputs[0]->ntensor());

    const TensorList<CPUBackend>& encoded_data = *inputs[0];

    const int c = this->GetNumColorComp();
    for (int i = 0; i < encoded_data.ntensor(); ++i) {
      auto *data = encoded_data.tensor<unsigned char>(i);
      auto data_size = Product(encoded_data.tensor_shape(i));

      this->DecodeImage(data, data_size, c, this->ImageType(), &out[i]);
    }

    vector<TensorList<CPUBackend>*> outputs(1);
    outputs[0] = new TensorList<CPUBackend>();
    outputs[0]->Copy(out, 0);
    return outputs;
  }

 protected:
  virtual const OpSpec DecodingOp() const   { return OpSpec(); }

  void RunTestDecode(t_imgType imageType, float eps = 5e-2) {
    TensorList<CPUBackend> encoded_data;
    switch (imageType) {
      case t_jpegImgType:
      case t_pngImgType:
        this->EncodedData(imageType, &encoded_data);
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
    const auto c = this->GetNumColorComp();
    const auto nImages = imgs.nImages();
    ImgSetDescr decodedImgsTst, decodedImgsRef;
    this->SetEps(eps);
    for (size_t imgIdx = 0; imgIdx < nImages; ++imgIdx) {
      const auto imgData = imgs.data(imgIdx);
      const auto imgSize = imgs.size(imgIdx);

      // Decoding image in two ways
      Tensor<CPUBackend> t, out;
      const auto imgType =  this->ImageType();
      DALI_CALL(DecodeJPEGHost(imgData, imgSize, imgType, &t));
      this->DecodeImage(imgData, imgSize, c, imgType, &out);

      // Save the results for comparison
      decodedImgsTst.addImage(t.dim(0), t.dim(1), c, t.data<uint8_t>());
      decodedImgsRef.addImage(out.dim(0), out.dim(1), c, out.mutable_data<uint8>());
    }

    // Prepare data for comparison:
    TensorList<CPUBackend> tlTst, tlRef;
    this->MakeEncodedBatch(&tlTst, nImages, decodedImgsTst);
    this->MakeEncodedBatch(&tlRef, nImages, decodedImgsRef);

    // Compare decoding results
    this->CheckTensorLists(&tlTst, &tlRef);
  }

  void VerifyDecode(const uint8 *img, int h, int w, const ImgSetDescr &imgs, int imgIdx) const {
    // Compare w/ opencv result
    const auto imgData = imgs.data(imgIdx);
    const auto imgSize = imgs.size(imgIdx);
    ASSERT_TRUE(CheckIsJPEG(imgData, imgSize));

    Tensor<CPUBackend> out;
    const int c = this->GetNumColorComp();
    this->DecodeImage(imgData, imgSize, c, this->ImageType(), &out);

    const auto imgSizeDecoded = h * w * c;
    ASSERT_TRUE(imgSizeDecoded == out.size());
    CheckBuffers<uint8>(imgSizeDecoded, out.mutable_data<uint8>(),
                        img, t_checkDefault, c, this->GetEps());
  }
};

}  // namespace dali
#endif  // DALI_TEST_DALI_TEST_DECODER_H_
