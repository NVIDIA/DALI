// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifndef DALI_TEST_DALI_TEST_DECODER_H_
#define DALI_TEST_DALI_TEST_DECODER_H_

#include "dali/test/dali_test_single_op.h"
#include <utility>
#include <vector>
#include <string>

namespace dali {

template <typename ImgType>
class GenericDecoderTest : public DALISingleOpTest {
 public:
  USING_DALI_SINGLE_OP_TEST();

  vector<TensorList<CPUBackend>*>
  Reference(const vector<TensorList<CPUBackend>*> &inputs,
            DeviceWorkspace *ws) {
    // single input - encoded images
    // single output - decoded images

    vector<Tensor<CPUBackend>> out(inputs[0]->ntensor());

    const TensorList<CPUBackend>& encoded_data = *inputs[0];

    c_ = (IsColor(img_type_) ? 3 : 1);
    for (int i = 0; i < encoded_data.ntensor(); ++i) {
      auto *data = encoded_data.tensor<unsigned char>(i);
      auto data_size = Product(encoded_data.tensor_shape(i));

      DecodeImage(data, data_size, c_, img_type_, &out[i]);
    }

    vector<TensorList<CPUBackend>*> outputs(1);
    outputs[0] = new TensorList<CPUBackend>();
    outputs[0]->Copy(out, 0);
    return outputs;
  }

 protected:
  virtual const OpSpec DecodingOp() const = 0;
  virtual uint8 TestCheckType() const       { return t_checkDefault; }

  void RunTestDecode(t_imgType imageType, float eps = 5e-2) {
#ifdef PIXEL_STAT_FILE
    FILE *file = fopen(PIXEL_STAT_FILE".txt", "a");
    fprintf(file, "Type of the files: %s   eps = %6.4f\n", jpegData? "JPEG" : "PNG", eps);
    fprintf(file, " Color#:       mean:        std:          eq.         pos.         neg.\n");
    fclose(file);
#endif
    TensorList<CPUBackend> encoded_data;
    switch (imageType) {
      case t_jpegImgType:
        EncodedJPEGData(&encoded_data, batch_size_);
        break;
      case t_pngImgType:
        EncodedPNGData(&encoded_data, batch_size_);
        break;
      default: {
        char buff[32];
        snprintf(buff, sizeof(buff), "%d", imageType);
        DALI_FAIL("Image of type `" + string(buff) + "` cannot be decoded");
      }
    }

    SetExternalInputs({std::make_pair("encoded", &encoded_data)});

    AddSingleOp(DecodingOp());

    DeviceWorkspace ws;
    RunOperator(&ws);

    SetEps(eps);
    SetTestCheckType(TestCheckType());
    CheckAnswers(&ws, {0});
  }

  const DALIImageType img_type_ = ImgType::type;
};

}  // namespace dali
#endif  // DALI_TEST_DALI_TEST_DECODER_H_
