// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/test/ndll_test_single_op.h"

namespace ndll {

template <typename ImgType>
class nvjpegDecodeTest : public NDLLSingleOpTest {
 public:
  USING_NDLL_SINGLE_OP_TEST();

  vector<TensorList<CPUBackend>*>
  Reference(const vector<TensorList<CPUBackend>*> &inputs,
            DeviceWorkspace *ws) {
    // single input - encoded images
    // single output - decoded images
    vector<TensorList<CPUBackend>*> outputs(1);

    vector<Tensor<CPUBackend>> out(inputs[0]->ntensor());

    const TensorList<CPUBackend>& encoded_data = *inputs[0];

    c_ = (IsColor(img_type_) ? 3 : 1);
    for (int i = 0; i < encoded_data.ntensor(); ++i) {
      auto *data = encoded_data.tensor<unsigned char>(i);
      auto data_size = Product(encoded_data.tensor_shape(i));

      cv::Mat tmp = cv::imdecode(cv::Mat(1, data_size,
                                         CV_8UC1,
                                         const_cast<unsigned char*>(data)),
                                 (c_ == 1) ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR);

      if (img_type_ == NDLL_RGB) {
        cv::cvtColor(tmp, tmp, cv::COLOR_BGR2RGB);
      }
      out[i].Resize({tmp.rows, tmp.cols, c_});
      auto *out_data = out[i].mutable_data<unsigned char>();

      std::memcpy(out_data, tmp.ptr(), tmp.rows * tmp.cols * c_);
    }
    outputs[0] = new TensorList<CPUBackend>();
    outputs[0]->Copy(out, 0);
    return outputs;
  }

 protected:
  void TestDecode(bool batched, int num_threads) {
    const int batch_size = 32;
    this->SetBatchSize(batch_size);
    this->SetNumThreads(num_threads);
    TensorList<CPUBackend> encoded_data;
    this->EncodedJPEGData(&encoded_data, batch_size);
    this->SetExternalInputs({std::make_pair("encoded", &encoded_data)});

    this->AddSingleOp(OpSpec("nvJPEGDecoder")
                .AddArg("device", "mixed")
                .AddArg("output_type", this->img_type_)
                .AddArg("use_batched_decode", batched)
                .AddInput("encoded", "cpu")
                .AddOutput("decoded", "gpu"));

    DeviceWorkspace ws;
    this->RunOperator(&ws);

    // Note: lower accuracy due to TJPG and OCV implementations for BGR/RGB.
    // Difference is consistent, deterministic and goes away if I force OCV
    // instead of TJPG decoding.
    this->SetEps(2.0);

    this->CheckAnswers(&ws, {0});
  }

  const NDLLImageType img_type_ = ImgType::type;
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(nvjpegDecodeTest, Types);

TYPED_TEST(nvjpegDecodeTest, TestSingleJPEGDecode) {
  this->TestDecode(false, 1);
}

TYPED_TEST(nvjpegDecodeTest, TestSingleJPEGDecode2T) {
  this->TestDecode(false, 2);
}

TYPED_TEST(nvjpegDecodeTest, TestSingleJPEGDecode3T) {
  this->TestDecode(false, 3);
}

TYPED_TEST(nvjpegDecodeTest, TestSingleJPEGDecode4T) {
  this->TestDecode(false, 4);
}

TYPED_TEST(nvjpegDecodeTest, TestBatchedJPEGDecode) {
  this->TestDecode(true, 1);
}

TYPED_TEST(nvjpegDecodeTest, TestBatchedJPEGDecode2T) {
  this->TestDecode(true, 2);
}

TYPED_TEST(nvjpegDecodeTest, TestBatchedJPEGDecode3T) {
  this->TestDecode(true, 3);
}

TYPED_TEST(nvjpegDecodeTest, TestBatchedJPEGDecode4T) {
  this->TestDecode(true, 4);
}

}  // namespace ndll

