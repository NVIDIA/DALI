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

#include "dali/test/dali_test_bboxes.h"

namespace dali {

using BBox = std::array<float, 4>;

template <typename ltrb_t>
class BBoxPasteTest;

template <bool ltrb>
class BBoxPasteTest<std::integral_constant<bool, ltrb>> : public DALISingleOpTest<RGB> {
 public:
  static const bool ltrb_ = ltrb;

  void SetUp() override {
  }

  static void ToTensorList(TensorList<CPUBackend> *out,
                           const std::vector<std::vector<BBox>> &boxes) {
    vector<Dims> dims(boxes.size());
    for (size_t i = 0; i < dims.size(); i++)
      dims[i] = { (Index)boxes[i].size(), 4 };

    out->Resize(dims);

    for (size_t i = 0; i < boxes.size(); i++) {
      auto *data = out->mutable_tensor<float>(i);
      for (size_t j = 0; j < boxes[i].size(); j++) {
        for (int k = 0; k < 4; k++)
          data[4*j+k] = boxes[i][j][k];
      }
    }
  }

  static std::unique_ptr<TensorList<CPUBackend>>
  ToTensorList(const std::vector<std::vector<BBox>> &boxes) {
    std::unique_ptr<TensorList<CPUBackend>> out(new TensorList<CPUBackend>());
    ToTensorList(out.get(), boxes);
    return out;
  }

  vector<TensorList<CPUBackend>*>
  Reference(const vector<TensorList<CPUBackend>*> &inputs,
            DeviceWorkspace *ws) override {
    (void)inputs;
    (void)ws;
    auto ref = ToTensorList(output_);
    return { ref.release() };
  }

  std::vector<std::vector<BBox>> input_, output_;

  void Run(float ratio, float paste_x, float paste_y) {
    OpSpec spec("BBoxPaste");
    input_tl_ = ToTensorList(input_);
    SetBatchSize(input_tl_->ntensor());

    spec.AddInput("bb_input", "cpu").AddOutput("bb_output", "cpu");

    spec.AddArg("ltrb", ltrb);
    spec.AddArg("ratio", ratio);
    spec.AddArg("paste_x", paste_x);
    spec.AddArg("paste_y", paste_y);
    SetExternalInputs({ { "bb_input", input_tl_.get() } });
    RunOperator(spec, 1e-7f);
  }

  std::vector<std::vector<BBox>> RandomBoxes(int num_sets, int max_per_set) {
    std::vector<std::vector<BBox>> out(num_sets);
    std::uniform_int_distribution<> num_box_dist(1, max_per_set);
    for (auto &boxes : out) {
      boxes.resize(num_box_dist(rand_gen_));
      MakeRandomBoxes(&boxes[0][0], boxes.size());
      if (!ltrb) {
        // convert to XYWH
        for (auto &box : boxes) {
          box[2] -= box[0];
          box[3] -= box[1];
        }
      }
    }
    return out;
  }

  static std::vector<std::vector<BBox>> CalculateReferenceOutput(
    const std::vector<std::vector<BBox>> &input,
    float ratio, float paste_x, float paste_y) {

    auto out = input;
    float scale = 1/ratio;     // scale factor - after pasting onto a <ratio> times larger
                               // canvas, the boxes become <ratio> times smaller
    float margin = ratio - 1;  // amount of free space after pasting onto a bigger canvas
    float ofs_x = paste_x * margin * scale;
    float ofs_y = paste_y * margin * scale;
    for (size_t i = 0; i < input.size(); i++) {
      for (size_t j = 0; j < input[i].size();  j++) {
        if (ltrb) {
          float l = input[i][j][0];
          float t = input[i][j][1];
          float r = input[i][j][2];
          float b = input[i][j][3];
          out[i][j][0] = l * scale + ofs_x;
          out[i][j][1] = t * scale + ofs_y;
          out[i][j][2] = r * scale + ofs_x;
          out[i][j][3] = b * scale + ofs_y;
        } else {
          float x = input[i][j][0];
          float y = input[i][j][1];
          float w = input[i][j][2];
          float h = input[i][j][3];
          out[i][j][0] = x * scale + ofs_x;
          out[i][j][1] = y * scale + ofs_y;
          out[i][j][2] = w * scale;
          out[i][j][3] = h * scale;
        }
      }
    }
    return out;
  }

 protected:
  unique_ptr<TensorList<CPUBackend>> input_tl_;
};

typedef ::testing::Types<std::true_type, std::false_type> Types;
TYPED_TEST_SUITE(BBoxPasteTest, Types);

TYPED_TEST(BBoxPasteTest, Identity) {
  this->input_ = this->RandomBoxes(10, 100);
  this->output_ = this->input_;
  this->Run(1, 0, 0);
}

TYPED_TEST(BBoxPasteTest, ZeroScaleOffsetOnly) {
  this->input_ = this->RandomBoxes(10, 100);
  this->output_ = this->input_;
  const float x = 0.123f;
  const float y = 0.567f;

  for (auto &boxes : this->output_)
    for (auto &box : boxes) {
      if (this->ltrb_) {
        box = { x, y, x, y };
      } else {
        box = { x, y, 0, 0 };
      }
    }
  this->Run(1e+38f, x, y);
}

TYPED_TEST(BBoxPasteTest, FullBoxPaste) {
  const float x = 0.876f;
  const float y = 0.321f;

  const float ratio = 1.789f;
  const float margin = (ratio-1)/ratio;

  this->input_ = { { { 0, 0, 1, 1 } } };

  if (this->ltrb_) {
    this->output_ = { { { x*margin, y*margin, x*margin+1/ratio, y*margin+1/ratio } } };
  } else {
    this->output_ = { { { x*margin, y*margin, 1/ratio, 1/ratio } } };
  }

  this->Run(ratio, x, y);
}


TYPED_TEST(BBoxPasteTest, Random) {
  const float x = 0.876f;
  const float y = 0.321f;

  const float ratio = 1.789f;
  const float margin = (ratio-1)/ratio;

  this->input_ = this->RandomBoxes(10, 100);
  this->output_ = this->CalculateReferenceOutput(this->input_, ratio, x, y);
  this->Run(ratio, x, y);
}

}  // namespace dali
