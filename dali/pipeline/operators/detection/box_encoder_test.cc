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

template <typename ImgType>
class SSDBoxEncoderTest : public GenericBBoxesTest<ImgType> {
 public:
  explicit SSDBoxEncoderTest() {
    CreateCocoObjects();
    CreateCocoAnchors();
  }

 protected:
  TensorList<CPUBackend> boxes_;
  TensorList<CPUBackend> labels_;
  vector<float> anchors_;

  void RunForCoco(vector<float> anchors, float criteria) {
    this->SetBatchSize(1);
    this->SetExternalInputs({{"bboxes", &this->boxes_}, {"labels", &this->labels_}});
    this->AddSingleOp(OpSpec("SSDBoxEncoder")
                          .AddArg("device", "cpu")
                          .AddArg("criteria", criteria)
                          .AddArg("anchors", anchors)
                          .AddInput("bboxes", "cpu")
                          .AddInput("labels", "cpu")
                          .AddOutput("encoded_bboxes", "cpu")
                          .AddOutput("encoded_labels", "cpu"));

    dali::DeviceWorkspace ws;
    this->RunOperator(&ws);
    this->CheckAnswersForCoco(&ws);
  }

  const int coco_object_count_ = 17;
  const vector<float> coco_boxes_ = {
      0.3070048094f, 0.1056995168f, 0.4433026910f, 0.2169388980f, 0.4286985993f, 0.7850852013f,
      0.4515353143f, 0.8401330113f, 0.2127676755f, 0.7940251827f, 0.3299510181f, 0.8681847453f,
      0.3589622080f, 0.7940251827f, 0.4503967762f, 0.8415408134f, 0.4777001441f, 0.7949931026f,
      0.5184909701f, 0.8232209086f, 0.6514389515f, 0.7885521054f, 0.7200586200f, 0.8309113979f,
      0.6264345646f, 0.7906638980f, 0.6599999666f, 0.8155832291f, 0.2899922132f, 0.7900479436f,
      0.3615239561f, 0.8523990512f, 0.4473752379f, 0.7978264093f, 0.4685260355f, 0.8227457404f,
      0.5104554296f, 0.7952923179f, 0.5309274793f, 0.8197364211f, 0.5255850554f, 0.7962073684f,
      0.5550341010f, 0.8153192401f, 0.5353721976f, 0.6945239305f, 0.5604422688f, 0.7127733827f,
      0.8001076579f, 0.7312869430f, 0.8142300248f, 0.7561182380f, 0.3099387884f, 0.7440986037f,
      0.3284183741f, 0.7708833218f, 0.7478436828f, 0.7215374112f, 0.7660385966f, 0.7546399832f,
      0.0784841254f, 0.7881297469f, 0.1185743213f, 0.8644539714f, 0.4970117211f, 0.6394585967f,
      0.5337957740f, 0.6653282046f,
  };
  const vector<int> coco_labels_ = {10, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 10, 10, 10, 10, 59, 10};

  void CreateCocoObjects() {
    boxes_.Resize({{coco_object_count_, 4}});
    auto boxes_data = boxes_.mutable_tensor<float>(0);
    MemCopy(boxes_data, coco_boxes_.data(), coco_object_count_ * 4 * sizeof(float));

    labels_.Resize({{coco_object_count_}});
    auto labels_data = labels_.mutable_tensor<int>(0);
    MemCopy(labels_data, coco_labels_.data(), coco_object_count_ * sizeof(int));
  }

  const vector<std::pair<int, float4>> coco_matches = {
      {1152, {0.3257580996f, 0.8212234974f, 0.0715317428f, 0.0623511076f}},
      {1155, {0.4046794772f, 0.8177829981f, 0.0914345682f, 0.0475156307f}},
      {1165, {0.6857488155f, 0.8097317219f, 0.0686196685f, 0.0423592925f}},
      {1647, {0.3751537502f, 0.1613192111f, 0.1362978816f, 0.1112393811f}},
      {1648, {0.3751537502f, 0.1613192111f, 0.1362978816f, 0.1112393811f}},
      {1685, {0.3751537502f, 0.1613192111f, 0.1362978816f, 0.1112393811f}},
      {1686, {0.3751537502f, 0.1613192111f, 0.1362978816f, 0.1112393811f}},
      {2593, {0.2713593543f, 0.8311049938f, 0.1171833426f, 0.0741595626f}},
      {2594, {0.2713593543f, 0.8311049938f, 0.1171833426f, 0.0741595626f}},
      {2631, {0.2713593543f, 0.8311049938f, 0.1171833426f, 0.0741595626f}},
      {2632, {0.2713593543f, 0.8311049938f, 0.1171833426f, 0.0741595626f}},
      {3818, {0.5154037476f, 0.6523934007f, 0.0367840528f, 0.0258696079f}},
      {3895, {0.5479072332f, 0.7036486864f, 0.0250700712f, 0.0182494521f}},
      {3941, {0.7569411397f, 0.738088727f, 0.0181949139f, 0.033102572f}},
      {3943, {0.8071688414f, 0.7437025905f, 0.0141223669f, 0.024831295f}},
      {3962, {0.3191785812f, 0.7574909925f, 0.0184795856f, 0.026784718f}},
      {4039, {0.3257580996f, 0.8212234974f, 0.0715317428f, 0.0623511076f}},
      {4040, {0.3257580996f, 0.8212234974f, 0.0715317428f, 0.0623511076f}},
      {4042, {0.4046794772f, 0.8177829981f, 0.0914345682f, 0.0475156307f}},
      {4043, {0.4046794772f, 0.8177829981f, 0.0914345682f, 0.0475156307f}},
      {4045, {0.4579506516f, 0.8102860451f, 0.0211507976f, 0.0249193311f}},
      {4046, {0.5206914544f, 0.8075143695f, 0.0204720497f, 0.0244441032f}},
      {4047, {0.5403095484f, 0.8057633042f, 0.0294490457f, 0.0191118717f}},
      {4051, {0.6432172656f, 0.8031235933f, 0.033565402f, 0.0249193311f}},
      {4053, {0.6857488155f, 0.8097317219f, 0.0686196685f, 0.0423592925f}},
      {4054, {0.6857488155f, 0.8097317219f, 0.0686196685f, 0.0423592925f}},
      {4076, {0.2713593543f, 0.8311049938f, 0.1171833426f, 0.0741595626f}},
      {5475, {0.0985292196f, 0.8262918591f, 0.0400901958f, 0.0763242245f}},
      {5488, {0.4401169419f, 0.8126090765f, 0.022836715f, 0.0550478101f}},
      {5513, {0.0985292196f, 0.8262918591f, 0.0400901958f, 0.0763242245f}}};

  const int coco_anchors_count_ = 8732;

  inline float Clamp(float val) { return val > 1.f ? 1.f : (val < 0.f ? 0.f : val); }

  void CreateCocoAnchors() {
    anchors_ = vector<float>(coco_anchors_count_ * 4);

    int fig_size = 300;
    vector<int> feat_sizes{38, 19, 10, 5, 3, 1};
    int feat_count = feat_sizes.size();
    vector<float> steps{8.f, 16.f, 32.f, 64.f, 100.f, 300.f};
    vector<float> scales = {21.f, 45.f, 99.f, 153.f, 207.f, 261.f, 315.f};
    vector<vector<int>> aspect_ratios = {{2}, {2, 3}, {2, 3}, {2, 3}, {2}, {2}};

    vector<float> fks;
    for (auto &step : steps) fks.push_back(fig_size / step);

    int anchor_idx = 0;
    for (int idx = 0; idx < feat_count; ++idx) {
      auto sk1 = scales[idx] / fig_size;
      auto sk2 = scales[idx + 1] / fig_size;
      auto sk3 = sqrt(sk1 * sk2);
      vector<std::pair<float, float>> all_sizes{{sk1, sk1}, {sk3, sk3}};

      for (auto &alpha : aspect_ratios[idx]) {
        auto w = sk1 * sqrt(alpha);
        auto h = sk1 / sqrt(alpha);
        all_sizes.push_back({w, h});
        all_sizes.push_back({h, w});
      }

      for (auto &sizes : all_sizes) {
        auto w = sizes.first;
        auto h = sizes.second;

        for (int i = 0; i < feat_sizes[idx]; ++i)
          for (int j = 0; j < feat_sizes[idx]; ++j) {
            auto cx = (j + 0.5f) / fks[idx];
            auto cy = (i + 0.5f) / fks[idx];

            cx = this->Clamp(cx);
            cy = this->Clamp(cy);
            w = this->Clamp(w);
            h = this->Clamp(h);

            anchors_[anchor_idx * 4] = cx - 0.5f * w;
            anchors_[anchor_idx * 4 + 1] = cy - 0.5f * h;
            anchors_[anchor_idx * 4 + 2] = cx + 0.5f * w;
            anchors_[anchor_idx * 4 + 3] = cy + 0.5f * h;

            ++anchor_idx;
          }
      }
    }
  }

  void CheckAnswersForCoco(DeviceWorkspace *ws) {
    TensorList<CPUBackend> *boxes = ws->Output<dali::CPUBackend>(0);
    TensorList<CPUBackend> *labels = ws->Output<dali::CPUBackend>(1);

    auto boxes_shape = boxes->shape();
    ASSERT_EQ(boxes_shape.size(), 1);
    ASSERT_EQ(boxes_shape[0].size(), 2);
    ASSERT_EQ(boxes_shape[0][0], coco_anchors_count_);
    ASSERT_EQ(boxes_shape[0][1], 4);

    auto labels_shape = labels->shape();
    ASSERT_EQ(labels_shape.size(), 1);
    ASSERT_EQ(labels_shape[0].size(), 1);
    ASSERT_EQ(labels_shape[0][0], coco_anchors_count_);

    vector<float4> boxes_data(coco_anchors_count_);
    MemCopy(boxes_data.data(), boxes->tensor<float>(0), coco_anchors_count_ * 4 * sizeof(float));

    vector<int> labels_data(coco_anchors_count_);
    MemCopy(labels_data.data(), labels->tensor<int>(0), coco_anchors_count_ * sizeof(int));

    int idx = 0;
    for (auto match : coco_matches) {
      while (idx < match.first) {
        ASSERT_EQ(labels_data[idx], 0);
        ++idx;
      }

      auto actual = boxes_data[match.first];
      auto expected = match.second;

      ASSERT_NE(labels_data[match.first], 0);
      ASSERT_FLOAT_EQ(actual.x, expected.x);
      ASSERT_FLOAT_EQ(actual.y, expected.y);
      ASSERT_FLOAT_EQ(actual.z, expected.z);
      ASSERT_FLOAT_EQ(actual.w, expected.w);

      ++idx;
    }

    while (idx < coco_anchors_count_) {
      ASSERT_EQ(labels_data[idx], 0);
      ++idx;
    }
  }
};

typedef ::testing::Types<Gray> Types;
TYPED_TEST_CASE(SSDBoxEncoderTest, Types);

TYPED_TEST(SSDBoxEncoderTest, TestOnCocoObjects) {
  this->RunForCoco(this->anchors_, 0.5f);
}

TYPED_TEST(SSDBoxEncoderTest, TestNegativeCriteria) {
  EXPECT_THROW(
    this->RunForCoco(this->anchors_, -0.5f),
    std::runtime_error);
}

TYPED_TEST(SSDBoxEncoderTest, TestCriteriaOverOne) {
  EXPECT_THROW(
    this->RunForCoco(this->anchors_, 1.5f),
    std::runtime_error);
}

TYPED_TEST(SSDBoxEncoderTest, TestInvalidAnchors) {
  vector<float> invalid_anchors(this->anchors_);
  invalid_anchors.pop_back();

  EXPECT_THROW(
    this->RunForCoco(invalid_anchors, 0.5f),
    std::runtime_error);
}

}  // namespace dali
