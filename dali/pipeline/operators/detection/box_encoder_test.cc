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
class BoxEncoderTest : public GenericBBoxesTest<ImgType> {
 public:
  BoxEncoderTest() {
    CreateCocoObjects();
    CreateCocoAnchors();
  }

 protected:
  TensorList<CPUBackend> boxes_;
  TensorList<CPUBackend> labels_;
  vector<float> anchors_;

  void RunForCoco(vector<float> anchors, float criteria) {
    this->SetBatchSize(coco_batch_size);
    this->SetExternalInputs({{"bboxes", &this->boxes_}, {"labels", &this->labels_}});
    this->AddSingleOp(OpSpec("BoxEncoder")
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

  const vector<int> coco_object_count = {9, 7, 3, 6};

  const int coco_batch_size = 4;

  const vector<vector<float>> coco_boxes = {
    {
      0.00800000f, 0.01348684f, 0.46112499f, 1.00000000f,
      0.52606249f, 0.59971493f, 0.62974995f, 0.64885968f,
      0.27350000f, 0.26585528f, 0.30095309f, 0.43789473f,
      0.27260938f, 0.08747807f, 0.62365627f, 1.00000000f,
      0.61110938f, 0.58271933f, 0.81009370f, 0.78265357f,
      0.82784379f, 0.88201755f, 0.90031254f, 1.00000000f,
      0.53293747f, 0.89311403f, 0.60610932f, 1.00000000f,
      0.63068753f, 0.82796049f, 0.69424999f, 0.91853064f,
      0.70648438f, 0.93995613f, 0.78620309f, 1.00000000f,
    },
    {
      0.10112500f, 0.21797222f, 0.88735944f, 0.77077770f,
      0.98060942f, 0.40358332f, 0.99967194f, 0.44494441f,
      0.46729690f, 0.43291667f, 0.49932814f, 0.51297224f,
      0.90595311f, 0.44258335f, 0.91853124f, 0.47866669f,
      0.77206248f, 0.00000000f, 0.81173438f, 0.08411111f,
      0.96995318f, 0.42811111f, 1.00000000f, 0.56316662f,
      0.88803130f, 0.46908331f, 0.98442191f, 0.58394444f,
    },
    {
      0.49365625f, 0.16355971f, 0.72104686f, 0.82276350f,
      0.29301563f, 0.13063231f, 0.56199998f, 0.87838411f,
      0.01050000f, 0.51911008f, 0.71517187f, 0.85843086f,
    },
    {
      0.62010938f, 0.70079625f, 0.71157813f, 0.94072598f,
      0.37431249f, 0.22718970f, 0.56432807f, 0.71327871f,
      0.51875001f, 0.20224825f, 0.67918748f, 0.69213110f,
      0.26292187f, 0.22128806f, 0.37992185f, 0.71955502f,
      0.42656249f, 0.42451993f, 0.68387496f, 0.67292738f,
      0.28239062f, 0.43517566f, 0.38075000f, 0.71091336f,
    },
  };

  const vector<vector<int>> coco_labels = {
    { 1, 44, 28, 1, 56, 56, 56, 56, 56, },
    { 6, 1, 12, 1, 1, 1, 59, },
    { 1, 1, 38, },
    { 17, 1, 1, 1, 14, 14, },
  };

  void CreateCocoObjects() {
    vector<Dims> boxes_shape;
    vector<Dims> labels_shape;
    for (auto &object_count : coco_object_count) {
      boxes_shape.push_back({object_count, 4});
      labels_shape.push_back({object_count});
    }

    boxes_.Resize(boxes_shape);

    for (int sample = 0; sample < coco_batch_size; ++sample) {
      auto boxes_data = boxes_.mutable_tensor<float>(sample);
      MemCopy(
        boxes_data, 
        coco_boxes[sample].data(), 
        coco_object_count[sample] * 4 * sizeof(float));

      labels_.Resize(labels_shape);
      auto labels_data = labels_.mutable_tensor<int>(sample);
      MemCopy(labels_data, coco_labels[sample].data(), coco_object_count[sample] * sizeof(int));
    }
  }

  // Expected matches : <idx of matched anchor, <matched object, matched object class>>
  const vector<vector<std::pair<int, std::pair<float4, int>>>> coco_matches = {
    {
      {895, {{0.57790625f, 0.62428731f, 0.10368747f, 0.04914474f}, 44}},
      {1240, {{0.66246879f, 0.87324560f, 0.06356245f, 0.09057015f}, 56}},
      {1324, {{0.86407816f, 0.94100881f, 0.07246876f, 0.11798245f}, 56}},
      {1351, {{0.56952339f, 0.94655704f, 0.07317185f, 0.10688597f}, 56}},
      {1362, {{0.86407816f, 0.94100881f, 0.07246876f, 0.11798245f}, 56}},
      {1395, {{0.74634373f, 0.96997809f, 0.07971871f, 0.06004387f}, 56}},
      {1396, {{0.74634373f, 0.96997809f, 0.07971871f, 0.06004387f}, 56}},
      {2684, {{0.66246879f, 0.87324560f, 0.06356245f, 0.09057015f}, 56}},
      {2685, {{0.66246879f, 0.87324560f, 0.06356245f, 0.09057015f}, 56}},
      {2768, {{0.86407816f, 0.94100881f, 0.07246876f, 0.11798245f}, 56}},
      {2794, {{0.56952339f, 0.94655704f, 0.07317185f, 0.10688597f}, 56}},
      {2795, {{0.56952339f, 0.94655704f, 0.07317185f, 0.10688597f}, 56}},
      {2805, {{0.86407816f, 0.94100881f, 0.07246876f, 0.11798245f}, 56}},
      {2806, {{0.86407816f, 0.94100881f, 0.07246876f, 0.11798245f}, 56}},
      {3783, {{0.57790625f, 0.62428731f, 0.10368747f, 0.04914474f}, 44}},
      {3784, {{0.57790625f, 0.62428731f, 0.10368747f, 0.04914474f}, 44}},
      {4283, {{0.74634373f, 0.96997809f, 0.07971871f, 0.06004387f}, 56}},
      {4284, {{0.74634373f, 0.96997809f, 0.07971871f, 0.06004387f}, 56}},
      {4874, {{0.28722656f, 0.35187501f, 0.02745309f, 0.17203945f}, 28}},
      {5572, {{0.66246879f, 0.87324560f, 0.06356245f, 0.09057015f}, 56}},
      {5683, {{0.56952339f, 0.94655704f, 0.07317185f, 0.10688597f}, 56}},
      {5694, {{0.86407816f, 0.94100881f, 0.07246876f, 0.11798245f}, 56}},
      {6017, {{0.71060157f, 0.68268645f, 0.19898432f, 0.19993424f}, 56}},
      {6377, {{0.71060157f, 0.68268645f, 0.19898432f, 0.19993424f}, 56}},
      {6378, {{0.71060157f, 0.68268645f, 0.19898432f, 0.19993424f}, 56}},
      {6397, {{0.71060157f, 0.68268645f, 0.19898432f, 0.19993424f}, 56}},
      {6739, {{0.71060157f, 0.68268645f, 0.19898432f, 0.19993424f}, 56}},
      {6758, {{0.71060157f, 0.68268645f, 0.19898432f, 0.19993424f}, 56}},
      {8629, {{0.44813281f, 0.54373902f, 0.35104689f, 0.91252196f}, 1}},
      {8678, {{0.23456249f, 0.50674343f, 0.45312500f, 0.98651314f}, 1}},
      {8679, {{0.44813281f, 0.54373902f, 0.35104689f, 0.91252196f}, 1}},
      {8695, {{0.23456249f, 0.50674343f, 0.45312500f, 0.98651314f}, 1}},
      {8704, {{0.23456249f, 0.50674343f, 0.45312500f, 0.98651314f}, 1}},
      {8722, {{0.23456249f, 0.50674343f, 0.45312500f, 0.98651314f}, 1}},
      {8723, {{0.44813281f, 0.54373902f, 0.35104689f, 0.91252196f}, 1}},
      {8731, {{0.44813281f, 0.54373902f, 0.35104689f, 0.91252196f}, 1}},
    },
    {
      {67, {{0.79189843f, 0.04205555f, 0.03967190f, 0.08411111f}, 1}},
      {2200, {{0.93622661f, 0.52651387f, 0.09639060f, 0.11486113f}, 59}},
      {2201, {{0.93622661f, 0.52651387f, 0.09639060f, 0.11486113f}, 59}},
      {2238, {{0.93622661f, 0.52651387f, 0.09639060f, 0.11486113f}, 59}},
      {2239, {{0.93622661f, 0.52651387f, 0.09639060f, 0.11486113f}, 59}},
      {4399, {{0.79189843f, 0.04205555f, 0.03967190f, 0.08411111f}, 1}},
      {4977, {{0.99014068f, 0.42426386f, 0.01906252f, 0.04136109f}, 1}},
      {4996, {{0.48331252f, 0.47294444f, 0.03203124f, 0.08005556f}, 12}},
      {5012, {{0.91224217f, 0.46062502f, 0.01257813f, 0.03608334f}, 1}},
      {5052, {{0.98497659f, 0.49563885f, 0.03004682f, 0.13505551f}, 1}},
      {8553, {{0.49424222f, 0.49437496f, 0.78623444f, 0.55280548f}, 6}},
      {8554, {{0.49424222f, 0.49437496f, 0.78623444f, 0.55280548f}, 6}},
      {8578, {{0.49424222f, 0.49437496f, 0.78623444f, 0.55280548f}, 6}},
      {8579, {{0.49424222f, 0.49437496f, 0.78623444f, 0.55280548f}, 6}},
      {8604, {{0.49424222f, 0.49437496f, 0.78623444f, 0.55280548f}, 6}},
      {8696, {{0.49424222f, 0.49437496f, 0.78623444f, 0.55280548f}, 6}},
      {8705, {{0.49424222f, 0.49437496f, 0.78623444f, 0.55280548f}, 6}},
      {8714, {{0.49424222f, 0.49437496f, 0.78623444f, 0.55280548f}, 6}},
      {8728, {{0.49424222f, 0.49437496f, 0.78623444f, 0.55280548f}, 6}},
      {8730, {{0.49424222f, 0.49437496f, 0.78623444f, 0.55280548f}, 6}},
    },
    {
      {8104, {{0.36283594f, 0.68877047f, 0.70467186f, 0.33932078f}, 38}},
      {8105, {{0.36283594f, 0.68877047f, 0.70467186f, 0.33932078f}, 38}},
      {8106, {{0.36283594f, 0.68877047f, 0.70467186f, 0.33932078f}, 38}},
      {8277, {{0.60735154f, 0.49316162f, 0.22739062f, 0.65920377f}, 1}},
      {8287, {{0.60735154f, 0.49316162f, 0.22739062f, 0.65920377f}, 1}},
      {8297, {{0.60735154f, 0.49316162f, 0.22739062f, 0.65920377f}, 1}},
      {8477, {{0.60735154f, 0.49316162f, 0.22739062f, 0.65920377f}, 1}},
      {8486, {{0.42750782f, 0.50450820f, 0.26898435f, 0.74775183f}, 1}},
      {8487, {{0.60735154f, 0.49316162f, 0.22739062f, 0.65920377f}, 1}},
      {8497, {{0.60735154f, 0.49316162f, 0.22739062f, 0.65920377f}, 1}},
      {8558, {{0.36283594f, 0.68877047f, 0.70467186f, 0.33932078f}, 38}},
      {8583, {{0.36283594f, 0.68877047f, 0.70467186f, 0.33932078f}, 38}},
      {8608, {{0.36283594f, 0.68877047f, 0.70467186f, 0.33932078f}, 38}},
      {8629, {{0.60735154f, 0.49316162f, 0.22739062f, 0.65920377f}, 1}},
      {8658, {{0.36283594f, 0.68877047f, 0.70467186f, 0.33932078f}, 38}},
    },
    {
      {6318, {{0.55521870f, 0.54872364f, 0.25731248f, 0.24840745f}, 14}},
      {6336, {{0.55521870f, 0.54872364f, 0.25731248f, 0.24840745f}, 14}},
      {6337, {{0.55521870f, 0.54872364f, 0.25731248f, 0.24840745f}, 14}},
      {6338, {{0.55521870f, 0.54872364f, 0.25731248f, 0.24840745f}, 14}},
      {6356, {{0.55521870f, 0.54872364f, 0.25731248f, 0.24840745f}, 14}},
      {7055, {{0.33157033f, 0.57304454f, 0.09835938f, 0.27573770f}, 14}},
      {7074, {{0.33157033f, 0.57304454f, 0.09835938f, 0.27573770f}, 14}},
      {7137, {{0.66584373f, 0.82076108f, 0.09146875f, 0.23992974f}, 17}},
      {7156, {{0.66584373f, 0.82076108f, 0.09146875f, 0.23992974f}, 17}},
      {7175, {{0.66584373f, 0.82076108f, 0.09146875f, 0.23992974f}, 17}},
      {7777, {{0.33157033f, 0.57304454f, 0.09835938f, 0.27573770f}, 14}},
      {7796, {{0.33157033f, 0.57304454f, 0.09835938f, 0.27573770f}, 14}},
      {7859, {{0.66584373f, 0.82076108f, 0.09146875f, 0.23992974f}, 17}},
      {7878, {{0.66584373f, 0.82076108f, 0.09146875f, 0.23992974f}, 17}},
      {7897, {{0.66584373f, 0.82076108f, 0.09146875f, 0.23992974f}, 17}},
      {7997, {{0.55521870f, 0.54872364f, 0.25731248f, 0.24840745f}, 14}},
      {8276, {{0.46932030f, 0.47023422f, 0.19001558f, 0.48608899f}, 1}},
      {8277, {{0.59896874f, 0.44718969f, 0.16043746f, 0.48988286f}, 1}},
      {8284, {{0.32142186f, 0.47042155f, 0.11699998f, 0.49826697f}, 1}},
      {8286, {{0.46932030f, 0.47023422f, 0.19001558f, 0.48608899f}, 1}},
      {8287, {{0.59896874f, 0.44718969f, 0.16043746f, 0.48988286f}, 1}},
      {8296, {{0.46932030f, 0.47023422f, 0.19001558f, 0.48608899f}, 1}},
      {8476, {{0.46932030f, 0.47023422f, 0.19001558f, 0.48608899f}, 1}},
      {8477, {{0.59896874f, 0.44718969f, 0.16043746f, 0.48988286f}, 1}},
      {8486, {{0.46932030f, 0.47023422f, 0.19001558f, 0.48608899f}, 1}},
      {8487, {{0.59896874f, 0.44718969f, 0.16043746f, 0.48988286f}, 1}},
      {8496, {{0.46932030f, 0.47023422f, 0.19001558f, 0.48608899f}, 1}},
      {8497, {{0.59896874f, 0.44718969f, 0.16043746f, 0.48988286f}, 1}},
    },
  };

  const int coco_anchors_count_ = 8732;

  inline float Clamp(float val) { return val > 1.f ? 1.f : (val < 0.f ? 0.f : val); }

  void CreateCocoAnchors() {
    anchors_ = vector<float>(coco_anchors_count_ * 4);

    int fig_size = 300;
    vector<int> feat_sizes {38, 19, 10, 5, 3, 1};
    int feat_count = feat_sizes.size();
    vector<float> steps {8.f, 16.f, 32.f, 64.f, 100.f, 300.f};
    vector<float> scales {21.f, 45.f, 99.f, 153.f, 207.f, 261.f, 315.f};
    vector<vector<int>> aspect_ratios {{2}, {2, 3}, {2, 3}, {2, 3}, {2}, {2}};

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
    ASSERT_EQ(boxes_shape.size(), coco_batch_size);
    for (auto& shape : boxes_shape) {
      ASSERT_EQ(shape.size(), 2);
      ASSERT_EQ(shape[0], coco_anchors_count_);
      ASSERT_EQ(shape[1], 4);
    }

    auto labels_shape = labels->shape();
    ASSERT_EQ(labels_shape.size(), coco_batch_size);
    for (auto& shape : labels_shape) {
      ASSERT_EQ(shape.size(), 1);
      ASSERT_EQ(shape[0], coco_anchors_count_);
    }

    vector<float4> boxes_data(coco_anchors_count_);
    vector<int> labels_data(coco_anchors_count_);
    auto anchors_data = reinterpret_cast<float4 *>(anchors_.data());

    for (int sample = 0; sample < coco_batch_size; ++sample) {
      MemCopy(
        boxes_data.data(),
        boxes->tensor<float>(sample),
        coco_anchors_count_ * 4 * sizeof(float));
      MemCopy(labels_data.data(), labels->tensor<int>(sample), coco_anchors_count_ * sizeof(int));

      int idx = 0;
      for (auto match : coco_matches[sample]) {
        while (idx < match.first) {
          ASSERT_EQ(labels_data[idx], 0);
          ++idx;
        }

        auto actual = boxes_data[match.first];
        auto expected = match.second.first;

        ASSERT_EQ(labels_data[match.first], match.second.second);
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
  }
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(BoxEncoderTest, Types);

TYPED_TEST(BoxEncoderTest, TestOnCocoObjects) {
  this->RunForCoco(this->anchors_, 0.5f);
}

TYPED_TEST(BoxEncoderTest, TestNegativeCriteria) {
  EXPECT_THROW(this->RunForCoco(this->anchors_, -0.5f), std::runtime_error);
}

TYPED_TEST(BoxEncoderTest, TestCriteriaOverOne) {
  EXPECT_THROW(this->RunForCoco(this->anchors_, 1.5f), std::runtime_error);
}

TYPED_TEST(BoxEncoderTest, TestInvalidAnchors) {
  vector<float> invalid_anchors(this->anchors_);
  invalid_anchors.pop_back();

  EXPECT_THROW(this->RunForCoco(invalid_anchors, 0.5f), std::runtime_error);
}

}  // namespace dali
