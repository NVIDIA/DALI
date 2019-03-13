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
class SliceTest : public GenericBBoxesTest<ImgType> {
  void SetUp() override {
    DALISingleOpTest<ImgType>::SetUp();

    this->MakeImageBatch(this->jpegs_.nImages(), &images_, DALI_RGB);
    this->MakeBBoxesAndLabelsBatch(&boxes_, &labels_, this->jpegs_.nImages());
  }

 protected:
  TensorList<CPUBackend> images_;
  TensorList<CPUBackend> boxes_;
  TensorList<CPUBackend> labels_;
  std::vector<TensorList<CPUBackend>*> outputs_;
};

typedef ::testing::Types<RGB> Types;
TYPED_TEST_SUITE(SliceTest, Types);

TYPED_TEST(SliceTest, RunCPUCheckImageCountMatches) {
  this->outputs_ = this->RunSliceCPU({{"images", &this->images_},
                                      {"boxes", &this->boxes_},
                                      {"labels", &this->labels_}});

  EXPECT_EQ(this->outputs_.at(0)->shape().size(), this->images_.shape().size());
}

TYPED_TEST(SliceTest, RunCPUCheckBoxesCountMatches) {
  this->outputs_ = this->RunSliceCPU({{"images", &this->images_},
                                      {"boxes", &this->boxes_},
                                      {"labels", &this->labels_}});

  EXPECT_EQ(this->outputs_.at(1)->shape().size(), this->boxes_.shape().size());
}

TYPED_TEST(SliceTest, RunCPUCheckImageOutputShapesMatch) {
  // By default, threshold for Bbox_crop should be zero, meaning that no crop
  //  will be performed.
  // Because of this, in this test we try to match the shape of each output
  //  image to each input
  // image as they must be identical
  this->outputs_ = this->RunSliceCPU({{"images", &this->images_},
                                      {"boxes", &this->boxes_},
                                      {"labels", &this->labels_}});

  for (size_t i = 0; i < this->images_.shape().size(); i++) {
    auto src_shape = this->images_.shape()[i];
    auto dst_shape = this->outputs_.at(0)->shape()[i];

    EXPECT_EQ(src_shape.size(), dst_shape.size());

    for (size_t j = 0; j < src_shape.size(); j++) {
      EXPECT_EQ(src_shape[j], dst_shape[j]);
    }
  }
}

TYPED_TEST(SliceTest, RunCPUCheckBoxesOutputShapesMatch) {
  // By default, threshold for Bbox_crop should be zero, meaning that no crop
  //  will be performed.
  // Because of this, in this test we try to match the shape of each output
  //  box to each input
  // box as they must be identical
  this->outputs_ = this->RunSliceCPU({{"images", &this->images_},
                                      {"boxes", &this->boxes_},
                                      {"labels", &this->labels_}});

  for (size_t i = 0; i < this->boxes_.shape().size(); i++) {
    auto src_shape = this->boxes_.shape()[i];
    auto dst_shape = this->outputs_.at(1)->shape()[i];

    EXPECT_EQ(src_shape.size(), dst_shape.size());

    for (size_t j = 0; j < src_shape.size(); j++) {
      EXPECT_EQ(src_shape[j], dst_shape[j]);
    }
  }
}

TYPED_TEST(SliceTest, RunGPUCheckImageCountMatches) {
  this->outputs_ = this->RunSliceGPU({{"images", &this->images_},
                                      {"boxes", &this->boxes_},
                                      {"labels", &this->labels_}});

  EXPECT_EQ(this->outputs_.at(0)->shape().size(), this->images_.shape().size());
}

TYPED_TEST(SliceTest, RunGPUCheckBoxesCountMatches) {
  this->outputs_ = this->RunSliceGPU({{"images", &this->images_},
                                      {"boxes", &this->boxes_},
                                      {"labels", &this->labels_}});

  EXPECT_EQ(this->outputs_.at(1)->shape().size(), this->boxes_.shape().size());
}

TYPED_TEST(SliceTest, RunGPUCheckImageOutputShapesMatch) {
  // By default, threshold for Bbox_crop should be zero, meaning that no crop
  //  will be performed.
  // Because of this, in this test we try to match the shape of each output
  //  image to each input
  // image as they must be identical
  this->outputs_ = this->RunSliceGPU({{"images", &this->images_},
                                      {"boxes", &this->boxes_},
                                      {"labels", &this->labels_}});

  for (size_t i = 0; i < this->images_.shape().size(); i++) {
    auto src_shape = this->images_.shape()[i];
    auto dst_shape = this->outputs_.at(0)->shape()[i];

    EXPECT_EQ(src_shape.size(), dst_shape.size());

    for (size_t j = 0; j < src_shape.size(); j++) {
      EXPECT_EQ(src_shape[j], dst_shape[j]);
    }
  }
}

TYPED_TEST(SliceTest, RunGPUCheckBoxesOutputShapesMatch) {
  // By default, threshold for Bbox_crop should be zero, meaning that no crop
  //  will be performed.
  // Because of this, in this test we try to match the shape of each output
  //  box to each input
  // box as they must be identical
  this->outputs_ = this->RunSliceGPU({{"images", &this->images_},
                                      {"boxes", &this->boxes_},
                                      {"labels", &this->labels_}});

  for (size_t i = 0; i < this->boxes_.shape().size(); i++) {
    auto src_shape = this->boxes_.shape()[i];
    auto dst_shape = this->outputs_.at(1)->shape()[i];

    EXPECT_EQ(src_shape.size(), dst_shape.size());

    for (size_t j = 0; j < src_shape.size(); j++) {
      EXPECT_EQ(src_shape[j], dst_shape[j]);
    }
  }
}

}  // namespace dali
