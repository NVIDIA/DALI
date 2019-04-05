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

#include <gtest/gtest.h>

#include "dali/test/dali_test_config.h"
#include "dali/pipeline/pipeline.h"

namespace dali {

class CocoReaderTest : public ::testing::Test {
 protected:
  std::vector<std::pair<std::string, std::string>> Outputs() {
    return {{"images", "cpu"}, {"boxes", "cpu"}, {"labels", "cpu"}, {"image_ids", "cpu"}};
  }

  OpSpec CocoReaderOpSpec() {
    return OpSpec("COCOReader")
          .AddArg("device", "cpu")
          .AddArg("file_root", file_root_)
          .AddArg("annotations_file", annotations_filename_)
          .AddArg("save_img_ids", true)
          .AddOutput("images", "cpu")
          .AddOutput("boxes", "cpu")
          .AddOutput("labels", "cpu")
          .AddOutput("image_ids", "cpu");
  }

  int SmallCocoSize() { return 64; }
  int EmptyImages() { return 2; }
  int ImagesWithBigObjects() { return 2; }
  int ObjectCount() { return 194; }

  std::vector<int> CopyIds(DeviceWorkspace &ws) {
    auto &output = ws.Output<dali::CPUBackend>(3);
    auto shape = output.shape();

    vector<int> ids(shape.size());

    MemCopy(
      ids.data(),
      output.data<int>(),
      shape.size() * sizeof(int));
    return ids;
  }

  void CheckInstances(DeviceWorkspace & ws) {
    const auto &boxes_output = ws.Output<dali::CPUBackend>(1);
    const auto &labels_output = ws.Output<dali::CPUBackend>(2);

    const auto boxes_shape = boxes_output.shape();
    const auto labels_shape = labels_output.shape();

    ASSERT_EQ(labels_shape.size(), SmallCocoSize());
    ASSERT_EQ(boxes_shape.size(), SmallCocoSize());

    for (int idx = 0; idx < SmallCocoSize(); ++idx) {
      ASSERT_EQ(boxes_shape[idx][0], objects_in_image_[idx]);
      ASSERT_EQ(labels_shape[idx][0], objects_in_image_[idx]);
      ASSERT_EQ(boxes_shape[idx][1], bbox_size);
      ASSERT_EQ(boxes_shape[idx][1], bbox_size);
    }

    vector<float> boxes(ObjectCount() * bbox_size);
    vector<int> labels(ObjectCount());

    MemCopy(
      boxes.data(),
      boxes_output.data<float>(),
      ObjectCount() * bbox_size * sizeof(float));

    for (int box_coord = 0; box_coord < ObjectCount() * bbox_size; ++box_coord) {
      ASSERT_EQ(boxes[box_coord], boxes_coords_[box_coord]);
    }

    MemCopy(
      labels.data(),
      labels_output.data<int>(),
      ObjectCount() * sizeof(int));

    for (int obj = 0; obj < ObjectCount(); ++obj) {
      ASSERT_EQ(labels[obj], categories_[obj]);
    }
  }

 private:
  std::string file_root_ = dali::testing::dali_extra_path() + "/db/coco/images";
  std::vector<std::string> annotations_filename_ =
    { dali::testing::dali_extra_path() + "/db/coco/instances.json" };

  const int bbox_size = 4;

  std::vector<int> objects_in_image_ = {
    1, 1, 1, 4, 1, 3, 3, 4, 5, 2, 3, 2, 5, 1, 5, 3,
    4, 5, 2, 4, 5, 2, 3, 1, 3, 3, 1, 4, 1, 5, 2, 5,
    5, 2, 3, 1, 3, 3, 5, 2, 5, 5, 2, 5, 4, 5, 4, 1,
    1, 2, 5, 1, 5, 2, 3, 5, 4, 2, 4, 3, 4, 4, 0, 0
  };

  std::vector<float> boxes_coords_ = {
    313, 168, 162, 120, 100, 216, 182, 237, 138, 15, 404, 172, 215,
    305, 69, 80, 248, 64, 344, 311, 123, 66, 95, 176, 194, 209, 48,
    207, 122, 178, 47, 248, 400, 115, 176, 158, 88, 217, 91, 114, 49,
    148, 257, 184, 99, 40, 361, 130, 89, 84, 259, 246, 213, 455, 270,
    158, 144, 137, 92, 150, 275, 39, 286, 32, 185, 78, 12, 90, 273,
    39, 275, 220, 180, 311, 226, 12, 351, 96, 85, 168, 178, 9, 23, 183,
    167, 194, 355, 90, 95, 193, 151, 226, 298, 315, 370, 63, 381, 311,
    210, 110, 247, 84, 385, 175, 137, 44, 161, 112, 282, 15, 336, 130,
    159, 332, 387, 97, 100, 285, 300, 116, 374, 73, 142, 20, 272, 93,
    348, 62, 22, 1, 266, 226, 376, 79, 143, 157, 285, 69, 280, 232, 208,
    143, 300, 107, 62, 129, 350, 171, 166, 93, 331, 183, 334, 7, 95,
    125, 221, 54, 354, 84, 240, 131, 258, 22, 290, 173, 337, 61, 460,
    144, 52, 187, 157, 221, 279, 150, 172, 306, 322, 38, 263, 143, 325,
    114, 82, 61, 317, 110, 280, 88, 162, 46, 222, 102, 258, 177, 103,
    135, 83, 200, 338, 105, 286, 288, 428, 229, 63, 30, 54, 3, 392, 338,
    498, 169, 63, 166, 86, 237, 61, 110, 397, 130, 13, 32, 8, 30, 232,
    142, 31, 189, 233, 29, 183, 76, 339, 79, 254, 23, 309, 231, 234, 316,
    262, 61, 110, 152, 339, 11, 188, 19, 136, 202, 498, 1, 159, 124, 392,
    197, 155, 41, 44, 70, 335, 126, 239, 159, 59, 344, 230, 8, 288, 324,
    185, 88, 233, 116, 124, 7, 90, 90, 24, 156, 363, 219, 484, 262, 198,
    186, 546, 381, 117, 60, 246, 96, 260, 248, 103, 108, 17, 184, 134,
    169, 236, 212, 177, 125, 268, 183, 95, 220, 298, 124, 143, 116, 247,
    222, 347, 44, 318, 80, 353, 211, 293, 53, 76, 29, 52, 172, 192, 83,
    198, 185, 33, 221, 329, 149, 181, 298, 396, 102, 202, 136, 269, 222,
    13, 229, 236, 149, 311, 14, 309, 183, 474, 359, 127, 79, 258, 143,
    189, 170, 348, 222, 211, 13, 129, 205, 190, 61, 391, 142, 14, 201, 12,
    172, 217, 16, 113, 90, 22, 149, 42, 62, 182, 225, 325, 68, 182, 48,
    90, 194, 235, 135, 504, 70, 24, 72, 556, 105, 87, 55, 220, 21, 73, 67,
    347, 168, 180, 237, 96, 278, 344, 137, 289, 110, 304, 33, 371, 222,
    228, 159, 453, 88, 105, 231, 241, 165, 339, 49, 274, 126, 108, 160,
    110, 338, 149, 87, 7, 304, 40, 20, 25, 314, 219, 88, 57, 18, 242, 249,
    46, 89, 284, 228, 364, 265, 163, 103, 148, 359, 80, 38, 409, 14, 181,
    130, 69, 46, 262, 247, 229, 270, 321, 37, 27, 170, 133, 204, 1, 150,
    262, 193, 356, 85, 89, 133, 76, 103, 142, 67, 143, 64, 333, 40, 122,
    240, 40, 45, 230, 61, 345, 115, 474, 183, 70, 107, 360, 73, 244, 99,
    418, 110, 109, 172, 197, 210, 251, 50, 84, 147, 148, 127, 272, 53, 203,
    154, 166, 49, 152, 110, 193, 160, 116, 139, 524, 82, 47, 152, 365, 3,
    115, 119, 357, 212, 17, 30, 402, 37, 138, 23, 421, 121, 94, 197, 74, 64,
    314, 125, 55, 59, 140, 76, 129, 173, 213, 132, 230, 106, 315, 219, 435,
    193, 38, 56, 2, 89, 261, 25, 147, 228, 152, 126, 10, 195, 223, 357, 28,
    163, 231, 305, 63, 416, 202, 76, 29, 150, 80, 129, 68, 510, 257, 10, 169,
    258, 295, 33, 313, 131, 88, 142, 392, 205, 91, 75, 9, 132, 317, 120,
    425, 2, 17, 200, 92, 162, 349, 128, 56, 5, 53, 107, 401, 154, 25, 103,
    122, 75, 293, 129, 10, 262, 195, 47, 389, 131, 205, 163, 149, 183, 154,
    144, 488, 139, 17, 137, 138, 82, 323, 57, 377, 33, 220, 216, 22, 268,
    243, 128, 311, 39, 99, 215, 221, 61, 85, 214, 46, 93, 192, 187, 91, 184,
    32, 12, 202, 203, 225, 171, 96, 207, 323, 64, 249, 106, 211, 254, 189,
    223, 316, 89, 184, 130, 322, 115, 336, 74, 244, 22, 519, 70, 139, 189,
    331, 84, 145, 181, 128, 189, 75, 152, 56, 408, 141, 26, 353, 217, 128, 69,
    167, 238, 331, 233, 93, 166, 285, 23, 22, 55, 68, 140, 260, 44, 41, 231,
    341, 180, 120, 91, 378, 208, 127, 112, 217, 24, 261, 160, 117, 94, 71, 86,
    46, 171, 313, 115, 199, 18, 65, 34, 105, 145, 322, 113, 270, 5, 31, 135,
    302, 337, 169, 76
  };

  std::vector<int> categories_ {
    33, 34, 12, 20, 8, 34, 28, 49, 36, 70, 56, 23, 25, 24, 64, 1,
    42, 44, 73, 72, 5, 39, 8, 10, 14, 75, 50, 22, 77, 71, 31, 63,
    32, 70, 59, 27, 69, 74, 37, 14, 22, 45, 16, 60, 16, 78, 15, 30,
    29, 58, 38, 25, 79, 28, 74, 47, 67, 28, 1, 27, 11, 25, 17, 39,
    31, 16, 32, 75, 59, 72, 15, 58, 11, 18, 25, 72, 32, 44, 17, 45,
    80, 77, 61, 68, 3, 20, 45, 70, 47, 2, 42, 73, 51, 64, 48, 19,
    69, 70, 65, 29, 39, 68, 45, 15, 15, 48, 49, 44, 43, 12, 50, 15,
    23, 54, 39, 62, 17, 6, 73, 58, 19, 41, 71, 61, 79, 32, 12, 27,
    33, 1, 19, 76, 15, 18, 34, 17, 76, 39, 44, 51, 7, 44, 36, 10, 3,
    32, 39, 46, 78, 38, 55, 68, 71, 29, 20, 80, 4, 66, 44, 1, 36, 54,
    24, 23, 20, 51, 46, 11, 71, 46, 49, 16, 12, 24, 54, 33, 46, 16,
    4, 64, 76, 2, 77, 17, 56, 5, 48, 1, 30, 19, 14, 5, 62, 31,
  };
};

TEST_F(CocoReaderTest, MutuallyExclusiveOptions) {
  Pipeline pipe(1, 1, 0);

  pipe.AddOperator(
    this->CocoReaderOpSpec()
    .AddArg("shuffle_after_epoch", true)
    .AddArg("stick_to_shard", true));

  EXPECT_THROW(pipe.Build(this->Outputs()), std::runtime_error);
}

TEST_F(CocoReaderTest, SkipEmpty) {
  Pipeline pipe(this->SmallCocoSize() - this->EmptyImages(), 1, 0);

  pipe.AddOperator(
    this->CocoReaderOpSpec().AddArg("skip_empty", true),
    "coco_reader");

  pipe.Build(this->Outputs());

  ASSERT_EQ(
    pipe.EpochSize()["coco_reader"],
    this->SmallCocoSize() - this->EmptyImages());

  DeviceWorkspace ws;
  pipe.RunCPU();
  pipe.RunGPU();
  pipe.Outputs(&ws);

  auto ids = this->CopyIds(ws);

  for (int id = 0; id < this->SmallCocoSize() - this->EmptyImages(); ++id) {
    ASSERT_EQ(ids[id], id);
  }
}

TEST_F(CocoReaderTest, IncludeEmpty) {
  Pipeline pipe(this->SmallCocoSize(), 1, 0);

  pipe.AddOperator(
    this->CocoReaderOpSpec(),
    "coco_reader");

  pipe.Build(this->Outputs());

  ASSERT_EQ(pipe.EpochSize()["coco_reader"], this->SmallCocoSize());

  DeviceWorkspace ws;
  pipe.RunCPU();
  pipe.RunGPU();
  pipe.Outputs(&ws);

  auto ids = this->CopyIds(ws);

  for (int id = 0; id < this->SmallCocoSize(); ++id) {
    ASSERT_EQ(ids[id], id);
  }

  this->CheckInstances(ws);
}

TEST_F(CocoReaderTest, IncludeEmptyLazy) {
  Pipeline pipe(this->SmallCocoSize(), 1, 0);

  pipe.AddOperator(
    this->CocoReaderOpSpec()
    .AddArg("lazy_init", true),
    "coco_reader");

  pipe.Build(this->Outputs());

  EXPECT_ANY_THROW(pipe.EpochSize()["coco_reader"]);

  DeviceWorkspace ws;
  pipe.RunCPU();
  pipe.RunGPU();
  pipe.Outputs(&ws);

  ASSERT_EQ(pipe.EpochSize()["coco_reader"], this->SmallCocoSize());

  auto ids = this->CopyIds(ws);

  for (int id = 0; id < this->SmallCocoSize(); ++id) {
    ASSERT_EQ(ids[id], id);
  }

  this->CheckInstances(ws);
}

TEST_F(CocoReaderTest, BigSizeThreshold) {
  Pipeline pipe(this->ImagesWithBigObjects(), 1, 0);

  pipe.AddOperator(
    this->CocoReaderOpSpec()
    .AddArg("skip_empty", true)
    .AddArg("size_threshold", 300.f),
    "coco_reader");

  pipe.Build(this->Outputs());

  ASSERT_EQ(pipe.EpochSize()["coco_reader"], this->ImagesWithBigObjects());

  DeviceWorkspace ws;
  pipe.RunCPU();
  pipe.RunGPU();
  pipe.Outputs(&ws);

  auto ids = this->CopyIds(ws);

  ASSERT_EQ(ids[0], 3);
  ASSERT_EQ(ids[1], 17);
}

TEST_F(CocoReaderTest, ShuffleAfterEpoch) {
  Pipeline pipe(this->SmallCocoSize(), 1, 0);

  pipe.AddOperator(
    this->CocoReaderOpSpec()
    .AddArg("shuffle_after_epoch", true),
    "coco_reader");

  pipe.Build(this->Outputs());

  DeviceWorkspace ws;
  pipe.RunCPU();
  pipe.RunGPU();
  pipe.Outputs(&ws);

  auto ids_epoch_1 = this->CopyIds(ws);

  pipe.RunCPU();
  pipe.RunGPU();
  pipe.Outputs(&ws);

  auto ids_epoch_2 = this->CopyIds(ws);

  bool difference = false;
  for (int id = 0; id < this->SmallCocoSize(); ++id) {
    difference = ids_epoch_1[id] != ids_epoch_2[id];
    if (difference) {
      break;
    }
  }
  ASSERT_TRUE(difference);
}

}  // namespace dali
