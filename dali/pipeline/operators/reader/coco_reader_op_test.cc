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
#include "dali/test/dali_test_config.h"

namespace dali {

template <typename Backend>
class CocoReaderTest : public DALITest {
 public:
  void SetUp() override {}
  void TearDown() override {}

  std::vector<std::pair<string, string>> Outputs() {
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

 private:
  std::string file_root_ = dali::testing::dali_extra_path() + "/db/coco/images";
  std::vector<std::string> annotations_filename_ =
    { dali::testing::dali_extra_path() + "/db/coco/instances.json" };
};

typedef ::testing::Types<CPUBackend> TestTypes;

TYPED_TEST_SUITE(CocoReaderTest, TestTypes);

TYPED_TEST(CocoReaderTest, MutuallyExclusiveOptions) {
  Pipeline pipe(1, 1, 0);

  pipe.AddOperator(
    this->CocoReaderOpSpec()
    .AddArg("shuffle_after_epoch", true)
    .AddArg("stick_to_shard", true));

  EXPECT_THROW(pipe.Build(this->Outputs()), std::runtime_error);
}

TYPED_TEST(CocoReaderTest, SkipEmpty) {
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

TYPED_TEST(CocoReaderTest, IncludeEmpty) {
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
}

TYPED_TEST(CocoReaderTest, BigSizeThreshold) {
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

TYPED_TEST(CocoReaderTest, ShuffleAfterEpoch) {
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

  for (int id = 0; id < this->SmallCocoSize(); ++id) {
    if (ids_epoch_1[id] != ids_epoch_2[id])
      return;
  }
  FAIL();
}

}  // namespace dali
