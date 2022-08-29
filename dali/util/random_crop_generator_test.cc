// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/util/random_crop_generator.h"
#include <gtest/gtest.h>

namespace dali {

class RandomCropGeneratorTest : public ::testing::Test {
 public:
    AspectRatioRange default_aspect_ratio_range_{3.0f/4.0f, 4.0f/3.0f};
    AreaRange default_area_range_{0.08f, 1.0f};
    RandomCropGenerator default_generator_{default_aspect_ratio_range_, default_area_range_};
    int default_H_ = 480;
    int default_W_ = 640;
    int64_t seed_ = 1234;

    RandomCropGenerator MakeGenerator(int64_t seed = 1234) {
        return RandomCropGenerator(
            default_aspect_ratio_range_, default_area_range_, seed);
    }
};

TEST_F(RandomCropGeneratorTest, GenerateOneCropWindow) {
    auto crop = MakeGenerator().GenerateCropWindow({default_H_, default_W_});
    EXPECT_TRUE(crop.IsInRange({default_H_, default_W_}));
}

TEST_F(RandomCropGeneratorTest, SameSeedProduceSameResult) {
    auto crop1 = MakeGenerator(seed_).GenerateCropWindow({default_H_, default_W_});
    EXPECT_TRUE(crop1.IsInRange({default_H_, default_W_}));
    auto crop2 = MakeGenerator(seed_).GenerateCropWindow({default_H_, default_W_});
    EXPECT_TRUE(crop2.IsInRange({default_H_, default_W_}));
    EXPECT_EQ(crop1, crop2);
}

TEST_F(RandomCropGeneratorTest, DifferentSeedProduceDifferentResult) {
    auto crop1 = MakeGenerator(seed_).GenerateCropWindow({default_H_, default_W_});
    EXPECT_TRUE(crop1.IsInRange({default_H_, default_W_}));
    auto crop2 = MakeGenerator(seed_+1).GenerateCropWindow({default_H_, default_W_});
    EXPECT_TRUE(crop2.IsInRange({default_H_, default_W_}));
    EXPECT_NE(crop1, crop2);
}

TEST_F(RandomCropGeneratorTest, GeneratingMultipleWindowsProduceDifferentResults) {
    auto crops = MakeGenerator().GenerateCropWindows({default_H_, default_W_}, 1000);
    for (std::size_t i = 1; i < crops.size(); i++) {
        EXPECT_TRUE(crops[i-1].IsInRange({default_H_, default_W_}));
        EXPECT_NE(crops[i-1], crops[i]);
    }
}

TEST_F(RandomCropGeneratorTest, DifferentSeedProduceDifferentResultBatchedVersion) {
    auto crops1 = MakeGenerator(seed_).GenerateCropWindows({default_H_, default_W_}, 1000);
    auto crops2 = MakeGenerator(seed_+1).GenerateCropWindows({default_H_, default_W_}, 1000);
    ASSERT_EQ(crops1.size(), crops2.size());
    for (std::size_t i = 0; i < crops1.size(); i++) {
        EXPECT_TRUE(crops1[i].IsInRange({default_H_, default_W_}));
        EXPECT_TRUE(crops2[i].IsInRange({default_H_, default_W_}));
        EXPECT_NE(crops1[i], crops2[i]);
    }
}

TEST_F(RandomCropGeneratorTest, DimensionH1W1) {
    for (const auto& crop : MakeGenerator().GenerateCropWindows({1, 1}, 1000)) {
        EXPECT_TRUE(crop.IsInRange({1, 1}));
        EXPECT_EQ(0, crop.anchor[1]);
        EXPECT_EQ(0, crop.anchor[0]);
        EXPECT_EQ(1, crop.shape[0]);
        EXPECT_EQ(1, crop.shape[1]);
    }
}

TEST_F(RandomCropGeneratorTest, DimensionH1) {
    for (const auto& crop : MakeGenerator().GenerateCropWindows({1, default_W_}, 1000)) {
        EXPECT_TRUE(crop.IsInRange({1, default_W_}));
        EXPECT_EQ(0, crop.anchor[0]);
        EXPECT_EQ(1, crop.shape[0]);
        EXPECT_EQ(1, crop.shape[1]);
    }
}

TEST_F(RandomCropGeneratorTest, DimensionW1) {
    for (const auto& crop : MakeGenerator().GenerateCropWindows({default_H_, 1}, 1000)) {
        EXPECT_TRUE(crop.IsInRange({default_H_, 1}));
        EXPECT_EQ(0, crop.anchor[1]);
        EXPECT_EQ(1, crop.shape[0]);
        EXPECT_EQ(1, crop.shape[1]);
    }
}


TEST_F(RandomCropGeneratorTest, AspectRatio) {
    float min_ratio = 0.8f;
    float max_ratio = 2.0f;
    RandomCropGenerator gen({ min_ratio, max_ratio }, { 0.1f, 0.9f }, 12345);

    std::mt19937_64 rng(4321);
    std::uniform_int_distribution<int> s_dist(1, 2048);
    std::uniform_real_distribution<float> r_dist(0.5f, 2.0f);
    for (int i = 0; i < 10000; i++) {
      float s = s_dist(rng);
      float r = r_dist(rng);
      int W = std::roundf(s*r);
      int H = std::roundf(s/r);
      if (!W) W = 1;
      if (!H) H = 1;
      CropWindow window = gen.GenerateCropWindow({H, W});
      auto shape_W = window.shape[1];
      auto shape_H = window.shape[0];
      auto anchor_x = window.anchor[1];
      auto anchor_y = window.anchor[0];
      float aspect = static_cast<float>(shape_W) / shape_H;
      EXPECT_GE(aspect, min_ratio) << shape_W << "x" << shape_H;
      EXPECT_LE(aspect, max_ratio) << shape_W << "x" << shape_H;
      EXPECT_GE(shape_W, 1) << shape_W << "x" << shape_H;
      EXPECT_GE(shape_H, 1) << shape_W << "x" << shape_H;
      EXPECT_LE(shape_W, W) << shape_W << "x" << shape_H;
      EXPECT_LE(shape_H, H) << shape_W << "x" << shape_H;
      EXPECT_LT(anchor_x, W) << anchor_x << "x" << anchor_y;
      EXPECT_LT(anchor_y, H) << anchor_x << "x" << anchor_y;
      EXPECT_LE(anchor_x + shape_W, W) << anchor_x << "x" << anchor_y;
      EXPECT_LE(anchor_y + shape_H, H) << anchor_x << "x" << anchor_y;
    }
}

TEST_F(RandomCropGeneratorTest, ConsecutiveGenerationsAreNotTheSame) {
    auto generator = MakeGenerator();
    auto prev_crop = generator.GenerateCropWindow({default_H_, default_W_});
    for (std::size_t i = 1; i < 100; i++) {
        auto crop = generator.GenerateCropWindow({default_H_, default_W_});
        EXPECT_NE(prev_crop, crop);
        prev_crop = crop;
    }
}

TEST_F(RandomCropGeneratorTest, SameSeedConsecutiveGenerations) {
    auto generator1 = MakeGenerator(seed_);
    auto generator2 = MakeGenerator(seed_);
    for (std::size_t i = 0; i < 100; i++) {
        EXPECT_EQ(
            generator1.GenerateCropWindow({default_H_, default_W_}),
            generator2.GenerateCropWindow({default_H_, default_W_}));
    }
}

}  // namespace dali
