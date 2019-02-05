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
    auto crop = MakeGenerator().GenerateCropWindow(default_H_, default_W_);
    EXPECT_TRUE(crop.IsInRange(default_H_, default_W_));
}

TEST_F(RandomCropGeneratorTest, SameSeedProduceSameResult) {
    auto crop1 = MakeGenerator(seed_).GenerateCropWindow(default_H_, default_W_);
    EXPECT_TRUE(crop1.IsInRange(default_H_, default_W_));
    auto crop2 = MakeGenerator(seed_).GenerateCropWindow(default_H_, default_W_);
    EXPECT_TRUE(crop2.IsInRange(default_H_, default_W_));
    EXPECT_EQ(crop1, crop2);
}

TEST_F(RandomCropGeneratorTest, DifferentSeedProduceDifferentResult) {
    auto crop1 = MakeGenerator(seed_).GenerateCropWindow(default_H_, default_W_);
    EXPECT_TRUE(crop1.IsInRange(default_H_, default_W_));
    auto crop2 = MakeGenerator(seed_+1).GenerateCropWindow(default_H_, default_W_);
    EXPECT_TRUE(crop2.IsInRange(default_H_, default_W_));
    EXPECT_NE(crop1, crop2);
}

TEST_F(RandomCropGeneratorTest, GeneratingMultipleWindowsProduceDifferentResults) {
    auto crops = MakeGenerator().GenerateCropWindows(default_H_, default_W_, 1000);
    for (std::size_t i = 1; i < crops.size(); i++) {
        EXPECT_TRUE(crops[i-1].IsInRange(default_H_, default_W_));
        EXPECT_NE(crops[i-1], crops[i]);
    }
}

TEST_F(RandomCropGeneratorTest, DifferentSeedProduceDifferentResultBatchedVersion) {
    auto crops1 = MakeGenerator(seed_).GenerateCropWindows(default_H_, default_W_, 1000);
    auto crops2 = MakeGenerator(seed_+1).GenerateCropWindows(default_H_, default_W_, 1000);
    ASSERT_EQ(crops1.size(), crops2.size());
    for (std::size_t i = 0; i < crops1.size(); i++) {
        EXPECT_TRUE(crops1[i].IsInRange(default_H_, default_W_));
        EXPECT_TRUE(crops2[i].IsInRange(default_H_, default_W_));
        EXPECT_NE(crops1[i], crops2[i]);
    }
}

TEST_F(RandomCropGeneratorTest, EmptyDimensions) {
    EXPECT_THROW(
        MakeGenerator().GenerateCropWindow(0, 0),
        std::runtime_error);
}

TEST_F(RandomCropGeneratorTest, DimensionH1W1) {
    for (auto crop : MakeGenerator().GenerateCropWindows(1, 1, 1000)) {
        EXPECT_TRUE(crop.IsInRange(1, 1));
        EXPECT_EQ(0, crop.x);
        EXPECT_EQ(0, crop.y);
        EXPECT_EQ(1, crop.h);
        EXPECT_EQ(1, crop.w);
    }
}

TEST_F(RandomCropGeneratorTest, DimensionH1) {
    for (auto crop : MakeGenerator().GenerateCropWindows(1, default_W_, 1000)) {
        EXPECT_TRUE(crop.IsInRange(1, default_W_));
        EXPECT_EQ(0, crop.y);
        EXPECT_EQ(1, crop.h);
        EXPECT_EQ(1, crop.w);
    }
}

TEST_F(RandomCropGeneratorTest, DimensionW1) {
    for (auto crop : MakeGenerator().GenerateCropWindows(default_H_, 1, 1000)) {
        EXPECT_TRUE(crop.IsInRange(default_H_, 1));
        EXPECT_EQ(0, crop.x);
        EXPECT_EQ(1, crop.h);
        EXPECT_EQ(1, crop.w);
    }
}

TEST_F(RandomCropGeneratorTest, ConsecutiveGenerationsAreNotTheSame) {
    auto generator = MakeGenerator();
    auto prev_crop = generator.GenerateCropWindow(default_H_, default_W_);
    for (std::size_t i = 1; i < 100; i++) {
        auto crop = generator.GenerateCropWindow(default_H_, default_W_);
        EXPECT_NE(prev_crop, crop);
        prev_crop = crop;
    }
}

TEST_F(RandomCropGeneratorTest, SameSeedConsecutiveGenerations) {
    auto generator1 = MakeGenerator(seed_);
    auto generator2 = MakeGenerator(seed_);
    for (std::size_t i = 0; i < 100; i++) {
        EXPECT_EQ(
            generator1.GenerateCropWindow(default_H_, default_W_),
            generator2.GenerateCropWindow(default_H_, default_W_));
    }
}

}  // namespace dali
