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

#include <ftw.h>
#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "dali/test/dali_test_config.h"
#include "dali/pipeline/pipeline.h"

namespace {

int Remove(const char *fpath, const struct stat *sb, int typeflag, struct FTW *ftwbuf) {
    int ret = remove(fpath);
    assert(!ret);
    return ret;
}

void RemoveAll(const char *dir) {
  nftw(dir, Remove, 64, FTW_DEPTH | FTW_PHYS);
}

}  // namespace

namespace dali {

class CocoReaderTest : public ::testing::Test {
 protected:
  void TearDown() override {
    std::remove("/tmp/boxes.txt");
    std::remove("/tmp/counts.txt");
    std::remove("/tmp/filenames.txt");
    std::remove("/tmp/labels.txt");
    std::remove("/tmp/offsets.txt");
    std::remove("/tmp/original_ids.txt");
  }

  std::vector<std::pair<std::string, std::string>> Outputs(bool polygon_masks = false,
                                                           bool pixelwise_masks = false,
                                                           bool polygon_masks_legacy = false) {
    std::vector<std::pair<std::string, std::string>> out = {
        {"images", "cpu"}, {"boxes", "cpu"}, {"labels", "cpu"}};
    if (polygon_masks) {
      out.push_back({"polygons", "cpu"});
      out.push_back({"vertices", "cpu"});
    } else if (polygon_masks_legacy) {
      out.push_back({"masks_meta", "cpu"});
      out.push_back({"masks_coords", "cpu"});
    } else if (pixelwise_masks) {
      out.push_back({"pixelwise_masks", "cpu"});
    }
    out.push_back({"image_ids", "cpu"});
    return out;
  }

  OpSpec BasicCocoReaderOpSpec(bool polygon_masks = false, bool pixelwise_masks = false,
                               bool polygon_masks_legacy = false) {
    OpSpec spec =  OpSpec("COCOReader")
      .AddArg("device", "cpu")
      .AddArg("file_root", file_root_)
      .AddArg("save_img_ids", true)
      .AddOutput("images", "cpu")
      .AddOutput("boxes", "cpu")
      .AddOutput("labels", "cpu");
      if (polygon_masks) {
        spec = spec.AddArg("polygon_masks", true)
                   .AddOutput("polygons", "cpu")
                   .AddOutput("vertices", "cpu");
      }
      if (pixelwise_masks) {
        spec = spec.AddArg("pixelwise_masks", true)
                   .AddOutput("pixelwise_masks", "cpu");
      }
      if (polygon_masks_legacy) {
        spec = spec.AddArg("masks", true)
                   .AddOutput("masks_meta", "cpu")
                   .AddOutput("masks_coords", "cpu");
      }
      spec = spec.AddOutput("image_ids", "cpu");
      return spec;
  }

  OpSpec CocoReaderOpSpec(bool polygon_masks = false, bool pixelwise_masks = false,
                          bool polygon_masks_legacy = false) {
    return BasicCocoReaderOpSpec(polygon_masks, pixelwise_masks, polygon_masks_legacy)
      .AddArg("annotations_file", annotations_filename_);
  }

  int SmallCocoSize(bool masks) { return masks ? 4 : 64; }
  int EmptyImages() { return 2; }
  int NonEmptyImages(bool masks) { return SmallCocoSize(masks) - EmptyImages(); }
  int ImagesWithBigObjects() { return 2; }
  int ObjectCount(bool masks) { return masks ? 7 : 194; }

  std::vector<int> CopyIds(DeviceWorkspace &ws, int ids_out_idx = 3) {
    auto &output = ws.Output<dali::CPUBackend>(ids_out_idx);
    const auto &shape = output.shape();

    vector<int> ids(shape.size());

    MemCopy(
      ids.data(),
      output.data<int>(),
      shape.size() * sizeof(int));
    return ids;
  }

  void RunTestForPipeline(Pipeline &pipe, bool ltrb, bool ratio, bool skip_empty, int expected_size,
                          bool polygon_masks = false, bool polygon_masks_legacy = false) {
    auto outs = Outputs(polygon_masks, false, polygon_masks_legacy);
    pipe.Build(outs);

    if (!polygon_masks) {
      ASSERT_EQ(pipe.GetReaderMeta("coco_reader").epoch_size, expected_size);
    }

    DeviceWorkspace ws;
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);
    auto ids = CopyIds(ws, outs.size()-1);

    for (int id = 0; id < expected_size; ++id) {
      ASSERT_EQ(ids[id], id);
    }

    CheckInstances(ws, ltrb, ratio, skip_empty, expected_size, polygon_masks, polygon_masks_legacy);
  }

  void RunTest(bool ltrb, bool ratio, bool skip_empty, bool polygon_masks = false,
               bool polygon_masks_legacy = false) {
    const auto expected_size =
        skip_empty ? NonEmptyImages(polygon_masks) : SmallCocoSize(polygon_masks);

    OpSpec spec = BasicCocoReaderOpSpec(polygon_masks, false, polygon_masks_legacy);

    std::string tmpl = "/tmp/coco_reader_test_XXXXXX";
    std::string tmp_dir = mkdtemp(&tmpl[0]);

    OpSpec spec1 = spec;
    spec1 = spec1.AddArg("annotations_file", annotations_filename_)
                 .AddArg("skip_empty", skip_empty)
                 .AddArg("ltrb", ltrb)
                 .AddArg("ratio", ratio)
                 .AddArg("save_preprocessed_annotations", true)
                 .AddArg("save_preprocessed_annotations_dir", tmp_dir);

    Pipeline pipe1(expected_size, 1, 0);
    pipe1.AddOperator(spec1, "coco_reader");
    RunTestForPipeline(pipe1, ltrb, ratio, skip_empty, expected_size, polygon_masks,
                       polygon_masks_legacy);

    OpSpec spec2 = spec;
    spec2.AddArg("preprocessed_annotations", tmp_dir);

    Pipeline pipe2(expected_size, 1, 0);
    pipe2.AddOperator(spec2, "coco_reader");
    RunTestForPipeline(pipe2, ltrb, ratio, skip_empty, expected_size, polygon_masks,
                       polygon_masks_legacy);

    RemoveAll(tmp_dir.c_str());
  }

  void CheckInstances(DeviceWorkspace &ws, bool ltrb, bool ratio, bool skip_empty,
                      int expected_size, bool polygon_masks, bool polygon_masks_legacy) {
    const auto &boxes_output = ws.Output<dali::CPUBackend>(1);
    const auto &labels_output = ws.Output<dali::CPUBackend>(2);

    const auto &boxes_shape = boxes_output.shape();
    const auto &labels_shape = labels_output.shape();

    ASSERT_EQ(labels_shape.size(), expected_size);
    ASSERT_EQ(boxes_shape.size(), expected_size);

    for (int idx = 0; idx < expected_size; ++idx) {
      ASSERT_EQ(boxes_shape[idx][0], objects_in_image_[idx]);
      ASSERT_EQ(labels_shape[idx][0], objects_in_image_[idx]);
      ASSERT_EQ(boxes_shape[idx][1], bbox_size_);
      ASSERT_EQ(boxes_shape[idx][1], bbox_size_);
    }

    auto obj_count = ObjectCount(polygon_masks);
    vector<float> boxes(obj_count * bbox_size_);
    vector<int> labels(obj_count);

    MemCopy(
      boxes.data(),
      boxes_output.data<float>(),
      obj_count * bbox_size_ * sizeof(float));

    for (int box_coord = 0, idx = 0;
         box_coord < obj_count * bbox_size_;
         box_coord += 4, idx++) {
      float v1 = boxes[box_coord];
      float v2 = boxes[box_coord + 1];
      float v3 = boxes[box_coord + 2];
      float v4 = boxes[box_coord + 3];

      if (ratio) {
        v1 *= widths_[idx];
        v2 *= heights_[idx];
        v3 *= widths_[idx];
        v4 *= heights_[idx];
      }

      ASSERT_FLOAT_EQ(v1, boxes_coords_[box_coord]);
      ASSERT_FLOAT_EQ(v2, boxes_coords_[box_coord + 1]);

      if (!ltrb) {
        ASSERT_FLOAT_EQ(v3, boxes_coords_[box_coord + 2]);
        ASSERT_FLOAT_EQ(v4, boxes_coords_[box_coord + 3]);
      } else {
        ASSERT_FLOAT_EQ(v3, boxes_coords_[box_coord] + boxes_coords_[box_coord + 2]);
        ASSERT_FLOAT_EQ(v4, boxes_coords_[box_coord + 1] + boxes_coords_[box_coord + 3]);
      }
    }

    MemCopy(
      labels.data(),
      labels_output.data<int>(),
      obj_count * sizeof(int));

    for (int obj = 0; obj < obj_count; ++obj) {
      ASSERT_EQ(labels[obj], categories_[obj]);
    }

    if (polygon_masks || polygon_masks_legacy) {
      const auto &polygons_output = ws.Output<dali::CPUBackend>(3);
      const auto &vertices_output = ws.Output<dali::CPUBackend>(4);

      const auto &polygons_shape = polygons_output.shape();
      const auto &vertices_shape = vertices_output.shape();

      ASSERT_EQ(polygons_shape.size(), expected_size);
      ASSERT_EQ(vertices_shape.size(), expected_size);

      int n_poly = 0;
      for (int idx = 0; idx < expected_size; ++idx) {
        n_poly += polygons_shape[idx][0];
      }
      ASSERT_EQ(n_poly, number_of_polygons_);

      vector<int> polygons(polygons_gt_.size());
      vector<float> vertices(vertices_gt_.size());

      MemCopy(
        polygons.data(),
        polygons_output.data<int>(),
        polygons_gt_.size() * sizeof(int));

      MemCopy(
        vertices.data(),
        vertices_output.data<float>(),
        vertices_gt_.size() * sizeof(float));

      if (polygon_masks_legacy) {
        for (size_t i = 0; i < polygons.size(); i+=3) {
          ASSERT_EQ(polygons[i], polygons_gt_[i]);
          ASSERT_EQ(polygons[i+1], 2 * polygons_gt_[i+1]);
          ASSERT_EQ(polygons[i+2], 2 * polygons_gt_[i+2]);
        }
      } else {
        for (size_t i = 0; i < polygons.size(); ++i) {
          ASSERT_EQ(polygons[i], polygons_gt_[i]);
        }
      }

      for (size_t i = 0; i < vertices.size(); ++i) {
        ASSERT_FLOAT_EQ(vertices[i], vertices_gt_[i]);
      }
    }
  }

  std::string file_list_ = dali::testing::dali_extra_path() + "/db/coco_dummy/file_list.txt";
  std::string file_root_ = dali::testing::dali_extra_path() + "/db/coco_dummy/images";
  std::string annotations_filename_ = dali::testing::dali_extra_path() +
                                      "/db/coco_dummy/instances.json";

  const int bbox_size_ = 4;

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

  std::vector<int> widths_ {
    619, 480, 691, 633, 633, 633, 633, 677, 690, 690, 690, 616,
    616, 616, 639, 639, 639, 639, 619, 619, 619, 619, 619, 668,
    668, 651, 651, 651, 696, 696, 630, 630, 630, 630, 630, 665,
    621, 621, 621, 621, 621, 642, 642, 642, 652, 652, 652, 652,
    658, 658, 658, 658, 658, 693, 693, 610, 610, 610, 610, 683,
    683, 683, 683, 683, 616, 616, 532, 532, 532, 611, 696, 696,
    696, 630, 630, 630, 618, 700, 700, 700, 700, 585, 669, 669,
    669, 669, 669, 669, 669, 661, 661, 661, 661, 661, 606, 606,
    606, 606, 606, 655, 655, 666, 666, 666, 626, 692, 692, 692,
    473, 473, 473, 616, 616, 616, 616, 616, 609, 609, 593, 593,
    593, 593, 593, 614, 614, 614, 614, 614, 592, 592, 656, 656,
    656, 656, 656, 595, 595, 595, 595, 630, 630, 630, 630, 630,
    402, 402, 402, 402, 650, 454, 608, 608, 586, 586, 586, 586,
    586, 582, 702, 702, 702, 702, 702, 651, 651, 598, 598, 598,
    658, 658, 658, 658, 658, 666, 666, 666, 666, 611, 611, 596,
    596, 596, 596, 690, 690, 690, 602, 602, 602, 602, 586, 586,
    586, 586
  };

  std::vector<int> heights_ {
    324, 632, 376, 605, 605, 605, 605, 447, 335, 335, 335, 614,
    614, 614, 387, 387, 387, 387, 419, 419, 419, 419, 419, 556,
    556, 384, 384, 384, 464, 464, 423, 423, 423, 423, 423, 467,
    352, 352, 352, 352, 352, 396, 396, 396, 371, 371, 371, 371,
    567, 567, 567, 567, 567, 392, 392, 391, 391, 391, 391, 393,
    393, 393, 393, 393, 366, 366, 642, 642, 642, 430, 478, 478,
    478, 429, 429, 429, 407, 391, 391, 391, 391, 447, 400, 400,
    400, 400, 400, 448, 448, 393, 393, 393, 393, 393, 403, 403,
    403, 403, 403, 389, 389, 419, 419, 419, 415, 401, 401, 401,
    618, 618, 618, 455, 455, 455, 455, 455, 413, 413, 436, 436,
    436, 436, 436, 306, 306, 306, 306, 306, 469, 469, 351, 351,
    351, 351, 351, 350, 350, 350, 350, 398, 398, 398, 398, 398,
    604, 604, 604, 604, 453, 600, 421, 421, 432, 432, 432, 432,
    432, 451, 347, 347, 347, 347, 347, 443, 443, 465, 465, 465,
    434, 434, 434, 434, 434, 391, 391, 391, 391, 399, 399, 526,
    526, 526, 526, 481, 481, 481, 381, 381, 381, 381, 417, 417,
    417, 417
  };

  const int number_of_polygons_ = 9;

  std::vector<int> polygons_gt_ {
    // image 0
    0, 0, 5,
    // image 1
    0, 0, 3,
    0, 3, 6,
    // image 2
    0, 0, 6,
    // image 3
    0, 0, 3,
    1, 3, 10,
    2, 10, 14,
    3, 14, 19,
    3, 19, 22,
  };

  std::vector<float> vertices_gt_ {
    // 1
    363.51, 278.77, 348.39, 378.6, 266.49, 288.67, 140.12, 184.81, 164.16, 313.07,
    // 2
    380.64, 447.57, 354.81, 476.05, 251.71, 146.86,
    138.6, 281.01, 391.44, 499.4, 218.37, 484.74,
    // 3
    427.52, 135.91, 131.18, 414.8, 221.47, 495.56, 107.92, 154.94, 486.7, 322.39, 389.64, 276.52,
    // 4
    452.11, 276.25, 343.76, 361.14, 391.68, 322.95,
    // 5
    312.17, 286.2, 197.92, 318.73, 304.82, 220.41, 218.68, 475.61, 420.28, 410.53,
    332.12, 438.03, 146.57, 252.33,
    // 6
    158.29, 130.46, 257.22, 140.59, 140.71, 129.21, 106.66, 197.4,
    // 7
    480.31, 277.87, 203.99, 147.71, 257.94, 115.94, 264.69, 111.48, 311.21, 329.63,
    158.43, 277.21, 384.77, 314.0, 241.69, 281.34,
  };
};

TEST_F(CocoReaderTest, NoDataSource) {
  Pipeline pipe(1, 1, 0);

  pipe.AddOperator(
    this->BasicCocoReaderOpSpec());

  EXPECT_THROW(pipe.Build(this->Outputs()), std::runtime_error);
}

TEST_F(CocoReaderTest, TwoDataSources) {
  Pipeline pipe(1, 1, 0);

  pipe.AddOperator(
    this->BasicCocoReaderOpSpec()
    .AddArg("annotations_file", this->annotations_filename_)
    .AddArg("meta_files_path", "/tmp/"));

  EXPECT_THROW(pipe.Build(this->Outputs()), std::runtime_error);
}

TEST_F(CocoReaderTest, MissingDumpPath) {
  Pipeline pipe(1, 1, 0);

  pipe.AddOperator(
    this->BasicCocoReaderOpSpec()
    .AddArg("annotations_file", this->annotations_filename_)
    .AddArg("save_preprocessed_annotations", true));

  EXPECT_THROW(pipe.Build(this->Outputs()), std::runtime_error);
}

TEST_F(CocoReaderTest, MutuallyExclusiveOptions) {
  Pipeline pipe(1, 1, 0);

  pipe.AddOperator(
    this->CocoReaderOpSpec()
    .AddArg("shuffle_after_epoch", true)
    .AddArg("stick_to_shard", true));

  EXPECT_THROW(pipe.Build(this->Outputs()), std::runtime_error);
}

TEST_F(CocoReaderTest, MutuallyExclusiveOptions2) {
  Pipeline pipe(1, 1, 0);

  pipe.AddOperator(
    this->CocoReaderOpSpec()
    .AddArg("random_shuffle", true)
    .AddArg("shuffle_after_epoch", true));

  EXPECT_THROW(pipe.Build(this->Outputs()), std::runtime_error);
}

TEST_F(CocoReaderTest, MutuallyExclusiveOptions3) {
  Pipeline pipe(1, 1, 0);

  pipe.AddOperator(
    this->BasicCocoReaderOpSpec()
    .AddArg("meta_files_path", "/tmp/")
    .AddArg("skip_empty", false));

  EXPECT_THROW(pipe.Build(this->Outputs()), std::runtime_error);
}

TEST_F(CocoReaderTest, SkipEmpty) {
  this->RunTest(false, false, true);
}

TEST_F(CocoReaderTest, IncludeEmpty) {
  this->RunTest(false, false, false);
}

TEST_F(CocoReaderTest, Ltrb) {
  this->RunTest(true, false, false);
}

TEST_F(CocoReaderTest, Ratio) {
  this->RunTest(false, true, false);
}

TEST_F(CocoReaderTest, LtrbRatio) {
  this->RunTest(true, true, false);
}

TEST_F(CocoReaderTest, LtrbRatioSkipEmpty) {
  this->RunTest(true, true, true);
}

TEST_F(CocoReaderTest, PolygonMasks) {
  this->RunTest(false, false, false, true);
}

TEST_F(CocoReaderTest, PolygonMasksLegacy) {
  this->RunTest(false, false, false, false, true);
}

TEST_F(CocoReaderTest, PixelwiseMasks) {
  this->file_root_ = dali::testing::dali_extra_path() + "/db/coco_pixelwise/images";
  this->annotations_filename_ = dali::testing::dali_extra_path() +
                                      "/db/coco_pixelwise/instances.json";
  int expected_size = 6;
  int kSeed = 12345;

  std::string tmpl = "/tmp/coco_reader_test_XXXXXX";
  std::string tmp_dir = mkdtemp(&tmpl[0]);

  Pipeline pipe1(expected_size, 1, 0, kSeed);
  pipe1.AddOperator(
    CocoReaderOpSpec(false, true)
    .AddArg("save_preprocessed_annotations", true)
    .AddArg("save_preprocessed_annotations_dir", tmp_dir),
    "coco_reader");
  pipe1.Build(Outputs(false, true));

  DeviceWorkspace ws1;
  pipe1.RunCPU();
  pipe1.RunGPU();
  pipe1.Outputs(&ws1);

  Pipeline pipe2(expected_size, 1, 0, kSeed);
  pipe2.AddOperator(
    BasicCocoReaderOpSpec(false, true)
    .AddArg("preprocessed_annotations", tmp_dir),
    "coco_reader");
  pipe2.Build(Outputs(false, true));

  DeviceWorkspace ws2;
  pipe2.RunCPU();
  pipe2.RunGPU();
  pipe2.Outputs(&ws2);

  for (auto *ws : {&ws1, &ws2}) {
    const auto &masks_output = ws->Output<dali::CPUBackend>(3);

    const auto &masks_shape = masks_output.shape();
    TensorListShape<3> pixelwise_masks_shape({
      {815, 1280, 1}, {853, 1280, 1}, {853, 1280, 1},
      {853, 1280, 1}, {853, 1280, 1}, {848, 1280, 1}
    });
    ASSERT_EQ(masks_shape.size(), expected_size);
    ASSERT_EQ(masks_shape, pixelwise_masks_shape);

    std::vector<std::string> files {
      "eat-1237431_1280.png", "home-office-336373_1280.png", "home-office-336377_1280.png",
      "home-office-336378_1280.png", "pizza-2000614_1280.png", "pizza-2068272_1280.png"
    };

    for (int i = 0; i < expected_size; ++i) {
      std::vector<uchar> labels(masks_output.tensor<int>(i),
        masks_output.tensor<int>(i) + pixelwise_masks_shape[i][0] * pixelwise_masks_shape[i][1]);

      std::string file_root = dali::testing::dali_extra_path() +
        "/db/coco_pixelwise/pixelwise_masks/";
      cv::Mat cv_mask =  cv::imread(file_root + files[i], cv::IMREAD_COLOR);
      cv::cvtColor(cv_mask, cv_mask, cv::COLOR_BGR2RGB);
      cv::Mat channels[3];
      split(cv_mask, channels);
      cv::Mat mask = channels[0] / 255 + 2 * channels[1] / 255 + 3 * channels[2] / 255;
      cv::Size s = mask.size();

      ASSERT_EQ(pixelwise_masks_shape[i][1], s.width);
      ASSERT_EQ(pixelwise_masks_shape[i][0], s.height);
      EXPECT_EQ(0, std::memcmp(mask.data, labels.data(), s.width * s.height * sizeof(uchar)));
    }
  }

  RemoveAll(tmp_dir.c_str());
}

TEST_F(CocoReaderTest, BigSizeThreshold) {
  Pipeline pipe(this->ImagesWithBigObjects(), 1, 0);

  pipe.AddOperator(
    this->CocoReaderOpSpec()
    .AddArg("skip_empty", true)
    .AddArg("size_threshold", 300.f),
    "coco_reader");

  pipe.Build(this->Outputs());

  ASSERT_EQ(pipe.GetReaderMeta("coco_reader").epoch_size, this->ImagesWithBigObjects());

  DeviceWorkspace ws;
  pipe.RunCPU();
  pipe.RunGPU();
  pipe.Outputs(&ws);

  auto ids = this->CopyIds(ws);

  ASSERT_EQ(ids[0], 3);
  ASSERT_EQ(ids[1], 17);
}

TEST_F(CocoReaderTest, ShuffleAfterEpoch) {
  Pipeline pipe(this->SmallCocoSize(false), 1, 0);

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
  for (int id = 0; id < this->SmallCocoSize(false); ++id) {
    difference = ids_epoch_1[id] != ids_epoch_2[id];
    if (difference) {
      break;
    }
  }
  ASSERT_TRUE(difference);
}

}  // namespace dali
