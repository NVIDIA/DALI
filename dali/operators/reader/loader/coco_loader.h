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

#ifndef DALI_OPERATORS_READER_LOADER_COCO_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_COCO_LOADER_H_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>
#include <utility>

#include "dali/operators/reader/loader/file_label_loader.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/geom/vec.h"
#include "dali/core/unique_handle.h"

extern "C" {
#include "third_party/cocoapi/common/maskApi.h"
}

namespace dali {

using ImageIdPairs = std::vector<std::pair<std::string, int>>;

inline bool OutPolygonMasksEnabled(const OpSpec &spec) {
  return spec.GetArgument<bool>("polygon_masks") ||
    (spec.HasArgument("masks") && spec.GetArgument<bool>("masks"));
}

inline bool OutPixelwiseMasksEnabled(const OpSpec &spec) {
  return spec.GetArgument<bool>("pixelwise_masks");
}

inline bool OutImageIdsEnabled(const OpSpec &spec) {
  return spec.GetArgument<bool>("image_ids") ||
    (spec.HasArgument("save_img_ids") && spec.GetArgument<bool>("save_img_ids"));
}

inline bool HasPreprocessedAnnotations(const OpSpec &spec) {
  return spec.HasArgument("preprocessed_annotations") ||
    (spec.HasArgument("meta_files_path") && spec.GetArgument<bool>("meta_files_path"));
}

inline bool HasSavePreprocessedAnnotations(const OpSpec &spec) {
  return spec.HasArgument("save_preprocessed_annotations") ||
    (spec.HasArgument("dump_meta_files") && spec.GetArgument<bool>("dump_meta_files"));
}

inline bool HasSavePreprocessedAnnotationsDir(const OpSpec &spec) {
  return spec.HasArgument("save_preprocessed_annotations_dir") ||
    (spec.HasArgument("dump_meta_files_path") && spec.GetArgument<bool>("dump_meta_files_path"));
}

struct RLEMask : public UniqueHandle<RLE, RLEMask> {
  DALI_INHERIT_UNIQUE_HANDLE(RLE, RLEMask)

  constexpr inline RLEMask() : UniqueHandle() {}

  RLEMask(siz h, siz w, siz m) {
    rleInit(&handle_, h, w, m, nullptr);
  }

  RLEMask(siz h, siz w, span<const uint> counts) {
    rleInit(&handle_, h, w, counts.size(), const_cast<uint*>(counts.data()));
  }

  RLEMask(siz h, siz w, const char* str) {
    rleFrString(&handle_, const_cast<char*>(str), h, w);
  }

  static constexpr bool is_null_handle(const RLE &handle) {
    return handle.cnts == nullptr;
  }

  static void DestroyHandle(RLE &handle) {
    if (handle.cnts)
      rleFree(&handle);
  }

  const RLE* operator->() const { return &handle_; }
  RLE* operator->() { return &handle_; }
};

using RLEMaskPtr = std::shared_ptr<RLEMask>;

class DLL_PUBLIC CocoLoader : public FileLabelLoader {
 public:
  explicit inline CocoLoader(const OpSpec &spec)
      : FileLabelLoader(spec, spec.GetArgument<bool>("shuffle_after_epoch")), spec_(spec) {
    has_preprocessed_annotations_ = HasPreprocessedAnnotations(spec);
    DALI_ENFORCE(has_preprocessed_annotations_ || spec.HasArgument("annotations_file"),
        "Either ``annotations_file`` or ``preprocessed_annotations`` must be provided");
    if (has_preprocessed_annotations_) {
      for (const char* arg_name : {"annotations_file", "skip_empty", "ratio", "ltrb", "images",
                                   "size_threshold", "dump_meta_files", "dump_meta_files_path"}) {
        if (spec.HasArgument(arg_name))
          DALI_FAIL(make_string("When reading data from preprocessed annotation files, \"",
                                arg_name, "\" is not supported."));
      }
    }

    spec.TryGetRepeatedArgument(images_, "images");
    output_polygon_masks_ = OutPolygonMasksEnabled(spec);
    output_pixelwise_masks_ = OutPixelwiseMasksEnabled(spec);
    output_image_ids_ = OutImageIdsEnabled(spec);
    if (output_polygon_masks_ && output_pixelwise_masks_) {
      DALI_FAIL("``pixelwise_masks`` and ``polygon_masks`` are mutually exclusive");
    }

    if (HasSavePreprocessedAnnotations(spec) != HasSavePreprocessedAnnotationsDir(spec)) {
      DALI_FAIL("``save_preprocessed_annotations`` and ``save_preprocessed_annotations_dir`` "
                "should be provided together");
    }
  }

  struct PixelwiseMasksInfo {
    TensorShape<3> shape;
    span<const RLEMaskPtr> rles;
    span<const int> mask_indices;
  };

  struct PolygonMasksInfo {
    span<const ivec3> polygons;
    span<const vec2> vertices;
  };

  span<const vec<4>> bboxes(int image_idx) const {
    return {reinterpret_cast<const vec<4>*>(boxes_.data()) + offsets_[image_idx],
            counts_[image_idx]};
  }

  span<const int> labels(int image_idx) const {
    return {labels_.data() + offsets_[image_idx], counts_[image_idx]};
  }

  int image_id(int image_idx) const {
    assert(output_image_ids_);
    return original_ids_[image_idx];
  }

  PixelwiseMasksInfo pixelwise_masks_info(int image_idx) const {
    assert(output_pixelwise_masks_);
    return {
      {heights_[image_idx], widths_[image_idx], 1},
      {masks_rles_.data() + mask_offsets_[image_idx], mask_counts_[image_idx]},
      {masks_rles_idx_.data() + mask_offsets_[image_idx], mask_counts_[image_idx]}
    };
  }

  span<const ivec3> polygons(int image_idx) const {
    assert(output_polygon_masks_ || output_pixelwise_masks_);
    if (polygon_data_.empty() || polygon_offset_.empty() || polygon_count_.empty())
      return {};
    return {polygon_data_.data() + polygon_offset_[image_idx], polygon_count_[image_idx]};
  }

  span<const vec2> vertices(int image_idx) const {
    assert(output_polygon_masks_ || output_pixelwise_masks_);
    if (vertices_data_.empty() || vertices_offset_.empty() || vertices_count_.empty())
      return {};
    return {vertices_data_.data() + vertices_offset_[image_idx], vertices_count_[image_idx]};
  }

 protected:
  void PrepareMetadataImpl() override {
    if (has_preprocessed_annotations_) {
      ParsePreprocessedAnnotations();
    } else {
      ParseJsonAnnotations();
    }

    DALI_ENFORCE(SizeImpl() > 0, "No files found.");
    if (shuffle_) {
      // seeded with hardcoded value to get
      // the same sequence on every shard
      std::mt19937 g(kDaliDataloaderSeed);
      std::shuffle(image_label_pairs_.begin(), image_label_pairs_.end(), g);
    }
    Reset(true);
  }

  void ParsePreprocessedAnnotations();

  void ParseJsonAnnotations();

  void SavePreprocessedAnnotations(const std::string &path, const ImageIdPairs &image_id_pairs);

 private:
  const OpSpec &spec_;

  std::vector<int> heights_;
  std::vector<int> widths_;
  std::vector<int> offsets_;
  std::vector<float> boxes_;
  std::vector<int> labels_;
  std::vector<int> counts_;
  std::vector<int> original_ids_;

  // polygons: (mask_idx, offset, size)
  std::vector<ivec3> polygon_data_;
  std::vector<int64_t> polygon_offset_;  // per-sample offset of polygons
  std::vector<int64_t> polygon_count_;   // number of polygon per sample
  // vertices: (all polygons concatenated)
  std::vector<vec2> vertices_data_;
  std::vector<int64_t> vertices_offset_;  // per-sample offset of vertices
  std::vector<int64_t> vertices_count_;   // number of vertices per sample

  // masks_rles: (run-length encodings)
  std::vector<RLEMaskPtr> masks_rles_;
  std::vector<int> masks_rles_idx_;
  std::vector<int64_t> mask_offsets_;  // per-sample offsets of masks
  std::vector<int64_t> mask_counts_;   // number of masks per sample

  bool output_polygon_masks_ = false;
  bool output_pixelwise_masks_ = false;
  bool output_image_ids_ = false;
  bool has_preprocessed_annotations_ = false;

  std::vector<std::string> images_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_COCO_LOADER_H_
