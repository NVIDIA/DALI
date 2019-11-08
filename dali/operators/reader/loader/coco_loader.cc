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

#include <map>
#include <iomanip>
#include <iostream>
#include <fstream>

#include "dali/operators/reader/loader/coco_loader.h"
#include "dali/pipeline/util/lookahead_parser.h"

namespace dali {
namespace detail {
struct ImageInfo {
  std::string filename_;
  int original_id_;
  int width_;
  int height_;
};

struct Annotation {
  int image_id_;
  int category_id_;
  std::array<float, 4> box_;
  std::vector<int> segm_meta_;
  std::vector<float> segm_coords_;

  void ToLtrb() {
    box_[2] += box_[0];
    box_[3] += box_[1];
  }

  bool IsOver(float min_size_threshold) {
    return box_[2] >= min_size_threshold && box_[3] >= min_size_threshold;
  }
};

template <typename T>
void dump_meta_file(std::vector<T> &input, const std::string path) {
  std::ofstream file(path, std::ios_base::binary | std::ios_base::out);
  DALI_ENFORCE(file, "CocoReader meta file error while saving: " + path);

  unsigned size = input.size();
  file.write(reinterpret_cast<char*>(&size), sizeof(unsigned));
  file.write(reinterpret_cast<char*>(input.data()), size * sizeof(T));
}

template <typename T>
void dump_meta_file(std::vector<std::vector<T> > &input, const std::string path) {
  std::ofstream file(path, std::ios_base::binary | std::ios_base::out);
  DALI_ENFORCE(file, "CocoReader meta file error while saving: " + path);

  unsigned size = input.size();
  file.write(reinterpret_cast<char*>(&size), sizeof(unsigned));
  for (auto& v : input) {
    size = v.size();
    file.write(reinterpret_cast<char*>(&size), sizeof(unsigned));
    file.write(reinterpret_cast<char*>(v.data()), size * sizeof(T));
  }
}

void dump_filenames(const ImageIdPairs &image_id_pairs, const std::string path) {
  std::ofstream file(path);
  DALI_ENFORCE(file, "CocoReader meta file error while saving: " + path);
  for (const auto &p : image_id_pairs) {
    file << p.first << std::endl;
  }
}

template <typename T>
void load_meta_file(std::vector<T> &output, const std::string path) {
  std::ifstream file(path);
  DALI_ENFORCE(file, "CocoReader meta file error while loading for path: " + path);

  unsigned size;
  file.read(reinterpret_cast<char*>(&size), sizeof(unsigned));
  output.resize(size);
  file.read(reinterpret_cast<char*>(output.data()), size * sizeof(T));
}

template <typename T>
void load_meta_file(std::vector<std::vector<T> > &output, const std::string path) {
  std::ifstream file(path);
  DALI_ENFORCE(file, "CocoReader meta file error while loading for path: " + path);

  unsigned size;
  file.read(reinterpret_cast<char*>(&size), sizeof(unsigned));
  output.resize(size);
  for (size_t i = 0; i < output.size(); ++i) {
    file.read(reinterpret_cast<char*>(&size), sizeof(unsigned));
    output[i].resize(size);
    file.read(reinterpret_cast<char*>(output[i].data()), size * sizeof(T));
  }
}

void load_filenames(ImageIdPairs &image_id_pairs, const std::string path) {
  std::ifstream file(path);
  DALI_ENFORCE(file, "CocoReader meta file error while loading for path: " + path);

  int id = 0;
  std::string filename;
  while (file >> filename) {
    image_id_pairs.emplace_back(std::move(filename), int{id});
    ++id;
  }
}

void parse_image_infos(LookaheadParser &parser, std::vector<ImageInfo> &image_infos) {
  RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
  parser.EnterArray();
  while (parser.NextArrayValue()) {
    if (parser.PeekType() != kObjectType) {
      continue;
    }
    parser.EnterObject();
    ImageInfo image_info;
    while (const char* internal_key = parser.NextObjectKey()) {
      if (0 == detail::safe_strcmp(internal_key, "id")) {
          image_info.original_id_ = parser.GetInt();
      } else if (0 == detail::safe_strcmp(internal_key, "width")) {
          image_info.width_ = parser.GetInt();
      } else if (0 == detail::safe_strcmp(internal_key, "height")) {
          image_info.height_ = parser.GetInt();
      } else if (0 == detail::safe_strcmp(internal_key, "file_name")) {
          image_info.filename_ = parser.GetString();
      } else {
        parser.SkipValue();
      }
    }
    image_infos.emplace_back(std::move(image_info));
  }
}

void parse_categories(LookaheadParser &parser, std::map<int, int> &category_ids) {
  RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
  parser.EnterArray();

  int id = -1;
  int new_id = 1;

  while (parser.NextArrayValue()) {
    if (parser.PeekType() != kObjectType) {
      continue;
    }
    id = -1;
    parser.EnterObject();
    while (const char* internal_key = parser.NextObjectKey()) {
      if (0 == detail::safe_strcmp(internal_key, "id")) {
        id = parser.GetInt();
      } else {
        parser.SkipValue();
      }
    }
    DALI_ENFORCE(id != -1, "Missing category ID in the JSON annotations file");
    category_ids.insert(std::make_pair(id, new_id));
    new_id++;
  }
}

void parse_annotations(
  LookaheadParser &parser,
  std::vector<Annotation> &annotations,
  float min_size_threshold,
  bool ltrb,
  bool read_masks) {
  RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
  parser.EnterArray();
  while (parser.NextArrayValue()) {
    detail::Annotation annotation;
    if (parser.PeekType() != kObjectType) {
      continue;
    }
    parser.EnterObject();
    while (const char* internal_key = parser.NextObjectKey()) {
      if (0 == detail::safe_strcmp(internal_key, "image_id")) {
        annotation.image_id_ = parser.GetInt();
      } else if (0 == detail::safe_strcmp(internal_key, "category_id")) {
        annotation.category_id_ = parser.GetInt();
      } else if (0 == detail::safe_strcmp(internal_key, "bbox")) {
        RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
        parser.EnterArray();
        int i = 0;
        while (parser.NextArrayValue()) {
          annotation.box_[i] = parser.GetDouble();
          ++i;
        }
      } else if (read_masks && 0 == detail::safe_strcmp(internal_key, "segmentation")) {
        // That means that the mask encoding is not polygons but RLE (iscrowd==1),
        // which is not needed for instance segmentation
        if (parser.PeekType() != kArrayType) {
          while (parser.NextObjectKey()) {}
          break;
        }

        int coord_offset = 0;
        auto& segm_meta = annotation.segm_meta_;
        auto& segm_coords = annotation.segm_coords_;
        parser.EnterArray();
        while (parser.NextArrayValue()) {
          segm_meta.push_back(coord_offset);
          parser.EnterArray();
          while (parser.NextArrayValue()) {
            segm_coords.push_back(parser.GetDouble());
            coord_offset++;
          }
          segm_meta.push_back(coord_offset - segm_meta.back());
        }
      } else {
        parser.SkipValue();
      }
    }
    if (!annotation.IsOver(min_size_threshold)) {
      continue;
    }
    if (ltrb) {
      annotation.ToLtrb();
    }
    annotations.emplace_back(std::move(annotation));
  }
}

void parse_json_file(
  const OpSpec &spec,
  std::vector<detail::ImageInfo> &image_infos,
  std::vector<detail::Annotation> &annotations,
  std::map<int, int> &category_ids) {
  const auto annotations_file = spec.GetArgument<string>("annotations_file");

  std::ifstream f(annotations_file);
  DALI_ENFORCE(f, "Could not open JSON annotations file");
  f.seekg(0, std::ios::end);
  size_t file_size = f.tellg();
  std::unique_ptr<char, std::function<void(char*)>> buff(
    new char[file_size + 1],
    [](char* data) {delete [] data;});
  f.seekg(0, std::ios::beg);
  buff.get()[file_size] = '\0';
  f.read(buff.get(), file_size);
  f.close();

  detail::LookaheadParser parser(buff.get());

  RAPIDJSON_ASSERT(parser.PeekType() == kObjectType);
  parser.EnterObject();
  while (const char* key = parser.NextObjectKey()) {
    if (0 == detail::safe_strcmp(key, "images")) {
      detail::parse_image_infos(parser, image_infos);
    } else if (0 == detail::safe_strcmp(key, "categories")) {
      detail::parse_categories(parser, category_ids);
    } else if (0 == detail::safe_strcmp(key, "annotations")) {
      parse_annotations(
        parser,
        annotations,
        spec.GetArgument<float>("size_threshold"),
        spec.GetArgument<bool>("ltrb"),
        spec.GetArgument<bool>("masks"));
    } else {
      parser.SkipValue();
    }
  }
}

}  // namespace detail

void CocoLoader::DumpMetaFiles(const std::string path, const ImageIdPairs &image_id_pairs) {
  detail::dump_meta_file(
    offsets_,
    path + "/offsets.dat");
  detail::dump_meta_file(
    boxes_,
    path + "/boxes.dat");
  detail::dump_meta_file(
    labels_,
    path + "/labels.dat");
  detail::dump_meta_file(
    counts_,
    path + "/counts.dat");
  detail::dump_filenames(
    image_id_pairs,
    path + "/filenames.dat");

  if (read_masks_) {
    detail::dump_meta_file(
      masks_meta_,
      path + "/masks_metas.dat");
    detail::dump_meta_file(
      masks_meta_,
      path + "/masks_coords.dat");
  }

  if (save_img_ids_) {
    detail::dump_meta_file(
      original_ids_,
      path + "/original_ids.dat");
  }
}

void CocoLoader::ParseMetafiles() {
  const auto meta_files_path = spec_.GetArgument<string>("meta_files_path");
  detail::load_meta_file(
    offsets_,
    meta_files_path + "/offsets.dat");
  detail::load_meta_file(
    boxes_,
    meta_files_path + "/boxes.dat");
  detail::load_meta_file(
    labels_,
    meta_files_path + "/labels.dat");
  detail::load_meta_file(
    counts_,
    meta_files_path + "/counts.dat");
  detail::load_filenames(
    image_label_pairs_,
    meta_files_path + "/filenames.dat");

  if (read_masks_) {
    detail::load_meta_file(
      masks_meta_,
      meta_files_path + "/masks_metas.dat");
    detail::load_meta_file(
      masks_meta_,
      meta_files_path + "/masks_coords.dat");
  }
  if (save_img_ids_) {
    detail::load_meta_file(
      original_ids_,
      meta_files_path + "/original_ids.dat");
  }
}

void CocoLoader::ParseJsonAnnotations() {
  std::vector<detail::ImageInfo> image_infos;
  std::vector<detail::Annotation> annotations;
  std::map<int, int> category_ids;

  detail::parse_json_file(
    spec_,
    image_infos,
    annotations,
    category_ids);

  bool skip_empty = spec_.GetArgument<bool>("skip_empty");
  bool ratio = spec_.GetArgument<bool>("ratio");

  std::sort(image_infos.begin(), image_infos.end(), [](auto &left, auto &right) {
    return left.original_id_ < right.original_id_;});
  std::stable_sort(annotations.begin(), annotations.end(), [](auto &left, auto &right) {
    return left.image_id_ < right.image_id_;});

  detail::Annotation sentinel;
  sentinel.image_id_ = -1;
  annotations.emplace_back(std::move(sentinel));

  int new_image_id = 0;
  int annotation_id = 0;
  int total_count = 0;

  for (auto &image_info : image_infos) {
    int objects_in_sample = 0;
    std::vector<int> sample_mask_meta;
    std::vector<float> sample_mask_coords;
    while (annotations[annotation_id].image_id_ == image_info.original_id_) {
      const auto &annotation = annotations[annotation_id];
      labels_.emplace_back(category_ids[annotation.category_id_]);
      if (ratio) {
        boxes_.push_back(annotation.box_[0] / image_info.width_);
        boxes_.push_back(annotation.box_[1] / image_info.height_);
        boxes_.push_back(annotation.box_[2] / image_info.width_);
        boxes_.push_back(annotation.box_[3] / image_info.height_);
      } else {
        boxes_.push_back(annotation.box_[0]);
        boxes_.push_back(annotation.box_[1]);
        boxes_.push_back(annotation.box_[2]);
        boxes_.push_back(annotation.box_[3]);
      }
      if (read_masks_) {
        auto obj_coords_offset = sample_mask_coords.size();
        for (size_t i = 0; i < annotation.segm_meta_.size(); i += 2) {
          sample_mask_meta.push_back(objects_in_sample);
          sample_mask_meta.push_back(obj_coords_offset + annotation.segm_meta_[i]);
          sample_mask_meta.push_back(obj_coords_offset + annotation.segm_meta_[i + 1]);
        }
        sample_mask_coords.insert(sample_mask_coords.end(),
                                  annotation.segm_coords_.begin(),
                                  annotation.segm_coords_.end());
      }
      ++annotation_id;
      ++objects_in_sample;
    }

    if (!skip_empty || objects_in_sample != 0) {
      offsets_.push_back(total_count);
      counts_.push_back(objects_in_sample);
      total_count += objects_in_sample;
      if (save_img_ids_) {
        original_ids_.push_back(image_info.original_id_);
      }
      if (read_masks_) {
        masks_meta_.emplace_back(std::move(sample_mask_meta));
        masks_coords_.emplace_back(std::move(sample_mask_coords));
      }
      image_label_pairs_.emplace_back(std::move(image_info.filename_), new_image_id);
      new_image_id++;
    }
  }

  if (spec_.GetArgument<bool>("dump_meta_files")) {
    DumpMetaFiles(
      spec_.GetArgument<std::string>("dump_meta_files_path"),
      image_label_pairs_);
  }
}

}  // namespace dali
