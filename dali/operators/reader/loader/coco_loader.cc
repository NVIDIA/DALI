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

struct Polygons {
  std::vector<int> segm_meta_;
  std::vector<float> segm_coords_;
};

struct RunLengthEncoding {
  int h_, w_;
  std::string rle_;
};

struct Annotation {
  enum {POLYGON, RLE} tag_;
  int image_id_;
  int category_id_;
  std::array<float, 4> box_;
  // union
  Polygons poly_;
  RunLengthEncoding rle_;

  void ToLtrb() {
    box_[2] += box_[0];
    box_[3] += box_[1];
  }

  bool IsOver(float min_size_threshold) {
    return box_[2] >= min_size_threshold && box_[3] >= min_size_threshold;
  }
};

template <typename T>
void SaveToFile(std::vector<T> &input, const std::string path) {
  std::ofstream file(path, std::ios_base::binary | std::ios_base::out);
  DALI_ENFORCE(file, "CocoReader meta file error while saving: " + path);

  unsigned size = input.size();
  file.write(reinterpret_cast<char*>(&size), sizeof(unsigned));
  file.write(reinterpret_cast<char*>(input.data()), size * sizeof(T));
  DALI_ENFORCE(file.good(), make_string("Error writing to path: ", path));
}

template <typename T>
void SaveToFile(std::vector<std::vector<T> > &input, const std::string path) {
  std::ofstream file(path, std::ios_base::binary | std::ios_base::out);
  DALI_ENFORCE(file, "CocoReader meta file error while saving: " + path);

  unsigned size = input.size();
  file.write(reinterpret_cast<char*>(&size), sizeof(unsigned));
  for (auto& v : input) {
    size = v.size();
    file.write(reinterpret_cast<char*>(&size), sizeof(unsigned));
    file.write(reinterpret_cast<char*>(v.data()), size * sizeof(T));
  }
  DALI_ENFORCE(file.good(), make_string("Error writing to path: ", path));
}

void SaveFilenamesToFile(const ImageIdPairs &image_id_pairs, const std::string path) {
  std::ofstream file(path);
  DALI_ENFORCE(file, "CocoReader meta file error while saving: " + path);
  for (const auto &p : image_id_pairs) {
    file << p.first << std::endl;
  }
  DALI_ENFORCE(file.good(), make_string("Error writing to path: ", path));
}

template <typename T>
void LoadFromFile(std::vector<T> &output, const std::string path) {
  std::ifstream file(path);
  DALI_ENFORCE(file.good(),
               make_string("CocoReader failed to read preprocessed annotation data from ", path));

  unsigned size;
  file.read(reinterpret_cast<char*>(&size), sizeof(unsigned));
  output.resize(size);
  file.read(reinterpret_cast<char*>(output.data()), size * sizeof(T));
}

template <typename T>
void LoadFromFile(std::vector<std::vector<T> > &output, const std::string path) {
  std::ifstream file(path);
  DALI_ENFORCE(file.good(),
               make_string("CocoReader failed to read preprocessed annotation data from ", path));

  unsigned size;
  file.read(reinterpret_cast<char*>(&size), sizeof(unsigned));
  output.resize(size);
  for (size_t i = 0; i < output.size(); ++i) {
    file.read(reinterpret_cast<char*>(&size), sizeof(unsigned));
    output[i].resize(size);
    file.read(reinterpret_cast<char*>(output[i].data()), size * sizeof(T));
  }
}

void LoadFilenamesFromFile(ImageIdPairs &image_id_pairs, const std::string path) {
  std::ifstream file(path);
  DALI_ENFORCE(file.good(),
               make_string("CocoReader failed to read preprocessed annotation data from ", path));

  int id = 0;
  std::string filename;
  while (file >> filename) {
    image_id_pairs.emplace_back(std::move(filename), int{id});
    ++id;
  }
}

void ParseImageInfo(LookaheadParser &parser, std::vector<ImageInfo> &image_infos) {
  RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
  parser.EnterArray();
  while (parser.NextArrayValue()) {
    if (parser.PeekType() != kObjectType) {
      continue;
    }
    parser.EnterObject();
    ImageInfo image_info;
    while (const char* internal_key = parser.NextObjectKey()) {
      if (0 == std::strcmp(internal_key, "id")) {
          image_info.original_id_ = parser.GetInt();
      } else if (0 == std::strcmp(internal_key, "width")) {
          image_info.width_ = parser.GetInt();
      } else if (0 == std::strcmp(internal_key, "height")) {
          image_info.height_ = parser.GetInt();
      } else if (0 == std::strcmp(internal_key, "file_name")) {
          image_info.filename_ = parser.GetString();
      } else {
        parser.SkipValue();
      }
    }
    image_infos.emplace_back(std::move(image_info));
  }
  DALI_ENFORCE(parser.IsValid(), "Error parsing JSON file.");
}

void ParseCategories(LookaheadParser &parser, std::map<int, int> &category_ids) {
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
      if (0 == std::strcmp(internal_key, "id")) {
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

void ParseAnnotations(LookaheadParser &parser, std::vector<Annotation> &annotations,
                      float min_size_threshold, bool ltrb, bool parse_segmentation) {
  RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
  parser.EnterArray();
  while (parser.NextArrayValue()) {
    detail::Annotation annotation;
    if (parser.PeekType() != kObjectType) {
      continue;
    }
    bool to_add = true;
    parser.EnterObject();
    while (const char* internal_key = parser.NextObjectKey()) {
      if (0 == std::strcmp(internal_key, "image_id")) {
        annotation.image_id_ = parser.GetInt();
      } else if (0 == std::strcmp(internal_key, "category_id")) {
        annotation.category_id_ = parser.GetInt();
      } else if (0 == std::strcmp(internal_key, "bbox")) {
        RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
        parser.EnterArray();
        int i = 0;
        while (parser.NextArrayValue()) {
          annotation.box_[i] = parser.GetDouble();
          ++i;
        }
      } else if (parse_segmentation && 0 == std::strcmp(internal_key, "segmentation")) {
        // That means that the mask encoding is not polygons but RLE
        // (iscrowd==1 in Object Detection task, or Stuff Segmentation)
        if (parser.PeekType() != kArrayType) {
          annotation.tag_ = Annotation::RLE;
          parser.EnterObject();
          while (const char* another_key = parser.NextObjectKey()) {
            if (0 == std::strcmp(another_key, "size")) {
              RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
              parser.EnterArray();
              parser.NextArrayValue();
              annotation.rle_.h_ = parser.GetInt();
              parser.NextArrayValue();
              annotation.rle_.w_ = parser.GetInt();
              parser.NextArrayValue();
            } else if (0 == std::strcmp(another_key, "counts")) {
              annotation.rle_.rle_ = parser.GetString();
            }
          }
        } else {
          annotation.tag_ = Annotation::POLYGON;
          int coord_offset = 0;
          auto& segm_meta = annotation.poly_.segm_meta_;
          auto& segm_coords = annotation.poly_.segm_coords_;
          parser.EnterArray();
          while (parser.NextArrayValue()) {
            segm_meta.push_back(coord_offset);
            parser.EnterArray();
            while (parser.NextArrayValue()) {
              segm_coords.push_back(parser.GetDouble());
              coord_offset++;
            }
            segm_meta.push_back(coord_offset);
          }
        }
      } else {
        parser.SkipValue();
      }
    }
    if (!annotation.IsOver(min_size_threshold)) {
      continue;
    }
    if (to_add) {
      if (ltrb) {
        annotation.ToLtrb();
      }
      annotations.emplace_back(std::move(annotation));
    }
  }
}

void ParseJsonFile(const OpSpec &spec, std::vector<detail::ImageInfo> &image_infos,
                   std::vector<detail::Annotation> &annotations,
                   std::map<int, int> &category_ids,
                   bool parse_masks) {
  const auto annotations_file = spec.GetArgument<string>("annotations_file");

  std::ifstream f(annotations_file);
  DALI_ENFORCE(f, "Could not open JSON annotations file: \"" + annotations_file + "\"");
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
  float sz_threshold = spec.GetArgument<float>("size_threshold");
  bool ltrb = spec.GetArgument<bool>("ltrb");
  while (const char* key = parser.NextObjectKey()) {
    if (0 == std::strcmp(key, "images")) {
      detail::ParseImageInfo(parser, image_infos);
    } else if (0 == std::strcmp(key, "categories")) {
      detail::ParseCategories(parser, category_ids);
    } else if (0 == std::strcmp(key, "annotations")) {
      ParseAnnotations(parser, annotations, sz_threshold, ltrb, parse_masks);
    } else {
      parser.SkipValue();
    }
  }
}

}  // namespace detail

void CocoLoader::SavePreprocessedAnnotations(const std::string &path,
                                             const ImageIdPairs &image_id_pairs) {
  using detail::SaveToFile;
  using detail::SaveFilenamesToFile;
  SaveToFile(offsets_, path + "/offsets.dat");
  SaveToFile(boxes_, path + "/boxes.dat");
  SaveToFile(labels_, path + "/labels.dat");
  SaveToFile(counts_, path + "/counts.dat");
  SaveFilenamesToFile(image_id_pairs, path + "/filenames.dat");

  if (output_polygon_masks_) {
    SaveToFile(polygon_data_, path + "/polygon_data.dat");
    SaveToFile(polygon_offset_, path + "/polygon_offset.dat");
    SaveToFile(polygon_count_, path + "/polygons_count.dat");
    SaveToFile(vertices_data_, path + "/vertices.dat");
    SaveToFile(vertices_offset_, path + "/vertices_offset.dat");
    SaveToFile(vertices_count_, path + "/vertices_count.dat");
  }

  if (output_pixelwise_masks_) {
    DALI_WARN("Warning: Saving preprocessed piwelwise masks is not supported");
  }

  if (output_image_ids_) {
    SaveToFile(original_ids_, path + "/original_ids.dat");
  }
}

void CocoLoader::ParsePreprocessedAnnotations() {
  assert(HasPreprocessedAnnotations(spec_));
  const auto path = spec_.HasArgument("meta_files_path")
      ? spec_.GetArgument<string>("meta_files_path")
      : spec_.GetArgument<string>("preprocessed_annotations");
  using detail::LoadFilenamesFromFile;
  using detail::LoadFromFile;
  LoadFromFile(offsets_, path + "/offsets.dat");
  LoadFromFile(boxes_, path + "/boxes.dat");
  LoadFromFile(labels_, path + "/labels.dat");
  LoadFromFile(counts_, path + "/counts.dat");
  LoadFilenamesFromFile(image_label_pairs_, path + "/filenames.dat");

  if (output_polygon_masks_) {
    LoadFromFile(polygon_data_, path + "/polygon_data.dat");
    LoadFromFile(polygon_offset_, path + "/polygon_offset.dat");
    LoadFromFile(polygon_count_, path + "/polygons_count.dat");
    LoadFromFile(vertices_data_, path + "/vertices.dat");
    LoadFromFile(vertices_offset_, path + "/vertices_offset.dat");
    LoadFromFile(vertices_count_, path + "/vertices_count.dat");
  }

  if (output_pixelwise_masks_) {
    DALI_WARN("Loading from preprocessed piwelwise masks is not supported");
  }

  if (output_image_ids_) {
    LoadFromFile(original_ids_, path + "/original_ids.dat");
  }
}

void CocoLoader::ParseJsonAnnotations() {
  std::vector<detail::ImageInfo> image_infos;
  std::vector<detail::Annotation> annotations;
  std::map<int, int> category_ids;

  bool parse_masks = output_polygon_masks_ || output_pixelwise_masks_;
  detail::ParseJsonFile(spec_, image_infos, annotations, category_ids, parse_masks);

  bool skip_empty = spec_.GetArgument<bool>("skip_empty");
  bool ratio = spec_.GetArgument<bool>("ratio");

  std::sort(image_infos.begin(), image_infos.end(), [](auto &left, auto &right) {
    return left.original_id_ < right.original_id_;
  });

  std::stable_sort(annotations.begin(), annotations.end(), [](auto &left, auto &right) {
    return left.image_id_ < right.image_id_;
  });

  detail::Annotation sentinel;
  sentinel.image_id_ = -1;
  annotations.emplace_back(std::move(sentinel));

  int new_image_id = 0;
  int annotation_id = 0;
  int total_count = 0;

  for (auto &image_info : image_infos) {
    int objects_in_sample = 0;
    std::vector<int> sample_rles_idx;
    std::vector<std::string> sample_rles;
    int64_t polygons_sample_offset = polygon_data_.size();
    int64_t polygons_sample_count = 0;
    int64_t vertices_sample_offset = vertices_data_.size();
    int64_t vertices_sample_count = 0;
    while (annotations[annotation_id].image_id_ == image_info.original_id_) {
      const auto &annotation = annotations[annotation_id];
      labels_.push_back(category_ids[annotation.category_id_]);
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
      if (parse_masks) {
        switch (annotation.tag_) {
          case detail::Annotation::POLYGON: {
            auto &segm_meta = annotation.poly_.segm_meta_;
            assert(segm_meta.size() % 2 == 0);
            polygons_sample_count += segm_meta.size() / 2;

            auto &coords = annotation.poly_.segm_coords_;
            assert(coords.size() % 2 == 0);
            vertices_sample_count += coords.size() / 2;

            for (size_t i = 0; i < segm_meta.size(); i += 2) {
              assert(segm_meta[i] % 2 == 0);
              assert(segm_meta[i + 1] % 2 == 0);
              int vertex_start_idx =
                  vertices_data_.size() - vertices_sample_offset + segm_meta[i] / 2;
              int vertex_end_idx =
                  vertices_data_.size() - vertices_sample_offset + segm_meta[i + 1] / 2;
              polygon_data_.push_back({objects_in_sample, vertex_start_idx, vertex_end_idx});
            }
            if (ratio) {
              for (size_t i = 0; i < coords.size(); i += 2) {
                vertices_data_.push_back({coords[i] / image_info.width_,
                                          coords[i + 1] / image_info.height_});
              }
            } else {
              for (size_t i = 0; i < coords.size(); i += 2) {
                vertices_data_.push_back({coords[i], coords[i + 1]});
              }
            }
            break;
          }
          case detail::Annotation::RLE: {
            sample_rles_idx.push_back(objects_in_sample);
            sample_rles.push_back(std::move(annotation.rle_.rle_));
            break;
          }
          default: {
            assert(false);
          }
        }
      }
      ++annotation_id;
      ++objects_in_sample;
    }

    if (!skip_empty || objects_in_sample != 0) {
      offsets_.push_back(total_count);
      counts_.push_back(objects_in_sample);
      total_count += objects_in_sample;
      if (output_image_ids_) {
        original_ids_.push_back(image_info.original_id_);
      }
      if (parse_masks) {
        polygon_offset_.push_back(polygons_sample_offset);
        polygon_count_.push_back(polygons_sample_count);
        vertices_offset_.push_back(vertices_sample_offset);
        vertices_count_.push_back(vertices_sample_count);
        masks_rles_.emplace_back(std::move(sample_rles));
        masks_rles_idx_.emplace_back(std::move(sample_rles_idx));
      }
      if (output_pixelwise_masks_) {
        heights_.push_back(image_info.height_);
        widths_.push_back(image_info.width_);
      }
      image_label_pairs_.emplace_back(std::move(image_info.filename_), new_image_id);
      new_image_id++;
    }
  }

  if (spec_.GetArgument<bool>("save_preprocessed_annotations")) {
    SavePreprocessedAnnotations(
      spec_.GetArgument<std::string>("save_preprocessed_annotations_dir"),
      image_label_pairs_);
  }
}

}  // namespace dali
