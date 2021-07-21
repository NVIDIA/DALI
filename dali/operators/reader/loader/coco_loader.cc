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

#include <list>
#include <map>
#include <unordered_map>
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

struct Annotation {
  enum {POLYGON, RLE} tag_;
  int image_id_;
  int category_id_;
  std::array<float, 4> box_;
  // union
  Polygons poly_;
  RLEMaskPtr rle_;

  void ToLtrb() {
    box_[2] += box_[0];
    box_[3] += box_[1];
  }

  bool IsOver(float min_size_threshold) {
    return box_[2] >= min_size_threshold && box_[3] >= min_size_threshold;
  }
};

template <typename T>
std::enable_if_t<std::is_pod<T>::value, void>
Read(std::ifstream& file, T& data, const char* filename) {
  int64_t bytes = sizeof(T);
  file.read(reinterpret_cast<char *>(&data), bytes);
  DALI_ENFORCE(file.gcount() == bytes,
               make_string("Error reading from path: ", filename, ". Read ", file.gcount(),
                           " bytes but requested ", bytes, " bytes."));
}

template <typename T>
void Read(std::ifstream& file, span<T> data, const char* filename) {
  if (data.empty())
    return;

  int64_t bytes = sizeof(T) * data.size();
  file.read(reinterpret_cast<char *>(data.data()), bytes);
  DALI_ENFORCE(file.gcount() == bytes,
               make_string("Error reading from path: ", filename, ". Read ", file.gcount(),
                           " bytes but requested ", bytes, " bytes."));
}

template <typename T>
void Write(std::ofstream& file, T data, const char* filename) {
  file.write(reinterpret_cast<const char*>(&data), sizeof(T));
  DALI_ENFORCE(file.good(), make_string("Error reading from path: ", filename));
}

template <typename T>
void Write(std::ofstream& file, span<const T> data, const char* filename) {
  if (data.empty())
    return;
  file.write(reinterpret_cast<const char*>(data.data()), sizeof(T) * data.size());
  DALI_ENFORCE(file.good(), make_string("Error reading from path: ", filename));
}

template <typename T>
void SaveToFile(const std::vector<T> &input, const std::string path) {
  if (input.empty())
    return;
  std::ofstream file(path, std::ios_base::binary | std::ios_base::out);
  DALI_ENFORCE(file, "CocoReader meta file error while saving: " + path);

  unsigned size = input.size();
  Write(file, size, path.c_str());
  Write(file, make_cspan(input), path.c_str());
  DALI_ENFORCE(file.good(), make_string("Error writing to path: ", path));
}

template <>
void SaveToFile(const std::vector<RLEMaskPtr> &input, const std::string path) {
  if (input.empty())
    return;
  std::ofstream file(path, std::ios_base::binary | std::ios_base::out);
  DALI_ENFORCE(file, "CocoReader meta file error while saving: " + path);

  unsigned size = input.size();
  Write(file, size, path.c_str());
  for (auto &rle : input) {
    assert((*rle)->h > 0 && (*rle)->w > 0 && (*rle)->m > 0);
    siz dims[3] = {(*rle)->h, (*rle)->w, (*rle)->m};
    Write(file, span<const siz>{&dims[0], 3}, path.c_str());
    Write(file, span<const uint>{(*rle)->cnts, static_cast<ptrdiff_t>((*rle)->m)}, path.c_str());
  }
}

template <typename T>
void SaveToFile(const std::vector<std::vector<T> > &input, const std::string path) {
  if (input.empty())
    return;
  std::ofstream file(path, std::ios_base::binary | std::ios_base::out);
  DALI_ENFORCE(file, "CocoReader meta file error while saving: " + path);

  unsigned size = input.size();
  Write(file, size, path.c_str());

  for (auto& v : input) {
    size = v.size();
    assert(size > 0);
    Write(file, size, path.c_str());
    Write(file, make_cspan(v), path.c_str());
  }
}

template <>
void SaveToFile(const ImageIdPairs &image_id_pairs, const std::string path) {
  if (image_id_pairs.empty())
    return;
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
  output.clear();
  if (!file.good())
    return;

  unsigned size;
  Read(file, size, path.c_str());
  output.resize(size);
  Read(file, make_span(output), path.c_str());
}

template <>
void LoadFromFile(std::vector<RLEMaskPtr> &output, const std::string path) {
  std::ifstream file(path);
  output.clear();
  if (!file.good())
    return;

  unsigned size;
  Read(file, size, path.c_str());
  output.clear();
  output.resize(size);
  for (auto &rle : output) {
    siz dims[3];
    Read(file, span<siz>{&dims[0], 3}, path.c_str());
    siz h = dims[0], w = dims[1], m = dims[2];
    rle = std::make_shared<RLEMask>(h, w, m);
    Read(file, span<uint>{(*rle)->cnts, static_cast<ptrdiff_t>((*rle)->m)}, path.c_str());
  }
}

template <typename T>
void LoadFromFile(std::vector<std::vector<T> > &output, const std::string path) {
  std::ifstream file(path);
  output.clear();
  if (!file.good())
    return;

  unsigned size;
  Read(file, size, path.c_str());
  output.resize(size);
  for (size_t i = 0; i < output.size(); ++i) {
    Read(file, size, path.c_str());
    output[i].resize(size);
    Read(file, make_span(output[i]), path.c_str());
  }
}

template <>
void LoadFromFile(ImageIdPairs &image_id_pairs, const std::string path) {
  std::ifstream file(path);
  image_id_pairs.clear();
  if (!file.good())
    return;

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
    category_ids.emplace(id, new_id);
    new_id++;
  }
}

void ParseAnnotations(LookaheadParser &parser, std::vector<Annotation> &annotations,
                      float min_size_threshold, bool ltrb,
                      bool parse_segmentation, bool parse_rle) {
  std::string rle_str;
  std::vector<uint32_t> rle_uints;
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
        if (parse_rle && parser.PeekType() == kObjectType) {
          annotation.tag_ = Annotation::RLE;
          parser.EnterObject();
          rle_str.clear();
          rle_uints.clear();
          int h = -1, w = -1;
          while (const char* another_key = parser.NextObjectKey()) {
            if (0 == std::strcmp(another_key, "size")) {
              RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
              parser.EnterArray();
              parser.NextArrayValue();
              h = parser.GetInt();
              parser.NextArrayValue();
              w = parser.GetInt();
              parser.NextArrayValue();
            } else if (0 == std::strcmp(another_key, "counts")) {
              if (parser.PeekType() == kStringType) {
                rle_str = parser.GetString();
              } else if (parser.PeekType() == kArrayType) {
                parser.EnterArray();
                while (parser.NextArrayValue()) {
                  rle_uints.push_back(parser.GetInt());
                }
              } else {
                parser.SkipValue();
              }
            } else {
              parser.SkipValue();
            }
          }
          DALI_ENFORCE(h > 0 && w > 0, "Invalid or missing mask sizes");
          if (!rle_str.empty()) {
            annotation.rle_ = std::make_shared<RLEMask>(h, w, rle_str.c_str());
          } else if (!rle_uints.empty()) {
            annotation.rle_ = std::make_shared<RLEMask>(h, w, make_cspan(rle_uints));
          } else {
            DALI_FAIL("Missing or invalid ``counts`` attribute.");
          }
        } else if (parser.PeekType() == kArrayType) {
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
        } else {
          parser.SkipValue();
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
                   bool parse_segmentation, bool parse_rle) {
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
      ParseAnnotations(parser, annotations, sz_threshold, ltrb, parse_segmentation, parse_rle);
    } else {
      parser.SkipValue();
    }
  }
}

}  // namespace detail

void CocoLoader::SavePreprocessedAnnotations(const std::string &path,
                                             const ImageIdPairs &image_id_pairs) {
  using detail::SaveToFile;
  SaveToFile(offsets_, path + "/offsets.dat");
  SaveToFile(boxes_, path + "/boxes.dat");
  SaveToFile(labels_, path + "/labels.dat");
  SaveToFile(counts_, path + "/counts.dat");
  SaveToFile(image_id_pairs, path + "/filenames.dat");

  if (output_polygon_masks_ || output_pixelwise_masks_) {
    SaveToFile(polygon_data_, path + "/polygon_data.dat");
    SaveToFile(polygon_offset_, path + "/polygon_offset.dat");
    SaveToFile(polygon_count_, path + "/polygons_count.dat");
    SaveToFile(vertices_data_, path + "/vertices.dat");
    SaveToFile(vertices_offset_, path + "/vertices_offset.dat");
    SaveToFile(vertices_count_, path + "/vertices_count.dat");
  }

  if (output_pixelwise_masks_) {
    SaveToFile(masks_rles_, path + "/masks_rles.dat");
    SaveToFile(masks_rles_idx_, path + "/masks_rles_idx.dat");
    SaveToFile(mask_offsets_, path + "/masks_offset.dat");
    SaveToFile(mask_counts_, path + "/mask_count.dat");
    SaveToFile(heights_, path + "/heights.dat");
    SaveToFile(widths_, path + "/widths.dat");
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
  using detail::LoadFromFile;
  LoadFromFile(offsets_, path + "/offsets.dat");
  LoadFromFile(boxes_, path + "/boxes.dat");
  LoadFromFile(labels_, path + "/labels.dat");
  LoadFromFile(counts_, path + "/counts.dat");
  LoadFromFile(image_label_pairs_, path + "/filenames.dat");

  if (output_polygon_masks_ || output_pixelwise_masks_) {
    LoadFromFile(polygon_data_, path + "/polygon_data.dat");
    LoadFromFile(polygon_offset_, path + "/polygon_offset.dat");
    LoadFromFile(polygon_count_, path + "/polygons_count.dat");
    LoadFromFile(vertices_data_, path + "/vertices.dat");
    LoadFromFile(vertices_offset_, path + "/vertices_offset.dat");
    LoadFromFile(vertices_count_, path + "/vertices_count.dat");
  }

  if (output_pixelwise_masks_) {
    LoadFromFile(masks_rles_, path + "/masks_rles.dat");
    LoadFromFile(masks_rles_idx_, path + "/masks_rles_idx.dat");
    LoadFromFile(mask_offsets_, path + "/masks_offset.dat");
    LoadFromFile(mask_counts_, path + "/mask_count.dat");
    LoadFromFile(heights_, path + "/heights.dat");
    LoadFromFile(widths_, path + "/widths.dat");
  }

  if (output_image_ids_) {
    LoadFromFile(original_ids_, path + "/original_ids.dat");
  }
}

void CocoLoader::ParseJsonAnnotations() {
  std::vector<detail::ImageInfo> image_infos;
  std::vector<detail::Annotation> annotations;
  std::map<int, int> category_ids;

  bool parse_segmentation = output_polygon_masks_ || output_pixelwise_masks_;
  detail::ParseJsonFile(spec_, image_infos, annotations, category_ids,
                        parse_segmentation, output_pixelwise_masks_);

  if (images_.empty()) {
    std::sort(image_infos.begin(), image_infos.end(), [&](auto &left, auto &right) {
      return left.original_id_ < right.original_id_;
    });
    for (auto &info : image_infos) {
      images_.push_back(info.filename_);
    }
  }

  std::unordered_map<std::string, const detail::ImageInfo*> img_infos_map;
  std::unordered_map<int, std::list<const detail::Annotation*>> img_annotations_map;
  img_infos_map.reserve(images_.size());
  img_annotations_map.reserve(images_.size());
  for (const auto &filename : images_) {
    img_infos_map[filename] = nullptr;
  }
  for (auto &info : image_infos) {
    auto it = img_infos_map.find(info.filename_);
    if (it != img_infos_map.end()) {
      it->second = &info;
      img_annotations_map[info.original_id_] = {};
    }
  }

  for (const auto &annotation : annotations) {
    auto it = img_annotations_map.find(annotation.image_id_);
    if (it != img_annotations_map.end()) {
      it->second.push_back(&annotation);
    }
  }

  bool skip_empty = spec_.GetArgument<bool>("skip_empty");
  bool ratio = spec_.GetArgument<bool>("ratio");

  int new_image_id = 0;
  int annotation_id = 0;
  int total_count = 0;

  for (auto &img_filename : images_) {
    auto img_info_ptr = img_infos_map[img_filename];
    if (!img_info_ptr)
      continue;
    const auto &image_info = *img_info_ptr;
    auto image_id = image_info.original_id_;
    int objects_in_sample = 0;
    int64_t sample_polygons_offset = polygon_data_.size();
    int64_t sample_polygons_count = 0;
    int64_t sample_vertices_offset = vertices_data_.size();
    int64_t sample_vertices_count = 0;
    int64_t mask_offset = masks_rles_.size();
    int64_t mask_count = 0;
    for (const auto* annotation_ptr : img_annotations_map[image_id]) {
      const auto &annotation = *annotation_ptr;
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
      if (parse_segmentation) {
        switch (annotation.tag_) {
          case detail::Annotation::POLYGON: {
            auto &segm_meta = annotation.poly_.segm_meta_;
            assert(segm_meta.size() % 2 == 0);
            sample_polygons_count += segm_meta.size() / 2;

            auto &coords = annotation.poly_.segm_coords_;
            assert(coords.size() % 2 == 0);
            sample_vertices_count += coords.size() / 2;

            for (size_t i = 0; i < segm_meta.size(); i += 2) {
              assert(segm_meta[i] % 2 == 0);
              assert(segm_meta[i + 1] % 2 == 0);
              int vertex_start_idx =
                  vertices_data_.size() - sample_vertices_offset + segm_meta[i] / 2;
              int vertex_end_idx =
                  vertices_data_.size() - sample_vertices_offset + segm_meta[i + 1] / 2;
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
            masks_rles_idx_.push_back(objects_in_sample);
            masks_rles_.push_back(annotation.rle_);
            mask_count++;
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
      if (parse_segmentation) {
        polygon_offset_.push_back(sample_polygons_offset);
        polygon_count_.push_back(sample_polygons_count);
        vertices_offset_.push_back(sample_vertices_offset);
        vertices_count_.push_back(sample_vertices_count);
        if (output_pixelwise_masks_) {
          mask_offsets_.push_back(mask_offset);
          mask_counts_.push_back(mask_count);
          heights_.push_back(image_info.height_);
          widths_.push_back(image_info.width_);
        }
      }

      image_label_pairs_.emplace_back(std::move(image_info.filename_), new_image_id);
      new_image_id++;
    }
  }

  // we don't need the list anymore and it can contain a lot of strings
  images_.clear();

  if (spec_.GetArgument<bool>("save_preprocessed_annotations")) {
    SavePreprocessedAnnotations(
      spec_.GetArgument<std::string>("save_preprocessed_annotations_dir"),
      image_label_pairs_);
  }
}

}  // namespace dali
