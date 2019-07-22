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

#include "dali/pipeline/operators/reader/coco_reader_op.h"

#include <map>
#include <unordered_map>
#include <iomanip>

#include "dali/pipeline/util/lookahead_parser.h"


namespace dali {
DALI_REGISTER_OPERATOR(COCOReader, COCOReader, CPU);
DALI_SCHEMA(COCOReader)
  .NumInput(0)
  .NumOutput(3)
  .DocStr(R"code(Read data from a COCO dataset composed of directory with images
and an annotation files. For each image, with `m` bboxes, returns its bboxes as (m,4)
Tensor (`m` * `[x, y, w, h] or `m` * [left, top, right, bottom]`) and labels as `(m,1)` Tensor (`m` * `category_id`).)code")
  .AddArg("file_root",
      R"code(Path to a directory containing data files.)code",
      DALI_STRING)
  .AddArg("annotations_file",
      R"code(List of paths to the JSON annotations files.)code",
      DALI_STRING_VEC)
  .AddOptionalArg("file_list",
      R"code(Path to the file with a list of pairs ``file label``
(leave empty to traverse the `file_root` directory to obtain files and labels))code",
      std::string())
  .AddOptionalArg("ltrb",
      R"code(If true, bboxes are returned as [left, top, right, bottom], else [x, y, width, height].)code",
      false)
  .AddOptionalArg("ratio",
      R"code(If true, bboxes returned values as expressed as ratio w.r.t. to the image width and height.)code",
      false)
  .AddOptionalArg("size_threshold",
      R"code(If width or height of a bounding box representing an instance of an object is under this value,
object will be skipped during reading. It is represented as absolute value.)code",
      0.1f,
      false)
  .AddOptionalArg("skip_empty",
      R"code(If true, reader will skip samples with no object instances in them)code",
      false)
  .AddOptionalArg("save_img_ids",
      R"code(If true, image IDs will also be returned.)code",
      false)
  .AddOptionalArg("shuffle_after_epoch",
      R"code(If true, reader shuffles whole dataset after each epoch.)code",
      false)
  .AdditionalOutputsFn([](const OpSpec& spec) {
    return static_cast<int>(spec.GetArgument<bool>("save_img_ids"));
  })
  .AddParent("LoaderBase");


DALI_REGISTER_OPERATOR(FastCocoReader, FastCocoReader, CPU);
DALI_SCHEMA(FastCocoReader)
  .NumInput(0)
  .NumOutput(3)
  .DocStr(R"code(Read data from a COCO dataset composed of directory with images
and an annotation files. For each image, with `m` bboxes, returns its bboxes as (m,4)
Tensor (`m` * `[x, y, w, h] or `m` * [left, top, right, bottom]`) and labels as `(m,1)` Tensor (`m` * `category_id`).)code")
  .AddOptionalArg(
    "meta_files_path",
    "Path to directory with meta files containing preprocessed COCO annotations.",
    std::string())
  .AddOptionalArg("annotations_file",
      R"code(List of paths to the JSON annotations files.)code",
      std::string())
  .AddOptionalArg("shuffle_after_epoch",
      R"code(If true, reader shuffles whole dataset after each epoch.)code",
      false)
  .AddArg("file_root",
      R"code(Path to a directory containing data files.)code",
      DALI_STRING)
  .AddOptionalArg("ltrb",
      R"code(If true, bboxes are returned as [left, top, right, bottom], else [x, y, width, height].)code",
      false)
  .AddOptionalArg("skip_empty",
      R"code(If true, reader will skip samples with no object instances in them)code",
      false)
  .AddOptionalArg("size_threshold",
      R"code(If width or height of a bounding box representing an instance of an object is under this value,
object will be skipped during reading. It is represented as absolute value.)code",
      0.1f,
      false)
  .AddOptionalArg("ratio",
      R"code(If true, bboxes returned values as expressed as ratio w.r.t. to the image width and height.)code",
      false)
  .AddOptionalArg("file_list",
      R"code(Path to the file with a list of pairs ``file id``
(leave empty to traverse the `file_root` directory to obtain files and labels))code",
      std::string())
  .AddOptionalArg("save_img_ids",
      R"code(If true, image IDs will also be returned.)code",
      false)
  .AddOptionalArg("dump_meta_files",
      R"code(If true, operator will dump meta files in folder provided with `dump_meta_files_path`.)code",
      false)
  .AddOptionalArg(
    "dump_meta_files_path",
    "Path to directory for saving meta files containing preprocessed COCO annotations.",
    std::string())
  .AdditionalOutputsFn([](const OpSpec& spec) {
    return static_cast<int>(spec.GetArgument<bool>("save_img_ids"));
  })
  .AddParent("LoaderBase");

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

  void ToLtrb() {
    box_[2] += box_[0];
    box_[3] += box_[1];
  }

  bool FitsUnder(float size_threshold) {
    return box_[2] >= size_threshold && box_[3] >= size_threshold;
  }
};

template<typename T>
void dump_meta_file(const std::vector<T> &input, const std::string path) {
  std::ofstream file(path);
  if (file) {
    for (const auto &val : input) {
      file << std::setprecision(9) << val << std::endl;
    }
  } else {
    DALI_FAIL("CocoReader meta file error while saving: " + path);
  }
}

void dump_filenames(const ImageIdPairs &image_id_pairs, const std::string path) {
  std::ofstream file(path);
  if (file) {
    for (const auto &p : image_id_pairs) {
      file << p.first << std::endl;
    }
  } else {
    DALI_FAIL("CocoReader meta file error while saving: " + path);
  }
}

template<typename T>
void load_meta_file(std::vector<T> &output, const std::string path) {
  std::ifstream file(path);
  if (file) {
    T val;
    while (file >> val)
      output.push_back(val);
  } else {
    DALI_FAIL("CocoReader meta file error while loading for path: " + path);
  }
}

void load_file_list(ImageIdPairs &image_id_pairs, const std::string &path) {
  std::ifstream file(path);
  if (file) {
    std::string filename;
    int id;
    while (file >> filename >> id) {
      image_id_pairs.push_back(std::make_pair(filename, id));
    }
  } else {
    DALI_FAIL("CocoReader file list error: " + path);
  }
}

ImageIdPairs load_filenames(const std::string path) {
  ImageIdPairs image_id_pairs;
  int id = 0;
  std::ifstream file(path);
  if (file) {
    std::string filename;
    while (file >> filename) {
      image_id_pairs.emplace_back(std::move(filename), id);
      ++id;
    }
  } else {
     DALI_FAIL("CocoReader meta file error while loading for path: " + path);
  }

  return image_id_pairs;
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
      if (0 == strcmp(internal_key, "id")) {
          image_info.original_id_ = parser.GetInt();
      } else if (0 == strcmp(internal_key, "width")) {
          image_info.width_ = parser.GetInt();
      } else if (0 == strcmp(internal_key, "height")) {
          image_info.height_ = parser.GetInt();
      } else if (0 == strcmp(internal_key, "file_name")) {
          image_info.filename_ = parser.GetString();
      } else {
        parser.SkipValue();
      }
    }
    image_infos.emplace_back(std::move(image_info));
  }
}

void parse_categories(LookaheadParser &parser, std::map<int, int> &category_ids) {
  RAPIDJSON_ASSERT(r.PeekType() == kArrayType);
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
      if (0 == strcmp(internal_key, "id")) {
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
  LookaheadParser &parser, std::vector<Annotation> &annotations, float size_threshold, bool ltrb) {
  RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
  parser.EnterArray();
  while (parser.NextArrayValue()) {
    detail::Annotation annotation;
    if (parser.PeekType() != kObjectType) {
      continue;
    }
    parser.EnterObject();
    while (const char* internal_key = parser.NextObjectKey()) {
      if (0 == strcmp(internal_key, "image_id")) {
        annotation.image_id_ = parser.GetInt();
      } else if (0 == strcmp(internal_key, "category_id")) {
        annotation.category_id_ = parser.GetInt();
      } else if (0 == strcmp(internal_key, "bbox")) {
        RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
        parser.EnterArray();
        int i = 0;
        while (parser.NextArrayValue()) {
          annotation.box_[i] = parser.GetDouble();
          ++i;
        }
      } else {
        parser.SkipValue();
      }
    }
    if (!annotation.FitsUnder(size_threshold)) {
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
    new char[file_size],
    [](char* data) {delete [] data;});
  f.seekg(0, std::ios::beg);
  f.read(buff.get(), file_size);

  detail::LookaheadParser parser(buff.get());

  RAPIDJSON_ASSERT(parser.PeekType() == kObjectType);
  parser.EnterObject();
  while (const char* key = parser.NextObjectKey()) {
    if (0 == strcmp(key, "images")) {
      detail::parse_image_infos(parser, image_infos);
    } else if (0 == strcmp(key, "categories")) {
      detail::parse_categories(parser, category_ids);
    } else if (0 == strcmp(key, "annotations")) {
      parse_annotations(
        parser,
        annotations,
        spec.GetArgument<float>("size_threshold"),
        spec.GetArgument<bool>("ltrb"));
    } else {
      parser.SkipValue();
    }
  }
  f.close();
}

}  // namespace detail


void FastCocoReader::DumpMetaFiles(const std::string path, const ImageIdPairs &image_id_pairs) {
  detail::dump_meta_file(
    offsets_,
    path + "offsets.txt");
  detail::dump_meta_file(
    boxes_,
    path + "boxes.txt");
  detail::dump_meta_file(
    labels_,
    path + "labels.txt");
  detail::dump_meta_file(
    counts_,
    path + "counts.txt");
  detail::dump_filenames(
    image_id_pairs,
    path + "filenames.txt");

  if (save_img_ids_) {
    detail::dump_meta_file(
      original_ids_,
      path + "original_ids.txt");
  }
}

ImageIdPairs FastCocoReader::ParseMetafiles(const OpSpec &spec) {
  const auto meta_files_path = spec.GetArgument<string>("meta_files_path");
  detail::load_meta_file(
    offsets_,
    meta_files_path + "offsets.txt");
  detail::load_meta_file(
    boxes_,
    meta_files_path + "boxes.txt");
  detail::load_meta_file(
    labels_,
    meta_files_path + "labels.txt");
  detail::load_meta_file(
    counts_,
    meta_files_path + "counts.txt");

  if (save_img_ids_) {
    detail::load_meta_file(
      original_ids_,
      meta_files_path + "original_ids.txt");
  }
  return detail::load_filenames(meta_files_path + "filenames.txt");
}

void FastCocoReader::ValidateOptions(const OpSpec &spec) {
  DALI_ENFORCE(
    spec.HasArgument("meta_files_path") || spec.HasArgument("annotations_file"),
    "`meta_files_path` or `annotations_file` must be provided");
  DALI_ENFORCE(
    !spec.HasArgument("file_list"),
    "Argument `file_list` is no longer supported for `COCOReader`."
    "The same functionality can be implemented with meta files option.");
  DALI_ENFORCE(!skip_cached_images_,
    "COCOReader doesn't support `skip_cached_images` option");

  if (spec.HasArgument("meta_files_path")) {
    DALI_ENFORCE(
      !spec.HasArgument("skip_empty"),
      "When reading data from meta files `skip_empty` is not working.");
    DALI_ENFORCE(
      !spec.HasArgument("ratio"),
      "When reading data from meta files `ratio` is not working.");
    DALI_ENFORCE(
      !spec.HasArgument("ltrb"),
      "When reading data from meta files `ltrb` is not working.");
    DALI_ENFORCE(
      !spec.HasArgument("size_threshold"),
      "When reading data from meta files `size_threshold` is not working.");
    DALI_ENFORCE(
      !spec.HasArgument("dump_meta_files"),
      "When reading data from meta files `dump_meta_files` is not working.");
    DALI_ENFORCE(
      !spec.HasArgument("dump_meta_files_path"),
      "When reading data from meta files `dump_meta_files_path` is not working.");
  }

  if (spec.HasArgument("dump_meta_files")) {
    DALI_ENFORCE(
      spec.HasArgument("dump_meta_files_path"),
      "When dumping meta files `dump_meta_files_path` must be provided.");
  }
}

std::vector<std::pair<std::string, int>> FastCocoReader::ParseJsonAnnotations(const OpSpec &spec) {
  std::vector<detail::ImageInfo> image_infos;
  std::vector<detail::Annotation> annotations;
  std::map<int, int> category_ids;

  detail::parse_json_file(
    spec,
    image_infos,
    annotations,
    category_ids);

  bool skip_empty = spec.GetArgument<bool>("skip_empty");
  bool ratio = spec.GetArgument<bool>("ratio");

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

  ImageIdPairs image_id_pairs;

  for (auto &image_info : image_infos) {
    int objects_in_sample = 0;
    while (annotations[annotation_id].image_id_ == image_info.original_id_) {
      const auto &annotation = annotations[annotation_id];
      labels_.emplace_back(category_ids[annotation.category_id_]);
      if (ratio) {
        boxes_.emplace_back(annotation.box_[0] / static_cast<float>(image_info.width_));
        boxes_.emplace_back(annotation.box_[1] / static_cast<float>(image_info.height_));
        boxes_.emplace_back(annotation.box_[2] / static_cast<float>(image_info.width_));
        boxes_.emplace_back(annotation.box_[3] / static_cast<float>(image_info.height_));
      } else {
        boxes_.emplace_back(annotation.box_[0]);
        boxes_.emplace_back(annotation.box_[1]);
        boxes_.emplace_back(annotation.box_[2]);
        boxes_.emplace_back(annotation.box_[3]);
      }
      ++annotation_id;
      ++objects_in_sample;
    }

    if (!skip_empty || objects_in_sample != 0) {
      offsets_.emplace_back(total_count);
      counts_.emplace_back(objects_in_sample);
      total_count += objects_in_sample;
      if (save_img_ids_) {
        original_ids_.emplace_back(image_info.original_id_);
      }

      image_id_pairs.emplace_back(std::move(image_info.filename_), new_image_id);
      new_image_id++;
    }
  }

  if (spec.GetArgument<bool>("dump_meta_files")) {
    DumpMetaFiles(
      spec.GetArgument<std::string>("dump_meta_files_path"),
      image_id_pairs);
  }

  return image_id_pairs;
}


}  // namespace dali
