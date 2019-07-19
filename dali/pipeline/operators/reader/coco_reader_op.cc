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
      image_id_pairs.push_back(std::make_pair(filename, id));
      ++id;
    }
  } else {
     DALI_FAIL("CocoReader meta file error while loading for path: " + path);
  }

  return image_id_pairs;
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
    DALI_ENFORCE(
      !spec.HasArgument("file_list"),
      "When reading data from meta files `file_list` is not working.");
  }

  if (spec.HasArgument("dump_meta_files")) {
    DALI_ENFORCE(
      spec.HasArgument("dump_meta_files_path"),
      "When dumping meta files `dump_meta_files_path` must be provided.");
  }

  DALI_ENFORCE(!skip_cached_images_,
    "COCOReader doesn't support `skip_cached_images` option");
}

using rapidjson::kObjectType;

std::vector<std::pair<std::string, int>> FastCocoReader::ParseJsonAnnotations(const OpSpec &spec) {
  std::vector<std::pair<std::string, int>> image_id_pairs;
  const auto annotations_file_path = spec.GetArgument<string>("annotations_file");
  bool ltrb = spec.GetArgument<bool>("ltrb");
  bool skip_empty = spec.GetArgument<bool>("skip_empty");
  float size_threshold = spec.GetArgument<float>("size_threshold");
  bool ratio = spec.GetArgument<bool>("ratio");
  string file_list = spec.GetArgument<string>("file_list");
  bool parse_file_list = file_list != "";


  std::ifstream f(annotations_file_path);
  DALI_ENFORCE(f, "Could not open JSON annotations file");
  f.seekg(0, std::ios::end);
  size_t file_size = f.tellg();
  std::unique_ptr<char, std::function<void(char*)>> buff(
    new char[file_size],
    [](char* data) {delete [] data;});
  f.seekg(0, std::ios::beg);
  f.read(buff.get(), file_size);

  detail::LookaheadParser r(buff.get());

  // mapping each image_id to its WH dimension
  std::unordered_map<int, std::pair<int, int> > image_id_to_wh;


  std::unordered_map<int, std::vector<int>> labels_map;
  std::unordered_map<int, std::vector<std::array<float, 4>>> boxes_map;

  std::vector<std::pair<int, std::pair<std::array<float, 4>, int>>> boxes_vector;

  // mapping each category_id to its actual category
  std::map<int, int> category_ids;


  if (parse_file_list) {
    detail::load_file_list(image_id_pairs, file_list);
  }

  RAPIDJSON_ASSERT(r.PeekType() == kObjectType);
  r.EnterObject();
  while (const char* key = r.NextObjectKey()) {
    if (0 == strcmp(key, "images")) {
        RAPIDJSON_ASSERT(r.PeekType() == kArrayType);
        r.EnterArray();
        string image_file_name;
        int width;
        int height;
        int id = 0;
        while (r.NextArrayValue()) {
          if (r.PeekType() != kObjectType) {
            continue;
          }
          r.EnterObject();
          while (const char* internal_key = r.NextObjectKey()) {
            if (0 == strcmp(internal_key, "id")) {
                id = r.GetInt();
            } else if (0 == strcmp(internal_key, "width")) {
                width = r.GetInt();
            } else if (0 == strcmp(internal_key, "height")) {
                height = r.GetInt();
            } else if (0 == strcmp(internal_key, "file_name")) {
                image_file_name = r.GetString();
            } else {
              r.SkipValue();
            }
          }
          if (!parse_file_list) {
            image_id_pairs.push_back(std::make_pair(image_file_name, id));
          }
          image_id_to_wh.insert(std::make_pair(id, std::make_pair(width, height)));
        }
      } else if (0 == strcmp(key, "categories")) { 
        detail::parse_categories(r, category_ids);
      } else if (0 == strcmp(key, "annotations")) {
        RAPIDJSON_ASSERT(r.PeekType() == kArrayType);
        r.EnterArray();
        int image_id;
        int category_id;
        std::array<float, 4> bbox = {0, };
        while (r.NextArrayValue()) {
          if (r.PeekType() != kObjectType) {
            continue;
          }
          r.EnterObject();
          while (const char* internal_key = r.NextObjectKey()) {
            if (0 == strcmp(internal_key, "image_id")) {
              image_id = r.GetInt();
            } else if (0 == strcmp(internal_key, "category_id")) {
              category_id = r.GetInt();
            } else if (0 == strcmp(internal_key, "bbox")) {
              RAPIDJSON_ASSERT(r.PeekType() == kArrayType);
              r.EnterArray();
              int i = 0;
              while (r.NextArrayValue()) {
                bbox[i] = r.GetDouble();
                ++i;
              }
            } else {
              r.SkipValue();
            }
          }

          if (bbox[2] < size_threshold || bbox[3] < size_threshold) {
            continue;
          }

          if (ltrb) {
            bbox[2] += bbox[0];
            bbox[3] += bbox[1];
          }

          labels_map[image_id].push_back(category_id);
          boxes_map[image_id].push_back(bbox);

          boxes_vector.emplace_back(std::make_pair(image_id, std::make_pair(bbox, category_id)));

        }
      } else {
        r.SkipValue();
      }
  }

  f.close();

  // ==============================================
  int total_count = 0;
  std::vector<std::pair<std::string, int>> image_id_pairs_2;
  std::vector<int> original_ids_2;
  int non_empty_id = 0;

  std::sort(image_id_pairs.begin(), image_id_pairs.end(), [](auto &left, auto &right) {
    return left.second < right.second;
  });
  std::sort(boxes_vector.begin(), boxes_vector.end(), [](auto &left, auto &right) {
    return left.first < right.first;
  });

  for (int i = 0; i < image_id_pairs.size(); ++i) {
    int id = image_id_pairs[i].second;

    if (save_img_ids_) {
      original_ids_.push_back(id);
    }

    image_id_pairs[i].second = i;
    
    for (int c : labels_map[id]) {
      labels_.push_back(category_ids[c]);
    }

    for (std::array<float, 4> &b : boxes_map[id]) {

      if (ratio) {
        const auto& wh = image_id_to_wh[id];
        boxes_.push_back(b[0] /= static_cast<float>(wh.first));
        boxes_.push_back(b[1] /= static_cast<float>(wh.second));
        boxes_.push_back(b[2] /= static_cast<float>(wh.first));
        boxes_.push_back(b[3] /= static_cast<float>(wh.second));

      } else {
        boxes_.push_back(b[0]);
        boxes_.push_back(b[1]);
        boxes_.push_back(b[2]);
        boxes_.push_back(b[3]);
      }
    }

    if (!skip_empty) {
      offsets_.push_back(total_count);
      counts_.push_back(labels_map[id].size());
      total_count += labels_map[id].size();
    } else {
      if (labels_map[id].size() != 0) {
        offsets_.push_back(total_count);
        counts_.push_back(labels_map[id].size());
        total_count += labels_map[id].size();
        if (save_img_ids_)
          original_ids_2.push_back(id);
        image_id_pairs_2.push_back(std::make_pair(image_id_pairs[i].first, non_empty_id));
        non_empty_id++;
      }

    }
  }

  if (skip_empty) {
    image_id_pairs = image_id_pairs_2;
    original_ids_ = original_ids_2;
  }

  if (spec.GetArgument<bool>("dump_meta_files")) {
    DumpMetaFiles(
      spec.GetArgument<std::string>("dump_meta_files_path"),
      image_id_pairs);
  }

  return image_id_pairs;
}


}  // namespace dali
