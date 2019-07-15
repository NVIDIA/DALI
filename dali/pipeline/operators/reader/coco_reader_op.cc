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

#include <rapidjson/reader.h>
#include <rapidjson/document.h>

#include <map>
#include <unordered_map>

RAPIDJSON_DIAG_PUSH
#ifdef __GNUC__
RAPIDJSON_DIAG_OFF(effc++)
#endif

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
  .AddOptionalArg("meta_files_path", "Path to directory with boxes and labels meta files",
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
  .AdditionalOutputsFn([](const OpSpec& spec) {
    return static_cast<int>(spec.GetArgument<bool>("save_img_ids"));
  })
  .AddParent("LoaderBase");


std::vector<std::pair<std::string, int>> FastCocoReader::ParseMetafiles(const OpSpec &spec) {
  const auto meta_files_path = spec.GetArgument<string>("meta_files_path");
  load_vector_from_file(
    offsets_,
    meta_files_path + "offsets.txt");
  load_vector_from_file(
    boxes_,
    meta_files_path + "boxes.txt");
  load_vector_from_file(
    labels_,
    meta_files_path + "labels.txt");
  load_vector_from_file(
    counts_,
    meta_files_path + "counts.txt");

  std::vector<std::pair<std::string, int>> image_id_pairs;
  int id = 0;
  std::ifstream file(meta_files_path + "filenames.txt");
  if (file) {
    std::string filename;
    while (file >> filename) {
      image_id_pairs.push_back(std::make_pair(filename, id));
      ++id;
    }
  } else {
    DALI_FAIL("TFRecord meta file error: " + meta_files_path + "filenames.txt");
  }

  return image_id_pairs;
}


using rapidjson::SizeType;
using rapidjson::Value;
using rapidjson::Reader;
using rapidjson::InsituStringStream;
using rapidjson::kParseDefaultFlags;
using rapidjson::kParseInsituFlag;
using rapidjson::kArrayType;
using rapidjson::kObjectType;

namespace {

// taken from https://github.com/Tencent/rapidjson/blob/master/example/lookaheadparser/lookaheadparser.cpp

class LookaheadParserHandler {
 public:
  bool Null() { st_ = kHasNull; v_.SetNull(); return true; }
  bool Bool(bool b) { st_ = kHasBool; v_.SetBool(b); return true; }
  bool Int(int i) { st_ = kHasNumber; v_.SetInt(i); return true; }
  bool Uint(unsigned u) { st_ = kHasNumber; v_.SetUint(u); return true; }
  bool Int64(int64_t i) { st_ = kHasNumber; v_.SetInt64(i); return true; }
  bool Uint64(uint64_t u) { st_ = kHasNumber; v_.SetUint64(u); return true; }
  bool Double(double d) { st_ = kHasNumber; v_.SetDouble(d); return true; }
  bool RawNumber(const char*, SizeType, bool) { return false; }
  bool String(const char* str, SizeType length, bool) {
    st_ = kHasString;
    v_.SetString(str, length);
    return true;
  }
  bool StartObject() { st_ = kEnteringObject; return true; }
  bool Key(const char* str, SizeType length, bool) {
    st_ = kHasKey;
    v_.SetString(str, length);
    return true;
  }
  bool EndObject(SizeType) { st_ = kExitingObject; return true; }
  bool StartArray() { st_ = kEnteringArray; return true; }
  bool EndArray(SizeType) { st_ = kExitingArray; return true; }

 protected:
  explicit LookaheadParserHandler(char* str);
  void ParseNext();

 protected:
  enum LookaheadParsingState {
    kInit,
    kError,
    kHasNull,
    kHasBool,
    kHasNumber,
    kHasString,
    kHasKey,
    kEnteringObject,
    kExitingObject,
    kEnteringArray,
    kExitingArray
  };

  Value v_;
  LookaheadParsingState st_;
  Reader r_;
  InsituStringStream ss_;

  static const int parseFlags = kParseDefaultFlags | kParseInsituFlag;
};

LookaheadParserHandler::LookaheadParserHandler(char* str) :
    v_(), st_(kInit), r_(), ss_(str) {
  r_.IterativeParseInit();
  ParseNext();
}

void LookaheadParserHandler::ParseNext() {
  if (r_.HasParseError()) {
    st_ = kError;
    return;
  }

  r_.IterativeParseNext<parseFlags>(ss_, *this);
}

class LookaheadParser : protected LookaheadParserHandler {
 public:
  explicit LookaheadParser(char* str) : LookaheadParserHandler(str) {}

  bool EnterObject();
  bool EnterArray();
  const char* NextObjectKey();
  bool NextArrayValue();
  int GetInt();
  double GetDouble();
  const char* GetString();
  bool GetBool();
  void GetNull();

  void SkipObject();
  void SkipArray();
  void SkipValue();
  Value* PeekValue();
  // returns a rapidjson::Type, or -1 for no value (at end of object/array)
  int PeekType();

  bool IsValid() { return st_ != kError; }

 protected:
  void SkipOut(int depth);
};

bool LookaheadParser::EnterObject() {
  if (st_ != kEnteringObject) {
    st_  = kError;
    return false;
  }

  ParseNext();
  return true;
}

bool LookaheadParser::EnterArray() {
  if (st_ != kEnteringArray) {
    st_  = kError;
    return false;
  }

  ParseNext();
  return true;
}

const char* LookaheadParser::NextObjectKey() {
  if (st_ == kHasKey) {
    const char* result = v_.GetString();
    ParseNext();
    return result;
  }

  if (st_ != kExitingObject) {
    st_ = kError;
    return 0;
  }

  ParseNext();
  return 0;
}

bool LookaheadParser::NextArrayValue() {
  if (st_ == kExitingArray) {
    ParseNext();
    return false;
  }

  if (st_ == kError || st_ == kExitingObject || st_ == kHasKey) {
    st_ = kError;
    return false;
  }

  return true;
}

int LookaheadParser::GetInt() {
  if (st_ != kHasNumber || !v_.IsInt()) {
    st_ = kError;
    return 0;
  }

  int result = v_.GetInt();
  ParseNext();
  return result;
}

double LookaheadParser::GetDouble() {
  if (st_ != kHasNumber) {
    st_  = kError;
    return 0.;
  }

  double result = v_.GetDouble();
  ParseNext();
  return result;
}

bool LookaheadParser::GetBool() {
  if (st_ != kHasBool) {
    st_  = kError;
    return false;
  }

  bool result = v_.GetBool();
  ParseNext();
  return result;
}

void LookaheadParser::GetNull() {
  if (st_ != kHasNull) {
    st_  = kError;
    return;
  }

  ParseNext();
}

const char* LookaheadParser::GetString() {
  if (st_ != kHasString) {
    st_  = kError;
    return 0;
  }

  const char* result = v_.GetString();
  ParseNext();
  return result;
}

void LookaheadParser::SkipOut(int depth) {
  do {
    if (st_ == kEnteringArray || st_ == kEnteringObject) {
      ++depth;
    } else if (st_ == kExitingArray || st_ == kExitingObject) {
      --depth;
    } else if (st_ == kError) {
      return;
    }

    ParseNext();
  } while (depth > 0);
}

void LookaheadParser::SkipValue() {
  SkipOut(0);
}

void LookaheadParser::SkipArray() {
  SkipOut(1);
}

void LookaheadParser::SkipObject() {
  SkipOut(1);
}

Value* LookaheadParser::PeekValue() {
  if (st_ >= kHasNull && st_ <= kHasKey) {
    return &v_;
  }

  return 0;
}

int LookaheadParser::PeekType() {
  if (st_ >= kHasNull && st_ <= kHasKey) {
    return v_.GetType();
  }

  if (st_ == kEnteringArray) {
    return kArrayType;
  }

  if (st_ == kEnteringObject) {
    return kObjectType;
  }

  return -1;
}

}  // namespace



std::vector<std::pair<std::string, int>> FastCocoReader::ParseJsonAnnotations(const OpSpec &spec) {
  std::vector<std::pair<std::string, int>> image_id_pairs;
  const auto annotations_file_path = spec.GetArgument<string>("annotations_file");
  bool ltrb = spec.GetArgument<bool>("ltrb");
  bool skip_empty = spec.GetArgument<bool>("skip_empty");
  float size_threshold = spec.GetArgument<float>("size_threshold");
  bool ratio = spec.GetArgument<bool>("ratio");
  std::ifstream f(annotations_file_path);
  DALI_ENFORCE(f, "Could not open JSON annotations file");
  f.seekg(0, std::ios::end);
  size_t file_size = f.tellg();
  std::unique_ptr<char, std::function<void(char*)>> buff(new char[file_size],
                        [](char* data) {delete [] data;});
  f.seekg(0, std::ios::beg);
  f.read(buff.get(), file_size);

  LookaheadParser r(buff.get());

  // mapping each image_id to its WH dimension
  std::unordered_map<int, std::pair<int, int> > image_id_to_wh;


  std::unordered_map<int, std::vector<int>> labels_map;
  std::unordered_map<int, std::vector<std::array<float, 4>>> boxes_map;

  // mapping each category_id to its actual category
  std::map<int, int> category_ids;
  int current_id = 1;

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
          image_id_pairs.push_back(std::make_pair(image_file_name, id));
          image_id_to_wh.insert(std::make_pair(id, std::make_pair(width, height)));
        }
      } else if (0 == strcmp(key, "categories")) { 
        RAPIDJSON_ASSERT(r.PeekType() == kArrayType);
        r.EnterArray();
        int id;
        while (r.NextArrayValue()) {
          if (r.PeekType() != kObjectType) {
            continue;
          }
          id = -1;
          r.EnterObject();
          while (const char* internal_key = r.NextObjectKey()) {
            if (0 == strcmp(internal_key, "id")) {
              id = r.GetInt();
            } else {
              r.SkipValue();
            }
          }
          DALI_ENFORCE(id != -1, "Missing category ID in the JSON annotations file");
          category_ids.insert(std::make_pair(id, current_id));
          current_id++;
        }
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

        }
      } else {
        r.SkipValue();
      }
  }

  f.close();

  // ==============================================
  int total_count = 0;
  for (int i = 0; i < image_id_pairs.size(); ++i) {
    int id = image_id_pairs[i].second;
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


    offsets_.push_back(total_count);
    counts_.push_back(labels_map[id].size());
    total_count += labels_map[id].size();
  }

  if (skip_empty) {
    std::vector<std::pair<std::string, int>> image_id_pairs_2;
    for (int i = 0; i < counts_.size(); ++i) {
      if (counts_[i] != 0) {
        image_id_pairs_2.push_back(image_id_pairs[i]);
      }
    }
    return image_id_pairs_2;
  }


  return image_id_pairs;
}


}  // namespace dali
