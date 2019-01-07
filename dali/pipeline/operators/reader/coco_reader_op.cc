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

#include "dali/util/rapidjson/reader.h"
#include "dali/util/rapidjson/document.h"

RAPIDJSON_DIAG_PUSH
#ifdef __GNUC__
RAPIDJSON_DIAG_OFF(effc++)
#endif

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

namespace dali {

DALI_REGISTER_OPERATOR(COCOReader, COCOReader, CPU);
DALI_SCHEMA(COCOReader)
  .NumInput(0)
  .NumOutput(3)
  .DocStr(R"code(Read data from a COCO dataset composed of directory with images
and an anotation files. For each image, with `m` bboxes, returns its bboxes as (m,4)
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
      R"code(If true, bboxes are returned as [left, top, right, bottom], else [x, y, width, height]. Default: False)code",
      false)
  .AddOptionalArg("ratio",
      R"code(If true, bboxes returned values as expressed as ratio w.r.t. to the image width and height. Default: False)code",
      false)
  .AddOptionalArg("save_img_ids",
      R"code(If true, image IDs will also be returned. Default: False)code",
      false)
  .AdditionalOutputsFn([](const OpSpec& spec) {
    return static_cast<int>(spec.GetArgument<bool>("save_img_ids"));
  })
  .AddParent("LoaderBase");


void ParseAnnotationFilesHelper(std::vector<std::string> &annotations_filename,
                                AnnotationMap &annotations_multimap,
                                std::vector<std::pair<std::string, int>> &image_id_pairs,
                                bool ltrb, bool ratio) {
  for (auto& file_name : annotations_filename) {
    // Loading raw json into the RAM
    std::ifstream f(file_name);
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

    // Change categories IDs to be in range [1, 80]
    std::vector<int> deleted_categories{ 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91 };
    std::map<int, int> new_category_ids;
    int current_id = 1;
    int vector_id = 0;
    for (int i = 1; i <= 90; i++) {
      if (i == deleted_categories[vector_id]) {
        vector_id++;
      } else {
        new_category_ids.insert(std::make_pair(i, current_id));
        current_id++;
      }
    }

    RAPIDJSON_ASSERT(r.PeekType() == kObjectType);
    r.EnterObject();
    while (const char* key = r.NextObjectKey()) {
      if (0 == strcmp(key, "images")) {
        RAPIDJSON_ASSERT(r.PeekType() == kArrayType);
        r.EnterArray();
        int id;
        string image_file_name;
        int width;
        int height;
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
          if (bbox[2] < 0.1 || bbox[3] < 0.1) {
            continue;
          }

          if (ltrb) {
            bbox[2] += bbox[0];
            bbox[3] += bbox[1];
          }

          annotations_multimap.insert(
            std::make_pair(image_id,
              Annotation(bbox[0], bbox[1], bbox[2], bbox[3], new_category_ids[category_id])));
        }
      } else {
        r.SkipValue();
      }
    }
    if (ratio) {
      for (auto& elm : annotations_multimap) {
        const auto& wh = image_id_to_wh[elm.first];
        elm.second.bbox[0] /= static_cast<float>(wh.first);
        elm.second.bbox[1] /= static_cast<float>(wh.second);
        elm.second.bbox[2] /= static_cast<float>(wh.first);
        elm.second.bbox[3] /= static_cast<float>(wh.second);
      }
    }
    f.close();
  }
}

RAPIDJSON_DIAG_POP

}  // namespace dali
