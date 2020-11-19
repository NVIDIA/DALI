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

#ifndef DALI_PIPELINE_UTIL_LOOKAHEAD_PARSER_H_
#define DALI_PIPELINE_UTIL_LOOKAHEAD_PARSER_H_

#include <rapidjson/reader.h>
#include <rapidjson/document.h>
#include "dali/core/api_helper.h"

RAPIDJSON_DIAG_PUSH
#ifdef __GNUC__
RAPIDJSON_DIAG_OFF(effc++)
#endif

namespace dali {
namespace detail {

using rapidjson::SizeType;
using rapidjson::Value;
using rapidjson::Reader;
using rapidjson::InsituStringStream;
using rapidjson::kParseDefaultFlags;
using rapidjson::kParseInsituFlag;
using rapidjson::kArrayType;
using rapidjson::kObjectType;
using rapidjson::kStringType;

// taken from https://github.com/Tencent/rapidjson/blob/master/example/lookaheadparser/lookaheadparser.cpp

class LookaheadParserHandler {
 public:
  inline bool Null() { st_ = kHasNull; v_.SetNull(); return true; }
  inline bool Bool(bool b) { st_ = kHasBool; v_.SetBool(b); return true; }
  inline bool Int(int i) { st_ = kHasNumber; v_.SetInt(i); return true; }
  inline bool Uint(unsigned u) { st_ = kHasNumber; v_.SetUint(u); return true; }
  inline bool Int64(int64_t i) { st_ = kHasNumber; v_.SetInt64(i); return true; }
  inline bool Uint64(uint64_t u) { st_ = kHasNumber; v_.SetUint64(u); return true; }
  inline bool Double(double d) { st_ = kHasNumber; v_.SetDouble(d); return true; }
  inline bool RawNumber(const char*, SizeType, bool) { return false; }
  inline bool String(const char* str, SizeType length, bool) {
    st_ = kHasString;
    v_.SetString(str, length);
    return true;
  }
  inline bool StartObject() { st_ = kEnteringObject; return true; }
  inline bool Key(const char* str, SizeType length, bool) {
    st_ = kHasKey;
    v_.SetString(str, length);
    return true;
  }
  inline bool EndObject(SizeType) { st_ = kExitingObject; return true; }
  inline bool StartArray() { st_ = kEnteringArray; return true; }
  inline bool EndArray(SizeType) { st_ = kExitingArray; return true; }

 protected:
  inline explicit LookaheadParserHandler(char* str);
  inline void ParseNext();

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

class DLL_PUBLIC LookaheadParser : protected LookaheadParserHandler {
 public:
  inline explicit LookaheadParser(char* str) : LookaheadParserHandler(str) {}

  inline bool EnterObject();
  inline bool EnterArray();
  inline const char* NextObjectKey();
  inline bool NextArrayValue();
  inline int GetInt();
  inline double GetDouble();
  inline const char* GetString();
  inline bool GetBool();
  inline void GetNull();

  inline void SkipObject();
  inline void SkipArray();
  inline void SkipValue();
  inline Value* PeekValue();
  // returns a rapidjson::Type, or -1 for no value (at end of object/array)
  inline int PeekType();

  inline bool IsValid() { return st_ != kError; }

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

}  // namespace detail

RAPIDJSON_DIAG_POP

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_LOOKAHEAD_PARSER_H_
