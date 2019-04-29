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

#ifndef DALI_PIPELINE_PROTO_DALI_PROTO_INTERN_H_
#define DALI_PIPELINE_PROTO_DALI_PROTO_INTERN_H_

#include <string>
#include <vector>
#include "dali/core/common.h"

namespace dali_proto {
class Argument;
}  // namespace dali_proto

namespace dali {

class DLL_PUBLIC DaliProtoPriv {
 public:
  DLL_PUBLIC explicit DaliProtoPriv(dali_proto::Argument *const);
  DLL_PUBLIC explicit DaliProtoPriv(const dali_proto::Argument *const);
  DLL_PUBLIC explicit DaliProtoPriv(DaliProtoPriv *const);
  DLL_PUBLIC void set_name(const string &);
  DLL_PUBLIC void set_type(const string &);
  DLL_PUBLIC void set_is_vector(const bool &);
  DLL_PUBLIC void add_ints(const int64 &);
  DLL_PUBLIC void add_floats(const float &);
  DLL_PUBLIC void add_bools(const bool &);
  DLL_PUBLIC void add_strings(const string &);
  DLL_PUBLIC DaliProtoPriv add_extra_args(void);

  DLL_PUBLIC string name(void) const;
  DLL_PUBLIC string type(void) const;
  DLL_PUBLIC bool is_vector(void) const;
  DLL_PUBLIC std::vector<int64> ints(void) const;
  DLL_PUBLIC int64 ints(int index) const;
  DLL_PUBLIC std::vector<float> floats(void) const;
  DLL_PUBLIC float floats(int index) const;
  DLL_PUBLIC std::vector<bool> bools(void) const;
  DLL_PUBLIC bool bools(int index) const;
  DLL_PUBLIC std::vector<string> strings(void) const;
  DLL_PUBLIC string strings(int index) const;
  DLL_PUBLIC DaliProtoPriv extra_args(int index) const;
  DLL_PUBLIC const std::vector<DaliProtoPriv> extra_args(void) const;

 private:
  dali_proto::Argument *const intern_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_PROTO_DALI_PROTO_INTERN_H_
