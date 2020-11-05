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

#include "dali/core/common.h"
#include "dali/pipeline/dali.pb.h"
#include "dali/pipeline/proto/dali_proto_intern.h"

namespace dali {

  DaliProtoPriv::DaliProtoPriv(dali_proto::Argument *const intern)
      : intern_(intern) {
  }

  DaliProtoPriv::DaliProtoPriv(const dali_proto::Argument *const intern)
      : DaliProtoPriv(const_cast<dali_proto::Argument *const>(intern)) {
  }

  DaliProtoPriv::DaliProtoPriv(DaliProtoPriv *const other)
      : intern_(other->intern_) {
  }

  void DaliProtoPriv::set_name(const string &val) {
    intern_->set_name(val);
  }

  void DaliProtoPriv::set_type(const string &val) {
    intern_->set_type(val);
  }

  void DaliProtoPriv::set_is_vector(const bool &val) {
    intern_->set_is_vector(val);
  }

  void DaliProtoPriv::add_ints(const int64 &val) {
    intern_->add_ints(val);
  }

  void DaliProtoPriv::add_floats(const float &val) {
    intern_->add_floats(val);
  }

  void DaliProtoPriv::add_bools(const bool &val) {
    intern_->add_bools(val);
  }

  void DaliProtoPriv::add_strings(const string &val) {
    intern_->add_strings(val);
  }

  DaliProtoPriv DaliProtoPriv::add_extra_args(void) {
    dali_proto::Argument* const tmp = intern_->add_extra_args();
    return DaliProtoPriv(tmp);
  }

  string DaliProtoPriv::name(void) const {
    return intern_->name();
  }

  string DaliProtoPriv::type(void) const {
    return intern_->type();
  }

  bool DaliProtoPriv::is_vector(void) const {
    return intern_->is_vector();
  }

  std::vector<int64> DaliProtoPriv::ints(void) const {
    std::vector<int64> tmp(intern_->ints().begin(), intern_->ints().end());
    return tmp;
  }

  int64 DaliProtoPriv::ints(int index) const {
    return intern_->ints(index);
  }

  std::vector<float> DaliProtoPriv::floats(void) const {
    std::vector<float> tmp(intern_->floats().begin(), intern_->floats().end());
    return tmp;
  }

  float DaliProtoPriv::floats(int index) const {
    return intern_->floats(index);
  }

  std::vector<bool> DaliProtoPriv::bools(void) const {
    std::vector<bool> tmp(intern_->bools().begin(), intern_->bools().end());
    return tmp;
  }

  bool DaliProtoPriv::bools(int index) const {
    return intern_->bools(index);
  }

  std::vector<string> DaliProtoPriv::strings(void) const {
    std::vector<string> tmp(intern_->strings().begin(), intern_->strings().end());
    return tmp;
  }

  string DaliProtoPriv::strings(int index) const {
    return intern_->strings(index);
  }

  const std::vector<DaliProtoPriv> DaliProtoPriv::extra_args(void) const {
    std::vector<DaliProtoPriv> tmp;
    for (auto& elm : intern_->extra_args()) {
      tmp.emplace_back(DaliProtoPriv(&elm));
    }
    return tmp;
  }

  DaliProtoPriv DaliProtoPriv::extra_args(int index) const {
    DaliProtoPriv tmp(&(intern_->extra_args(index)));
    return tmp;
  }

}  // namespace dali
