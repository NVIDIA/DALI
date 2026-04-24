// Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATOR_OPERATOR_FACTORY_H_
#define DALI_PIPELINE_OPERATOR_OPERATOR_FACTORY_H_

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/core/string_map.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/operator/op_schema.h"

namespace dali {

class OpSpec;

template <typename OpType>
class OperatorRegistry {
 public:
  typedef std::function<std::unique_ptr<OpType> (const OpSpec &spec)> Creator;
  typedef unordered_string_map<Creator> CreatorRegistry;

  OperatorRegistry() {}

  template <typename Backend>
  void Register(std::string name, Creator creator) {
    Register(name, std::move(creator), BackendDeviceName<Backend>);
  }

  void Register(
        std::string name,
        Creator creator,
        std::optional<std::string_view> device_name = {}) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto [it, inserted] = registry_.emplace(std::move(name), std::move(creator));
    DALI_ENFORCE(inserted, make_string(
        "Operator \"", it->first, "\" already registered",  // it->first because `name` is moved out
        (device_name ? make_string(" for \"", *device_name, "\"") : "")));
  }

  template <typename Backend>
  std::unique_ptr<OpType> Create(
      std::string_view name, const OpSpec &spec) {
    return Create(name, spec, BackendDeviceName<Backend>);
  }

  std::unique_ptr<OpType> Create(
        std::string_view name,
        const OpSpec &spec,
        std::optional<std::string_view> device_name = {}) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto creator_it = registry_.find(name);
    DALI_ENFORCE(creator_it != registry_.end(), make_string(
        "Operator \"", name, "\" not registered",
        (device_name? make_string(" for \"", *device_name, "\"") : "")));

    return creator_it->second(spec);
  }

  vector<std::string> RegisteredNames(bool internal_ops) {
    vector<std::string> names;
    for (const auto &pair : registry_) {
      auto& schema = SchemaRegistry::GetSchema(pair.first);
      if (internal_ops || !schema.IsInternal())
        names.push_back(schema.name().length() ? schema.name() : pair.first);
    }
    return names;
  }

  bool IsRegistered(const std::string &name) {
    std::lock_guard<std::mutex> lock(mutex_);
    return registry_.count(name) > 0;
  }

 private:
  CreatorRegistry registry_;
  std::mutex mutex_;
};

template <typename OpType>
class Registerer {
 public:
  Registerer(const std::string &name,
      OperatorRegistry<OpType> *registry,
      typename OperatorRegistry<OpType>::Creator creator,
      std::string_view devName = "") {
    registry->Register(name, creator, devName);
  }

  // Standard creator function used by all operators
  template <typename DerivedType>
  static std::unique_ptr<OpType> OperatorCreator(const OpSpec &spec) {
    return std::unique_ptr<OpType>(new DerivedType(spec));
}
};


// Creators a registry object for a specific op type
#define DALI_DECLARE_OPTYPE_REGISTRY(RegistryName, OpType)            \
  class DLL_PUBLIC RegistryName##Registry {                           \
   public:                                                            \
    DLL_PUBLIC static ::dali::OperatorRegistry<OpType>& Registry();     \
  };

#define DALI_DEFINE_OPTYPE_REGISTRY(RegistryName, OpType)               \
  dali::OperatorRegistry<OpType>& RegistryName##Registry::Registry() {  \
    static ::dali::OperatorRegistry<OpType> registry;                     \
    return registry;                                                    \
  }

// Helper to define a registerer for a specific op type. Each op type
// defines its own, more aptly named, registration macros on top of this
#define DALI_DEFINE_OPTYPE_REGISTERER(OpName, DerivedType,              \
    RegistryName, OpType, dev)                                          \
  namespace {                                                           \
    static ::dali::Registerer<OpType> ANONYMIZE_VARIABLE(anon##OpName)( \
        #OpName, &RegistryName##Registry::Registry(),                   \
        ::dali::Registerer<OpType>::OperatorCreator<DerivedType>, dev); \
  }

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_OPERATOR_FACTORY_H_
