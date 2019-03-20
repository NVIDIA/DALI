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

#ifndef DALI_PIPELINE_OPERATORS_OPERATOR_FACTORY_H_
#define DALI_PIPELINE_OPERATORS_OPERATOR_FACTORY_H_

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <functional>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/op_schema.h"

namespace dali {

class OpSpec;

template <typename OpType>
class OperatorRegistry {
 public:
  typedef std::function<std::unique_ptr<OpType> (const OpSpec &spec)> Creator;
  typedef std::unordered_map<std::string, Creator> CreatorRegistry;

  OperatorRegistry() {}

  void Register(const std::string &name, Creator creator, const std::string &devName = "") {
      std::lock_guard<std::mutex> lock(mutex_);
    DALI_ENFORCE(registry_.count(name) == 0,
        "Operator \"" + name + "\" already registered" +
        (devName != ""? (" for " + devName) : "") + ".");
    registry_[name] = creator;
  }

  std::unique_ptr<OpType> Create(
      const std::string &name, const OpSpec &spec, const std::string *devName = NULL) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto creator_it = registry_.find(name);
    DALI_ENFORCE(creator_it != registry_.end(),
        "Operator \"" + name + "\" not registered" + (devName? (" for " + *devName) : "") + ".");
    return registry_[name](spec);
  }

  vector<std::string> RegisteredNames(bool internal_ops) {
    vector<std::string> names;
    for (const auto &pair : registry_) {
      auto& schema = SchemaRegistry::GetSchema(pair.first);
      if (internal_ops || !schema.IsInternal())
        names.push_back(pair.first);
    }
    return names;
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
      typename OperatorRegistry<OpType>::Creator creator, const std::string &devName = "") {
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
    static ::dali::Registerer<OpType> ANONYMIZE_VARIABLE(anon##OpName)(   \
        #OpName, &RegistryName##Registry::Registry(),                   \
        ::dali::Registerer<OpType>::OperatorCreator<DerivedType>, dev);   \
  }

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_OPERATOR_FACTORY_H_
