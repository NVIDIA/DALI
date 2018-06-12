// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_OPERATOR_FACTORY_H_
#define NDLL_PIPELINE_OPERATORS_OPERATOR_FACTORY_H_

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <functional>

#include "ndll/common.h"
#include "ndll/error_handling.h"

namespace ndll {

class OpSpec;

template <typename OpType>
class OperatorRegistry {
 public:
  typedef std::function<std::unique_ptr<OpType> (const OpSpec &spec)> Creator;
  typedef std::unordered_map<std::string, Creator> CreatorRegistry;

  OperatorRegistry() {}

  void Register(const std::string &name, Creator creator) {
    std::lock_guard<std::mutex> lock(mutex_);
    NDLL_ENFORCE(registry_.count(name) == 0,
        "Operator \"" + name + "\" already registered.");
    registry_[name] = creator;
  }

  std::unique_ptr<OpType> Create(
      const std::string &name, const OpSpec &spec, const std::string *devName = NULL) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto creator_it = registry_.find(name);
    NDLL_ENFORCE(creator_it != registry_.end(),
        "Operator \"" + name + "\" not registered" + (devName? (" for " + *devName) : "") + ".");
    return registry_[name](spec);
  }

  vector<std::string> RegisteredNames() {
    vector<std::string> names;
    for (const auto &pair : registry_) {
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
      typename OperatorRegistry<OpType>::Creator creator) {
    registry->Register(name, creator);
  }

  // Standard creator function used by all operators
  template <typename DerivedType>
  static std::unique_ptr<OpType> OperatorCreator(const OpSpec &spec) {
    return std::unique_ptr<OpType>(new DerivedType(spec));
}
};


// Creators a registry object for a specific op type
#define NDLL_DECLARE_OPTYPE_REGISTRY(RegistryName, OpType)            \
  class RegistryName##Registry {                                      \
   public:                                                            \
    static ndll::OperatorRegistry<OpType>& Registry();                \
  };

#define NDLL_DEFINE_OPTYPE_REGISTRY(RegistryName, OpType)               \
  ndll::OperatorRegistry<OpType>& RegistryName##Registry::Registry() {  \
    static ndll::OperatorRegistry<OpType> registry;                     \
    return registry;                                                    \
  }

// Helper to define a registerer for a specific op type. Each op type
// defines its own, more aptly named, registration macros on top of this
#define NDLL_DEFINE_OPTYPE_REGISTERER(OpName, DerivedType,              \
    RegistryName, OpType)                                               \
  namespace {                                                           \
    static ndll::Registerer<OpType> ANONYMIZE_VARIABLE(anon##OpName)(   \
        #OpName, &RegistryName##Registry::Registry(),                   \
        ndll::Registerer<OpType>::OperatorCreator<DerivedType>);        \
  }

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_OPERATOR_FACTORY_H_
