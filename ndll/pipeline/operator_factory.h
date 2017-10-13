#ifndef NDLL_PIPELINE_OPERATOR_FACTORY_H_
#define NDLL_PIPELINE_OPERATOR_FACTORY_H_

#include <mutex>
#include <unordered_map>

namespace ndll {

class OpSpec;

template <typename OpType>
class OperatorRegistry {
public:
  typedef std::function<unique_ptr<OpType> (const OpSpec &spec)> Creator;
  typedef std::unordered_map<string, Creator> CreatorRegistry;

  OperatorRegistry() {}
  
  void Register(const string &name, Creator creator) {
    std::lock_guard<std::mutex> lock(mutex_);
    NDLL_ENFORCE(registry_.count(name) == 0,
        "Operator \"" + name + "\" already registered.");
    registry_[name] = creator;
  }

  unique_ptr<OpType> Create(
      const string &name, const OpSpec &spec) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto creator_it = registry_.find(name);
    NDLL_ENFORCE(creator_it != registry_.end(),
        "Operator \"" + name + "\" not registered.");
    return registry_[name](spec);
  }
  
private:
  CreatorRegistry registry_;
  std::mutex mutex_;
};

template <typename OpType>
class Registerer {
public:
  Registerer(const string &name,
      OperatorRegistry<OpType> *registry,
      typename OperatorRegistry<OpType>::Creator creator) {
    registry->Register(name, creator);
  }

  // Standard creator function used by all operators
  template <typename DerivedOpType>
  static std::unique_ptr<OpType> OperatorCreator(const OpSpec &spec) {
    return std::unique_ptr<OpType>(new DerivedOpType(spec));
}
};


// Creators a registry object for a specific op type
#define NDLL_DEFINE_OPTYPE_REGISTRY(RegistryName, OpType)           \
  class RegistryName##Registry {                                    \
  public:                                                           \
    static OperatorRegistry<OpType>& Registry() {                   \
      static OperatorRegistry<OpType> registry;                     \
      return registry;                                              \
    }                                                               \
  };

// Helper to define a registerer for a specific op type. Each op type
// defines its own, more aptly named, registration macros on top of this
#define NDLL_DEFINE_OPTYPE_REGISTERER(OpName, RegistryName, OpType, Suffix) \
  namespace {                                                               \
    Registerer<OpType> hidden_var(#OpName,                                  \
      &RegistryName##Registry::Registry(),                                  \
      Registerer<OpType>::OperatorCreator<OpName##Suffix>);                 \
  }

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATOR_FACTORY_H_
