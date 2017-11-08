#include "ndll/pipeline/operator_factory.h"

#include <gtest/gtest.h>

#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/op_spec.h"
#include "ndll/test/ndll_test.h"

namespace ndll {

template <typename Backend>
class OperatorFactoryTest : public NDLLTest {
public:
  // Don't do any setup
  void SetUp() override {}
  void TearDown() override {}
};

// Some dummy operators for us to test with
template <typename Backend>
class DummyBase {
public:
  DummyBase() {}

  virtual int GetId() const = 0;
};

template <typename Backend>
class DummyDerivedOne : public DummyBase<Backend> {
public:
  DummyDerivedOne(const OpSpec &spec) {}

  int GetId() const override { return 1; }
};

template <typename Backend>
class DummyDerivedTwo : public DummyBase<Backend>  {
public:
  DummyDerivedTwo(const OpSpec &spec) {}

  int GetId() const override { return 2; }
};

template <typename Backend>
class DummyDerivedThree : public DummyBase<Backend> {
public:
  DummyDerivedThree(const OpSpec &spec) {}

  int GetId() const override { return 3; }
};

// Create some dummy factories (lol)
NDLL_DECLARE_OPTYPE_REGISTRY(CPUDummy, DummyBase<CPUBackend>);
NDLL_DECLARE_OPTYPE_REGISTRY(GPUDummy, DummyBase<GPUBackend>);
NDLL_DEFINE_OPTYPE_REGISTRY(CPUDummy, DummyBase<CPUBackend>);
NDLL_DEFINE_OPTYPE_REGISTRY(GPUDummy, DummyBase<GPUBackend>);

// Some registration macros
#define NDLL_REGISTER_CPU_DUMMY(OpName, OpType)        \
  NDLL_DEFINE_OPTYPE_REGISTERER(OpName, OpType,        \
      CPUDummy, DummyBase<CPUBackend>) 

#define NDLL_REGISTER_GPU_DUMMY(OpName, OpType)         \
  NDLL_DEFINE_OPTYPE_REGISTERER(OpName, OpType,         \
      GPUDummy, DummyBase<GPUBackend>) 


// Register the classes
NDLL_REGISTER_CPU_DUMMY(DummyDerivedOne, DummyDerivedOne<CPUBackend>);
NDLL_REGISTER_GPU_DUMMY(DummyDerivedOne, DummyDerivedOne<GPUBackend>);
NDLL_REGISTER_CPU_DUMMY(DummyDerivedTwo, DummyDerivedTwo<CPUBackend>);
NDLL_REGISTER_GPU_DUMMY(DummyDerivedTwo, DummyDerivedTwo<GPUBackend>);
NDLL_REGISTER_CPU_DUMMY(DummyDerivedThree, DummyDerivedThree<CPUBackend>);
NDLL_REGISTER_GPU_DUMMY(DummyDerivedThree, DummyDerivedThree<GPUBackend>);

// For now just test w/ CPU backend
typedef ::testing::Types<CPUBackend> TestTypes;
TYPED_TEST_CASE(OperatorFactoryTest, TestTypes);

TYPED_TEST(OperatorFactoryTest, TestRegisterAndConstruct) {
  OperatorRegistry<DummyBase<CPUBackend>> &registry = CPUDummyRegistry::Registry();
  vector<string> names = registry.RegisteredNames();
  vector<string> val_names = {"DummyDerivedOne",
                              "DummyDerivedTwo",
                              "DummyDerivedThree"};

  // Check that all names are registered
  for (auto &target : val_names) {
    bool found = false;
    for (auto &n : names) {
      if (n == target) {
        found = true;
        break;
      }
    }
    ASSERT_TRUE(found) << "Did not find name " << target << " in registry.";
  }

  // Try constructing the ops an validating them by ID
  int val_id = 1;
  for (auto &target : val_names) {
    std::unique_ptr<DummyBase<CPUBackend>> dummy =
      registry.Create(target, OpSpec("Dummy"));

    ASSERT_EQ(val_id, dummy->GetId());
    ++val_id;
  }
}

} // namespace ndll
