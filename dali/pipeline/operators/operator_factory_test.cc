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

#include "dali/pipeline/operators/operator_factory.h"

#include <gtest/gtest.h>

#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/operators/op_schema.h"
#include "dali/test/dali_test.h"

namespace dali {

template <typename Backend>
class OperatorFactoryTest : public DALITest {
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
  virtual ~DummyBase() {}

  virtual int GetId() const = 0;
};

template <typename Backend>
class DummyDerivedOne : public DummyBase<Backend> {
 public:
  explicit DummyDerivedOne(const OpSpec &) {}

  int GetId() const override { return 1; }
};

template <typename Backend>
class DummyDerivedTwo : public DummyBase<Backend>  {
 public:
  explicit DummyDerivedTwo(const OpSpec &) {}

  int GetId() const override { return 2; }
};

template <typename Backend>
class DummyDerivedThree : public DummyBase<Backend> {
 public:
  explicit DummyDerivedThree(const OpSpec &) {}

  int GetId() const override { return 3; }
};

// Create some dummy factories (lol)
DALI_DECLARE_OPTYPE_REGISTRY(CPUDummy, DummyBase<CPUBackend>);
DALI_DECLARE_OPTYPE_REGISTRY(GPUDummy, DummyBase<GPUBackend>);
DALI_DEFINE_OPTYPE_REGISTRY(CPUDummy, DummyBase<CPUBackend>);
DALI_DEFINE_OPTYPE_REGISTRY(GPUDummy, DummyBase<GPUBackend>);


// Some registration macros
#define DALI_REGISTER_CPU_DUMMY(OpName, OpType)        \
  DALI_DEFINE_OPTYPE_REGISTERER(OpName, OpType,        \
      CPUDummy, DummyBase<CPUBackend>, "CPU");         \
  DALI_SCHEMA_REG(OpName)

#define DALI_REGISTER_GPU_DUMMY(OpName, OpType)         \
  DALI_DEFINE_OPTYPE_REGISTERER(OpName, OpType,         \
      GPUDummy, DummyBase<GPUBackend>, "GPU")


// Register the classes
DALI_REGISTER_CPU_DUMMY(DummyDerivedOne, DummyDerivedOne<CPUBackend>);
DALI_REGISTER_GPU_DUMMY(DummyDerivedOne, DummyDerivedOne<GPUBackend>);
DALI_REGISTER_CPU_DUMMY(DummyDerivedTwo, DummyDerivedTwo<CPUBackend>);
DALI_REGISTER_GPU_DUMMY(DummyDerivedTwo, DummyDerivedTwo<GPUBackend>);
DALI_REGISTER_CPU_DUMMY(DummyDerivedThree, DummyDerivedThree<CPUBackend>);
DALI_REGISTER_GPU_DUMMY(DummyDerivedThree, DummyDerivedThree<GPUBackend>);

// For now just test w/ CPU backend
typedef ::testing::Types<CPUBackend> TestTypes;
TYPED_TEST_SUITE(OperatorFactoryTest, TestTypes);

TYPED_TEST(OperatorFactoryTest, TestRegisterAndConstruct) {
  OperatorRegistry<DummyBase<CPUBackend>> &registry = CPUDummyRegistry::Registry();
  vector<string> names = registry.RegisteredNames(true);
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

}  // namespace dali
