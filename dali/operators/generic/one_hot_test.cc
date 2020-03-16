#include <gtest/gtest.h>
#include "dali/operators/generic/one_hot.h"
#include "dali/kernels/kernel_params.h"
#include "utility"
#include "dali/test/tensor_test_utils.h"

#include <iostream>

namespace dali {
namespace testing {

class OneHotOpTest : public ::testing::Test {
  protected:
    void SetUp() final {
        for (int i = 0; i < nclasses_; ++i) output_.push_back(0);
    }

    int buffer_length_ = 10;
    int nclasses_ = 20;
    std::vector<float> input_{1, 3, 5, 0, 1, 2, 10, 15, 7, 6};
    std::vector<float> output_;
    TensorShape<1> input_shape_ = {1};
    TensorShape<1> output_shape_ = {nclasses_};
};

TEST_F(OneHotOpTest, TestDoOneHotForFloats) {

    for (auto it = input_.begin(); it != input_.end(); ++it) {
        auto input = make_tensor_cpu(reinterpret_cast<const float*>(&(*it)), 
                                                                this->input_shape_);
        auto output = make_tensor_cpu(reinterpret_cast<float*>(this->output_.data()), 
                                                                  this->output_shape_);
        detail::DoOneHot(output, input, 20, 1, 0);
        auto ptr = output.data;
        for (int i = 0; i < nclasses_; ++i) {
            if (i == *it) ASSERT_EQ(ptr[i], 1);
            else ASSERT_EQ(ptr[i], 0);
        }
    }
}

} // namespace testing
} // namespace dali
