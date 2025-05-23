// Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_TF_PLUGIN_DALI_HELPER_H_
#define DALI_TF_PLUGIN_DALI_HELPER_H_

#include <memory>
#include <utility>
#include <vector>
#include <string>

#include "dali/dali.h"
#include "dali/core/format.h"
#include "dali/dali_cpp_wrappers.h"

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

namespace dali_tf_impl {

class DALIException : public std::runtime_error {
 public:
  explicit DALIException(
        daliResult_t result,
        std::string message,
        const char *expression,
        const char *file,
        int line)
  : runtime_error(MakeErrorString(result, message, expression, file, line)) {}

  static daliResult_t Rethrow(
        daliResult_t result,
        const char *expression,
        const char *file,
        int line) {
    if (!(result & DALI_ERROR))  // return non-error results
      return result;
    throw DALIException(
      result,
      daliGetLastErrorMessage(),
      expression,
      file,
      line);
  }

  static std::string MakeErrorString(
        daliResult_t result,
        const std::string &message,
        const char *expression,
        const char *file,
        int line) {
    std::stringstream ss;
    ss << "Error ";
    if (const char *err = daliGetErrorName(result))
      ss << err;
    else
      ss << "<unknown error " << static_cast<int>(result) << ">";

    ss << ":\n" << message;

    if (expression)
      ss << "\nwhile executing: " << expression;

    if (file && line > 0)
      ss << "\nin " << file << ":" << line;

    ss << std::endl;
    return ss.str();
  }

  daliResult_t code;
  std::string message;
  const char *expression = nullptr;
  const char *file = nullptr;
  int line = -1;
};

#define DALI_RETHROW(...) DALIException::Rethrow((__VA_ARGS__), #__VA_ARGS__, __FILE__, __LINE__)

inline tensorflow::TensorShape ToTfShape(const int64_t *shape, int ndim) {
  tensorflow::TensorShape ts;
  auto status = tensorflow::TensorShape::BuildTensorShape(absl::Span(shape, ndim), &ts);
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
  return ts;
}

inline tensorflow::TensorShape GetTfShape(const daliTensorDesc_t &desc) {
  return ToTfShape(desc.shape, desc.ndim);
}

inline tensorflow::TensorShape GetTfShape(daliTensor_h tensor) {
  daliTensorDesc_t desc{};
  DALI_RETHROW(daliTensorGetDesc(tensor, &desc));
  return ToTfShape(desc.shape, desc.ndim);
}

constexpr tensorflow::DataType DaliToTfType(daliDataType_t dali_type) {
  switch (dali_type) {
    case DALI_UINT8:
      return tensorflow::DT_UINT8;
    case DALI_UINT16:
      return tensorflow::DT_UINT16;
    case DALI_UINT32:
      return tensorflow::DT_UINT32;
    case DALI_UINT64:
      return tensorflow::DT_UINT64;
    case DALI_INT8:
      return tensorflow::DT_INT8;
    case DALI_INT16:
      return tensorflow::DT_INT16;
    case DALI_INT32:
      return tensorflow::DT_INT32;
    case DALI_INT64:
      return tensorflow::DT_INT64;
    case DALI_FLOAT16:
      return tensorflow::DT_HALF;
    case DALI_FLOAT:
      return tensorflow::DT_FLOAT;
    case DALI_FLOAT64:
      return tensorflow::DT_DOUBLE;
    case DALI_BOOL:
      return tensorflow::DT_BOOL;
    default:
      return tensorflow::DT_INVALID;
  }
}

constexpr daliDataType_t TfToDaliType(tensorflow::DataType tf_type) {
  switch (tf_type) {
    case tensorflow::DT_UINT8:
      return DALI_UINT8;
    case tensorflow::DT_UINT16:
      return DALI_UINT16;
    case tensorflow::DT_UINT32:
      return DALI_UINT32;
    case tensorflow::DT_UINT64:
      return DALI_UINT64;
    case tensorflow::DT_INT8:
      return DALI_INT8;
    case tensorflow::DT_INT16:
      return DALI_INT16;
    case tensorflow::DT_INT32:
      return DALI_INT32;
    case tensorflow::DT_INT64:
      return DALI_INT64;
    case tensorflow::DT_HALF:
      return DALI_FLOAT16;
    case tensorflow::DT_FLOAT:
      return DALI_FLOAT;
    case tensorflow::DT_DOUBLE:
      return DALI_FLOAT64;
    case tensorflow::DT_BOOL:
      return DALI_BOOL;
    default:
      return DALI_NO_TYPE;
  }
}


static const void* GetTensorData(const tensorflow::Tensor& t) {
#if TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 2
  return t.data();
#else
  return t.tensor_data().data();
#endif
}


// TensorFlow treats a sample/example as a vector of Tensors (they flatten everything),
// and if a dataset has multiple outputs it means, that it returned a tuple that maps
// to a vector of Tensors.
using TfExample = std::vector<tensorflow::Tensor>;

// Batch is a list of Tensors - we repack the TfExamples into a Batch.
// We need to build batches sample by sample to advance the input iterators in sync.
using BatchStorage = std::vector<tensorflow::Tensor>;

inline bool IsUniform(int num_samples, int ndim, const int64_t *shape) {
  if (num_samples < 2 || ndim == 0)
    return true;
  for (int s = 1; s < num_samples; s++) {
    for (int i = 0; i < ndim; i++)
      if (shape[i] != shape[s * ndim + i])
        return false;
  }
  return true;
}

inline std::string ShapeToString(daliTensorList_h tl) {
  const int64_t *shape = nullptr;
  int ndim = 0, num_samples = 0;
  DALI_RETHROW(daliTensorListGetShape(tl, &num_samples, &ndim, &shape));
  std::stringstream ss;
  for (int s = 0; s < num_samples; s++) {
    if (s)
      ss << ", ";
    ss << ToTfShape(shape + s * ndim, ndim);
  }
  return ss.str();
}

class Batch {
 public:
  Batch() = default;

  explicit Batch(BatchStorage &&sample_tensors)
      : storage(std::move(sample_tensors)), is_per_sample(true) {}

  explicit Batch(tensorflow::Tensor &batch_tensor) : storage{batch_tensor}, is_per_sample(false) {}

  int64_t ndim() const {
    if (is_per_sample) {
      return storage[0].dims();
    } else {
      // remove the batch dimension if present
      return storage[0].dims() - 1;
    }
  }

  daliDataType_t dtype() const {
    return TfToDaliType(storage[0].dtype());
  }

  int64_t batch_size() const {
    if (is_per_sample) {
      return storage.size();
    } else {
      return storage[0].dim_size(0);
    }
  }

  void GetShapes(std::vector<int64_t> &shapes) const {
    shapes.clear();
    shapes.reserve(batch_size() * ndim());
    if (is_per_sample) {
      for (int sample_idx = 0; sample_idx < batch_size(); sample_idx++) {
        for (int d = 0; d < ndim(); d++) {
          shapes.push_back(storage[sample_idx].dim_size(d));
        }
      }
    } else {
      for (int sample_idx = 0; sample_idx < batch_size(); sample_idx++) {
        for (int d = 0; d < ndim(); d++) {
          shapes.push_back(storage[0].dim_size(d + 1));
        }
      }
    }
  }

  tensorflow::Status GetPtrs(std::vector<const void *> &ptrs) const {
    if (!is_per_sample) {
      return tensorflow::errors::Internal("Internal mismatch of batch and per-sample mode.");
    }
    ptrs.clear();
    ptrs.resize(batch_size(), nullptr);
    for (int sample_idx = 0; sample_idx < batch_size(); sample_idx++) {
      ptrs[sample_idx] = GetTensorData(storage[sample_idx]);
    }
    return tensorflow::Status();
  }


  tensorflow::Status GetPtr(const void *&ptr) const {
    if (is_per_sample) {
      return tensorflow::errors::Internal("Internal mismatch of batch and per-sample mode.");
    }
    ptr = GetTensorData(storage[0]);
    return tensorflow::Status();
  }


  /**
   * @brief Check if samples have the same ndim, dtype etc.
   *
   * This probably is already handled by TF, as dataset probably can't
   * dynamically change its shape. And we can probably check the declared output
   * structure in Python level.
   *
   * TODO(klecki): add those checks in Python for clear errors
   */
  tensorflow::Status VerifyUniform(int input_idx) {
    if (storage.empty()) {
      return tensorflow::errors::InvalidArgument("Empty batch for input: ", input_idx, ".");
    }
    if (!is_per_sample) {
      return tensorflow::Status();
    }
    int ndim = storage[0].dims();
    auto dtype = storage[0].dtype();
    for (auto &sample : storage) {
      if (sample.dims() != ndim) {
        return tensorflow::errors::InvalidArgument(
            "Inconsistent dimensionality of samples in a batch for input: ", input_idx,
            ", got sample with: ", sample.dims(), " dimensions while the first one has: ", ndim,
            " dimensions.");
      }
      if (sample.dtype() != dtype) {
        return tensorflow::errors::InvalidArgument(
            "Inconsistent dtype of samples in a batch for input: ", input_idx,
            ", got sample with: ", sample.dtype(), " dtype while the first one has: ", dtype,
            " dtype.");
      }
    }
    return tensorflow::Status();
  }

  void clear() {
    storage.clear();
  }

 private:
  BatchStorage storage;
  bool is_per_sample = true;
};

// Represents tuple of inputs for one iteration
using ListOfBatches = std::vector<Batch>;


}  // namespace dali_tf_impl


#endif  // DALI_TF_PLUGIN_DALI_HELPER_H_
