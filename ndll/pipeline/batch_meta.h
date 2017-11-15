#ifndef NDLL_PIPELINE_BATCH_META_H_
#define NDLL_PIPELINE_BATCH_META_H_

#include <cuda_runtime_api.h>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/tensor.h"
#include "ndll/pipeline/data/tensor_list.h"

namespace ndll {

// Forward declare BatchWorkspace
class BatchWorkspace;

/**
 * @brief Stores all Input/Output meta-data from a BatchWorkspace.
 */
class BatchMeta {
public:
  /**
   * @brief Sets the stream to 0.
   */
  inline BatchMeta() : stream_(0) {}

  inline BatchMeta(BatchWorkspace *ws) {
    SetMeta(ws);
  }

  ~BatchMeta() = default;

  /**
   * @brief Stores all meta-data from the input BatchWorkspace.
   */
  void SetMeta(BatchWorkspace *ws);

  /**
   * @brief Returns the number of input Tensors.
   */
  inline int NumInput() const { return input_shapes_.size(); }

  /**
   * @brief Returns the number of output Tensors.
   */
  inline int NumOutput() const { return output_shapes_.size(); }

  /**
   * @brief Returns a vector of shapes of the input TensorLists.
   */
  const vector<vector<Dims>>& InputShapes() const { return input_shapes_; }

  /**
   * @brief Returns a vector of shapes of the output TensorLists.
   */
  const vector<vector<Dims>>& OutputShapes() const { return output_shapes_; }

  /**
   * @brief Returns the shape of the input TensorList at index = `idx`.
   */
  const vector<Dims>& InputShape(int idx) const;

  /**
   * @brief Returns the shape of the output TensorList at index = `idx`.
   */
  const vector<Dims>& OutputShape(int idx) const;

  /**
   * @brief Returns a vector of the types of the input TensorLists.
   */
  inline const vector<TypeInfo>& InputTypes() const { return input_types_; }

  /**
   * @brief Returns a vector of the types of the output TensorLists.
   */
  inline const vector<TypeInfo>& OutputTypes() const { return output_types_; }

  /**
   * @brief Returns the type of the input TensorList at index = `idx`.
   */
  const TypeInfo& InputType(int idx) const;

  /**
   * @brief Returns the type of the output TensorList at index = `idx`.
   */
  const TypeInfo& OutputType(int idx) const;

  /**
   * @brief Returns the cuda stream that this work is to be done in.
   */
  inline cudaStream_t stream() const { return stream_; }
  
private:
  vector<vector<Dims>> input_shapes_, output_shapes_;
  vector<TypeInfo> input_types_, output_types_;

  cudaStream_t stream_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_BATCH_META_H_
