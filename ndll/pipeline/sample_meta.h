#ifndef NDLL_PIPELINE_SAMPLE_META_H_
#define NDLL_PIPELINE_SAMPLE_META_H_

#include <cuda_runtime_api.h>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/tensor.h"
#include "ndll/pipeline/data/tensor_list.h"

namespace ndll {

// Forward declare SampleWorkspace
class SampleWorkspace;

/**
 * @brief Stores all Input/Output meta-data from a SampleWorkspace.
 */
class SampleMeta {
public:
  /**
   * @brief Sets data index & thread index to -1.
   */
  inline SampleMeta() : data_idx_(-1), thread_idx_(-1) {}

  /**
   * @brief Stores all meta-data from the input SampleWorkspace.
   */
  inline SampleMeta(SampleWorkspace *ws) {
    SetMeta(ws);
  }
  
  ~SampleMeta() = default;

  /**
   * @brief Stores all meta-data from the input SampleWorkspace.
   */
  void SetMeta(SampleWorkspace *ws);

  /**
   * @brief Returns the number of input Tensors.
   */
  inline int NumInput() const { return input_shapes_.size(); }

  /**
   * @brief Returns the number of output Tensors.
   */
  inline int NumOutput() const { return output_shapes_.size(); }
  
  /**
   * @brief Returns a vector of the shapes of the input Tensors
   */
  const vector<Dims>& InputShapes() const { return input_shapes_; }

  /**
   * @brief Returns a vector of the shapes of the output Tensors
   */
  const vector<Dims>& OutputShapes() const { return output_shapes_; }

  /**
   * @brief Returns the shape of the input tensor at index = `idx`.
   */
  const vector<Index>& InputShape(int idx) const;

  /**
   * @brief Returns the shape of the output tensor at index = `idx`.
   */
  const vector<Index>& OutputShape(int idx) const;

  /**
   * @brief Returns a vector of the types of the input Tensors.
   */
  inline const vector<TypeInfo>& InputTypes() const { return input_types_; }

  /**
   * @brief Returns a vector of the types of the output Tensors.
   */
  inline const vector<TypeInfo>& OutputTypes() const { return output_types_; }

  /**
   * @brief Returns the type of the input tensor at index = `idx`.
   */
  const TypeInfo& InputType(int idx) const;

  /**
   * @brief Returns the type of the output tensor at index = `idx`.
   */
  const TypeInfo& OutputType(int idx) const;

  /**
   * @brief Returns the index of the sample that this workspace stores
   * in the input/output batch.
   */
  inline int data_idx() const { return data_idx_; }

  /**
   * @brief Returns the index of the thread that will process this data.
   */
  inline int thread_idx() const { return thread_idx_; }
  
private:
  vector<Dims> input_shapes_, output_shapes_;
  vector<TypeInfo> input_types_, output_types_;
  
  int data_idx_, thread_idx_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_SAMPLE_META_H_
