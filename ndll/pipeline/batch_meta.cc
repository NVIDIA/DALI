#include "ndll/pipeline/batch_meta.h"

#include "ndll/pipeline/batch_workspace.h"

namespace ndll {

void BatchMeta::SetMeta(BatchWorkspace *ws) {
  // Clean up current settings
  input_shapes_.clear();
  output_shapes_.clear();
  input_types_.clear();
  output_types_.clear();

  // Save execution data
  stream_ = ws->stream();

  // Collect the shapes and types for each input TensorList
  for (int i = 0; i < ws->NumInput(); ++i) {
    if (ws->InputIsType<CPUBackend>(i)) {
      auto& tensor_list = ws->Input<CPUBackend>(i);
      input_shapes_.push_back(tensor_list.shape());
      input_types_.push_back(tensor_list.type());
    } else {
      auto& tensor_list = ws->Input<GPUBackend>(i);
      input_shapes_.push_back(tensor_list.shape());
      input_types_.push_back(tensor_list.type());
    }
  }

  // Collect the shapes and types for each output TensorList
  for (int i = 0; i < ws->NumOutput(); ++i) {
    if (ws->OutputIsType<CPUBackend>(i)) {
      auto tensor_list = ws->Output<CPUBackend>(i);
      output_shapes_.push_back(tensor_list->shape());
      output_types_.push_back(tensor_list->type());
    } else {
      auto tensor_list = ws->Output<GPUBackend>(i);
      output_shapes_.push_back(tensor_list->shape());
      output_types_.push_back(tensor_list->type());
    }
  }
}

const vector<Dims>& BatchMeta::InputShape(int idx) const {
  NDLL_ENFORCE(idx >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)idx < input_shapes_.size(),
      "Index out of range." + std::to_string(idx) +
      " not in range [0, " + std::to_string(input_shapes_.size())
      + ")");
  return input_shapes_[idx];
}

const vector<Dims>& BatchMeta::OutputShape(int idx) const {
  NDLL_ENFORCE(idx >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)idx < output_shapes_.size(),
      "Index out of range." + std::to_string(idx) +
      " not in range [0, " + std::to_string(output_shapes_.size())
      + ")");
  return output_shapes_[idx];
}

const TypeInfo& BatchMeta::InputType(int idx) const {
  NDLL_ENFORCE(idx >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)idx < input_types_.size(),
      "Index out of range." + std::to_string(idx) +
      " not in range [0, " + std::to_string(input_types_.size())
      + ")");
  return input_types_[idx];
}

const TypeInfo& BatchMeta::OutputType(int idx) const {
  NDLL_ENFORCE(idx >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)idx < output_types_.size(),
      "Index out of range." + std::to_string(idx) +
      " not in range [0, " + std::to_string(output_types_.size())
      + ")");
  return output_types_[idx];
}

} // namespace ndll
