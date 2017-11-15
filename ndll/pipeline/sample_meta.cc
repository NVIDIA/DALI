#include "ndll/pipeline/sample_meta.h"

#include "ndll/pipeline/sample_workspace.h"

namespace ndll {

void SampleMeta::SetMeta(SampleWorkspace *ws) {
  // Clean up current settings
  input_shapes_.clear();
  output_shapes_.clear();
  input_types_.clear();
  output_types_.clear();

  // Save execution data
  data_idx_ = ws->data_idx();
  thread_idx_ = ws->thread_idx();

  // Collect the shapes and types for each input Tensor
  for (int i = 0; i < ws->NumInput(); ++i) {
    if (ws->InputIsType<CPUBackend>(i)) {
      auto &tensor = ws->Input<CPUBackend>(i);
      input_shapes_.push_back(tensor.shape());
      input_types_.push_back(tensor.type());
    } else {
      auto &tensor = ws->Input<GPUBackend>(i);
      input_shapes_.push_back(tensor.shape());
      input_types_.push_back(tensor.type());
    }
  }

  // Collect the shapes and types for each output Tensor
  for (int i = 0; i < ws->NumOutput(); ++i) {
    if (ws->OutputIsType<CPUBackend>(i)) {
      auto tensor = ws->Output<CPUBackend>(i);
      output_shapes_.push_back(tensor->shape());
      output_types_.push_back(tensor->type());
    } else {
      auto tensor = ws->Output<GPUBackend>(i);
      output_shapes_.push_back(tensor->shape());
      output_types_.push_back(tensor->type());
    }
  }
}

const vector<Index>& SampleMeta::InputShape(int idx) const {
  NDLL_ENFORCE(idx >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)idx < input_shapes_.size(),
      "Index out of range." + std::to_string(idx) +
      " not in range [0, " + std::to_string(input_shapes_.size())
      + ")");
  return input_shapes_[idx];
}

const vector<Index>& SampleMeta::OutputShape(int idx) const {
  NDLL_ENFORCE(idx >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)idx < output_shapes_.size(),
      "Index out of range." + std::to_string(idx) +
      " not in range [0, " + std::to_string(output_shapes_.size())
      + ")");
  return output_shapes_[idx];
}

const TypeInfo& SampleMeta::InputType(int idx) const {
  NDLL_ENFORCE(idx >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)idx < input_types_.size(),
      "Index out of range." + std::to_string(idx) +
      " not in range [0, " + std::to_string(input_types_.size())
      + ")");
  return input_types_[idx];
}

const TypeInfo& SampleMeta::OutputType(int idx) const {
  NDLL_ENFORCE(idx >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)idx < output_types_.size(),
      "Index out of range." + std::to_string(idx) +
      " not in range [0, " + std::to_string(output_types_.size())
      + ")");
  return output_types_[idx];
}

} // namespace ndll
