#ifndef NDLL_PIPELINE_OPERATORS_RESIZE_H_
#define NDLL_PIPELINE_OPERATORS_RESIZE_H_

#include <random>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/image/transform.h"
#include "ndll/pipeline/operator.h"

namespace ndll {

template <typename Backend>
class Resize : public Operator<Backend> {
public:
  inline Resize(const OpSpec &spec) :
    Operator<Backend>(spec),
    rand_gen_(time(nullptr)),
    random_resize_(spec.GetArgument<bool>("random_resize", false)),
    warp_resize_(spec.GetArgument<bool>("warp_resize", false)),
    resize_a_(spec.GetArgument<int>("resize_a", -1)),
    resize_b_(spec.GetArgument<int>("resize_b", -1)),
    image_type_(spec.GetArgument<NDLLImageType>("image_type", NDLL_RGB)),
    color_(IsColor(image_type_)), C_(color_ ? 3 : 1),
    type_(spec.GetArgument<NDLLInterpType>("interp_type", NDLL_INTERP_LINEAR)) {
    // Validate input parameters
    NDLL_ENFORCE(resize_a_ > 0 && resize_b_ > 0);
    NDLL_ENFORCE(resize_a_ <= resize_b_);

    // Resize per-image data
    input_ptrs_.resize(batch_size_);
    output_ptrs_.resize(batch_size_);
    input_sizes_.resize(batch_size_);
    output_sizes_.resize(batch_size_);    
  }
    
  virtual inline ~Resize() = default;

protected:
  inline void RunBatchedGPU(DeviceWorkspace *ws) override {
    DataDependentSetup(ws);

    // Run the kernel
    cudaStream_t old_stream = nppGetStream();
    nppSetStream(ws->stream());
    BatchedResize(
        (const uint8**)input_ptrs_.data(),
        batch_size_, C_, input_sizes_.data(),
        output_ptrs_.data(), output_sizes_.data(),
        type_
        );
    nppSetStream(old_stream);
  }

  inline void DataDependentSetup(DeviceWorkspace *ws) {
    auto &input = ws->Input<GPUBackend>(0);
    auto output = ws->Output<GPUBackend>(0);
    NDLL_ENFORCE(IsType<uint8>(input.type()),
        "Expected input data stored in uint8.");

    vector<Dims> output_shape(batch_size_);
    for (int i = 0; i < batch_size_; ++i) {
      // Verify the inputs
      vector<Index> input_shape = input.tensor_shape(i);
      NDLL_ENFORCE(input_shape.size() == 3,
          "Expects 3-dimensional image input.");
      NDLL_ENFORCE(input_shape[2] == C_,
          "Input channel dimension does not match "
          "the output channel argument.");
      
      // Select resize dimensions for the output
      NDLLSize &in_size = input_sizes_[i];
      in_size.height = input_shape[0];
      in_size.width = input_shape[1];
      
      NDLLSize &out_size = output_sizes_[i];
      if (random_resize_ && warp_resize_) {
        // random resize + warp. Select a new size for both dims of
        // the image uniformly from the range [resize_a_, resize_b_]
        out_size.height =
          std::uniform_int_distribution<>(resize_a_, resize_b_)(rand_gen_);
        out_size.width =
          std::uniform_int_distribution<>(resize_a_, resize_b_)(rand_gen_);
      } else if (random_resize_) {
        // random + no warp. We select a new size of the smallest side
        // of the image uniformly in the range [resize_a_, resize_b_]
        if (in_size.width < in_size.height) {
          out_size.width =
            std::uniform_int_distribution<>(resize_a_, resize_b_)(rand_gen_);
          out_size.height =
            static_cast<float>(in_size.height) / in_size.width * out_size.width;
        } else {
          out_size.height =
            std::uniform_int_distribution<>(resize_a_, resize_b_)(rand_gen_);
          out_size.width =
            static_cast<float>(in_size.width) / in_size.height * out_size.height;
        }
      } else if (warp_resize_) {
        // no random + warp. We take the new dims to be h = resize_a_
        // and w = resize_b_
        out_size.height = resize_a_;
        out_size.width = resize_b_;
      } else { 
        // no random + no warp. In this mode resize_b_ is ignored and
        // the input image is resizes such that the smallest side is
        // >= resize_a_
        if (in_size.width < in_size.height) {
          if (in_size.width < resize_a_) {
            out_size.width = resize_a_;
            out_size.height =
              static_cast<float>(in_size.height) / in_size.width * out_size.width;
          }
        } else {
          if (in_size.height < resize_a_) {
            out_size.height = resize_a_;
            out_size.width =
              static_cast<float>(in_size.width) / in_size.height * out_size.height;
          }
        }
      }

      // Collect the output shapes
      output_shape[i] = {out_size.height, out_size.width, C_};
    }

    // Resize the output
    output->Resize(output_shape);

    // Collect the pointers for execution
    for (int i = 0; i < batch_size_; ++i) {
      input_ptrs_[i] = input.template tensor<uint8>(i);
      output_ptrs_[i] = output->template mutable_tensor<uint8>(i);
    }
  }
  
  std::mt19937 rand_gen_;
  
  // Resize meta-data
  bool random_resize_;
  bool warp_resize_;
  int resize_a_, resize_b_;

  // Input/output channels meta-data
  NDLLImageType image_type_;
  bool color_;
  int C_;

  // Interpolation type
  NDLLInterpType type_;
  
  vector<const uint8*> input_ptrs_;
  vector<uint8*> output_ptrs_;
  
  vector<NDLLSize> input_sizes_, output_sizes_;

  USE_OPERATOR_MEMBERS();
};

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_RESIZE_H_
