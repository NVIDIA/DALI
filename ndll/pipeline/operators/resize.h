// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_RESIZE_H_
#define NDLL_PIPELINE_OPERATORS_RESIZE_H_

#include <random>
#include <utility>
#include <vector>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/image/transform.h"
#include "ndll/pipeline/operator.h"

namespace ndll {

typedef enum {
    input_t,
    output_t
} io_type;

typedef std::pair<int, int> resize_t;

class ResizeAttr {
 public:
    ResizeAttr(const OpSpec &spec) :
            rand_gen_(time(nullptr)),
            random_resize_(spec.GetArgument<bool>("random_resize", false)),
            warp_resize_(spec.GetArgument<bool>("warp_resize", false)),
            image_type_(spec.GetArgument<NDLLImageType>("image_type", NDLL_RGB)),
            color_(IsColor(image_type_)), C_(color_ ? 3 : 1),
            random_crop_(spec.GetArgument<bool>("random_crop", false)),
            crop_h_(spec.GetArgument<int>("crop_h", -1)),
            crop_w_(spec.GetArgument<int>("crop_w", -1)),
            mirror_prob_(spec.GetArgument<float>("mirror_prob", 0.5f)),
            type_(spec.GetArgument<NDLLInterpType>("interp_type", NDLL_INTERP_LINEAR)) {
        resize_.first = spec.GetArgument<int>("resize_a", -1);
        resize_.second = spec.GetArgument<int>("resize_b", -1);

        // Validate input parameters
        NDLL_ENFORCE(resize_.first > 0 && resize_.second > 0);
        NDLL_ENFORCE(resize_.first <= resize_.second);
        NDLL_ENFORCE(mirror_prob_ <= 1.f && mirror_prob_ >= 0.f);
    }

    void SetSize(NDLLSize &in_size, const vector<Index> &shape,
                 const resize_t &rand, NDLLSize &out_size) const;

    inline vector<NDLLSize> &sizes(io_type type)            { return sizes_[type]; }
    inline NDLLSize &size(io_type type, size_t idx)         { return sizes(type)[idx]; }
    inline const resize_t &newSizes(size_t idx) const       { return per_sample_rand_[idx]; }
    inline int randomUniform(int max, int min = 0) const    {
                return std::uniform_int_distribution<>(min, max)(rand_gen_);
            }

    void DefineCrop(NDLLSize &out_size, int *pCropX, int *pCropY) const;

    bool CropNeeded(const NDLLSize &out_size) const {
        return 0 < crop_h_ && crop_h_ <= out_size.height &&
               0 < crop_w_ && crop_w_ <= out_size.width;
    }

 protected:
    inline vector<const uint8*> *inputImages()              { return &input_ptrs_; }
    inline vector<uint8 *> *outputImages()                  { return &output_ptrs_; }
    inline const resize_t &resize() const                   { return resize_; };

    mutable std::mt19937 rand_gen_;

    // Resize meta-data
    bool random_resize_;
    bool warp_resize_;
    resize_t resize_;

    // Input/output channels meta-data
    NDLLImageType image_type_;
    bool color_;
    int C_;

    bool random_crop_;
    int crop_h_, crop_w_;
    float mirror_prob_;

    // Interpolation type
    NDLLInterpType type_;

    // store per-thread data for same resize on multiple data
    std::vector<resize_t> per_sample_rand_;

    vector<const uint8*> input_ptrs_;
    vector<uint8*> output_ptrs_;

    vector<NDLLSize> sizes_[2];
};

template <typename Backend>
class Resize : public Operator, public ResizeAttr {
 public:
  explicit inline Resize(const OpSpec &spec) :
    Operator(spec), ResizeAttr(spec) {
      // Resize per-image data
      input_ptrs_.resize(batch_size_);
      output_ptrs_.resize(batch_size_);
      sizes_[0].resize(batch_size_);
      sizes_[1].resize(batch_size_);

      // Per set-of-samples random numbers
      per_sample_rand_.resize(batch_size_);
  }

  virtual inline ~Resize() = default;

 protected:
  inline void SetupSharedSampleParams(DeviceWorkspace* ws) override {
    const int resize_a = resize_.first;
    const int resize_b = resize_.second;
    for (int i = 0; i < batch_size_; ++i) {
      auto rand_a = randomUniform(resize_b, resize_a);
      auto rand_b = randomUniform(resize_b, resize_a);

      per_sample_rand_[i] = std::make_pair(rand_a, rand_b);
    }
  }

  void RunPerSampleCPU(SampleWorkspace *ws, const int idx) override {
        const auto &input = ws->Input<CPUBackend>(idx);
        auto output = ws->Output<CPUBackend>(idx);

        const vector <Index> &input_shape = input.shape();
        NDLLSize &out_size = size(output_t, 0);
        SetSize(size(input_t, 0), input_shape, resize(), out_size);

        DataDependentSetupCPU(input, output, "Resize", &input_ptrs_,
                              &output_ptrs_, NULL, &out_size);
        NDLL_CALL(BatchedResize((const uint8 **) input_ptrs_.data(), 1, C_, sizes(input_t).data(),
                                    output_ptrs_.data(), sizes(output_t).data(), type_));
  }

  inline void RunBatchedGPU(DeviceWorkspace *ws, const int idx) override {
    const auto &input = ws->Input<GPUBackend>(idx);
    auto output = ws->Output<GPUBackend>(idx);

    DataDependentSetupGPU(input, output, batch_size_, false,
                            inputImages(), outputImages(), NULL, this);

    // Run the kernel
    cudaStream_t old_stream = nppGetStream();
    nppSetStream(ws->stream());
    BatchedResize(
        (const uint8**)input_ptrs_.data(),
        batch_size_, C_, sizes(input_t).data(),
        output_ptrs_.data(), sizes(output_t).data(),
        type_);
    nppSetStream(old_stream);
  }

  USE_OPERATOR_MEMBERS();
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_RESIZE_H_
