// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// limitations under the License.

#include <vector>
#include <string>
#include <map>
#include <memory>

#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/operators/imgcodec/operator_utils.h"
#include "dali/operators/image/crop/crop_attr.h"
#include "dali/operators/generic/slice/slice_attr.h"
#include "dali/operators/image/crop/random_crop_attr.h"
#include "dali/imgcodec/image_decoder_interfaces.h"
#include "dali/imgcodec/util/output_shape.h"

#ifndef DALI_OPERATORS_IMGCODEC_DECODER_H_
#define DALI_OPERATORS_IMGCODEC_DECODER_H_

namespace dali {
namespace imgcodec {

template <typename Backend>
class DecoderBase : public Operator<Backend> {
 public:
  ~DecoderBase() override = default;

 protected:
  explicit DecoderBase(const OpSpec &spec) : Operator<Backend>(spec) {
    device_id_ = spec.GetArgument<int>("device_id");
    opts_.format = spec.GetArgument<DALIImageType>("output_type");
    opts_.dtype = spec.GetArgument<DALIDataType>("dtype");
    opts_.use_orientation = spec.GetArgument<bool>("adjust_orientation");
    GetDecoderSpecificArguments(spec);
  }

  virtual void SetupRoiGenerator(const OpSpec &spec, const Workspace &ws) {}

  virtual ROI GetRoi(const OpSpec &spec, const Workspace &ws, std::size_t data_idx,
                     TensorShape<> shape) {
    return {};
  }

  template<typename T>
  void GetDecoderSpecificArgument(const OpSpec &spec, const std::string &name,
                                  std::function<bool(const T&)> validator = [](T x){return true;}) {
    T value;
    if (spec.TryGetArgument(value, name)) {
      auto value = spec.GetArgument<T>(name);
      if (!validator(value)) {
        DALI_FAIL(make_string("Invalid value for decoder-specific parameter ", name));
      }
      decoder_params_[name] = value;
    }
  }

  void GetDecoderSpecificArguments(const OpSpec &spec) {
    GetDecoderSpecificArgument<uint64_t>(spec, "hybrid_huffman_threshold");
    GetDecoderSpecificArgument<size_t>(spec, "device_memory_padding");
    GetDecoderSpecificArgument<size_t>(spec, "host_memory_padding");
    GetDecoderSpecificArgument<size_t>(spec, "device_memory_padding_jpeg2k");
    GetDecoderSpecificArgument<size_t>(spec, "host_memory_padding_jpeg2k");
    GetDecoderSpecificArgument<float>(spec, "hw_decoder_load");
    GetDecoderSpecificArgument<size_t>(spec, "preallocate_width_hint");
    GetDecoderSpecificArgument<size_t>(spec, "preallocate_height_hint");
    GetDecoderSpecificArgument<bool>(spec, "use_fast_idct");
    GetDecoderSpecificArgument<bool>(spec, "jpeg_fancy_upsampling");
    GetDecoderSpecificArgument<int>(spec, "num_threads");

    if (decoder_params_.count("nvjpeg_num_threads") == 0)
      decoder_params_["nvjpeg_num_threads"] = decoder_params_["num_threads"];

    if (decoder_params_.count("nvjpeg2k_num_threads") == 0)
      decoder_params_["nvjpeg2k_num_threads"] = decoder_params_["nvjpeg2k_num_threads"];
  }

  bool CanInferOutputs() const override { return true; }

  void SetupShapes(const OpSpec &spec, const Workspace &ws,
                    std::vector<OutputDesc> &output_descs, ThreadPool &tp) {
    output_descs.resize(1);
    auto &input = ws.template Input<CPUBackend>(0);
    int nsamples = input.num_samples();
    srcs_.resize(nsamples);
    src_ptrs_.resize(nsamples);
    rois_.resize(nsamples);
    SetupRoiGenerator(spec, ws);
    TensorListShape<> shapes;
    shapes.resize(nsamples, 3);
    auto *decoder = GetDecoderInstance();

    for (int i = 0; i < shapes.size(); i++) {
      tp.AddWork([i, decoder, &input, &shapes, &ws, &spec, this] (int tid) {
        srcs_[i] = SampleAsImageSource(input[i], input.GetMeta(i).GetSourceInfo());
        src_ptrs_[i] = &srcs_[i];
        auto info = decoder->GetInfo(src_ptrs_[i]);
        ROI roi = GetRoi(spec, ws, i, info.shape);
        rois_[i] = roi;
        OutputShape(shapes.tensor_shape_span(i), info, this->opts_, roi);
      }, 0);
    }
    tp.RunAll();
    output_descs[0] = {shapes, opts_.dtype};
  }

  ImageDecoder *GetDecoderInstance() {
    if (!decoder_ptr_)
      decoder_ptr_ = std::make_unique<ImageDecoder>(device_id_, false, decoder_params_);
    return decoder_ptr_.get();
  }

  std::map<std::string, std::any> decoder_params_;
  std::vector<ImageSource> srcs_;
  std::vector<ImageSource *> src_ptrs_;
  std::vector<ROI> rois_;
  DecodeParams opts_;
  std::unique_ptr<ImageDecoder> decoder_ptr_;
  int device_id_;
};

inline ROI RoiFromCropWindowGenerator(const CropWindowGenerator& generator, TensorShape<> shape) {
  shape = shape.first(2);
  auto crop_window = generator(shape, "HW");
  crop_window.EnforceInRange(shape);
  TensorShape<> end = crop_window.anchor;
  for (int d = 0; d < end.size(); d++) {
    end[d] += crop_window.shape[d];
  }
  return {crop_window.anchor, end};
}

template <typename Decoder, typename Backend>
class WithCropAttr : public Decoder, CropAttr {
 public:
  explicit WithCropAttr(const OpSpec &spec) : Decoder(spec), CropAttr(spec) {}

 protected:
  void SetupRoiGenerator(const OpSpec &spec, const Workspace &ws) override {
    CropAttr::ProcessArguments(spec, ws);
  }

  ROI GetRoi(const OpSpec &spec, const Workspace &ws, std::size_t data_idx,
             TensorShape<> shape) override {
    return RoiFromCropWindowGenerator(GetCropWindowGenerator(data_idx), shape);
  }
};

template <typename Decoder, typename Backend>
class WithSliceAttr : public Decoder, SliceAttr {
 public:
  explicit WithSliceAttr(const OpSpec &spec) : Decoder(spec), SliceAttr(spec) {}
 protected:
  void SetupRoiGenerator(const OpSpec &spec, const Workspace &ws) override {
    SliceAttr::ProcessArguments<Backend>(spec, ws);
  }

  ROI GetRoi(const OpSpec &spec, const Workspace &ws, std::size_t data_idx,
             TensorShape<> shape) override {
    return RoiFromCropWindowGenerator(SliceAttr::GetCropWindowGenerator(data_idx), shape);
  }
};

template <typename Decoder, typename Backend>
class WithRandomCropAttr : public Decoder, RandomCropAttr {
 public:
  explicit WithRandomCropAttr(const OpSpec &spec) : Decoder(spec), RandomCropAttr(spec) {}
 protected:
  void SetupRoiGenerator(const OpSpec &spec, const Workspace &ws) override {}

  ROI GetRoi(const OpSpec &spec, const Workspace &ws, std::size_t data_idx,
             TensorShape<> shape) override {
    return RoiFromCropWindowGenerator(RandomCropAttr::GetCropWindowGenerator(data_idx), shape);
  }
};

class ImgcodecHostDecoder : public DecoderBase<CPUBackend> {
 public:
  explicit ImgcodecHostDecoder(const OpSpec &spec);

  ~ImgcodecHostDecoder() override = default;

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_descs, const Workspace &ws) override;
  void RunImpl(Workspace &ws) override;

  USE_OPERATOR_MEMBERS();
  using Operator<CPUBackend>::RunImpl;
};

class ImgcodecMixedDecoder : public DecoderBase<MixedBackend> {
 public:
  explicit ImgcodecMixedDecoder(const OpSpec &spec);
  ~ImgcodecMixedDecoder() override = default;

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_descs,
                 const Workspace &ws) override;
  void Run(Workspace &ws) override;

  USE_OPERATOR_MEMBERS();
  using Operator<MixedBackend>::Run;

  ThreadPool thread_pool_;
};

}  // namespace imgcodec
}  // namespace dali
#endif  // DALI_OPERATORS_IMGCODEC_DECODER_H_
