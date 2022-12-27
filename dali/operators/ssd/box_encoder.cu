// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/ssd/box_encoder.cuh"
#include <cuda.h>
#include <vector>
#include <utility>

namespace dali {
__host__ __device__ inline float4 ToCenterWidthHeight(const float4 &box) {
    return {
      0.5f * (box.x + box.z),
      0.5f * (box.y + box.w),
      box.z - box.x,
      box.w - box.y};
}

void BoxEncoder<GPUBackend>::PrepareAnchors(const vector<float> &anchors) {
  DALI_ENFORCE(
    (anchors.size() % BoundingBox::size) == 0,
    "Anchors size must be divisible by 4, actual value = " + std::to_string(anchors.size()));

  anchor_count_ = anchors.size() / BoundingBox::size;
  anchors_.Resize({anchor_count_, static_cast<int64_t>(BoundingBox::size)}, DALI_FLOAT);
  anchors_as_center_wh_.Resize({anchor_count_, static_cast<int64_t>(BoundingBox::size)},
                               DALI_FLOAT);

  auto anchors_data_cpu = reinterpret_cast<const float4 *>(anchors.data());

  vector<float4> anchors_as_center_wh(anchor_count_);
  for (unsigned int anchor = 0; anchor < anchor_count_; ++anchor)
    anchors_as_center_wh[anchor] = ToCenterWidthHeight(anchors_data_cpu[anchor]);

  auto anchors_data = anchors_.mutable_data<float>();
  auto anchors_as_center_wh_data = anchors_as_center_wh_.mutable_data<float>();
  MemCopy(anchors_data, anchors.data(), anchor_count_ * BoundingBox::size * sizeof(float));
  MemCopy(
    anchors_as_center_wh_data,
    anchors_as_center_wh.data(),
    anchor_count_ * BoundingBox::size * sizeof(float));
}

__device__ __forceinline__ float CalculateIou(const float4 &b1, const float4 &b2) {
  float l = cuda_max(b1.x, b2.x);
  float t = cuda_max(b1.y, b2.y);
  float r = cuda_min(b1.z, b2.z);
  float b = cuda_min(b1.w, b2.w);
  float first = cuda_max(r - l, 0.0f);
  float second = cuda_max(b - t, 0.0f);
  volatile float intersection = first * second;
  volatile float area1 = (b1.w - b1.y) * (b1.z - b1.x);
  volatile float area2 = (b2.w - b2.y) * (b2.z - b2.x);

  return intersection / (area1 + area2 - intersection);
}

__device__ inline void FindBestMatch(const int N, volatile float *vals, volatile int *idx) {
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      if (vals[threadIdx.x] <= vals[threadIdx.x + stride]) {
        if (vals[threadIdx.x] == vals[threadIdx.x + stride]) {
          idx[threadIdx.x] = cuda_max(idx[threadIdx.x], idx[threadIdx.x + stride]);
        } else {
          vals[threadIdx.x] = vals[threadIdx.x + stride];
          idx[threadIdx.x] = idx[threadIdx.x + stride];
        }
      }
    }
    __syncthreads();
  }
}

// Scale argument is used to maintain numerical consistency with reference implementation:
// https://github.com/mlcommons/training/blob/master/single_stage_detector/ssd/utils.py
__device__ float4 MatchOffsets(
  float4 box, float4 anchor, const float *means, const float *stds, float scale) {
  box.x *= scale; box.y *= scale; box.z *= scale; box.w *= scale;
  anchor.x *= scale; anchor.y *= scale; anchor.z *= scale; anchor.w *= scale;

  float x = ((box.x - anchor.x) / anchor.z - means[0]) / stds[0];
  float y = ((box.y - anchor.y) / anchor.w - means[1]) / stds[1];
  float z = (log(box.z / anchor.z) - means[2]) / stds[2];
  float w = (log(box.w / anchor.w) - means[3]) / stds[3];

  return {x, y, z, w};
}

__device__ void WriteMatchesToOutput(
  unsigned int anchor_count, float criteria, int *labels_out, const int *labels_in,
  float4 *boxes_out, const float4 *boxes_in,
  volatile int *best_box_idx, volatile float *best_box_iou, bool offset,
  const float* means, const float* stds, float scale, const float4 *anchors_as_cwh) {
  for (unsigned int anchor = threadIdx.x; anchor < anchor_count; anchor += blockDim.x) {
    if (best_box_iou[anchor] > criteria) {
      int box_idx = best_box_idx[anchor];
      labels_out[anchor] = labels_in[box_idx];
      float4 box = boxes_in[box_idx];

      if (!offset)
        boxes_out[anchor] = ToCenterWidthHeight(box);
      else
        boxes_out[anchor] = MatchOffsets(
          ToCenterWidthHeight(box), anchors_as_cwh[anchor], means, stds, scale);
    }
  }
}

__device__ void MatchBoxWithAnchors(
  const float4 &box, const int box_idx, unsigned int anchor_count, const float4 *anchors,
  volatile int *best_anchor_idx_tmp, volatile float *best_anchor_iou_tmp,
  volatile int *best_box_idx, volatile float *best_box_iou) {
  float best_anchor_iou = -1.0f;
  int best_anchor_idx = -1;

  for (unsigned int anchor = threadIdx.x; anchor < anchor_count; anchor += blockDim.x) {
    float new_val = CalculateIou(box, anchors[anchor]);

    if (new_val >= best_anchor_iou) {
        best_anchor_iou = new_val;
        best_anchor_idx = anchor;
    }

    if (new_val >= best_box_iou[anchor]) {
        best_box_iou[anchor] = new_val;
        best_box_idx[anchor] = box_idx;
    }
  }

  best_anchor_iou_tmp[threadIdx.x] = best_anchor_iou;
  best_anchor_idx_tmp[threadIdx.x] = best_anchor_idx;
}

template <int BLOCK_SIZE>
__global__ void Encode(const BoxEncoderSampleDesc *samples, const int anchor_count,
                       const float4 *anchors, const float criteria, int *box_idx_buffer,
                       float *box_iou_buffer, bool offset, const float *means, const float *stds,
                       float scale, const float4 *anchors_as_cwh) {
  const int sample_idx = blockIdx.x;
  const auto &sample = samples[sample_idx];

  // Remark: This algorithm is very fragile to floating point arithmetic effects.
  // For now, excessive use of volatile in this code,
  // makes it conform to reference solution in terms of resulting encoding.

  __shared__ volatile int best_anchor_idx_tmp[BLOCK_SIZE];
  __shared__ volatile float best_anchor_iou_tmp[BLOCK_SIZE];

  volatile int *best_box_idx = box_idx_buffer + sample_idx * anchor_count;
  volatile float *best_box_iou = box_iou_buffer + sample_idx * anchor_count;

  for (int box_idx = 0; box_idx < sample.in_box_count; ++box_idx) {
    MatchBoxWithAnchors(
      sample.boxes_in[box_idx],
      box_idx,
      anchor_count,
      anchors,
      best_anchor_idx_tmp,
      best_anchor_iou_tmp,
      best_box_idx,
      best_box_iou);

    __syncthreads();

    FindBestMatch(blockDim.x, best_anchor_iou_tmp, best_anchor_idx_tmp);
    __syncthreads();

    if (threadIdx.x == 0) {
      int idx = best_anchor_idx_tmp[0];
      best_box_idx[idx] = box_idx;
      best_box_iou[idx] = 2.f;
    }
    __syncthreads();
  }
  __syncthreads();

  WriteMatchesToOutput(
    anchor_count,
    criteria,
    sample.labels_out,
    sample.labels_in,
    sample.boxes_out,
    sample.boxes_in,
    best_box_idx,
    best_box_iou,
    offset,
    means,
    stds,
    scale,
    anchors_as_cwh);
}

std::pair<int *, float *> BoxEncoder<GPUBackend>::ClearBuffers(const cudaStream_t &stream) {
  auto best_box_idx_data = best_box_idx_.mutable_data<int>();
  auto best_box_iou_data = best_box_iou_.mutable_data<float>();

  CUDA_CALL(cudaMemsetAsync(
    best_box_idx_data, 0, curr_batch_size_ * anchor_count_ * sizeof(int), stream));
  CUDA_CALL(cudaMemsetAsync(
    best_box_iou_data, 0, curr_batch_size_ * anchor_count_ * sizeof(float), stream));

  return {best_box_idx_data, best_box_iou_data};
}

void BoxEncoder<GPUBackend>::ClearLabels(TensorList<GPUBackend> &labels_out,
                                         const cudaStream_t &stream) {
  for (int sample = 0; sample < curr_batch_size_; ++sample) {
    CUDA_CALL(cudaMemsetAsync(labels_out.mutable_tensor<int>(sample), 0,
                              anchor_count_ * sizeof(int), stream));
  }
}

void BoxEncoder<GPUBackend>::WriteAnchorsToOutput(TensorList<GPUBackend> &boxes_out,
                                                  const cudaStream_t &stream) {
  if (!curr_batch_size_)
    return;
  const auto *anchors_to_copy = anchors_as_center_wh_.data<float>();
  auto *first_sample_boxes_out = boxes_out.mutable_tensor<float>(0);
  // Host -> device copy of anchors for first sample
  MemCopy(first_sample_boxes_out, anchors_to_copy,
          anchor_count_ * BoundingBox::size * sizeof(float), stream);
  // Device -> device copy for the rest
  for (int sample = 1; sample < curr_batch_size_; ++sample) {
    auto *boxes_out_data = boxes_out.mutable_tensor<float>(sample);
    MemCopy(boxes_out_data, first_sample_boxes_out,
            anchor_count_ * BoundingBox::size * sizeof(float), stream);
  }
}

void BoxEncoder<GPUBackend>::ClearOutput(TensorList<GPUBackend> &boxes_out,
                                         const cudaStream_t &stream) {
  for (int sample = 0; sample < curr_batch_size_; ++sample) {
    auto *boxes_out_data = boxes_out.mutable_tensor<float>(sample);
    CUDA_CALL(cudaMemsetAsync(boxes_out_data, 0, anchor_count_ * BoundingBox::size * sizeof(float),
                              stream));
  }
}

std::pair<TensorListShape<>, TensorListShape<>>
BoxEncoder<GPUBackend>::CalculateDims(
  const TensorList<GPUBackend> &boxes_input) {
  TensorListShape<> boxes_output_shape(boxes_input.num_samples(), kBoxesOutputDim);
  TensorListShape<> labels_output_shape(boxes_input.num_samples(), kLabelsOutputDim);

  for (int i = 0; i < boxes_input.num_samples(); i++) {
    boxes_output_shape.set_tensor_shape(i,
        {anchor_count_, static_cast<int64_t>(BoundingBox::size)});
    labels_output_shape.set_tensor_shape(i, {anchor_count_});
  }

  return {boxes_output_shape, labels_output_shape};
}

void BoxEncoder<GPUBackend>::RunImpl(Workspace &ws) {
  const auto &boxes_input = ws.Input<GPUBackend>(kBoxesInId);
  const auto &labels_input = ws.Input<GPUBackend>(kLabelsInId);
  assert(ws.GetInputBatchSize(kBoxesInId) == ws.GetInputBatchSize(kLabelsInId));
  auto curr_batch_size = ws.GetInputBatchSize(kBoxesInId);

  const auto anchors_data = reinterpret_cast<const float4 *>(anchors_.data<float>());
  const auto anchors_as_cwh_data =
    reinterpret_cast<const float4 *>(anchors_as_center_wh_.data<float>());

  const auto buffers = ClearBuffers(ws.stream());

  auto dims = CalculateDims(boxes_input);

  auto &boxes_output = ws.Output<GPUBackend>(kBoxesOutId);
  boxes_output.Resize(dims.first, boxes_input.type());

  auto &labels_output = ws.Output<GPUBackend>(kLabelsOutId);
  labels_output.Resize(dims.second, labels_input.type());

  samples.resize(curr_batch_size_);
  for (int sample_idx = 0; sample_idx < curr_batch_size_; sample_idx++) {
    auto &sample = samples[sample_idx];
    sample.boxes_out = reinterpret_cast<float4 *>(boxes_output.mutable_tensor<float>(sample_idx));
    sample.labels_out = labels_output.mutable_tensor<int>(sample_idx);
    sample.boxes_in = reinterpret_cast<const float4 *>(boxes_input.tensor<float>(sample_idx));
    sample.labels_in = labels_input.tensor<int>(sample_idx);
    sample.in_box_count = boxes_input.shape().tensor_shape_span(sample_idx)[0];
  }

  const auto means_data = means_.data<float>();
  const auto stds_data = stds_.data<float>();

  ClearLabels(labels_output, ws.stream());

  if (!offset_)
    WriteAnchorsToOutput(boxes_output, ws.stream());
  else
    ClearOutput(boxes_output, ws.stream());

  samples_dev.from_host(samples, ws.stream());

  Encode<BlockSize><<<curr_batch_size, BlockSize, 0, ws.stream()>>>(
    samples_dev.data(),
    anchor_count_,
    anchors_data,
    criteria_,
    buffers.first,
    buffers.second,
    offset_,
    means_data,
    stds_data,
    scale_,
    anchors_as_cwh_data);
}

DALI_REGISTER_OPERATOR(BoxEncoder, BoxEncoder<GPUBackend>, GPU);

}  // namespace dali
