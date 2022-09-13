// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <map>
#include <vector>
#include "dali/core/cuda_error.h"
#include "dali/core/mm/memory.h"
#include "dali/imgcodec/image_decoder.h"
#include "dali/imgcodec/util/output_shape.h"

namespace dali {
namespace imgcodec {

ImageDecoder::ImageDecoder(int device_id,
                           bool lazy_init,
                           const std::map<std::string, any> &params,
                           std::function<bool(ImageDecoderFactory *)> decoder_filter) {
  if (device_id == -1)
    CUDA_CALL(cudaGetDevice(&device_id));
  device_id_ = device_id;
  DeviceGuard dg(device_id);

  for (auto *format : FormatRegistry().Formats()) {
    for (auto *factory : format->Decoders()) {
      if (!factory->IsSupported(device_id) && !factory->IsSupported(CPU_ONLY_DEVICE_ID))
        continue;
      if (decoder_filter(factory))
        filtered_.emplace(format, factory);
    }
  }

  InitWorkers(lazy_init);
}

ImageDecoder::~ImageDecoder() {
  for (auto &[_, w] : workers_)
    w->stop();
}


bool ImageDecoder::CanDecode(DecodeContext, ImageSource *, DecodeParams, const ROI &) {
  return true;
}

std::vector<bool> ImageDecoder::CanDecode(DecodeContext,
                                          cspan<ImageSource *> in,
                                          DecodeParams,
                                          cspan<ROI>) {
  return std::vector<bool>(in.size(), true);
}

ImageInfo ImageDecoder::GetInfo(ImageSource *encoded) const {
  if (auto *format = FormatRegistry().GetImageFormat(encoded))
    return format->Parser()->GetInfo(encoded);
  else
    DALI_FAIL(make_string("Cannot parse the image: ", encoded->SourceInfo()));
}


std::vector<bool> ImageDecoder::GetInfo(span<ImageInfo> info, span<ImageSource*> sources) const {
  assert(info.size() == sources.size());
  std::vector<bool> ret(size(info), false);
  for (int i = 0, n = info.size(); i < n; i++) {
    if (auto *format = FormatRegistry().GetImageFormat(sources[i])) {
      info[i] = format->Parser()->GetInfo(sources[i]);
      ret[i] = true;
    } else {
      info[i] = {};
      ret[i] = false;
    }
  }
  return ret;
}


void ImageDecoder::CalculateOutputShape(TensorListShape<> &shape,
                                        span<SampleView<CPUBackend>> out,
                                        cspan<ImageSource *> in,
                                        DecodeParams opts,
                                        cspan<ROI> rois) const {
  int n = in.size();
  for (int i = 0; i < n; i++) {
    if (auto *format = FormatRegistry().GetImageFormat(in[i])) {
      ImageInfo info = format->Parser()->GetInfo(in[i]);
      if (i == 0) {
        shape.resize(n, info.shape.size());
      }
      OutputShape(shape.tensor_shape_span(i), info, opts, rois.empty() ? ROI() : rois[i]);
    } else {
      if (const char *info = in[i]->SourceInfo())
        throw std::runtime_error(make_string("Cannot get image info for \"", info, "\""));
      else
        throw std::runtime_error(make_string("Cannot get image info for image # ", i));
    }
  }
}

bool ImageDecoder::SetParam(const char *key, const any &value) {
  params_[key] = value;

  for (auto &[_, worker] : workers_)
    if (auto *dec = worker->decoder(false))
      dec->SetParam(key, value);

  return true;
}

any ImageDecoder::GetParam(const char *key) const {
  auto it = params_.find(key);
  return it != params_.end() ? it->second : any{};
}

int ImageDecoder::SetParams(const std::map<std::string, any> &params) {
  for (auto &[k, v] : params)
    params_[k] = v;

  for (auto &[_, worker] : workers_)
    if (auto *dec = worker->decoder(false))
      dec->SetParams(params);

  return params.size();
}


DecodeResult ImageDecoder::Decode(DecodeContext ctx,
                                  SampleView<CPUBackend> out,
                                  ImageSource *in,
                                  DecodeParams opts,
                                  const ROI &roi) {
  return ScheduleDecode(ctx, out, in, opts, roi).get_one(0);
}

DecodeResult ImageDecoder::Decode(DecodeContext ctx,
                                  SampleView<GPUBackend> out,
                                  ImageSource *in,
                                  DecodeParams opts,
                                  const ROI &roi) {
  return ScheduleDecode(ctx, out, in, opts, roi).get_one(0);
}


/**
 * @brief Decodes a single image to device buffers
 */
std::vector<DecodeResult> ImageDecoder::Decode(DecodeContext ctx,
                                               span<SampleView<CPUBackend>> out,
                                               cspan<ImageSource *> in,
                                               DecodeParams opts,
                                               cspan<ROI> rois) {
  return ScheduleDecode(ctx, out, in, opts, rois).get_all_copy();
}


/**
 * @brief Decodes a single image to device buffers
 */
std::vector<DecodeResult> ImageDecoder::Decode(DecodeContext ctx,
                                               span<SampleView<GPUBackend>> out,
                                               cspan<ImageSource *> in,
                                               DecodeParams opts,
                                               cspan<ROI> rois) {
  return ScheduleDecode(ctx, out, in, opts, rois).get_all_copy();
}



FutureDecodeResults ImageDecoder::ScheduleDecode(DecodeContext ctx,
                                                 SampleView<CPUBackend> out,
                                                 ImageSource *in,
                                                 DecodeParams opts,
                                                 const ROI &roi) {
  return ScheduleDecode(ctx, make_span(&out, 1), make_span(&in, 1), opts, make_span(&roi, 1));
}

FutureDecodeResults ImageDecoder::ScheduleDecode(DecodeContext ctx,
                                                 SampleView<GPUBackend> out,
                                                 ImageSource *in,
                                                 DecodeParams opts,
                                                 const ROI &roi) {
  return ScheduleDecode(ctx, make_span(&out, 1), make_span(&in, 1), opts, make_span(&roi, 1));
}

FutureDecodeResults ImageDecoder::ScheduleDecode(DecodeContext ctx,
                                                span<SampleView<CPUBackend>> out,
                                                cspan<ImageSource *> in,
                                                DecodeParams opts,
                                                cspan<ROI> rois) {
  temp_buffers_.clear();
  int N = out.size();
  assert(in.size() == N);
  assert(rois.size() == N || rois.empty());
  DecodeResultsPromise results(N);

  auto work = new_work(ctx, results);
  work->cpu_outputs.reserve(N);
  work->indices.reserve(N);
  work->sources.reserve(N);

  for (auto &o : out)
    work->cpu_outputs.push_back(o);
  for (auto *src : in)
    work->sources.push_back(src);
  for (int i = 0; i < N; i++)
    work->indices.push_back(i);

  DistributeWork(std::move(work));

  temp_buffers_.clear();

  return results.get_future();
}


/**
 * @brief Decodes a single image to device buffers
 */
FutureDecodeResults ImageDecoder::ScheduleDecode(DecodeContext ctx,
                                                 span<SampleView<GPUBackend>> out,
                                                 cspan<ImageSource *> in,
                                                 DecodeParams opts,
                                                 cspan<ROI> rois) {
  int N = out.size();
  assert(in.size() == N);
  assert(rois.size() == N || rois.empty());

  temp_buffers_.resize(N);

  DecodeResultsPromise results(N);

  auto work = new_work(ctx, results);
  work->gpu_outputs.reserve(N);
  work->indices.reserve(N);
  work->sources.reserve(N);

  for (auto &o : out)
    work->gpu_outputs.push_back(o);
  for (auto *src : in)
    work->sources.push_back(src);
  for (int i = 0; i < N; i++)
    work->indices.push_back(i);

  DistributeWork(std::move(work));

  return results.get_future();
}

void ImageDecoder::DistributeWork(std::unique_ptr<ScheduledWork> work) {
  std::map<const ImageFormat*, std::unique_ptr<ScheduledWork>> dist;
  for (int i = 0; i < work->num_samples(); i++) {
    auto *f = FormatRegistry().GetImageFormat(work->sources[i]);
    if (!f || filtered_.find(f) == filtered_.end()) {
      work->results.set(i, DecodeResult::Failure(
        std::make_exception_ptr(std::runtime_error("Image not supported."))));
      continue;
    }
    auto &w = dist[f];
    if (!w)
      w = new_work(work->ctx, work->results);
    w->move_entry(*work, i);
  }
  for (auto &[f, w] : dist) {
    auto &worker = workers_[filtered_.find(f)->second];
    worker->add_work(std::move(w));
  }
}

void ImageDecoder::InitWorkers(bool lazy_init) {
  const ImageFormat *prev_format = nullptr;
  DecoderWorker *prev_worker = nullptr;
  for (auto [format, factory] : filtered_) {
    auto [it, inserted] = workers_.emplace(
        factory, std::make_unique<DecoderWorker>(this, factory, !lazy_init));

    DecoderWorker *worker = it->second.get();

    if (prev_format == format) {
      assert(prev_worker);
      prev_worker->set_fallback(worker);
    }

    prev_format = format;
    prev_worker = worker;
  }
}

ImageDecoder::DecoderWorker &ImageDecoder::GetWorker(ImageDecoderFactory *factory) {
  auto it = workers_.find(factory);
  assert(it != workers_.end());
  auto *worker = it->second.get();
  worker->start();
  return *worker;
}

ImageDecoderInstance *ImageDecoder::DecoderWorker::decoder(bool create_if_null) {
  if (!decoder_) {
    decoder_ = factory_->Create(owner_->device_id_, owner_->params_);
    produces_gpu_output_ = factory_->GetProperties().gpu_output;
  }
  return decoder_.get();
}


void ImageDecoder::DecoderWorker::start() {
  std::call_once(started_, [&]() {
    (void)decoder(true);
    worker_ = std::thread(&DecoderWorker::run, this);
  });
}

ImageDecoder::DecoderWorker::~DecoderWorker() {
  stop();
}

void ImageDecoder::DecoderWorker::stop() {
  if (worker_.joinable()) {
    {
      std::lock_guard lock(mtx_);
      stop_requested_ = true;
      work_.reset();
    }
    cv_.notify_all();
    worker_.join();
    worker_ = {};
  }
}

void ImageDecoder::DecoderWorker::run() {
  DeviceGuard dg(owner_->device_id_);
  while (!stop_requested_) {
    std::unique_lock lock(mtx_);
    cv_.wait(lock, [&]() {
      return stop_requested_ || work_ != nullptr;
    });
    if (stop_requested_)
      break;
    assert(work_ != nullptr);
    auto w = std::move(work_);
    lock.unlock();
    process_batch(std::move(w));
  }
}

void ImageDecoder::move_to_fallback(ScheduledWork *fb,
                                    ScheduledWork &work,
                                    const vector<bool> &keep) {
    int moved = 0;

    int n = work.sources.size();

    for (int i = 0; i < n; i++) {
      if (keep[i]) {
        if (moved) {
          // compact
          if (!work.cpu_outputs.empty())
            work.cpu_outputs[i - moved] = std::move(work.cpu_outputs[i]);
          if (!work.gpu_outputs.empty())
            work.gpu_outputs[i - moved] = std::move(work.gpu_outputs[i]);
          if (!work.rois[i])
            work.rois[i - moved] = std::move(work.rois[i]);
          work.sources[i - moved] = work.sources[i];
          work.indices[i - moved] = work.indices[i];
        }
      } else {
        if (fb)
          fb->move_entry(work, i);
        moved++;
      }
    }
    if (fb)
      fb->resize(fb->num_samples() - moved);
}

void ImageDecoder::ScheduledWork::alloc_temp_cpu_outputs(ImageDecoder &owner) {
  cpu_outputs.resize(indices.size());
  for (int i = 0, n = indices.size(); i < n; i++) {
    int idx = indices[i];

    SampleView<GPUBackend> &gpu_out = gpu_outputs[i];

    // TODO(michalz): Add missing utility functions to SampleView - or just use Tensor again...
    size_t size = volume(gpu_out.shape()) + TypeTable::GetTypeInfo(gpu_out.type()).size();
    constexpr int kTempBufferAlignment = 256;

    auto &buf_ptr = owner.temp_buffers_[idx];
    buf_ptr = mm::alloc_raw_async_shared<char, mm::memory_kind::pinned>(
      size, mm::host_sync, ctx.stream, kTempBufferAlignment);

    SampleView<CPUBackend> &cpu_out = cpu_outputs[i];
    cpu_out = SampleView<CPUBackend>(buf_ptr.get(), gpu_out.shape(), gpu_out.type());
  }
}

void ImageDecoder::DecoderWorker::process_batch(std::unique_ptr<ScheduledWork> work) {
  auto mask = decoder_->CanDecode(work->ctx,
                                  make_span(work->sources),
                                  work->params,
                                  make_span(work->rois));

  std::unique_ptr<ScheduledWork> fallback_work;
  if (fallback_) {
    fallback_work = owner_->new_work(work->ctx, work->results);
    move_to_fallback(fallback_work.get(), *work, mask);
    if (!fallback_work->empty())
      fallback_->add_work(std::move(fallback_work));
  } else {
    filter(*work, mask);
    for (size_t i = 0; i < mask.size(); i++) {
      if (!mask[i])
        work->results.set(i, DecodeResult::Failure(nullptr));
    }
  }

  if (fallback_work)
    fallback_->add_work(std::move(fallback_work));

  if (!work->sources.empty()) {
    bool decode_to_gpu = produces_gpu_output_;

    if (!decode_to_gpu && work->cpu_outputs.empty()) {
      work->alloc_temp_cpu_outputs(*owner_);
    }

    auto future = decode_to_gpu
      ? decoder_->ScheduleDecode(work->ctx,
                                 make_span(work->gpu_outputs),
                                 make_span(work->sources),
                                 work->params,
                                 make_span(work->rois))
      : decoder_->ScheduleDecode(work->ctx,
                                 make_span(work->cpu_outputs),
                                 make_span(work->sources),
                                 work->params,
                                 make_span(work->rois));

    if (fallback_) {
      for (;;) {
        auto indices = future.wait_new();
        if (indices.empty())
          break;

        auto r = future.get_all_ref();

        for (int sub_idx : indices) {
          if (r[sub_idx].success) {
            work->results.set(work->indices[sub_idx], r[sub_idx]);
          } else {
            if (!fallback_work)
              fallback_work = owner_->new_work(work->ctx, work->results);
            fallback_work->move_entry(*work, sub_idx);
          }
        }

        if (fallback_work)
          fallback_->add_work(std::move(fallback_work));
      }
    } else {
      // no fallback - the results are final
      future.wait_all();
      auto r = future.get_all();
      // we may need to copy to the gpu
      if (!decode_to_gpu && !work->gpu_outputs.empty()) {
        for (int i = 0; i < r.size(); i++) {
          if (r[i].success) {  // TODO(michalz): bulk copy
            copy(work->gpu_outputs[i], work->cpu_outputs[i], work->ctx.stream);
          }
        }
      }
      // just copy the result to the final promise
      for (int i = 0; i < work->num_samples(); i++) {
        work->results.set(work->indices[i], r[i]);
      }
    }
  }
  owner_->recycle_work(std::move(work));
}

void ImageDecoder::copy(SampleView<GPUBackend> &out,
                        const ConstSampleView<CPUBackend> &in,
                        cudaStream_t stream) {
  assert(out.shape() == in.shape());
  assert(out.type() == in.type());

  size_t nbytes = TypeTable::GetTypeInfo(out.type()).size() * volume(out.shape());

  CUDA_CALL(cudaMemcpyAsync(out.raw_mutable_data(), in.raw_data(), nbytes,
                            cudaMemcpyHostToDevice, stream));
}

////////////////////////////////////////////////////////////////////////////
// Work item management

std::unique_ptr<ImageDecoder::ScheduledWork> ImageDecoder::new_work(
    DecodeContext ctx, DecodeResultsPromise results) {
  if (free_work_items_) {
    std::lock_guard<std::mutex> g(work_mutex_);
    if (free_work_items_) {
      auto ptr = std::move(free_work_items_);
      free_work_items_ = std::move(ptr->next);
      ptr->results = std::move(results);
      ptr->ctx = ctx;
      return ptr;
    }
  }

  return std::make_unique<ScheduledWork>(ctx, results);
}


void ImageDecoder::recycle_work(std::unique_ptr<ScheduledWork> work) {
  std::lock_guard<std::mutex> g(work_mutex_);
  work->clear();
  work->next = std::move(free_work_items_);
  free_work_items_ = std::move(work);
}

void ImageDecoder::combine_work(ScheduledWork &target, std::unique_ptr<ScheduledWork> source) {
  assert(target.results == source->results);
  assert(target.ctx.stream == source->ctx.stream);
  assert(target.ctx.tp == source->ctx.tp);
  // if the target has gpu outputs, then the source cannot have any
  assert(!target.cpu_outputs.empty() || target.gpu_outputs.empty() || source->cpu_outputs.empty());
  // if the target has cpu outputs, then the source cannot have any
  assert(!target.gpu_outputs.empty() || target.cpu_outputs.empty() || source->gpu_outputs.empty());

  auto move_append = [](auto &dst, auto &src) {
    dst.reserve(dst.size() + src.size());
    for (auto &x : src)
      dst.emplace_back(std::move(x));
  };

  move_append(target.cpu_outputs, source->cpu_outputs);
  move_append(target.gpu_outputs, source->gpu_outputs);
  move_append(target.sources, source->sources);
  move_append(target.indices, source->indices);
  move_append(target.rois, source->rois);
  recycle_work(std::move(source));
}

void ImageDecoder::ScheduledWork::move_entry(ScheduledWork &from, int which) {
  sources.push_back(from.sources[which]);
  indices.push_back(from.indices[which]);
  if (!from.cpu_outputs.empty())
    cpu_outputs.push_back(std::move(from.cpu_outputs[which]));
  if (!from.gpu_outputs.empty())
    gpu_outputs.push_back(std::move(from.gpu_outputs[which]));
  if (!from.rois.empty())
    rois.push_back(std::move(from.rois[which]));
}

void ImageDecoder::DecoderWorker::add_work(std::unique_ptr<ScheduledWork> work) {
  {
    std::lock_guard guard(mtx_);
    assert((work->cpu_outputs.empty() && work->gpu_outputs.size() == work->sources.size()) ||
           (work->gpu_outputs.empty() && work->cpu_outputs.size() == work->sources.size()));
    assert(work->rois.empty() || work->rois.size() == work->sources.size());
    if (work_) {
      owner_->combine_work(*work_, std::move(work));
      // no need to notify - a work item was already there, so it will be picked up regardless
    } else {
      work_ = std::move(work);
      cv_.notify_one();
    }
  }
  start();
}

std::vector<DecodeResult> ImageDecoder::Decode(DecodeContext ctx,
                                               TensorList<CPUBackend> &out,
                                               cspan<ImageSource *> in,
                                               DecodeParams opts,
                                               cspan<ROI> rois) {
  BatchVector<SampleView<CPUBackend>> samples;
  int N = out.num_samples();
  samples.resize(N);
  for (int i = 0; i < N; i++)
    samples[i] = out[i];
  return Decode(ctx, make_span(samples), in, opts, rois);
}

std::vector<DecodeResult> ImageDecoder::Decode(DecodeContext ctx,
                                               TensorList<GPUBackend> &out,
                                               cspan<ImageSource *> in,
                                               DecodeParams opts,
                                               cspan<ROI> rois) {
  BatchVector<SampleView<GPUBackend>> samples;
  int N = out.num_samples();
  samples.resize(N);
  for (int i = 0; i < N; i++)
    samples[i] = out[i];
  return Decode(ctx, make_span(samples), in, opts, rois);
}


}  // namespace imgcodec
}  // namespace dali
