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
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dali/imgcodec/image_decoder.h"
#include <cassert>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <typeinfo>
#include <vector>
#include "dali/core/cuda_error.h"
#include "dali/core/mm/memory.h"
#include "dali/core/nvtx.h"
#include "dali/imgcodec/util/output_shape.h"

namespace dali {
namespace imgcodec {

///////////////////////////////////////////////////////////////////////////////
// ScheduledWork
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief Describes a sub-batch of work to be processed
 *
 * This object contains a (shared) promise from the original request containing the full batch
 * and a mapping of indices from this sub-batch to the full batch.
 * It also contains a subset of relevant sample views, sources, etc.
 */
struct ImageDecoder::ScheduledWork {
  ScheduledWork(DecodeContext ctx, DecodeResultsPromise results, DecodeParams params)
  : ctx(std::move(ctx)), results(std::move(results)), params(std::move(params)) {}

  void clear() {
    indices.clear();
    sources.clear();
    cpu_outputs.clear();
    gpu_outputs.clear();
    rois.clear();
    temp_buffers.clear();
  }

  int num_samples() const {
    return indices.size();
  }

  bool empty() const {
    return indices.empty();
  }

  void resize(int num_samples) {
    indices.resize(num_samples);
    sources.resize(num_samples);
    if (!cpu_outputs.empty())
      cpu_outputs.resize(num_samples);
    if (!gpu_outputs.empty())
      gpu_outputs.resize(num_samples);
    if (!rois.empty())
      rois.resize(num_samples);
    if (!temp_buffers.empty())
      temp_buffers.resize(num_samples);
  }

  template <typename Backend>
  void init(span<SampleView<Backend>> out,
            cspan<ImageSource *> in,
            cspan<ROI> rois) {
    int N = out.size();

    this->indices.reserve(N);
    for (int i = 0; i < N; i++)
      this->indices.push_back(i);

    if constexpr (std::is_same_v<Backend, CPUBackend>) {
      cpu_outputs.reserve(N);
      for (auto &o : out)
        cpu_outputs.push_back(o);
    } else {
      gpu_outputs.reserve(N);
      for (auto &o : out)
        gpu_outputs.push_back(o);
    }

    this->sources.reserve(N);
    for (auto *src : in)
      this->sources.push_back(src);

    this->rois.reserve(rois.size());
    for (const auto &roi : rois)
      this->rois.push_back(roi);
  }

  /**
   * @brief Moves one work entry from another work to this one
   */
  void move_entry(ScheduledWork &from, int which);

  /**
   * @brief Allocates temporary CPU outputs for this sub-batch
   *
   * This function is used when falling back from GPU to CPU decoder.
   */
  void alloc_temp_cpu_outputs();

  DecodeContext ctx;
  // The original promise
  DecodeResultsPromise results;
  // The indices in the original request
  BatchVector<int> indices;
  BatchVector<ImageSource *> sources;
  BatchVector<SampleView<CPUBackend>> cpu_outputs;
  BatchVector<SampleView<GPUBackend>> gpu_outputs;
  BatchVector<mm::async_uptr<void>> temp_buffers;
  DecodeParams params;
  BatchVector<ROI> rois;

  std::unique_ptr<ScheduledWork> next;
};

void ImageDecoder::ScheduledWork::move_entry(ScheduledWork &from, int which) {
  sources.push_back(from.sources[which]);
  indices.push_back(from.indices[which]);
  if (!from.cpu_outputs.empty())
    cpu_outputs.push_back(std::move(from.cpu_outputs[which]));
  if (!from.gpu_outputs.empty())
    gpu_outputs.push_back(std::move(from.gpu_outputs[which]));
  if (!from.temp_buffers.empty())
    temp_buffers.push_back(std::move(from.temp_buffers[which]));
  if (!from.rois.empty())
    rois.push_back(std::move(from.rois[which]));
}

void ImageDecoder::ScheduledWork::alloc_temp_cpu_outputs() {
  cpu_outputs.resize(indices.size());
  temp_buffers.clear();
  for (int i = 0, n = indices.size(); i < n; i++) {
    SampleView<GPUBackend> &gpu_out = gpu_outputs[i];

    // TODO(michalz): Add missing utility functions to SampleView - or just use Tensor again...
    size_t size = volume(gpu_out.shape()) * TypeTable::GetTypeInfo(gpu_out.type()).size();
    constexpr int kTempBufferAlignment = 256;

    temp_buffers.emplace_back();
    auto &buf_ptr = temp_buffers.back();
    buf_ptr = mm::alloc_raw_async_unique<char, mm::memory_kind::pinned>(
      size, mm::host_sync, ctx.stream, kTempBufferAlignment);

    SampleView<CPUBackend> &cpu_out = cpu_outputs[i];
    cpu_out = SampleView<CPUBackend>(buf_ptr.get(), gpu_out.shape(), gpu_out.type());
  }
}


///////////////////////////////////////////////////////////////////////////////
// DecoderWorker
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief A worker that processes sub-batches of work to be processed by a particular decoder.
 *
 * A DecoderWorker waits for incoming ScheduledWork objects and processes them by running
 * `decoder_->ScheduleDecode` and waiting for partial results, scheduling the failed
 * samples to a fallback decoder, if present.
 *
 * When a sample is successfully decoded, it is marked as a success in the parent
 * DecodeResultsPromise. If it fails, it goes to fallback and only if all fallbacks fail, it is
 * marked in the DecodeResultsPromise as a failure.
 */
class ImageDecoder::DecoderWorker {
 public:
  /**
   * @brief Constructs a decoder worker for a given decoder.
   *
   * @param owner   - the parent decoder object
   * @param factory - the factory that constructs the decoder for this worker
   * @param start   - if true, the decoder is immediately instantiated and the worker thread
   *                  is launched; otherwise a call to `start` is delayed until the first
   *                  work that's relevant for this decoder.
   */
  DecoderWorker(ImageDecoder *owner, const ImageDecoderFactory *factory, bool start) {
    owner_ = owner;
    factory_ = factory;
    if (start)
      this->start();
  }
  ~DecoderWorker();

  void start();
  void stop();
  void add_work(std::unique_ptr<ScheduledWork> work);

  void set_fallback(DecoderWorker *fallback) {
    fallback_ = fallback;
  }

  ImageDecoderInstance *decoder(bool create_if_null = true);

 private:
  std::mutex mtx_;
  std::condition_variable cv_;

  std::unique_ptr<ScheduledWork> work_;
  std::thread worker_;
  bool stop_requested_ = false;
  std::once_flag started_;
  std::string thread_name_;
  std::string nvtx_marker_str_;

  ImageDecoder *owner_ = nullptr;
  const ImageDecoderFactory *factory_ = nullptr;
  std::shared_ptr<ImageDecoderInstance> decoder_;
  bool produces_gpu_output_ = false;

  /**
   * @brief Processes a (sub)batch of work.
   *
   * The work is scheduled and the results are waited for. Any failed samples will be added
   * to a fallback work, if a fallback decoder is present.
   */
  void process_batch(std::unique_ptr<ScheduledWork> work) noexcept;

  /**
   * @brief The main loop of the worker thread.
   */
  void run();

  // Fallback worker or null, if no fallback is available
  DecoderWorker *fallback_ = nullptr;
};

ImageDecoderInstance *ImageDecoder::DecoderWorker::decoder(bool create_if_null) {
  if (!decoder_) {
    decoder_ = factory_->Create(owner_->device_id_, owner_->params_);
    produces_gpu_output_ = factory_->GetProperties().gpu_output;
    thread_name_ = make_string("[DALI][WT]ImageDecoder");
    auto &dec_ref = *this->decoder_;
    nvtx_marker_str_ = make_string(typeid(dec_ref).name(), "/ process_batch");
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
  SetThreadName(thread_name_.c_str());
  DeviceGuard dg(owner_->device_id_);
  std::unique_lock lock(mtx_, std::defer_lock);
  while (!stop_requested_) {
    lock.lock();
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

void ImageDecoder::DecoderWorker::add_work(std::unique_ptr<ScheduledWork> work) {
  assert(work->num_samples() > 0);
  {
    std::lock_guard guard(mtx_);
    assert((work->cpu_outputs.empty() && work->gpu_outputs.size() == work->sources.size()) ||
           (work->gpu_outputs.empty() && work->cpu_outputs.size() == work->sources.size()) ||
           (work->cpu_outputs.size() == work->sources.size() &&
            work->gpu_outputs.size() == work->sources.size()));
    assert(work->rois.empty() || work->rois.size() == work->sources.size());
    assert(work->temp_buffers.empty() || work->temp_buffers.size() == work->cpu_outputs.size());
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

// The main processing function
void ImageDecoder::DecoderWorker::process_batch(std::unique_ptr<ScheduledWork> work) noexcept {
  DomainTimeRange tr(nvtx_marker_str_, DomainTimeRange::kCyan);
  assert(work->num_samples() > 0);
  assert((work->cpu_outputs.empty() && work->gpu_outputs.size() == work->sources.size()) ||
         (work->gpu_outputs.empty() && work->cpu_outputs.size() == work->sources.size()) ||
         (work->cpu_outputs.size() == work->sources.size() &&
          work->gpu_outputs.size() == work->sources.size()));
  assert(work->rois.empty() || work->rois.size() == work->sources.size());
  assert(work->temp_buffers.empty() || work->temp_buffers.size() == work->cpu_outputs.size());

  auto mask = decoder_->CanDecode(work->ctx,
                                  make_span(work->sources),
                                  work->params,
                                  make_span(work->rois));

  std::unique_ptr<ScheduledWork> fallback_work;
  if (fallback_) {
    fallback_work = owner_->new_work(work->ctx, work->results, work->params);
    move_to_fallback(fallback_work.get(), *work, mask);
    if (!fallback_work->empty())
      fallback_->add_work(std::move(fallback_work));
  } else {
    for (size_t i = 0; i < mask.size(); i++) {
      if (!mask[i])
        work->results.set(work->indices[i], DecodeResult::Failure(nullptr));
    }
    filter(*work, mask);
  }

  if (!work->sources.empty()) {
    bool decode_to_gpu = produces_gpu_output_;

    if (!decode_to_gpu && work->cpu_outputs.empty()) {
      work->alloc_temp_cpu_outputs();
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

    for (;;) {
      auto indices = future.wait_new();
      if (indices.empty())
        break;  // if wait_new returns with an empty result, it means that everything is ready

      for (int sub_idx : indices) {
        DecodeResult r = future.get_one(sub_idx);
        if (r.success) {
          if (!decode_to_gpu && !work->gpu_outputs.empty()) {
            try {
              copy(work->gpu_outputs[sub_idx], work->cpu_outputs[sub_idx], work->ctx.stream);
            } catch (...) {
              r = DecodeResult::Failure(std::current_exception());
            }
          }
          work->results.set(work->indices[sub_idx], r);
        } else {  // failed to decode
          if (fallback_) {
            // if there's fallback, we don't set the result, but try to use the fallback first
            if (!fallback_work)
              fallback_work = owner_->new_work(work->ctx, work->results, work->params);
            fallback_work->move_entry(*work, sub_idx);
          } else {
            // no fallback - just propagate the result to the original promise
            work->results.set(work->indices[sub_idx], r);
          }
        }
      }

      if (fallback_work && !fallback_work->empty())
        fallback_->add_work(std::move(fallback_work));
    }
  }
  owner_->recycle_work(std::move(work));
}


///////////////////////////////////////////////////////////////////////////////
// ImageDecoder
///////////////////////////////////////////////////////////////////////////////


ImageDecoder::ImageDecoder(int device_id,
                           bool lazy_init,
                           const std::map<std::string, std::any> &params,
                           std::function<bool(ImageDecoderFactory *)> decoder_filter) {
  SetParams(params);
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

bool ImageDecoder::CanParse(ImageSource *encoded) const {
  auto *f = FormatRegistry().GetImageFormat(encoded);
  return f && filtered_.find(f) != filtered_.end();
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

bool ImageDecoder::SetParam(const char *key, const std::any &value) {
  params_[key] = value;

  for (auto &[_, worker] : workers_)
    if (auto *dec = worker->decoder(false))
      dec->SetParam(key, value);

  return true;
}

std::any ImageDecoder::GetParam(const char *key) const {
  auto it = params_.find(key);
  return it != params_.end() ? it->second : std::any{};
}

int ImageDecoder::SetParams(const std::map<std::string, std::any> &params) {
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
  auto rois = make_span(&roi, roi.use_roi() ? 1 : 0);
  return ScheduleDecode(ctx, make_span(&out, 1), make_span(&in, 1), opts, rois);
}

FutureDecodeResults ImageDecoder::ScheduleDecode(DecodeContext ctx,
                                                 SampleView<GPUBackend> out,
                                                 ImageSource *in,
                                                 DecodeParams opts,
                                                 const ROI &roi) {
  auto rois = make_span(&roi, roi.use_roi() ? 1 : 0);
  return ScheduleDecode(ctx, make_span(&out, 1), make_span(&in, 1), opts, rois);
}

FutureDecodeResults ImageDecoder::ScheduleDecode(DecodeContext ctx,
                                                 span<SampleView<CPUBackend>> out,
                                                 cspan<ImageSource *> in,
                                                 DecodeParams opts,
                                                 cspan<ROI> rois) {
  int N = out.size();
  assert(in.size() == N);
  assert(rois.size() == N || rois.empty());
  DecodeResultsPromise results(N);

  auto work = new_work(ctx, results, opts);
  work->init(out, in, rois);

  DistributeWork(std::move(work));

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

  DecodeResultsPromise results(N);

  auto work = new_work(ctx, results, opts);
  work->init(out, in, rois);
  DistributeWork(std::move(work));

  return results.get_future();
}

void ImageDecoder::DistributeWork(std::unique_ptr<ScheduledWork> work) {
  std::map<const ImageFormat*, std::unique_ptr<ScheduledWork>> dist;
  for (int i = 0; i < work->num_samples(); i++) {
    auto *f = FormatRegistry().GetImageFormat(work->sources[i]);
    if (!f || filtered_.find(f) == filtered_.end()) {
      std::string msg;
      if (work->sources[i]->SourceInfo())
        msg = make_string("Image not supported: ", work->sources[i]->SourceInfo());
      else
        msg = make_string("Image #", work->indices[i], " not supported");

      work->results.set(i, DecodeResult::Failure(
        std::make_exception_ptr(std::runtime_error(msg))));
      continue;
    }
    auto &w = dist[f];
    if (!w)
      w = new_work(work->ctx, work->results, work->params);
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
    auto it = workers_.find(factory);
    if (it == workers_.end()) {
      it = workers_.emplace(
        factory, std::make_unique<DecoderWorker>(this, factory, !lazy_init)).first;
    }

    DecoderWorker *worker = it->second.get();

    if (prev_format == format) {
      assert(prev_worker);
      prev_worker->set_fallback(worker);
    }

    prev_format = format;
    prev_worker = worker;
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
        if (!work.temp_buffers.empty())
          work.temp_buffers[i - moved] = std::move(work.temp_buffers[i]);
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
  if (moved)
    work.resize(n - moved);
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

////////////////////////////////////////////////////////////////////////////
// Work item management
////////////////////////////////////////////////////////////////////////////

std::unique_ptr<ImageDecoder::ScheduledWork> ImageDecoder::new_work(
    DecodeContext ctx, DecodeResultsPromise results, DecodeParams params) {
  if (free_work_items_) {
    std::lock_guard<std::mutex> g(work_mutex_);
    if (free_work_items_) {
      auto ptr = std::move(free_work_items_);
      free_work_items_ = std::move(ptr->next);
      ptr->results = std::move(results);
      ptr->ctx = std::move(ctx);
      ptr->params = std::move(params);
      return ptr;
    }
  }

  return std::make_unique<ScheduledWork>(ctx, results, params);
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

  // if only one has temporary CPU storage, allocate it in the other
  if (target.temp_buffers.empty() && !source->temp_buffers.empty())
    target.alloc_temp_cpu_outputs();
  else if (!target.temp_buffers.empty() && source->temp_buffers.empty())
    source->alloc_temp_cpu_outputs();

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
  move_append(target.temp_buffers, source->temp_buffers);
  recycle_work(std::move(source));
}

}  // namespace imgcodec
}  // namespace dali
