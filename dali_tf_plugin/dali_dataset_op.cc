// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <chrono>
#include <queue>
#include <sstream>
#include <vector>


#include "tensorflow/core/public/version.h"

#if TF_MAJOR_VERSION == 2 || (TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION >= 15)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"

#define EIGEN_USE_GPU

#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

#include "tensorflow/core/framework/tensor.h"

#include "dali/c_api.h"
#include "dali/core/common.h"
#include "dali/core/format.h"
#include "dali_tf_plugin/dali_helper.h"

#include "dali_tf_plugin/dali_dataset.h"

#define TF_DALI_CALL(FUNC)                                                                    \
  do {                                                                                        \
    try {                                                                                     \
      FUNC;                                                                                   \
    } catch (std::exception & e) {                                                            \
      std::string error = "DALI " + std::string(#FUNC) + " failed: " + std::string(e.what()); \
      std::cout << error << std::endl;                                                        \
      return ::tensorflow::errors::Internal(error);                                           \
    }                                                                                         \
  } while (0)

using namespace tensorflow;        // NOLINT(build/namespaces)
using namespace tensorflow::data;  // NOLINT(build/namespaces)

namespace dali_tf_impl {

class DALIDatasetOp::Dataset : public DatasetBase {
 public:
  explicit Dataset(OpKernelContext *context, const PipelineDef pipeline_def,
                   const Inputs &inputs, const InputAttrs &input_attrs,
                   const std::vector<PartialTensorShape> &shapes, const DataTypeVector &dtypes,
                   const bool is_gpu_device, const bool fail_on_device_mismatch)
      : DatasetBase(DatasetContext(context)),
        input_desc_(inputs, input_attrs),
        pipeline_def_(pipeline_def),
        shapes_(shapes),
        dtypes_(dtypes),
        device_type_(is_gpu_device ? device_type_t::GPU : device_type_t::CPU),
        fail_on_device_mismatch_(fail_on_device_mismatch) {
    for (const auto &input : input_desc_.inputs) {
      input->Ref();
    }
    if (is_gpu_device) {
      stream_ = context->eigen_gpu_device().stream();
    }
  }

  ~Dataset() override {
    for (const auto &input : input_desc_.inputs) {
      input->Unref();
    }
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(const string &prefix) const override;


#if TF_MAJOR_VERSION > 2 || (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 4)

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override;

#endif


#if TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 4 && TF_MINOR_VERSION < 6

  /**
   * @brief Current implementation disables splits. For newer TF versions, it is
   * necessary to implement InputDatasets to get rid of the warnings, but adding it would enable
   * automatic SplitProvider for DALIDataset. As DALI has its own concept of shards, we do not
   * handle splits as of now, so it is disabled explicitly.
   */
  Status MakeSplitProvider(std::unique_ptr<SplitProvider> *split_provider) const override;

#elif TF_MAJOR_VERSION > 2 || (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 6)

  /**
   * @brief Current implementation disables splits. For newer TF versions, it is
   * necessary to implement InputDatasets to get rid of the warnings, but adding it would enable
   * automatic SplitProvider for DALIDataset. As DALI has its own concept of shards, we do not
   * handle splits as of now, so it is disabled explicitly.
   */
  Status MakeSplitProviders(
      std::vector<std::unique_ptr<SplitProvider>> *split_providers) const override;

#endif


  const DataTypeVector &output_dtypes() const override {
    return dtypes_;
  }

  const std::vector<PartialTensorShape> &output_shapes() const override {
    return shapes_;
  }

  string DebugString() const override {
    return "DALI::DatasetOp()::Dataset";
  }

  tensorflow::int64 Cardinality() const override {
    return data::kInfiniteCardinality;
  }

 protected:
  PipelineDef pipeline_def_;
  std::vector<PartialTensorShape> shapes_;
  const DataTypeVector dtypes_;
  cudaStream_t stream_ = 0;
  const device_type_t device_type_;
  const bool fail_on_device_mismatch_;

  const InputDescs input_desc_;

  Status AsGraphDefInternal(SerializationContext *context, DatasetGraphDefBuilder *b,
                            Node **output) const override {
    std::vector<Node *> inputs;
    TF_RETURN_IF_ERROR(InputsToNodeList(context, b, input_desc_, inputs));

    AttrSerializationContainer attrs;
    PipelineDefToGraphDefAttrs(b, pipeline_def_, attrs);
    InputDescsToGraphDefAttrs(b, input_desc_, attrs);

    SerializeField(attrs, b, "output_shapes", shapes_);
    SerializeField(attrs, b, "output_dtypes", dtypes_);
    SerializeField(attrs, b, "fail_on_device_mismatch", fail_on_device_mismatch_);

    // with the {std::make_pair(0, inputs)} below we wrap the data into view
    // (that is gtl::ArraySlice<Node*>), returning view directly from the helper is not an option
    TF_RETURN_IF_ERROR(b->AddDataset(this, {}, {std::make_pair(0, inputs)}, attrs, output));

    return Status::OK();
  }

#if TF_MAJOR_VERSION > 2 || (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 3)
  Status CheckExternalState() const override {
    return errors::Unimplemented("CheckExternalState is not supported for DALI dataset.");
  }
#endif

 private:
  using AttrSerializationContainer = std::vector<std::pair<StringPiece, AttrValue>>;

  /**
   * @brief Append the AttrValue created from `filed` under `name` to `attrs` vector
   */
  template <typename T>
  void SerializeField(AttrSerializationContainer &attrs, DatasetGraphDefBuilder *b,
                      const StringPiece &name, const T &field) const {
    AttrValue field_attr;
    b->BuildAttrValue(field, &field_attr);
    attrs.push_back(std::make_pair(name, field_attr));
  }

  void PipelineDefToGraphDefAttrs(DatasetGraphDefBuilder *b, const PipelineDef &def,
                                  AttrSerializationContainer &attrs) const {
    SerializeField(attrs, b, kPipeline, pipeline_def_.pipeline);
    SerializeField(attrs, b, kBatchSize, pipeline_def_.batch_size);
    SerializeField(attrs, b, kNumThreads, pipeline_def_.num_threads);
    SerializeField(attrs, b, kDeviceId, pipeline_def_.device_id);
    SerializeField(attrs, b, kExecSeparated, pipeline_def_.exec_separated);
    SerializeField(attrs, b, kPrefetchQueueDepth, pipeline_def_.prefetch_queue_depth);
    SerializeField(attrs, b, kCpuPrefetchQueueDepth, pipeline_def_.cpu_prefetch_queue_depth);
    SerializeField(attrs, b, kGpuPrefetchQueueDepth, pipeline_def_.gpu_prefetch_queue_depth);
    SerializeField(attrs, b, kGpuMemoryStats, pipeline_def_.enable_memory_stats);
  }

  void InputDescsToGraphDefAttrs(DatasetGraphDefBuilder *b, const InputDescs &input_desc,
                                AttrSerializationContainer &attrs) const {
    SerializeField(attrs, b, kInputNames, input_desc.input_names);
    SerializeField(attrs, b, kInputLayouts, input_desc.input_layouts);
    SerializeField(attrs, b, kInputBatched, input_desc.input_batched);
  }

  Status InputsToNodeList(SerializationContext *context, DatasetGraphDefBuilder *b,
                          const InputDescs &input_desc,
                          std::vector<Node *> &input_graph_nodes) const {
    // Based on ZipDataset
    input_graph_nodes.clear();
    input_graph_nodes.reserve(input_desc.inputs.size());
    for (const auto &input : input_desc.inputs) {
      Node *input_node;
      TF_RETURN_IF_ERROR(b->AddInputDataset(context, input, &input_node));
      input_graph_nodes.emplace_back(input_node);
    }
    return Status::OK();
  }

  Status InitPipeline(daliPipelineHandle *pipeline_handle) const {
    TF_DALI_CALL(daliCreatePipeline(
        pipeline_handle, pipeline_def_.pipeline.c_str(), pipeline_def_.pipeline.length(),
        pipeline_def_.batch_size, pipeline_def_.num_threads, pipeline_def_.device_id,
        pipeline_def_.exec_separated, pipeline_def_.prefetch_queue_depth,
        pipeline_def_.cpu_prefetch_queue_depth, pipeline_def_.gpu_prefetch_queue_depth,
        pipeline_def_.enable_memory_stats));
    return Status::OK();
  }

  class Iterator;

  bool HasInputs() const {
    return !input_desc_.inputs.empty();
  }


  int NumInputs() const {
    return input_desc_.inputs.size();
  }
};


class DALIDatasetOp::Dataset::Iterator : public DatasetIterator<Dataset> {
 public:
  explicit Iterator(const Params &params, daliPipelineHandle pipeline_handle,
                    bool enable_memory_stats = false)
      : DatasetIterator<Dataset>(params),
        pipeline_handle_(pipeline_handle),
        enable_memory_stats_(enable_memory_stats) {}

  Status Initialize(IteratorContext *context) override {
    // Based on ZipDataset
    mutex_lock l(mu_);
    iterator_state_ = InputState::in_progress;
    if (dataset()->HasInputs()) {
      input_impls_.resize(dataset()->NumInputs());
      for (size_t i = 0; i < input_impls_.size(); ++i) {
#if TF_MAJOR_VERSION > 2 || (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 3)
        TF_RETURN_IF_ERROR(dataset()->input_desc_.inputs[i]->MakeIterator(
            context, this, strings::StrCat(prefix(), "[", i, "]"), &input_impls_[i]));
#else
        // Older TF versions (2.2.0 and earlier) use 3 arguments
        TF_RETURN_IF_ERROR(dataset()->input_desc_.inputs[i]->MakeIterator(
            context, strings::StrCat(prefix(), "[", i, "]"), &input_impls_[i]));
#endif
      }
      input_ext_src_devices_.resize(dataset()->NumInputs());
      for (size_t i = 0; i < input_ext_src_devices_.size(); i++) {
        TF_DALI_CALL(input_ext_src_devices_[i] = daliGetOperatorBackend(
                         &pipeline_handle_, dataset()->input_desc_.input_names[i].c_str()));
      }
    }
    TF_RETURN_IF_ERROR(PrefetchPipeline(context, &pipeline_handle_));
    return CheckOutputDevices();
  }

  Status CheckOutputDevices() {
    auto num_outputs = daliGetNumOutput(&pipeline_handle_);
    for (auto i = 0; i < num_outputs; ++i) {
      auto dali_device_type = daliGetOutputDevice(&pipeline_handle_, i);

      if (dali_device_type != dataset()->device_type_) {
        auto msg = dali::make_string(
            "TF device and DALI device mismatch. TF device: ",
            (dataset()->device_type_ == device_type_t::CPU ? "CPU" : "GPU"),
            ", DALI device: ", (dali_device_type == device_type_t::CPU ? "CPU" : "GPU"),
            " for output ", i);

        if (dataset()->fail_on_device_mismatch_) {
          return Status(tensorflow::error::Code::INTERNAL, msg);
        }
      }
    }
    return Status::OK();
  }

  Status GetNextInternal(IteratorContext *context, std::vector<Tensor> *out_tensors,
                         bool *end_of_sequence) override {
    tensorflow::mutex_lock l(mu_);
    *end_of_sequence = false;

    if (dataset()->HasInputs()) {
      // if someone is stubborn enough to query us after the end of input data
      // TODO(klecki): we should raise a warning that reinitializing DALI Pipeline is not efficient
      if (iterator_state_ == InputState::stop_signaled) {
        *end_of_sequence = true;
        return Status::OK();
      }

      // Obtain the inputs and if end wasn't reached feed it.
      if (iterator_state_ == InputState::in_progress) {
        bool end_of_input_sequence;
        ListOfBatches batches;
        TF_RETURN_IF_ERROR(PrepareBatches(context, batches, end_of_input_sequence));
        if (end_of_input_sequence) {
          iterator_state_ = InputState::stop_pending;
        } else {
          TF_RETURN_IF_ERROR(FeedInputs(&pipeline_handle_, std::move(batches)));
        }
      }

      // We run out of data, indicate the end
      if (iterator_state_ == InputState::stop_pending && InputsScheduled() == 0) {
        iterator_state_ = InputState::stop_signaled;
        *end_of_sequence = true;
        for (auto &input : input_impls_) {
          input.reset();
        }
        return Status::OK();
      }
    }

    TF_RETURN_IF_ERROR(ProduceOutputs(context, out_tensors, *end_of_sequence));

    // we produced output, we can safely release the input that was used to produce it
    if (dataset()->HasInputs()) {
      assert(iterator_state_ != InputState::stop_signalled && InputsScheduled() > 0);
      ReleaseInputs();
    }
    // We schedule next run always when we don't have inputs or when we have inputs
    // and they produced something - which happens only in `in_progress` state.
    if (!dataset()->HasInputs() || iterator_state_ == InputState::in_progress) {
      TF_DALI_CALL(daliRun(&pipeline_handle_));
    }
    return Status::OK();
  }

  ~Iterator() {
    if (enable_memory_stats_) {
      size_t N;
      daliExecutorMetadata *meta;
      daliGetExecutorMetadata(&pipeline_handle_, &meta, &N);
      std::cout << "DALI operator memory statistics: " << std::endl;
      for (size_t i = 0; i < N; ++i) {
        std::cout << "Operator " << meta[i].operator_name;
        for (size_t j = 0; j < meta[i].out_num; ++j) {
          std::cout << "   output [ " << j << " ] : " << meta[i].real_size[j] << "B allocated "
                    << meta[i].max_real_size[j] << "B max allocated " << meta[i].reserved[j]
                    << "B reserved" << meta[i].max_reserved[j] << "B max reserved";
          if (j != meta[i].out_num - 1) {
            std::cout << ",";
          }
        }
        std::cout << std::endl;
      }
      daliFreeExecutorMetadata(meta, N);
    }
    daliDeletePipeline(&pipeline_handle_);
  }

#if TF_MAJOR_VERSION > 2 || (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 3)
  Status SaveInternal(SerializationContext *ctx, IteratorStateWriter *writer) override {
    return errors::Unimplemented("SaveInternal is not supported for DALI dataset.");
  }

  Status RestoreInternal(IteratorContext *ctx, IteratorStateReader *reader) override {
    return errors::Unimplemented("RestoreInternal is not supported for DALI dataset");
  }
#endif

 private:
  /**
   * @brief Schedule required number of runs of DALI Pipeline to fill the prefetch queue.
   *
   * When there are input datasets, feed the pipeline required number of input batches.
   *
   * TODO(klecki): Inputs handled only for an uniform executor
   */
  Status PrefetchPipeline(IteratorContext *context, daliPipelineHandle *pipeline_handle) {
    if (!dataset()->pipeline_def_.exec_separated) {
      int prefetch_depth = dataset()->pipeline_def_.prefetch_queue_depth;
      int actual_prefetch_depth = 0;
      if (dataset()->HasInputs()) {
        for (int i = 0; i < prefetch_depth; i++) {
          bool end_of_sequence = false;
          ListOfBatches batches;
          TF_RETURN_IF_ERROR(PrepareBatches(context, batches, end_of_sequence));
          if (end_of_sequence) {
            iterator_state_ = InputState::stop_pending;
            break;
          } else {
            // we only feed and don't release during warmup.
            TF_RETURN_IF_ERROR(FeedInputs(pipeline_handle, std::move(batches)));
            actual_prefetch_depth++;
          }
        }
      } else {
        actual_prefetch_depth = prefetch_depth;
      }
      TF_DALI_CALL(daliPrefetchUniform(pipeline_handle, actual_prefetch_depth));
    } else {
      if (dataset()->HasInputs()) {
        return errors::InvalidArgument("Input datasets are not compatible with split executor.");
      }
      TF_DALI_CALL(daliPrefetchSeparate(pipeline_handle,
                                        dataset()->pipeline_def_.cpu_prefetch_queue_depth,
                                        dataset()->pipeline_def_.gpu_prefetch_queue_depth));
    }
    return Status::OK();
  }

  /**
   * @brief Call GetNext on given input and extract the only Tensor Example that is expected as
   * output or signall stop.
   */
  Status GetExampleFromInput(IteratorContext *context, int input_idx, Tensor &example,
                             bool &end_of_sequence) {
    TfExample input_example;
    end_of_sequence = false;
    auto &input = input_impls_[input_idx];
    // TODO(klecki): ZipDataset just goes to next iteration on Error.
    // Desync of input datasets is not desired, we just report the problem fast
    TF_RETURN_IF_ERROR(input->GetNext(context, &input_example, &end_of_sequence));
    if (end_of_sequence) {
      return Status::OK();
    }

    // Repack the single Tensor from TfExample to Batch
    if (input_example.size() != 1) {
      return errors::InvalidArgument("Got an example consisting of ", input_example.size(),
                                     " elements for input: ", input_idx,
                                     ". Only examples of 1 element are supported.");
    }
    // Extract the obtained example
    example = input_example[0];
    return Status::OK();
  }

  /**
   * @brief Obtain samples from input of given index, operating in batch mode
   *
   * out_batch will contain one output sample representing whole batch
   */
  Status PrepareBatchesBatchMode(IteratorContext *context, int input_idx, Batch &out_batch,
                                 bool &end_of_sequence) {
    int next_batch_size = dataset()->pipeline_def_.batch_size;
    end_of_sequence = false;

    Tensor example;
    TF_RETURN_IF_ERROR(GetExampleFromInput(context, input_idx, example, end_of_sequence));
    if (end_of_sequence) {
      return Status::OK();
    }
    // Batch mode, only one tensor for whole batch
    out_batch = Batch{example};
    return Status::OK();
  }

  /**
   * @brief Obtain samples from input of given index, operation in sample mode.
   *
   * out_batch will contain sample tensors.
   */
  Status PrepareBatchesSampleMode(IteratorContext *context, int input_idx, Batch &out_batch,
                                  bool &end_of_sequence) {
    int next_batch_size = dataset()->pipeline_def_.batch_size;
    // Sample mode, so we have batch of actual sample tensors
    BatchStorage in_batch;
    in_batch.resize(next_batch_size);
    end_of_sequence = false;

    // We fail fast, we will either bubble up the error or set the state of dataset to stop_pending
    // and just use up what is queued.
    for (int sample_idx = 0; sample_idx < next_batch_size; sample_idx++) {
      TF_RETURN_IF_ERROR(
          GetExampleFromInput(context, input_idx, in_batch[sample_idx], end_of_sequence));
      if (end_of_sequence) {
        return Status::OK();
      }
    }
    out_batch = Batch{std::move(in_batch)};
    return Status::OK();
  }


  /**
   * @brief Obtain samples from input interators and build collection of batches representing
   * one input iteration.
   *
   * Get the input in either batch or sample mode, report errors fast and return empty batches.
   * In case of end_of_sequence or error we expect to start over.
   */
  Status PrepareBatches(IteratorContext *context, ListOfBatches &out_batches,
                        bool &end_of_sequence) {
    out_batches.clear();
    end_of_sequence = false;
    ListOfBatches input_batches(dataset()->NumInputs());
    int next_batch_size = dataset()->pipeline_def_.batch_size;

    for (int input_idx = 0; input_idx < dataset()->NumInputs(); input_idx++) {
      bool batched = dataset()->input_desc_.input_batched[input_idx];
      if (batched) {
        TF_RETURN_IF_ERROR(
            PrepareBatchesBatchMode(context, input_idx, input_batches[input_idx], end_of_sequence));
      } else {
        TF_RETURN_IF_ERROR(PrepareBatchesSampleMode(context, input_idx, input_batches[input_idx],
                                                    end_of_sequence));
      }
      if (end_of_sequence) {
        return Status::OK();
      }
      TF_RETURN_IF_ERROR(input_batches[input_idx].VerifyUniform(input_idx));
    }
    out_batches = std::move(input_batches);
    return Status::OK();
  }

  /**
   * @brief Obtain the last computed outputs from DALI Pipeline and copy them to the TF Tensors
   * that we allocated for outputs. Release the DALI Pipeline Outputs.
   */
  Status ProduceOutputs(IteratorContext *context, std::vector<Tensor> *out_tensors,
                        bool &end_of_sequence) {
    TF_DALI_CALL(daliShareOutput(&pipeline_handle_));

    auto num_outputs = 0;
    TF_DALI_CALL(num_outputs = daliGetNumOutput(&pipeline_handle_));

    for (int out_id = 0; out_id < num_outputs; ++out_id) {
      TensorShape output_shape;
      bool is_uniform = false;
      TF_DALI_CALL(is_uniform = daliOutputHasUniformShape(&pipeline_handle_, out_id));

      if (!is_uniform) {
        std::stringstream shapes;
        for (int sample_id = 0; sample_id < dataset()->pipeline_def_.batch_size; sample_id++) {
          AutoCPtr<int64_t> dali_shape;
          TF_DALI_CALL(dali_shape = AutoCPtr<int64_t>(
                           daliShapeAtSample(&pipeline_handle_, out_id, sample_id)));

          shapes << DaliToShape(dali_shape);
          if (sample_id < dataset()->pipeline_def_.batch_size - 1) {
            shapes << ", ";
          }
        }
        return errors::FailedPrecondition(
            "Batch output at index '", out_id,
            "' from DALI pipeline is not uniform - individual samples have different dimensions. "
            "This output cannot be represented as single, dense Tensor, which is required by "
            "TensorFlow. Ensure that all the samples that you produce in given batch have equal "
            "shape. Got shapes: ", shapes.str());
      }
      AutoCPtr<int64_t> dali_batch_shape;
      TF_DALI_CALL(dali_batch_shape = AutoCPtr<int64_t>(daliShapeAt(&pipeline_handle_, out_id)));
      auto dali_shape = DaliToShape(dali_batch_shape);
      auto status = GetCompatibleShape(output_shape, dataset()->shapes_[out_id], dali_shape,
                                       dataset()->pipeline_def_.batch_size, out_id);
      if (status != Status::OK()) {
        return status;
      }

      dali_data_type_t dali_type = DALI_NO_TYPE;
      TF_DALI_CALL(dali_type = daliTypeAt(&pipeline_handle_, out_id));
      auto tf_type = DaliToTfType(dali_type);

      if (tf_type != dataset()->dtypes_[out_id]) {
        std::stringstream ss;
        ss << "The type provided for output `" << out_id << "` is not compatible with "
           << "the type returned by DALI Pipeline. Expected (output_types[" << out_id
           << "]): " << DataTypeString(dataset()->dtypes_[out_id])
           << ", got from Pipeline: " << DataTypeString(tf_type) << ".";
        return errors::InvalidArgument(ss.str());
      }

      out_tensors->emplace_back(context->allocator({}), dataset()->dtypes_[out_id], output_shape);
      tensorflow::Tensor &output = out_tensors->operator[](out_id);

      void *dst = nullptr;  // TODO(klecki): output.data();
      switch (dataset()->dtypes_[out_id]) {
        case DT_BOOL:
          dst = reinterpret_cast<void *>(output.flat<bool>().data());
          break;
        case DT_HALF:
          dst = reinterpret_cast<void *>(output.flat<Eigen::half>().data());
          break;
        case DT_FLOAT:
          dst = reinterpret_cast<void *>(output.flat<float>().data());
          break;
        case DT_DOUBLE:
          dst = reinterpret_cast<void *>(output.flat<double>().data());
          break;
        case DT_UINT8:
          dst = reinterpret_cast<void *>(output.flat<uint8_t>().data());
          break;
        case DT_UINT16:
          dst = reinterpret_cast<void *>(output.flat<uint16_t>().data());
          break;
        case DT_UINT32:
          dst = reinterpret_cast<void *>(output.flat<uint32_t>().data());
          break;
        case DT_UINT64:
          dst = reinterpret_cast<void *>(output.flat<uint64>().data());
          break;
        case DT_INT8:
          dst = reinterpret_cast<void *>(output.flat<int8_t>().data());
          break;
        case DT_INT16:
          dst = reinterpret_cast<void *>(output.flat<int16_t>().data());
          break;
        case DT_INT32:
          dst = reinterpret_cast<void *>(output.flat<int32_t>().data());
          break;
        case DT_INT64:
          dst = reinterpret_cast<void *>(output.flat<int64>().data());
          break;
        default:
          return errors::InvalidArgument(
              "Unsupported type: " + DataTypeString(dataset()->dtypes_[out_id]) + " for tensor " +
              std::to_string(out_id));
      }

      TF_DALI_CALL(daliOutputCopy(&pipeline_handle_, dst, out_id, dataset()->device_type_,
                                  dataset()->stream_, false));
    }

    end_of_sequence = false;

    TF_DALI_CALL(daliOutputRelease(&pipeline_handle_));
    return Status::OK();
  }

  /**
   * @brief Feed a batches into coresponding inputs (External Source nodes).
   *
   * The batches are kept in queue to keep them alive long enough for DALI to process them.
   */
  Status FeedInputs(daliPipelineHandle *pipeline_handle, ListOfBatches &&batches) {
    // Keep alive the prefetch_queue_depth of batches - this corresponds to the number of batches
    // that we insert during warmup
    alive_batches_.push(std::move(batches));
    auto &current_batches = alive_batches_.back();

    // reuse the allocations for all inputs
    std::vector<const void *> ptrs;
    dali_data_type_t dtype;
    std::vector<int64_t> shapes;
    int64_t ndim;

    for (int input_idx = 0; input_idx < dataset()->NumInputs(); input_idx++) {
      auto &input_batch = current_batches[input_idx];
      bool batched = dataset()->input_desc_.input_batched[input_idx];

      auto &input_name = dataset()->input_desc_.input_names[input_idx];
      // TODO(klecki): Currently we are restricted to supporting input memory on the same
      // device as the DALIDataset placement
      auto input_device = dataset()->device_type_;
      auto &input_layout = dataset()->input_desc_.input_layouts[input_idx];
      auto ext_src_device = input_ext_src_devices_[input_idx];


      unsigned int flag = 0u;
      // We can share the data without copy when the backends are the same, otherwise
      // we need to copy.
      if ((input_device == CPU && ext_src_device == DALI_BACKEND_CPU) ||
          (input_device == GPU && ext_src_device == DALI_BACKEND_GPU)) {
        flag = DALI_ext_force_no_copy;
      } else {
        flag = DALI_ext_force_copy;
      }

      // TODO(klecki): Consider using other stream here: Dataset's stream_ or stream 0.
      if (batched) {
        const void *ptr = nullptr;
        TF_RETURN_IF_ERROR(input_batch.GetPtr(ptr));
        input_batch.GetShapes(shapes);
        TF_DALI_CALL(daliSetExternalInput(pipeline_handle, input_name.c_str(), input_device, ptr,
                                          input_batch.dtype(), shapes.data(), input_batch.ndim(),
                                          input_layout.c_str(), flag));
      } else {
        TF_RETURN_IF_ERROR(input_batch.GetPtrs(ptrs));
        input_batch.GetShapes(shapes);
        TF_DALI_CALL(daliSetExternalInputTensors(pipeline_handle, input_name.c_str(), input_device,
                                                 ptrs.data(), input_batch.dtype(), shapes.data(),
                                                 input_batch.ndim(), input_layout.c_str(), flag));
      }

      // No need keep the data if we did the copy
      if ((input_device == CPU && ext_src_device != DALI_BACKEND_CPU) ||
          (input_device == GPU && ext_src_device != DALI_BACKEND_GPU)) {
        input_batch.clear();
      }
    }
    return Status::OK();
  }

  void ReleaseInputs() {
    alive_batches_.pop();
  }

  int InputsScheduled() {
    return alive_batches_.size();
  }

  /**
   * @brief Get a shape that is compatible with the partial required shape (set for TF dataset)
   *        and the shape of batch returned from DALI pipeline.
   *        Allow to squeeze excess dimensions from the DALI shape but we cannot modify the
   *        rank of TF shape.
   *        If there is not unambiguous matching return error status.
   *
   * @param result the matching shape if the function returned Status::OK()
   */
  Status GetCompatibleShape(TensorShape &result, const PartialTensorShape &required_shape,
                            const TensorShape &dali_shape, int batch_size, int output_idx) {
    if (required_shape.IsCompatibleWith(dali_shape)) {
      result = dali_shape;
      return Status::OK();
    }

    // both ranks should be known at this points (otherwise shapes are compatible)
    // If they are not compatible and have the same rank or the required shape is the bigger one
    // we cannot make them compatible.
    if (required_shape.dims() >= dali_shape.dims()) {
      std::stringstream ss;
      ss << "The shape provided for output `" << output_idx << "` is not compatible with "
         << "the shape returned by DALI Pipeline. Expected (output_shapes[" << output_idx
         << "]): " << required_shape << ", got from Pipeline: " << dali_shape << ".";
      return errors::InvalidArgument(ss.str());
    }
    for (int i = 0; i < required_shape.dims(); i++) {
      result.AddDim(0);
    }
    // Non-trivial batch size case, different dims
    if (batch_size != 1) {
      // Should not happen, batch size from DALI will always match
      if (dali_shape.dim_size(0) != batch_size) {
        std::stringstream ss;
        ss << "The shape returned by DALI Pipeline for output `" << output_idx
           << "` has different `batch_size` than the one specified in `DALIDataset`. "
           << "Specified `batch_size`: " << batch_size
           << ", got from Pipeline: " << dali_shape.dim_size(0) << " in shape: " << dali_shape
           << ".";
        return errors::InvalidArgument(ss.str());
      }
      // It's easier to check it in C++ as in Python the structure is a bit convoluted
      if (!DimSizeMatch(required_shape.dim_size(0), batch_size)) {
        std::stringstream ss;
        ss << "The shape provided for output `" << output_idx << "` is not compatible with "
           << "the `batch_size` argument that was specified in `DALIDataset`. "
           << "Specified `batch_size`: " << batch_size << ", got: " << required_shape.dim_size(0)
           << " in shape: " << required_shape << ".";
        return errors::InvalidArgument(ss.str());
      }
    }
    // Special case for 1-element tensors
    if (dali_shape.num_elements() == 1) {
      TensorShape regular_shape;
      if (required_shape.AsTensorShape(&regular_shape) && regular_shape.num_elements() == 1) {
        result = regular_shape;
        return Status::OK();
      }
    }
    int matches = CountShapeMatches(result, required_shape, dali_shape);
    if (matches != 1) {
      std::stringstream ss;
      ss << "The shape provided for output `" << output_idx << "` is not compatible with "
         << "the shape returned by DALI Pipeline in an umabigous way. Expected (output_shapes["
         << output_idx << "]): " << required_shape << ", got from Pipeline: " << dali_shape << ".";
      return errors::InvalidArgument(ss.str());
    }
    return Status::OK();
  }

  /**
   * @brief Check if the given dimension sizes match. Negative value represent `None`
   *        placeholder. `dali` values are always concrete
   */
  bool DimSizeMatch(int64_t required, int64_t dali) {
    return required < 0 || required == dali;
  }

  /**
   * @brief Calculate the matching shapes by recursively matching possible. Count the number
   *        of possible matches. 1-sized dimensions in `dali_shape` can be skipped.
   *
   * If there is only 1 match it will be stored in the `result`.
   */
  int CountShapeMatches(TensorShape &result, const PartialTensorShape &required_shape,
                        const TensorShape &dali_shape, int req_pos = 0, int dali_pos = 0) {
    // We went over the whole shapes and they matched on the way
    if (req_pos == required_shape.dims() && dali_pos == dali_shape.dims()) {
      return 1;
    }
    // We have only DALI shape elements left, if they are all `1` it's ok.
    if (req_pos == required_shape.dims() && dali_pos < dali_shape.dims()) {
      if (dali_shape.dim_size(dali_pos) == 1) {
        return CountShapeMatches(result, required_shape, dali_shape, req_pos, dali_pos + 1);
      } else {
        return 0;
      }
    }
    if (req_pos < required_shape.dims() && dali_pos < dali_shape.dims()) {
      int total = 0;
      // We match exactly or to a "None" position
      if (DimSizeMatch(required_shape.dim_size(req_pos), dali_shape.dim_size(dali_pos))) {
        int matches =
            CountShapeMatches(result, required_shape, dali_shape, req_pos + 1, dali_pos + 1);
        // If we are the only exact match when backing up from recursion, save the result
        if (matches == 1) {
          result.set_dim(req_pos, dali_shape.dim_size(dali_pos));
        }
        total += matches;
      }
      // If DALI returned 1, we can skip this position an try other match
      if (dali_shape.dim_size(dali_pos) == 1) {
        total += CountShapeMatches(result, required_shape, dali_shape, req_pos, dali_pos + 1);
      }
      return total;
    }
    return 0;
  }

  enum class InputState {
    in_progress,   // we can still use inputs, none have ended
    stop_pending,  // input signalled end, we stop reading them, some might be in pipeline
    stop_signaled  // we ran out of batches ahead, time to raise stop ourselves
  };

  tensorflow::mutex mu_;
  std::vector<std::unique_ptr<IteratorBase>> input_impls_;
  // Obtained from pipeline, the `device` parameter of external source nodes used for inputs
  std::vector<dali_backend_t> input_ext_src_devices_;
  std::queue<ListOfBatches> alive_batches_;
  InputState iterator_state_ = InputState::in_progress;
  daliPipelineHandle pipeline_handle_;
  bool enable_memory_stats_;
};

void DALIDatasetOp::MakeDataset(OpKernelContext *context, DatasetBase **output) {
  Inputs inputs;
  FillInputs(context, inputs);
  ValidateInputs(context, inputs, input_attrs_);
  *output = new Dataset(context, pipeline_def_, inputs, input_attrs_, shapes_, dtypes_,
                        is_gpu_device_, fail_on_device_mismatch_);
}

void DALIDatasetOp::FillPipelineDef(OpKernelConstruction *context, PipelineDef &def) {
  OP_REQUIRES_OK(context, context->GetAttr(kPipeline, &def.pipeline));
  OP_REQUIRES_OK(context, context->GetAttr(kBatchSize, &def.batch_size));
  OP_REQUIRES_OK(context, context->GetAttr(kNumThreads, &def.num_threads));
  OP_REQUIRES_OK(context, context->GetAttr(kDeviceId, &def.device_id));
  OP_REQUIRES_OK(context, context->GetAttr(kExecSeparated, &def.exec_separated));
  OP_REQUIRES_OK(context, context->GetAttr(kPrefetchQueueDepth, &def.prefetch_queue_depth));
  OP_REQUIRES_OK(context, context->GetAttr(kCpuPrefetchQueueDepth, &def.cpu_prefetch_queue_depth));
  OP_REQUIRES_OK(context, context->GetAttr(kGpuPrefetchQueueDepth, &def.gpu_prefetch_queue_depth));
  OP_REQUIRES_OK(context, context->GetAttr(kGpuMemoryStats, &def.enable_memory_stats));
}

void DALIDatasetOp::FillInputAttrs(OpKernelConstruction *context, InputAttrs &def) {
  OP_REQUIRES_OK(context, context->GetAttr(kInputNames, &def.input_names));
  OP_REQUIRES_OK(context, context->GetAttr(kInputLayouts, &def.input_layouts));
  OP_REQUIRES_OK(context, context->GetAttr(kInputBatched, &def.input_batched));
}

void DALIDatasetOp::FillInputs(OpKernelContext *context, Inputs &def) {
  def.inputs.clear();
  def.inputs.reserve(context->num_inputs());
  for (size_t i = 0; i < context->num_inputs(); ++i) {
    DatasetBase *input;
    OP_REQUIRES_OK(context, GetDatasetFromVariantTensor(context->input(i), &input));
    def.inputs.push_back(input);
    // TODO(klecki): If possible, obtain the device placement of the inputs for verification,
    // we may try to make sure that the inputs to GPU dataset used copy_to_device.
  }
}

void DALIDatasetOp::ValidateInputs(OpKernelContext *context, Inputs &inputs,
                                   InputAttrs &input_attrs) {
  // This is not really needed - we do it in Python side already, but just in case:
  OP_REQUIRES(context, inputs.inputs.size() == input_attrs.input_names.size(),
              errors::InvalidArgument("Number of inputs and input names provided must match, got ",
                                      inputs.inputs.size(), " inputs and ",
                                      input_attrs.input_names.size(), " input names."));
  OP_REQUIRES(
      context, inputs.inputs.size() == input_attrs.input_layouts.size(),
      errors::InvalidArgument("Number of inputs and input layouts provided must match, got ",
                              inputs.inputs.size(), " inputs and ",
                              input_attrs.input_layouts.size(), " input layouts."));
  OP_REQUIRES(
      context, inputs.inputs.size() == input_attrs.input_batched.size(),
      errors::InvalidArgument("Number of inputs and input batched specification must match, got ",
                              inputs.inputs.size(), " inputs and ",
                              input_attrs.input_batched.size(), " input batched."));
  // TODO(klecki): Validate the input devices against the current device
}


std::unique_ptr<IteratorBase> DALIDatasetOp::Dataset::MakeIteratorInternal(
    const string &prefix) const {
  daliPipelineHandle pipeline_handle;
  TF_CHECK_OK(InitPipeline(&pipeline_handle));

  return absl::make_unique<Iterator>(Iterator::Params{this, strings::StrCat(prefix, "::DALI")},
                                     pipeline_handle, pipeline_def_.enable_memory_stats);
}


#if TF_MAJOR_VERSION > 2 || (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 4)

namespace {

Status MakeSplitProvidersImpl() {
  return errors::Unimplemented(
      "Cannot create split providers for dataset of type DALIDataset, "
      ", because the dataset does not support this functionality yet. "
      "Please use DALI sharding for iterating over parts of the dataset.");
}

}  // namespace


Status DALIDatasetOp::Dataset::InputDatasets(std::vector<const DatasetBase *> *inputs) const {
  if (!HasInputs()) {
    inputs->clear();
    return Status::OK();
  }
  inputs->resize(NumInputs());
  for (int i = 0; i < NumInputs(); i++) {
    inputs->operator[](i) = input_desc_.inputs[i];
  }
  return Status::OK();
}

#endif

#if TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 4 && TF_MINOR_VERSION < 6

Status DALIDatasetOp::Dataset::MakeSplitProvider(std::unique_ptr<SplitProvider> *) const {
  return MakeSplitProvidersImpl();
}

#elif TF_MAJOR_VERSION > 2 || (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 6)

Status DALIDatasetOp::Dataset::MakeSplitProviders(
    std::vector<std::unique_ptr<SplitProvider>> *) const {
  return MakeSplitProvidersImpl();
}

#endif


// Regestrations
REGISTER_KERNEL_BUILDER(Name("DALIDataset").Device(tensorflow::DEVICE_CPU), DALIDatasetOp);

REGISTER_KERNEL_BUILDER(
    Name("DALIDataset").Device(DEVICE_GPU).HostMemory("handle").HostMemory("input_datasets"),
    DALIDatasetOp);

REGISTER_INPUT_COLOCATION_EXEMPTION("DALIDataset");

REGISTER_OP("DALIDataset")
    .Input("input_datasets: N * variant")
    .Output("handle: variant")
    // the input_* attrs must match the input_datasets length
    .Attr("input_names: list(string)")
    .Attr("input_layouts: list(string)")
    .Attr("input_batched: list(int)")  // use vector<int> instead of vector<bool>
    .Attr("pipeline: string")
    .Attr("batch_size: int")
    .Attr("num_threads: int")
    .Attr("device_id: int")
    .Attr("exec_separated: bool")
    .Attr("prefetch_queue_depth: int")
    .Attr("cpu_prefetch_queue_depth: int")
    .Attr("gpu_prefetch_queue_depth: int")
    .Attr("enable_memory_stats: bool = false")
    .Attr("N: int >= 0")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr(
        "output_dtypes: "
        "list({bool, half, float, uint8, uint16, uint32, uint64, int8, int16, int32, int64}) >= 1")
    .Attr("fail_on_device_mismatch: bool = true")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
DALI Dataset plugin
Creates a DALI dataset compatible with tf.data.Dataset from a DALI pipeline.
`output_shapes` must match the shape of the corresponding DALI Pipeline output tensor shape.
`output_dtypes` must match the type of the corresponding DALI Pipeline output tensors type.
)doc");

}  // namespace dali_tf_impl

#endif  // TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION >= 15
