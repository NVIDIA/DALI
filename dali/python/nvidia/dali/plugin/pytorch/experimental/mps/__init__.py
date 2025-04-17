# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ["DALIPipelineRunner"]

from torch.utils.data._utils.collate import default_collate_fn_map as _default_collate_fn_map
from nvidia.dali.external_source import ExternalSource as _ExternalSource
from nvidia.dali.plugin.pytorch.torch_utils import to_torch_tensor
from inspect import Parameter, Signature


def _external_source_node_names(pipeline):
    """
    extract the names of all the ExternalSource nodes in the pipeline
    """
    # TODO(janton): Add a native function to query those names, so that we can do it
    # also on deserialized pipelines
    if pipeline._deserialized:
        raise RuntimeError(
            "Not able to find the external source "
            "operator names, since the pipeline was deserialized"
        )
    if not pipeline._py_graph_built:
        pipeline._build_graph()
    input_node_names = []
    for op in pipeline._ops:
        if isinstance(op._op, _ExternalSource):
            input_node_names.append(op.name)
    return input_node_names


class DALIOutputSampleRef:
    """
    Reference for a single sample output bound to a pipeline run.
    """

    def __init__(self, pipe, output_idx, sample_idx):
        """
        Args:
            pipe (DALIPipeline): The pipeline object used.
            output_idx (int): The index of the output in the pipeline.
            sample_idx (int): The index of the sample within the batch.
        """
        self.pipe = pipe
        self.output_idx = output_idx
        self.sample_idx = sample_idx

    def __repr__(self):
        return (
            f"DALIOutputSampleRef(pipe={self.pipe}, "
            + f"output_idx={self.output_idx}, sample_idx={self.sample_idx})"
        )


class DALIOutputBatchRef:
    """
    Reference for a batched output bound to a pipeline run.
    """

    def __init__(self, pipe, output_idx):
        """
        Args:
            pipe (_DALIPipeline): A reference to the pipeline.
            output_idx (int): The index of the output in the pipeline.
        """
        self.pipe = pipe
        self.output_idx = output_idx

    def __repr__(self):
        return f"DALIOutputBatchRef(pipe={self.pipe}, output_idx={self.output_idx})"


def _collate_dali_output_sample_ref_fn(samples, *, collate_fn_map=None):
    """
    Special collate function that schedules a DALI iteration for execution
    """
    assert len(samples) > 0
    pipe = samples[0].pipe
    output_idx = samples[0].output_idx
    for i, sample in enumerate(samples):
        if sample.pipe != pipe or sample.output_idx != output_idx:
            raise RuntimeError("All samples should belong to the same batch")

        if sample.sample_idx != i:
            raise RuntimeError("Unexpected sample order")

    return pipe._complete_batch()[output_idx]


# In-place modify `default_collate_fn_map` to handle DALIOutputSampleRef
_default_collate_fn_map.update({DALIOutputSampleRef: _collate_dali_output_sample_ref_fn})


class DALIPipelineRunner:
    def __init__(self, pipeline_fn, pipeline_kwargs):
        # Pipeline function
        self._pipeline_fn = pipeline_fn
        # Pipeline kwargs
        self._pipeline_kwargs = pipeline_kwargs
        # get pipeline
        self._pipe = None
        self._signature = None
        self._num_outputs = None
        # Current batch
        self._curr_batch_params = {}
        # Whether the current batch is complete
        self._batch_complete = False
        # batch idx
        self._batch_idx = None
        self._batch_sample_idx = None
        # Outputs of the current batch
        self._batch_outputs = []

        self._callable = None

    def init_pipeline(self):
        if self._pipe is not None:
            return self._pipe

        self._pipe = self._pipeline_fn(**self._pipeline_kwargs)

        # Override callable signature
        self._dali_input_names = _external_source_node_names(self._pipe)
        num_inputs = len(self._dali_input_names)
        if num_inputs == 0:
            raise RuntimeError("The provided pipeline doesn't have any inputs")

        parameters = [Parameter("self", Parameter.POSITIONAL_OR_KEYWORD)]
        parameter_kind = (
            Parameter.POSITIONAL_OR_KEYWORD if num_inputs == 1 else Parameter.KEYWORD_ONLY
        )
        for input_name in self._dali_input_names:
            parameters.append(Parameter(input_name, parameter_kind))
        return_annotation = tuple(DALIOutputSampleRef for _ in range(self._pipe.num_outputs))
        self._signature = Signature(parameters, return_annotation=return_annotation)
        self._num_outputs = self._pipe.num_outputs
        return self._pipe

    def _add_sample(self, inputs):
        """
        Adds a sample to the current batch. In the collate function, we mark the batch as
        complete and submit it for execution.
        When a completed batch is encountered, a new batch should be started.
        """
        if self._batch_idx is None or self._batch_complete:
            self._batch_idx = self._batch_idx + 1 if self._batch_idx is not None else 0
            self._batch_sample_idx = 0
            self._curr_batch_params = {}
            self._batch_complete = False

        for name, value in inputs.items():
            # we want to transfer only the arguments to the caller side, not the the self reference
            if name == "self":
                continue
            if name not in self._curr_batch_params:
                self._curr_batch_params[name] = []
            self._curr_batch_params[name].append(value)

        ret = tuple(
            DALIOutputSampleRef(self, output_idx=i, sample_idx=self._batch_sample_idx)
            for i in range(self._num_outputs)
        )

        # unpack single element tuple
        if len(ret) == 1:
            ret = ret[0]
        self._batch_sample_idx += 1
        return ret

    def _complete_batch(self):
        """
        Complete the current batch and submit it for execution.
        """
        if self._batch_complete is False:
            self._batch_complete = True
            for key, value in self._curr_batch_params.items():
                self._pipe.feed_input(key, value)
            self._pipe._run_once()
            dali_outputs = self._pipe.outputs()
            self._batch_outputs = tuple(
                to_torch_tensor(out.as_tensor(), not self._pipe.exec_dynamic)
                for out in dali_outputs
            )
        return self._batch_outputs

    def __call__(self, *args, **kwargs):
        self.init_pipeline()
        bound_args = self._signature.bind(self, *args, **kwargs)
        return self._add_sample(bound_args.arguments)
