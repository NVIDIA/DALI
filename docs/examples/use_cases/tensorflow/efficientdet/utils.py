# Copyright 2020 Google Research. All Rights Reserved.
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
# ==============================================================================
"""Common utils."""
from typing import Text, Tuple, Union
import tensorflow as tf
import argparse
import tensorflow.compat.v1 as tf1
from enum import Enum
from collections import namedtuple
from tensorflow.python.eager import (
    tape as tape_lib,
)  # pylint:disable=g-direct-tensorflow-import
from tensorflow.python.tpu import (
    tpu_function,
)  # pylint:disable=g-direct-tensorflow-import

# pylint: disable=logging-format-interpolation


class PipelineType(Enum):
    synthetic = 0
    tensorflow = 1
    dali_cpu = 2
    dali_gpu = 3


class InputType(Enum):
    tfrecord = 0
    coco = 1


# argparse multiline argument help according to: https://stackoverflow.com/a/22157136
class SmartFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith("R|"):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


def dict_to_namedtuple(dict_instance):
    NamedTuple = namedtuple("NamedTuple", dict_instance.keys())
    return NamedTuple._make(dict_instance.values())


def setup_gpus():
    for gpu_instance in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu_instance, True)
    tf.config.set_soft_device_placement(True)


def get_dataset(args, total_batch_size, is_training, params, strategy=None):
    pipeline = args.pipeline_type

    if strategy and not is_training and pipeline == PipelineType.dali_gpu:
        strategy = None
        pipeline = PipelineType.dali_cpu

    if pipeline in [PipelineType.tensorflow, PipelineType.synthetic]:
        from pipeline.tf.dataloader import InputReader

        if args.input_type != InputType.tfrecord:
            raise ValueError(
                "tensorflow and syntax pipelines are only compatible with tfrecord input type :<"
            )

        if is_training:
            file_pattern = args.train_file_pattern
        else:
            file_pattern = args.eval_file_pattern or args.train_file_pattern

        dataset = InputReader(
            params,
            file_pattern,
            is_training=is_training,
            use_fake_data=(pipeline == PipelineType.synthetic),
        ).get_dataset(total_batch_size)
    elif strategy:
        from pipeline.dali.efficientdet_pipeline import EfficientDetPipeline

        if pipeline == PipelineType.dali_cpu:
            raise ValueError(
                "dali_cpu pipeline is not compatible with multi_gpu mode :<"
            )

        def dali_dataset_fn(input_context):
            with tf.device(f"/gpu:{input_context.input_pipeline_id}"):
                device_id = input_context.input_pipeline_id
                num_shards = input_context.num_input_pipelines
                return EfficientDetPipeline(
                    params,
                    int(total_batch_size / num_shards),
                    args,
                    is_training=is_training,
                    num_shards=num_shards,
                    device_id=device_id,
                ).get_dataset()

        input_options = tf.distribute.InputOptions(
            experimental_place_dataset_on_device=True,
            experimental_fetch_to_device=False,
            experimental_replication_mode=tf.distribute.InputReplicationMode.PER_REPLICA,
        )

        dataset = strategy.distribute_datasets_from_function(
            dali_dataset_fn, input_options
        )

    else:
        from pipeline.dali.efficientdet_pipeline import EfficientDetPipeline

        cpu_only = pipeline == PipelineType.dali_cpu
        device = "/cpu:0" if cpu_only else "/gpu:0"
        with tf.device(device):
            dataset = EfficientDetPipeline(
                params,
                total_batch_size,
                args,
                is_training=is_training,
                cpu_only=cpu_only,
            ).get_dataset()

    return dataset


def srelu_fn(x):
    """Smooth relu: a smooth version of relu."""
    with tf.name_scope("srelu"):
        beta = tf.Variable(20.0, name="srelu_beta", dtype=tf.float32) ** 2
        beta = tf.cast(beta ** 2, x.dtype)
        safe_log = tf.math.log(tf.where(x > 0.0, beta * x + 1.0, tf.ones_like(x)))
        return tf.where((x > 0.0), x - (1.0 / beta) * safe_log, tf.zeros_like(x))


def activation_fn(features: tf1.Tensor, act_type: Text):
    """Customized non-linear activation type."""
    if act_type in ("silu", "swish"):
        return tf.keras.activations.swish(features)
    elif act_type == "swish_native":
        return features * tf.keras.activations.sigmoid(features)
    elif act_type == "hswish":
        return features * tf.nn.relu6(features + 3) / 6
    elif act_type == "relu":
        return tf.nn.relu(features)
    elif act_type == "relu6":
        return tf.nn.relu6(features)
    elif act_type == "mish":
        return features * tf.math.tanh(tf.math.softplus(features))
    elif act_type == "srelu":
        return srelu_fn(features)
    else:
        raise ValueError("Unsupported act_type {}".format(act_type))


def cross_replica_mean(t, num_shards_per_group=None):
    """Calculates the average value of input tensor across TPU replicas."""
    num_shards = tpu_function.get_tpu_context().number_of_shards
    if not num_shards_per_group:
        return tf1.tpu.cross_replica_sum(t) / tf.cast(num_shards, t.dtype)

    group_assignment = None
    if num_shards_per_group > 1:
        if num_shards % num_shards_per_group != 0:
            raise ValueError(
                "num_shards: %d mod shards_per_group: %d, should be 0"
                % (num_shards, num_shards_per_group)
            )
        num_groups = num_shards // num_shards_per_group
        group_assignment = [
            [x for x in range(num_shards) if x // num_shards_per_group == y]
            for y in range(num_groups)
        ]
    return tf1.tpu.cross_replica_sum(t, group_assignment) / tf.cast(
        num_shards_per_group, t.dtype
    )


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Fixed default name of BatchNormalization to match TpuBatchNormalization."""

    def __init__(self, **kwargs):
        if not kwargs.get("name", None):
            kwargs["name"] = "tpu_batch_normalization"
        super().__init__(**kwargs)

    def call(self, inputs, training=None):
        outputs = super().call(inputs, training)
        # A temporary hack for tf. compatibility with keras batch norm.
        for u in self.updates:
            tf1.add_to_collection(tf1.GraphKeys.UPDATE_OPS, u)
        return outputs


def build_batch_norm(
    beta_initializer: Text = "zeros",
    gamma_initializer: Text = "ones",
    data_format: Text = "channels_last",
    momentum: float = 0.99,
    epsilon: float = 1e-3,
    name: Text = "tpu_batch_normalization",
):
    """Build a batch normalization layer.

    Args:
      beta_initializer: `str`, beta initializer.
      gamma_initializer: `str`, gamma initializer.
      data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
      momentum: `float`, momentume of batch norm.
      epsilon: `float`, small value for numerical stability.
      name: the name of the batch normalization layer

    Returns:
      A normalized `Tensor` with the same `data_format`.
    """
    axis = 1 if data_format == "channels_first" else -1

    bn_layer = BatchNormalization(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=True,
        scale=True,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        name=name,
    )

    return bn_layer


def drop_connect(inputs, is_training, survival_prob):
    """Drop the entire conv with given survival probability."""
    # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    if not is_training:
        return inputs

    # Compute tensor.
    batch_size = tf.shape(inputs)[0]
    random_tensor = survival_prob
    random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    # Unlike conventional way that multiply survival_prob at test time, here we
    # divide survival_prob at training time, such that no addition compute is
    # needed at test time.
    output = inputs / survival_prob * binary_tensor
    return output


def parse_image_size(image_size: Union[Text, int, Tuple[int, int]]):
    """Parse the image size and return (height, width).

    Args:
      image_size: A integer, a tuple (H, W), or a string with HxW format.

    Returns:
      A tuple of integer (height, width).
    """
    if isinstance(image_size, int):
        # image_size is integer, with the same width and height.
        return (image_size, image_size)

    if isinstance(image_size, str):
        # image_size is a string with format WxH
        width, height = image_size.lower().split("x")
        return (int(height), int(width))

    if isinstance(image_size, tuple):
        return image_size

    raise ValueError(
        "image_size must be an int, WxH string, or (height, width)"
        "tuple. Was %r" % image_size
    )


def get_feat_sizes(image_size: Union[Text, int, Tuple[int, int]], max_level: int):
    """Get feat widths and heights for all levels.

    Args:
      image_size: A integer, a tuple (H, W), or a string with HxW format.
      max_level: maximum feature level.

    Returns:
      feat_sizes: a list of tuples (height, width) for each level.
    """
    image_size = parse_image_size(image_size)
    feat_sizes = [{"height": image_size[0], "width": image_size[1]}]
    feat_size = image_size
    for _ in range(1, max_level + 1):
        feat_size = ((feat_size[0] - 1) // 2 + 1, (feat_size[1] - 1) // 2 + 1)
        feat_sizes.append({"height": feat_size[0], "width": feat_size[1]})
    return feat_sizes


def _recompute_grad(f):
    """An eager-compatible version of recompute_grad.

    For f(*args, **kwargs), this supports gradients with respect to args or
    kwargs, but kwargs are currently only supported in eager-mode.
    Note that for keras layer and model objects, this is handled automatically.

    Warning: If `f` was originally a tf.keras Model or Layer object, `g` will not
    be able to access the member variables of that object, because `g` returns
    through the wrapper function `inner`.  When recomputing gradients through
    objects that inherit from keras, we suggest keeping a reference to the
    underlying object around for the purpose of accessing these variables.

    Args:
      f: function `f(*x)` that returns a `Tensor` or sequence of `Tensor` outputs.

    Returns:
     A function `g` that wraps `f`, but which recomputes `f` on the backwards
     pass of a gradient call.
    """

    @tf1.custom_gradient
    def inner(*args, **kwargs):
        """Inner function closure for calculating gradients."""
        current_var_scope = tf1.get_variable_scope()
        with tape_lib.stop_recording():
            result = f(*args, **kwargs)

        def grad_wrapper(*wrapper_args, **grad_kwargs):
            """Wrapper function to accomodate lack of kwargs in graph mode decorator."""

            @tf1.custom_gradient
            def inner_recompute_grad(*dresult):
                """Nested custom gradient function for computing grads in reverse and forward mode autodiff."""
                # Gradient calculation for reverse mode autodiff.
                variables = grad_kwargs.get("variables")
                with tf.GradientTape() as t:
                    id_args = tf.nest.map_structure(tf.identity, args)
                    t.watch(id_args)
                    if variables is not None:
                        t.watch(variables)
                    with tf.control_dependencies(dresult):
                        with tf1.variable_scope(current_var_scope):
                            result = f(*id_args, **kwargs)
                kw_vars = []
                if variables is not None:
                    kw_vars = list(variables)
                grads = t.gradient(
                    result,
                    list(id_args) + kw_vars,
                    output_gradients=dresult,
                    unconnected_gradients=tf.UnconnectedGradients.ZERO,
                )

                def transpose(*t_args, **t_kwargs):
                    """Gradient function calculation for forward mode autodiff."""
                    # Just throw an error since gradients / activations are not stored on
                    # tape for recompute.
                    raise NotImplementedError(
                        "recompute_grad tried to transpose grad of {}. "
                        "Consider not using recompute_grad in forward mode"
                        "autodiff".format(f.__name__)
                    )

                return (grads[: len(id_args)], grads[len(id_args) :]), transpose

            return inner_recompute_grad(*wrapper_args)

        return result, grad_wrapper

    return inner


def recompute_grad(recompute=False):
    """Decorator determine whether use gradient checkpoint."""

    def _wrapper(f):
        if recompute:
            return _recompute_grad(f)
        return f

    return _wrapper
