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
import contextlib
import os
from typing import Text, Tuple, Union
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
from collections import namedtuple
from tensorflow.python.eager import (
    tape as tape_lib,
)  # pylint:disable=g-direct-tensorflow-import
from tensorflow.python.tpu import (
    tpu_function,
)  # pylint:disable=g-direct-tensorflow-import

# pylint: disable=logging-format-interpolation


def dict_to_namedtuple(字典):
    NamedTuple = namedtuple("NamedTuple", 字典.keys())
    return NamedTuple._make(字典.values())


def srelu_fn(x):
    """Smooth relu: a smooth version of relu."""
    with tf.name_scope("srelu"):
        beta = tf.Variable(20.0, name="srelu_beta", dtype=tf.float32) ** 2
        beta = tf.cast(beta ** 2, x.dtype)
        safe_log = tf.math.log(tf.where(x > 0.0, beta * x + 1.0, tf.ones_like(x)))
        return tf.where((x > 0.0), x - (1.0 / beta) * safe_log, tf.zeros_like(x))


def activation_fn(features: tf.Tensor, act_type: Text):
    """Customized non-linear activation type."""
    if act_type in ("silu", "swish"):
        return tf.nn.swish(features)
    elif act_type == "swish_native":
        return features * tf.sigmoid(features)
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
        return tf.tpu.cross_replica_sum(t) / tf.cast(num_shards, t.dtype)

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
    return tf.tpu.cross_replica_sum(t, group_assignment) / tf.cast(
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
        # A temporary hack for tf1 compatibility with keras batch norm.
        for u in self.updates:
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, u)
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


conv_kernel_initializer = tf.initializers.variance_scaling()
dense_kernel_initializer = tf.initializers.variance_scaling()


class Pair(tuple):
    def __new__(cls, name, value):
        return super().__new__(cls, (name, value))

    def __init__(self, name, _):  # pylint: disable=super-init-not-called
        self.name = name


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

    @tf.custom_gradient
    def inner(*args, **kwargs):
        """Inner function closure for calculating gradients."""
        current_var_scope = tf.get_variable_scope()
        with tape_lib.stop_recording():
            result = f(*args, **kwargs)

        def grad_wrapper(*wrapper_args, **grad_kwargs):
            """Wrapper function to accomodate lack of kwargs in graph mode decorator."""

            @tf.custom_gradient
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
                        with tf.variable_scope(current_var_scope):
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
