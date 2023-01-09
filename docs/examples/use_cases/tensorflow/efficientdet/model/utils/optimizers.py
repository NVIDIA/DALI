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
"""Keras efficientdet optimizers."""

from absl import logging
import numpy as np
import tensorflow as tf

_DEFAULT_BATCH_SIZE = 64


def get_optimizer(params, *args):
    """Get optimizer."""
    learning_rate = learning_rate_schedule(params, *args)

    if params["optimizer"].lower() == "sgd":
        logging.info("Use SGD optimizer")
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate, momentum=params["momentum"])
    elif params["optimizer"].lower() == "adam":
        logging.info("Use Adam optimizer")
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)
    else:
        raise ValueError("optimizers should be adam or sgd")
    return optimizer


def update_learning_rate_schedule_parameters(
    params, epochs, global_batch_size, steps_per_epoch
):
    """Updates params that are related to the learning rate schedule."""
    # Learning rate is proportional to the batch size
    params["adjusted_learning_rate"] = (
        params["learning_rate"] * global_batch_size / _DEFAULT_BATCH_SIZE
    )

    if "lr_warmup_init" in params:
        params["adjusted_lr_warmup_init"] = (
            params["lr_warmup_init"] * global_batch_size / _DEFAULT_BATCH_SIZE
        )

    params["lr_warmup_step"] = int(params["lr_warmup_epoch"] * steps_per_epoch)
    params["first_lr_drop_step"] = int(params["first_lr_drop_epoch"] * steps_per_epoch)
    params["second_lr_drop_step"] = int(
        params["second_lr_drop_epoch"] * steps_per_epoch
    )
    params["total_steps"] = epochs * steps_per_epoch


def learning_rate_schedule(params, *args):
    """Learning rate schedule based on global step."""
    update_learning_rate_schedule_parameters(params, *args)
    lr_decay_method = params["lr_decay_method"]
    if lr_decay_method == "stepwise":
        return StepwiseLrSchedule(
            params["adjusted_learning_rate"],
            params["adjusted_lr_warmup_init"],
            params["lr_warmup_step"],
            params["first_lr_drop_step"],
            params["second_lr_drop_step"],
        )

    if lr_decay_method == "cosine":
        return CosineLrSchedule(
            params["adjusted_learning_rate"],
            params["adjusted_lr_warmup_init"],
            params["lr_warmup_step"],
            params["total_steps"],
        )

    if lr_decay_method == "polynomial":
        return PolynomialLrSchedule(
            params["adjusted_learning_rate"],
            params["adjusted_lr_warmup_init"],
            params["lr_warmup_step"],
            params["poly_lr_power"],
            params["total_steps"],
        )

    if lr_decay_method == "constant":
        return params["adjusted_learning_rate"]

    raise ValueError("unknown lr_decay_method: {}".format(lr_decay_method))


class StepwiseLrSchedule(tf.optimizers.schedules.LearningRateSchedule):
    """Stepwise learning rate schedule."""

    def __init__(
        self,
        adjusted_lr: float,
        lr_warmup_init: float,
        lr_warmup_step: int,
        first_lr_drop_step: int,
        second_lr_drop_step: int,
    ):
        """Build a StepwiseLrSchedule.

        Args:
          adjusted_lr: `float`, The initial learning rate.
          lr_warmup_init: `float`, The warm up learning rate.
          lr_warmup_step: `int`, The warm up step.
          first_lr_drop_step: `int`, First lr decay step.
          second_lr_drop_step: `int`, Second lr decay step.
        """
        super().__init__()
        logging.info("LR schedule method: stepwise")
        self.adjusted_lr = adjusted_lr
        self.lr_warmup_init = lr_warmup_init
        self.lr_warmup_step = lr_warmup_step
        self.first_lr_drop_step = first_lr_drop_step
        self.second_lr_drop_step = second_lr_drop_step

    def __call__(self, step):
        linear_warmup = self.lr_warmup_init + (
            tf.cast(step, dtype=tf.float32)
            / self.lr_warmup_step
            * (self.adjusted_lr - self.lr_warmup_init)
        )
        learning_rate = tf.where(
            step < self.lr_warmup_step, linear_warmup, self.adjusted_lr
        )
        lr_schedule = [
            [1.0, self.lr_warmup_step],
            [0.1, self.first_lr_drop_step],
            [0.01, self.second_lr_drop_step],
        ]
        for mult, start_global_step in lr_schedule:
            learning_rate = tf.where(
                step < start_global_step, learning_rate, self.adjusted_lr * mult
            )
        return learning_rate


class CosineLrSchedule(tf.optimizers.schedules.LearningRateSchedule):
    """Cosine learning rate schedule."""

    def __init__(
        self,
        adjusted_lr: float,
        lr_warmup_init: float,
        lr_warmup_step: int,
        total_steps: int,
    ):
        """Build a CosineLrSchedule.

        Args:
          adjusted_lr: `float`, The initial learning rate.
          lr_warmup_init: `float`, The warm up learning rate.
          lr_warmup_step: `int`, The warm up step.
          total_steps: `int`, Total train steps.
        """
        super().__init__()
        logging.info("LR schedule method: cosine")
        self.adjusted_lr = adjusted_lr
        self.lr_warmup_init = lr_warmup_init
        self.lr_warmup_step = lr_warmup_step
        self.decay_steps = tf.cast(total_steps - lr_warmup_step, tf.float32)

    def __call__(self, step):
        linear_warmup = self.lr_warmup_init + (
            tf.cast(step, dtype=tf.float32)
            / self.lr_warmup_step
            * (self.adjusted_lr - self.lr_warmup_init)
        )
        cosine_lr = (
            0.5
            * self.adjusted_lr
            * (1 + tf.cos(np.pi * tf.cast(step, tf.float32) / self.decay_steps))
        )
        return tf.where(step < self.lr_warmup_step, linear_warmup, cosine_lr)


class PolynomialLrSchedule(tf.optimizers.schedules.LearningRateSchedule):
    """Polynomial learning rate schedule."""

    def __init__(
        self,
        adjusted_lr: float,
        lr_warmup_init: float,
        lr_warmup_step: int,
        power: float,
        total_steps: int,
    ):
        """Build a PolynomialLrSchedule.

        Args:
          adjusted_lr: `float`, The initial learning rate.
          lr_warmup_init: `float`, The warm up learning rate.
          lr_warmup_step: `int`, The warm up step.
          power: `float`, power.
          total_steps: `int`, Total train steps.
        """
        super().__init__()
        logging.info("LR schedule method: polynomial")
        self.adjusted_lr = adjusted_lr
        self.lr_warmup_init = lr_warmup_init
        self.lr_warmup_step = lr_warmup_step
        self.power = power
        self.total_steps = total_steps

    def __call__(self, step):
        linear_warmup = self.lr_warmup_init + (
            tf.cast(step, dtype=tf.float32)
            / self.lr_warmup_step
            * (self.adjusted_lr - self.lr_warmup_init)
        )
        polynomial_lr = self.adjusted_lr * tf.pow(
            1 - (tf.cast(step, dtype=tf.float32) / self.total_steps), self.power
        )
        return tf.where(step < self.lr_warmup_step, linear_warmup, polynomial_lr)
