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
"""Keras implementation of efficientdet."""
from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import re

import utils
from . import efficientdet_net
import pipeline.anchors as anchors

# pylint: disable=arguments-differ  # fo keras layers.

_DEFAULT_BATCH_SIZE = 64


def get_optimizer(params, *args):
    """Get optimizer."""
    learning_rate = learning_rate_schedule(params, *args)

    if params["optimizer"].lower() == "sgd":
        logging.info("Use SGD optimizer")
        optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=params["momentum"])
    elif params["optimizer"].lower() == "adam":
        logging.info("Use Adam optimizer")
        optimizer = tf.keras.optimizers.Adam(learning_rate)
    else:
        raise ValueError("optimizers should be adam or sgd")

    moving_average_decay = params["moving_average_decay"]
    if moving_average_decay:
        from tensorflow_addons import (
            optimizers as tfa_optimizers,
        )  # pylint: disable=g-import-not-at-top

        optimizer = tfa_optimizers.MovingAverage(
            optimizer,
            average_decay=moving_average_decay,
            name="ExponentialMovingAverage",
        )

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


class FocalLoss(tf.keras.losses.Loss):
    """Compute the focal loss between `logits` and the golden `target` values.

    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    """

    def __init__(self, alpha, gamma, label_smoothing=0.0, **kwargs):
        """Initialize focal loss.

        Args:
          alpha: A float32 scalar multiplying alpha to the loss from positive
            examples and (1-alpha) to the loss from negative examples.
          gamma: A float32 scalar modulating loss from hard and easy examples.
          label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.
          **kwargs: other params.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    @tf.autograph.experimental.do_not_convert
    def call(self, y, y_pred):
        """Compute focal loss for y and y_pred.

        Args:
          y: A tuple of (normalizer, y_true), where y_true is the target class.
          y_pred: A float32 tensor [batch, height_in, width_in, num_predictions].

        Returns:
          the focal loss.
        """
        normalizer, y_true = y
        alpha = tf.convert_to_tensor(self.alpha, dtype=y_pred.dtype)
        gamma = tf.convert_to_tensor(self.gamma, dtype=y_pred.dtype)
        normalizer = tf.cast(normalizer, dtype=y_pred.dtype)

        # compute focal loss multipliers before label smoothing, such that it will
        # not blow up the loss.
        pred_prob = tf.sigmoid(y_pred)
        p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = (1.0 - p_t) ** gamma

        # apply label smoothing for cross_entropy for each entry.
        if label_smoothing:
            y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

        # compute the final loss and return
        return alpha_factor * modulating_factor * ce / normalizer


class BoxLoss(tf.keras.losses.Loss):
    """L2 box regression loss."""

    def __init__(self, delta=0.1, **kwargs):
        """Initialize box loss.

        Args:
          delta: `float`, the point where the huber loss function changes from a
            quadratic to linear. It is typically around the mean value of regression
            target. For instances, the regression targets of 512x512 input with 6
            anchors on P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
          **kwargs: other params.
        """
        super().__init__(**kwargs)
        self.huber = tf.keras.losses.Huber(
            delta, reduction=tf.keras.losses.Reduction.SUM
        )

    @tf.autograph.experimental.do_not_convert
    def call(self, y_true, box_outputs):
        num_positives, box_targets = y_true
        normalizer = num_positives * 4.0
        mask = tf.cast(tf.not_equal(box_targets, 0.0), box_outputs.dtype)
        box_loss = tf.cast(
            self.huber(box_targets, box_outputs, sample_weight=mask), box_outputs.dtype
        )
        box_loss /= normalizer
        return box_loss


def focal_loss(y_pred, y_true, alpha, gamma, normalizer, label_smoothing=0.0):
    """Compute the focal loss between `logits` and the golden `target` values.

    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.

    Args:
      y_pred: A float tensor of size [batch, height_in, width_in,
        num_predictions].
      y_true: A float tensor of size [batch, height_in, width_in,
        num_predictions].
      alpha: A float scalar multiplying alpha to the loss from positive examples
        and (1-alpha) to the loss from negative examples.
      gamma: A float scalar modulating loss from hard and easy examples.
      normalizer: Divide loss by this value.
      label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.

    Returns:
      loss: A float32 scalar representing normalized total loss.
    """
    with tf1.name_scope("focal_loss"):
        normalizer = tf.cast(normalizer, dtype=y_pred.dtype)

        # compute focal loss multipliers before label smoothing, such that it will
        # not blow up the loss.
        pred_prob = tf.math.sigmoid(y_pred)
        p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = (1.0 - p_t) ** gamma

        # apply label smoothing for cross_entropy for each entry.
        if label_smoothing:
            y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

        # compute the final loss and return
        return (1 / normalizer) * alpha_factor * modulating_factor * ce


def _box_loss(box_outputs, box_targets, num_positives, delta=0.1):
    """Computes box regression loss."""
    # delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
    normalizer = num_positives * 4.0
    mask = tf.not_equal(box_targets, 0.0)
    box_loss = tf1.losses.huber_loss(
        box_targets,
        box_outputs,
        weights=mask,
        delta=delta,
        reduction=tf1.losses.Reduction.SUM,
    )
    box_loss /= normalizer
    return box_loss


def detection_loss(cls_outputs, box_outputs, labels, config):
    """Computes total detection loss.

    Computes total detection loss including box and class loss from all levels.
    Args:
      cls_outputs: an OrderDict with keys representing levels and values
        representing logits in [batch_size, height, width, num_anchors].
      box_outputs: an OrderDict with keys representing levels and values
        representing box regression targets in [batch_size, height, width,
        num_anchors * 4].
      labels: the dictionary that returned from dataloader that includes
        groundtruth targets.
      config: the dictionary including training parameters specified in
        default_haprams function in this file.

    Returns:
      total_loss: an integer tensor representing total loss reducing from
        class and box losses from all levels.
      cls_loss: an integer tensor representing total class loss.
      box_loss: an integer tensor representing total box regression loss.
    """
    # Sum all positives in a batch for normalization and avoid zero
    # num_positives_sum, which would lead to inf loss during training
    num_positives_sum = tf.math.reduce_sum(labels["mean_num_positives"]) + 1.0
    positives_momentum = config.get("positives_momentum", None) or 0
    if positives_momentum > 0:
        # normalize the num_positive_examples for training stability.
        moving_normalizer_var = tf.Variable(
            0.0,
            name="moving_normalizer",
            dtype=tf.float32,
            synchronization=tf.VariableSynchronization.ON_READ,
            trainable=False,
            aggregation=tf.VariableAggregation.MEAN,
        )
        num_positives_sum = tf.keras.backend.moving_average_update(
            moving_normalizer_var, num_positives_sum, momentum=config.positives_momentum
        )
    elif positives_momentum < 0:
        num_positives_sum = utils.cross_replica_mean(num_positives_sum)

    levels = cls_outputs.keys()
    cls_losses = []
    box_losses = []
    for level in levels:
        # Onehot encoding for classification labels.
        cls_targets_at_level = tf.one_hot(
            labels["cls_targets_%d" % level],
            config.num_classes,
            dtype=cls_outputs[level].dtype,
        )

        if config.data_format == "channels_first":
            bs, _, width, height, _ = cls_targets_at_level.get_shape().as_list()
            cls_targets_at_level = tf.reshape(
                cls_targets_at_level, [bs, -1, width, height]
            )
        else:
            bs, width, height, _, _ = cls_targets_at_level.get_shape().as_list()
            cls_targets_at_level = tf.reshape(
                cls_targets_at_level, [bs, width, height, -1]
            )
        box_targets_at_level = labels["box_targets_%d" % level]

        cls_loss = focal_loss(
            cls_outputs[level],
            cls_targets_at_level,
            config.alpha,
            config.gamma,
            normalizer=num_positives_sum,
            label_smoothing=config.label_smoothing,
        )

        if config.data_format == "channels_first":
            cls_loss = tf.reshape(cls_loss, [bs, -1, width, height, config.num_classes])
        else:
            cls_loss = tf.reshape(cls_loss, [bs, width, height, -1, config.num_classes])

        cls_loss *= tf.cast(
            tf.expand_dims(tf.not_equal(labels["cls_targets_%d" % level], -2), -1),
            cls_loss.dtype,
        )
        cls_loss_sum = tf.reduce_sum(cls_loss)
        cls_losses.append(tf.cast(cls_loss_sum, tf.float32))

        if config.box_loss_weight:
            box_losses.append(
                _box_loss(
                    box_outputs[level],
                    box_targets_at_level,
                    num_positives_sum,
                    delta=config.delta,
                )
            )

    # Sum per level losses to total loss.
    cls_loss = tf.math.add_n(cls_losses)
    box_loss = tf.math.add_n(box_losses) if box_losses else tf.constant(0.0)

    total_loss = cls_loss + config.box_loss_weight * box_loss

    return total_loss, cls_loss, box_loss


class EfficientDetTrain(efficientdet_net.EfficientDetNet):
    """EfficientDet keras network defining own train_step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_metrics = {
            "mean_loss_tracker": tf.keras.metrics.Mean(name="mean_loss"),
            "loss_tracker": tf.keras.metrics.Mean(name="loss"),
            "lr_tracker": tf.keras.metrics.Mean(name="lr"),
        }
        self.train_metrics = utils.dict_to_namedtuple(self.train_metrics)

        self.test_metrics = {
            "cls_loss_tracker": tf.keras.metrics.Mean(name="cls_loss"),
            "box_loss_tracker": tf.keras.metrics.Mean(name="box_loss"),
            # "accuracy_tracker": tf.keras.metrics.Mean(name="accuracy"),
        }
        self.test_metrics = utils.dict_to_namedtuple(self.test_metrics)

    def _freeze_vars(self):
        if self.config.var_freeze_expr:
            return [
                v
                for v in self.trainable_variables
                if not re.match(self.config.var_freeze_expr, v.name)
            ]
        return self.trainable_variables

    def _reg_l2_loss(self, weight_decay, regex=r".*(kernel|weight):0$"):
        """Return regularization l2 loss loss."""
        var_match = re.compile(regex)
        return weight_decay * tf.add_n(
            [
                tf.nn.l2_loss(v)
                for v in self.trainable_variables
                if var_match.match(v.name)
            ]
        )

    def _unpack_inputs(self, inputs):
        config = self.config

        features, num_pos, *targets = inputs
        labels = {}
        for level in range(config.min_level, config.max_level + 1):
            i = 2 * (level - config.min_level)
            labels["cls_targets_%d" % level] = targets[i]
            labels["box_targets_%d" % level] = targets[i + 1]
        labels["mean_num_positives"] = tf.reshape(
            tf.tile(
                tf.expand_dims(tf.reduce_mean(num_pos), 0), [config.train_batch_size]
            ),
            [config.train_batch_size, 1],
        )

        return features, labels

    def _unpack_outputs(self, cls_out_list, box_out_list):
        config = self.config
        min_level = config.min_level
        max_level = config.max_level
        cls_outputs, box_outputs = {}, {}

        for i in range(min_level, max_level + 1):
            cls_outputs[i] = cls_out_list[i - min_level]
            box_outputs[i] = box_out_list[i - min_level]

        return cls_outputs, box_outputs

    def train_step(self, inputs):
        config = self.config

        features, labels = self._unpack_inputs(inputs)

        with tf.GradientTape() as tape:
            cls_out_list, box_out_list = self.call(features, training=True)
            cls_outputs, box_outputs = self._unpack_outputs(cls_out_list, box_out_list)

            # cls_loss and box_loss are for logging. only total_loss is optimized.
            det_loss, cls_loss, box_loss = detection_loss(
                cls_outputs, box_outputs, labels, config
            )
            reg_l2loss = self._reg_l2_loss(config.weight_decay)
            total_loss = det_loss + reg_l2loss

        trainable_vars = self._freeze_vars()
        gradients = tape.gradient(total_loss, trainable_vars)

        if config.clip_gradients_norm:
            clip_norm = abs(config.clip_gradients_norm)
            gradients = [
                tf.clip_by_norm(g, clip_norm) if g is not None else None
                for g in gradients
            ]
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.train_metrics.mean_loss_tracker.update_state(total_loss)
        self.train_metrics.loss_tracker.reset_states()
        self.train_metrics.loss_tracker.update_state(total_loss)
        self.train_metrics.lr_tracker.reset_states()
        self.train_metrics.lr_tracker.update_state(
            self.optimizer.lr(self.optimizer.iterations)
        )

        return {m.name: m.result() for m in self.train_metrics}

    def test_step(self, inputs):
        config = self.config
        nms_configs = config.nms_configs
        params = config.as_dict()
        params["nms_configs"]["max_nms_inputs"] = anchors.MAX_DETECTION_POINTS

        features, labels = self._unpack_inputs(inputs)

        cls_out_list, box_out_list = self.call(features, training=False)
        cls_outputs, box_outputs = self._unpack_outputs(cls_out_list, box_out_list)

        det_loss, cls_loss, box_loss = detection_loss(
            cls_outputs, box_outputs, labels, self.config
        )

        # cls_outputs = postprocess.to_list(cls_outputs)
        # box_outputs = postprocess.to_list(box_outputs)
        # boxes, scores, classes = postprocess.pre_nms(params, cls_outputs, box_outputs)

        # metric_fn_inputs = {
        #    "image_ids": labels["source_ids"],
        #    "groundtruth_data": labels["groundtruth_data"],
        #    "image_scales": labels["image_scales"],
        # }

        # nms_boxes, nms_scores, nms_classes, _ = postprocess.per_class_nms(
        #    params,
        #    boxes,
        #    scores,
        #    classes,
        #    image_scales,
        # )
        # img_ids = tf.cast(tf.expand_dims(kwargs["image_ids"], -1), nms_scores.dtype)
        # detections_bs = [
        #    img_ids * tf.ones_like(nms_scores),
        #    nms_boxes[:, :, 1],
        #    nms_boxes[:, :, 0],
        #    nms_boxes[:, :, 3] - nms_boxes[:, :, 1],
        #    nms_boxes[:, :, 2] - nms_boxes[:, :, 0],
        #    nms_scores,
        #    nms_classes,
        # ]
        # detections_bs = tf.stack(detections_bs, axis=-1, name="detnections")

        # if params.get("testdev_dir", None):
        #    logging.info("Eval testdev_dir %s", params["testdev_dir"])
        #    eval_metric = coco_metric.EvaluationMetric(
        #        testdev_dir=params["testdev_dir"]
        #    )
        #    coco_metrics = eval_metric.estimator_metric_fn(detections_bs, tf.zeros([1]))
        # else:
        #    logging.info("Eval val with groudtruths %s.", params["val_json_file"])
        #    eval_metric = coco_metric.EvaluationMetric(
        #        filename=params["val_json_file"], label_map=params["label_map"]
        #    )
        #    coco_metrics = eval_metric.estimator_metric_fn(
        #        detections_bs, kwargs["groundtruth_data"]
        #    )

        # Add metrics to output.
        self.test_metrics.cls_loss_tracker.update_state(
            cls_loss, config.eval_batch_size
        )
        self.test_metrics.box_loss_tracker.update_state(
            box_loss, config.eval_batch_size
        )
        # self.test_metrics.accuracy_tracker.update_state()
        # output_metrics.update(coco_metrics)
        return {m.name: m.result() for m in self.test_metrics}

    @property
    def metrics(self):
        return list(self.train_metrics) + list(self.test_metrics)
