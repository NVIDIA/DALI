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
"""Loss functions for efficientdet."""

import tensorflow as tf
import tensorflow.compat.v1 as tf1
import utils

# pylint: disable=arguments-differ  # fo keras layers.


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
