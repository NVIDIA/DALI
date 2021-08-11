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
import functools
from absl import logging
import tensorflow as tf

import hparams_config
import utils
from .backbone import efficientnet_builder
from .utils import postprocess
from .utils import layers
from .utils import losses
import re


# pylint: disable=arguments-differ  # fo keras layers.


class EfficientDetNet(tf.keras.Model):
    """EfficientDet keras network with train and test step."""

    def __init__(self, model_name=None, params=None, name=""):
        """Initialize model."""
        super().__init__(name=name)

        self.train_metrics = {
            "mean_loss_tracker": tf.keras.metrics.Mean(name="mean_loss"),
            "loss_tracker": tf.keras.metrics.Mean(name="loss"),
            "lr_tracker": tf.keras.metrics.Mean(name="lr"),
        }
        self.train_metrics = utils.dict_to_namedtuple(self.train_metrics)
        self.mAP_tracker = tf.keras.metrics.Mean(name="mAP")

        if params:
            self.config = hparams_config.Config(params)
        else:
            self.config = hparams_config.get_efficientdet_config(model_name)

        config = self.config

        # Backbone.
        backbone_name = config.backbone_name
        if "efficientnet" in backbone_name:
            override_params = {
                "relu_fn": functools.partial(
                    utils.activation_fn, act_type=config.act_type
                ),
                "grad_checkpoint": self.config.grad_checkpoint,
            }
            if "b0" in backbone_name:
                override_params["survival_prob"] = 0.0
            if config.backbone_config is not None:
                override_params[
                    "blocks_args"
                ] = efficientnet_builder.BlockDecoder().encode(
                    config.backbone_config.blocks
                )
            override_params["data_format"] = config.data_format
            self.backbone = efficientnet_builder.get_model(
                backbone_name, override_params=override_params
            )

        # Feature network.
        self.resample_layers = []  # additional resampling layers.
        for level in range(6, config.max_level + 1):
            # Adds a coarser level by downsampling the last feature map.
            self.resample_layers.append(
                layers.ResampleFeatureMap(
                    feat_level=(level - config.min_level),
                    target_num_channels=config.fpn_num_filters,
                    apply_bn=config.apply_bn_for_resampling,
                    conv_after_downsample=config.conv_after_downsample,
                    data_format=config.data_format,
                    name="resample_p%d" % level,
                )
            )
        self.fpn_cells = layers.FPNCells(config)

        # class/box output prediction network.
        num_anchors = len(config.aspect_ratios) * config.num_scales
        num_filters = config.fpn_num_filters
        self.class_net = layers.ClassNet(
            num_classes=config.num_classes,
            num_anchors=num_anchors,
            num_filters=num_filters,
            min_level=config.min_level,
            max_level=config.max_level,
            act_type=config.act_type,
            repeats=config.box_class_repeats,
            separable_conv=config.separable_conv,
            survival_prob=config.survival_prob,
            grad_checkpoint=config.grad_checkpoint,
            data_format=config.data_format,
        )

        self.box_net = layers.BoxNet(
            num_anchors=num_anchors,
            num_filters=num_filters,
            min_level=config.min_level,
            max_level=config.max_level,
            act_type=config.act_type,
            repeats=config.box_class_repeats,
            separable_conv=config.separable_conv,
            survival_prob=config.survival_prob,
            grad_checkpoint=config.grad_checkpoint,
            data_format=config.data_format,
        )

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
        features, num_pos, _, _, *targets = inputs

        labels = {}
        for level in range(config.min_level, config.max_level + 1):
            i = 2 * (level - config.min_level)
            labels["cls_targets_%d" % level] = targets[i]
            labels["box_targets_%d" % level] = targets[i + 1]
        labels["mean_num_positives"] = tf.reshape(
            tf.tile(tf.expand_dims(tf.reduce_mean(num_pos), 0), [config.batch_size]),
            [config.batch_size, 1],
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

    def _calc_mAP(self, pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes):
        def iou(box1, box2):
            l = max(box1[0], box2[0])
            t = max(box1[1], box2[1])
            r = min(box1[2], box2[2])
            b = min(box1[3], box2[3])
            i = max(0, r - l) * max(0, b - t)
            u = (
                (box1[2] - box1[0]) * (box1[3] - box1[1])
                + (box2[2] - box2[0]) * (box2[3] - box2[1])
                - i
            )
            return i / u

        batch_size = pred_boxes.shape[0]
        num_pred_boxes = 0
        num_gt_boxes = 0
        num_true_positives = 0

        stats = []
        for batch_idx in range(batch_size):

            pred_num_positives = tf.math.count_nonzero(pred_scores[batch_idx, :] > 0.25)
            gt_num = tf.math.count_nonzero(gt_classes > -1)

            gt_used_idx = []
            num_pred_boxes += pred_num_positives
            num_gt_boxes += gt_num

            # for pred_idx, (pred_box, pred_class) in enumerate(zip(pred_boxes, pred_classes)):
            for pred_idx in range(pred_num_positives):
                pred_box = pred_boxes[batch_idx, pred_idx]
                pred_class = pred_classes[batch_idx, pred_idx]

                found = False
                # for gt_idx, (gt_box, gt_class) in enumerate(zip(boxes, classes)):
                for gt_idx in range(gt_num):

                    if gt_idx in gt_used_idx:
                        continue

                    gt_box = gt_boxes[batch_idx, gt_idx]
                    gt_class = gt_classes[batch_idx, gt_idx]

                    if pred_class != gt_class:
                        continue

                    if iou(pred_box, gt_box) < 0.5:
                        continue

                    found = True
                    num_true_positives += 1
                    break

                stats.append((pred_scores[batch_idx, pred_idx], found))

        if num_pred_boxes == 0:
            return 0.0

        ap = 0.0
        max_prec = num_true_positives / num_pred_boxes
        for _, found in sorted(stats):
            if found:
                ap += max_prec / tf.cast(num_gt_boxes, dtype=tf.float64)
                num_true_positives -= 1
            num_pred_boxes -= 1
            if num_pred_boxes == 0:
                break
            max_prec = tf.math.maximum(max_prec, num_true_positives / num_pred_boxes)

        return ap

    def call(self, inputs, training):
        config = self.config
        # call backbone network.
        all_feats = self.backbone(inputs, training=training, features_only=True)
        feats = all_feats[config.min_level : config.max_level + 1]

        # Build additional input features that are not from backbone.
        for resample_layer in self.resample_layers:
            feats.append(resample_layer(feats[-1], training, None))

        # call feature network.
        fpn_feats = self.fpn_cells(feats, training)

        # call class/box output network.
        class_outputs = self.class_net(fpn_feats, training)
        box_outputs = self.box_net(fpn_feats, training)
        return (class_outputs, box_outputs)

    def train_step(self, inputs):
        config = self.config

        features, labels = self._unpack_inputs(inputs)

        with tf.GradientTape() as tape:
            cls_out_list, box_out_list = self.call(features, training=True)
            cls_outputs, box_outputs = self._unpack_outputs(cls_out_list, box_out_list)

            # cls_loss and box_loss are for logging. only total_loss is optimized.
            det_loss, cls_loss, box_loss = losses.detection_loss(
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
        features, _, gt_boxes, gt_classes, *_ = inputs
        # tf.print(gt_boxes, gt_classes)
        # gt_boxes = tf.stack(
        #    [
        #        gt_boxes[..., 0] * self.config.image_size[0],
        #        gt_boxes[..., 1] * self.config.image_size[1],
        #        gt_boxes[..., 2] * self.config.image_size[0],
        #        gt_boxes[..., 3] * self.config.image_size[1],
        #    ],
        #    axis=-1,
        # )

        cls_out_list, box_out_list = self.call(features, training=False)
        ltrb, scores, classes, _ = postprocess.postprocess_per_class(
            self.config, cls_out_list, box_out_list
        )
        classes = tf.cast(classes, dtype=tf.int32)

        ap = tf.py_function(
            func=self._calc_mAP,
            inp=[ltrb, scores, classes, gt_boxes, gt_classes],
            Tout=tf.float64,
        )
        self.mAP_tracker.update_state(ap)

        return {m.name: m.result() for m in [self.mAP_tracker]}

    @property
    def metrics(self):
        return list(self.train_metrics) + [self.mAP_tracker]
