#!/usr/bin/env python
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import print_function
from builtins import range
import tensorflow as tf

try:
    from tensorflow.compat.v1.train import Optimizer
except:
    # Older TF versions don't have compat.v1 layer
    from tensorflow.train import Optimizer

class LarcOptimizer(Optimizer):
    def __init__(self, optimizer, learning_rate, eta, clip=True, epsilon=1.,
                 name="LarcOptimizer", use_locking=False):
        super(LarcOptimizer, self).__init__(
            name=name, use_locking=use_locking)
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._eta = float(eta) if eta is not None else None
        self._clip = clip
        self._epsilon = float(epsilon)

    def compute_gradients(self, *args, **kwargs):
        return self._optimizer.compute_gradients(*args, **kwargs)

    def apply_gradients(self, gradvars, *args, **kwargs):
        if self._eta is not None:
            v_list = [tf.norm(tensor=v, ord=2) for _, v in gradvars]
            g_list = [tf.norm(tensor=g, ord=2) if g is not None else 0.0
                      for g, _ in gradvars]
            v_norms = tf.stack(v_list)
            g_norms = tf.stack(g_list)
            zeds = tf.zeros_like(v_norms)
            cond = tf.logical_and(
                tf.not_equal(v_norms, zeds),
                tf.not_equal(g_norms, zeds))
            true_vals = tf.scalar_mul(self._eta, tf.div(v_norms, g_norms))
            false_vals = tf.fill(tf.shape(v_norms), self._epsilon)
            larc_local_lr = tf.where(cond, true_vals, false_vals)
            if self._clip:
                ones = tf.ones_like(v_norms)
                lr = tf.fill(tf.shape(v_norms), self._learning_rate)
                larc_local_lr = tf.minimum(tf.div(larc_local_lr, lr), ones)
            gradvars = [(tf.multiply(larc_local_lr[i], g), v)
                        if g is not None else (None, v)
                        for i, (g, v) in enumerate(gradvars) ]
        return self._optimizer.apply_gradients(gradvars, *args, **kwargs)


class LossScalingOptimizer(Optimizer):
    """An optimizer that scales loss and un-scales gradients."""

    def __init__(self, optimizer,
                 scale=None,
                 name="LossScalingOptimizer",
                 use_locking=False):
        super(LossScalingOptimizer, self).__init__(
            name=name, use_locking=use_locking)
        self._optimizer=optimizer
        self._scale = float(scale) if scale is not None else 1.0

    def compute_gradients(self, loss, var_list=None, *args, **kwargs):
        if self._scale != 1.0:
            loss = tf.scalar_mul(self._scale, loss)
        gradvar = self._optimizer.compute_gradients(loss, var_list, *args, **kwargs)
        gradvar = [(tf.scalar_mul(1./self._scale, g), v) for g, v in gradvar]
        return gradvar

    def apply_gradients(self, *args, **kwargs):
        return self._optimizer.apply_gradients(*args, **kwargs)

