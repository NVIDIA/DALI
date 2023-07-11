# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


# This file is modified version of mnist_classifier_fromscratch.py from JAX codebase.
# File in its unchanged form is included.

import numpy.random as npr
from jax import jit, grad
from jax.scipy.special import logsumexp
import jax.numpy as jnp

layer_sizes = [784, 1024, 1024, 10]
param_scale = 0.1
step_size = 0.001


def init_random_params(scale=param_scale, layer_sizes=layer_sizes, rng=npr.RandomState(0)):
    return [(scale * rng.randn(m, n), scale * rng.randn(n))
            for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]


def predict(params, inputs):
    activations = inputs
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = jnp.tanh(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w) + final_b
    return logits - logsumexp(logits, axis=1, keepdims=True)


def loss(params, batch):
    inputs = batch['images']
    targets = batch['labels']

    preds = predict(params, inputs)
    return -jnp.mean(jnp.sum(preds * targets, axis=1))


def accuracy(params, iterator):
    correct_predictions_num = 0
    for batch in iterator:
        inputs = batch['images']
        targets = batch['labels']

        predicted_class = jnp.argmax(predict(params, inputs), axis=1)
        correct_predictions_num = correct_predictions_num + \
            jnp.sum(predicted_class == targets.ravel())

    return correct_predictions_num / iterator.size


@jit
def update(params, batch, step_size=step_size):
    grads = grad(loss)(params, batch)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]
