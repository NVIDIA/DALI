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


from jax import jit, grad, lax, pmap
from functools import partial
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy.random as npr


layers = [784, 1024, 1024, 10]


def init_model(layers=layers, rng=npr.RandomState(0)):
    model = []
    for (
        in_size,
        out_size,
    ) in zip(layers[:-1], layers[1:]):
        new_w = 0.1 * rng.randn(in_size, out_size)
        new_b = 0.1 * rng.randn(out_size)
        new_layer = (new_w, new_b)
        model.append(new_layer)

    return model


def predict(model, images):
    input = images
    for w, b in model[:-1]:
        output = jnp.dot(input, w) + b
        input = jnp.tanh(output)

    last_w, last_b = model[-1]
    last_output = jnp.dot(input, last_w) + last_b
    return last_output - logsumexp(last_output, axis=1, keepdims=True)


def loss(model, batch):
    predicted_labels = predict(model, batch["images"])
    return -jnp.mean(jnp.sum(predicted_labels * batch["labels"], axis=1))


def accuracy(model, iterator):
    correct_predictions_num = 0
    for batch in iterator:
        images = batch["images"]
        labels = batch["labels"]

        predicted_class = jnp.argmax(predict(model, images), axis=1)
        correct_predictions_num = correct_predictions_num + jnp.sum(
            predicted_class == labels.ravel()
        )

    return correct_predictions_num / iterator.size


@jit
def update(model, batch, learning_rate=0.001):
    grads = grad(loss)(model, batch)

    updated_model = []

    for model, updates in zip(model, grads):
        w, b = model
        dw, db = updates

        new_w = w - learning_rate * dw
        new_b = b - learning_rate * db

        updated_model.append((new_w, new_b))

    return updated_model


@partial(pmap, axis_name="data")
def update_parallel(model, batch, learning_rate=0.001):
    grads = grad(loss)(model, batch)

    grads = lax.pmean(grads, axis_name="data")

    updated_model = []

    for model, updates in zip(model, grads):
        w, b = model
        dw, db = updates

        new_w = w - learning_rate * dw
        new_b = b - learning_rate * db

        updated_model.append((new_w, new_b))

    return updated_model
