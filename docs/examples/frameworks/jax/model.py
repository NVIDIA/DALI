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

from jax import jit, grad
import jax.numpy as jnp
from mnist_classifier_fromscratch import init_random_params as jax_init_random_params
from mnist_classifier_fromscratch import loss, predict


layer_sizes = [784, 1024, 1024, 10]
param_scale = 0.1
step_size = 0.001


def init_random_params():
    return jax_init_random_params(param_scale, layer_sizes)


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
    grads = grad(loss)(params, (batch['images'], batch['labels']))
    
    updated_params = []
    
    for params, updates in zip(params, grads):
        w, b = params
        dw, db = updates
        
        new_w = w - step_size * dw
        new_b = b - step_size * db
        
        updated_params.append((new_w, new_b))
    
    return updated_params
