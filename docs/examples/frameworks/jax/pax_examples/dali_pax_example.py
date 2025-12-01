# coding=utf-8
# Copyright 2022 The Pax Authors.
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


"""A self-contained example of a CNN classifier on the MNIST dataset."""

from typing import cast

import jax
import jax.numpy as jnp
from paxml import base_experiment
from paxml import base_task
from paxml import experiment_registry
from paxml import learners
from paxml import tasks_lib
from praxis import base_input
from praxis import base_layer
from praxis import base_model
from praxis import optimizers
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis import schedules
from praxis.layers import activations
from praxis.layers import convolutions
from praxis.layers import linears
from praxis.layers import poolings

from pax_examples.dali_pax_input import MnistDaliInput


NestedMap = py_utils.NestedMap
template_field = base_layer.template_field
JTensor = pytypes.JTensor
WeightedScalars = pytypes.WeightedScalars
Predictions = base_model.Predictions


class CNN(base_layer.BaseLayer):
    """CNN layer."""

    height: int = 28
    width: int = 28
    num_classes: int = 10
    kernel_size: int = 3
    activation_tpl: pax_fiddle.Config[activations.BaseActivation] = (
        template_field(activations.ReLU)
    )

    def setup(self) -> None:
        super().setup()

        conv1_p = pax_fiddle.Config(
            convolutions.ConvBNAct,
            name="conv1",
            filter_shape=(self.kernel_size, self.kernel_size, 1, 32),
            filter_stride=(1, 1),
            batch_norm_tpl=None,
            activation_tpl=self.activation_tpl.clone(),
            bias=True,
        )
        self.create_child("conv1", conv1_p)
        pooling1_p = pax_fiddle.Config(
            poolings.Pooling,
            window_shape=(2, 2),
            window_stride=(2, 2),
            pooling_type="AVG",
        )
        self.create_child("pooling1", pooling1_p)
        conv2_p = pax_fiddle.Config(
            convolutions.ConvBNAct,
            name="conv2",
            filter_shape=(self.kernel_size, self.kernel_size, 32, 64),
            filter_stride=(1, 1),
            batch_norm_tpl=None,
            activation_tpl=self.activation_tpl.clone(),
            bias=True,
        )
        self.create_child("conv2", conv2_p)
        pooling2_p = pooling1_p.clone()
        self.create_child("pooling2", pooling2_p)

        dense_p = pax_fiddle.Config(
            linears.FeedForward,
            name="dense1",
            input_dims=4 * self.height * self.width,
            output_dims=256,
            has_bias=True,
            activation_tpl=self.activation_tpl.clone(),
        )
        linear_p = pax_fiddle.Config(
            linears.Linear,
            name="linear",
            input_dims=256,
            output_dims=self.num_classes,
        )
        bias_p = pax_fiddle.Config(
            linears.Bias, name="bias", dims=self.num_classes
        )
        self.create_children("dense", [dense_p, linear_p, bias_p])

    def __call__(self, inputs: JTensor) -> JTensor:
        batch_size = inputs.shape[0]
        outputs = inputs
        outputs = self.conv1(outputs)
        outputs = self.pooling1(outputs)[0]
        outputs = self.conv2(outputs)
        outputs = self.pooling2(outputs)[0]
        outputs = jnp.reshape(outputs, [batch_size, -1])
        for _, dense_layer in enumerate(self.dense):
            outputs = dense_layer(outputs)
        return outputs


def _cross_entropy_loss(targets, preds):
    num_classes = preds.shape[-1]
    log_preds = jax.nn.log_softmax(preds)
    one_hot_targets = jax.nn.one_hot(targets, num_classes)
    loss = jnp.mean(-jnp.sum(one_hot_targets * log_preds, axis=-1))
    return loss


def _compute_accuracy(targets, preds):
    accuracy = (targets == jnp.argmax(preds, axis=-1)).astype(jnp.float32)
    return jnp.mean(accuracy)


class CNNModel(base_model.BaseModel):
    """CNN model."""

    network_tpl: pax_fiddle.Config[base_layer.BaseLayer] = template_field(CNN)

    def setup(self) -> None:
        # Construct the model.
        super().setup()
        self.create_child("network", self.network_tpl)

    def compute_predictions(self, input_batch: NestedMap) -> Predictions:
        return self.network(input_batch["inputs"])

    def compute_loss(
        self, predictions: Predictions, input_batch: NestedMap
    ) -> tuple[WeightedScalars, NestedMap]:
        predictions = cast(NestedMap, predictions)
        loss = _cross_entropy_loss(input_batch["labels"], predictions)
        accuracy = _compute_accuracy(input_batch["labels"], predictions)
        return NestedMap(loss=(loss, 1.0), accuracy=(accuracy, 1.0)), NestedMap(
            predictions=predictions
        )


@experiment_registry.register
class MnistExperiment(base_experiment.BaseExperiment):
    """MNIST model based on CNN. xid/62622458."""

    BATCH_SIZE = 1024
    MOMENTUM = 0.1
    LEARNING_RATE = 0.01
    TRAIN_STEPS = 1000
    HEIGHT = 28
    WIDTH = 28
    NUM_CLASSES = 10
    KERNEL_SIZE = 3

    # SPMD related hparams.
    MESH_AXIS_NAMES = ("replica", "data", "mdl")
    ICI_MESH_SHAPE = [1, 1, 1]

    def task(self) -> pax_fiddle.Config[base_task.BaseTask]:
        model_p = pax_fiddle.Config(
            CNNModel,
            network_tpl=pax_fiddle.Config(
                CNN,
                name="mnist_model",
                height=self.HEIGHT,
                width=self.WIDTH,
                num_classes=self.NUM_CLASSES,
                kernel_size=self.KERNEL_SIZE,
            ),
        )
        model_p.ici_mesh_shape = self.ICI_MESH_SHAPE
        model_p.mesh_axis_names = self.MESH_AXIS_NAMES

        optimizer_p = pax_fiddle.Config(
            optimizers.ShardedSgd,
            momentum=self.MOMENTUM,
            lr_schedule=pax_fiddle.Config(schedules.Constant, value=1),
        )
        optimizer_p.learning_rate = self.LEARNING_RATE
        learner_p = pax_fiddle.Config(
            learners.Learner,
            loss_name="loss",
            optimizer=optimizer_p,
        )
        task_p = pax_fiddle.Config(
            tasks_lib.SingleTask,
            name="mnist_task",
            model=model_p,
            summary_verbosity=0,
            train=pax_fiddle.Config(
                tasks_lib.SingleTask.Train,
                learner=learner_p,
                num_train_steps=self.TRAIN_STEPS,
                eval_skip_train=True,
            ),
        )
        return task_p

    def datasets(self) -> list[pax_fiddle.Config[base_input.BaseInput]]:
        return [
            pax_fiddle.Config(
                MnistDaliInput, batch_size=self.BATCH_SIZE, is_training=True
            ),
            # Eval dataset is not infinite.
            pax_fiddle.Config(
                MnistDaliInput,
                batch_size=self.BATCH_SIZE,
                is_training=False,
                reset_for_eval=True,
            ),
        ]
