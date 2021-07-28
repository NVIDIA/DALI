# Copyright 2021 Jagoda Kamińska. All Rights Reserved.
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

import tensorflow as tf
import tensorflow_addons as tfa


class Mish(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return tfa.activations.mish(inputs)


class ScaledRandomUniform(tf.keras.initializers.RandomUniform):
    def __init__(self, scale=1, **kwags):
        super().__init__(**kwags)
        self.scale = scale

    def __call__(self, *args, **kwargs):
        return tf.math.scalar_mul(self.scale, super().__call__(*args, **kwargs))

    def get_config(self):  # To support serialization
        return {"scale": self.scale} | super().get_config()
