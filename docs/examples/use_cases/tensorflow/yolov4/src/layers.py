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
