import tensorflow as tf

from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers

from nvutils import image_processing

layers = tf.keras.layers

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def _gen_l2_regularizer(use_l2_regularizer=True):
  return regularizers.l2(L2_WEIGHT_DECAY) if use_l2_regularizer else None


def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   use_l2_regularizer=True):
  """The identity block is the block that has no conv layer at shortcut.
  Args:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    use_l2_regularizer: whether to use L2 regularizer on Conv layer.
  Returns:
    Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  if backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = layers.Conv2D(
      filters1, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2a')(
          input_tensor)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2a')(
          x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(
      filters2,
      kernel_size,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2b')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2b')(
          x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(
      filters3, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2c')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2c')(
          x)

  x = layers.add([x, input_tensor])
  x = layers.Activation('relu')(x)
  return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               use_l2_regularizer=True):
  """A block that has a conv layer at shortcut.
  Note that from stage 3,
  the second conv layer at main path is with strides=(2, 2)
  And the shortcut should have strides=(2, 2) as well
  Args:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    strides: Strides for the second conv layer in the block.
    use_l2_regularizer: whether to use L2 regularizer on Conv layer.
  Returns:
    Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  if backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = layers.Conv2D(
      filters1, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2a')(
          input_tensor)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2a')(
          x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(
      filters2,
      kernel_size,
      strides=strides,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2b')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2b')(
          x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(
      filters3, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2c')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2c')(
          x)

  shortcut = layers.Conv2D(
      filters3, (1, 1),
      strides=strides,
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '1')(
          input_tensor)
  shortcut = layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '1')(
          shortcut)

  x = layers.add([x, shortcut])
  x = layers.Activation('relu')(x)
  return x


def resnet50(num_classes,
             batch_size=None,
             use_l2_regularizer=True,
             rescale_inputs=False):
  """Instantiates the ResNet50 architecture.
  Args:
    num_classes: `int` number of classes for image classification.
    batch_size: Size of the batches for each step.
    use_l2_regularizer: whether to use L2 regularizer on Conv/Dense layer.
    rescale_inputs: whether to rescale inputs from 0 to 1.
  Returns:
      A Keras model instance.
  """
  input_shape = (224, 224, 3)
  img_input = layers.Input(shape=input_shape, batch_size=batch_size)
  if rescale_inputs:
    # Hub image modules expect inputs in the range [0, 1]. This rescales these
    # inputs to the range expected by the trained model.
    x = layers.Lambda(
        lambda x: x * 255.0 - backend.constant(
            image_processing.CHANNEL_MEANS,
            shape=[1, 1, 3],
            dtype=x.dtype),
        name='rescale')(
            img_input)
  else:
    x = img_input

  if backend.image_data_format() == 'channels_first':
    x = layers.Lambda(
        lambda x: backend.permute_dimensions(x, (0, 3, 1, 2)),
        name='transpose')(x)
    bn_axis = 1
  else:  # channels_last
    bn_axis = 3

  x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
  x = layers.Conv2D(
      64, (7, 7),
      strides=(2, 2),
      padding='valid',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='conv1')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name='bn_conv1')(
          x)
  x = layers.Activation('relu')(x)
  x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

  x = conv_block(
      x,
      3, [64, 64, 256],
      stage=2,
      block='a',
      strides=(1, 1),
      use_l2_regularizer=use_l2_regularizer)
  x = identity_block(
      x,
      3, [64, 64, 256],
      stage=2,
      block='b',
      use_l2_regularizer=use_l2_regularizer)
  x = identity_block(
      x,
      3, [64, 64, 256],
      stage=2,
      block='c',
      use_l2_regularizer=use_l2_regularizer)

  x = conv_block(
      x,
      3, [128, 128, 512],
      stage=3,
      block='a',
      use_l2_regularizer=use_l2_regularizer)
  x = identity_block(
      x,
      3, [128, 128, 512],
      stage=3,
      block='b',
      use_l2_regularizer=use_l2_regularizer)
  x = identity_block(
      x,
      3, [128, 128, 512],
      stage=3,
      block='c',
      use_l2_regularizer=use_l2_regularizer)
  x = identity_block(
      x,
      3, [128, 128, 512],
      stage=3,
      block='d',
      use_l2_regularizer=use_l2_regularizer)

  x = conv_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='a',
      use_l2_regularizer=use_l2_regularizer)
  x = identity_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='b',
      use_l2_regularizer=use_l2_regularizer)
  x = identity_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='c',
      use_l2_regularizer=use_l2_regularizer)
  x = identity_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='d',
      use_l2_regularizer=use_l2_regularizer)
  x = identity_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='e',
      use_l2_regularizer=use_l2_regularizer)
  x = identity_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='f',
      use_l2_regularizer=use_l2_regularizer)

  x = conv_block(
      x,
      3, [512, 512, 2048],
      stage=5,
      block='a',
      use_l2_regularizer=use_l2_regularizer)
  x = identity_block(
      x,
      3, [512, 512, 2048],
      stage=5,
      block='b',
      use_l2_regularizer=use_l2_regularizer)
  x = identity_block(
      x,
      3, [512, 512, 2048],
      stage=5,
      block='c',
      use_l2_regularizer=use_l2_regularizer)

  rm_axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]
  x = layers.Lambda(lambda x: backend.mean(x, rm_axes), name='reduce_mean')(x)
  x = layers.Dense(
      num_classes,
      kernel_initializer=initializers.RandomNormal(stddev=0.01),
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='fc1000')(
          x)

  # A softmax that is followed by the model loss must be done cannot be done
  # in float16 due to numeric issues. So we pass dtype=float32.
  x = layers.Activation('softmax', dtype='float32')(x)

  # Create model.
  return models.Model(img_input, x, name='resnet50')

def trivial(num_classes,
            batch_size=None,
            use_l2_regularizer=True):
  input_shape = (224, 224, 3)
  img_input = layers.Input(shape=input_shape, batch_size=batch_size)
  x = img_input

  if backend.image_data_format() == 'channels_first':
    x = layers.Lambda(
        lambda x: backend.permute_dimensions(x, (0, 3, 1, 2)),
        name='transpose')(x)
    bn_axis = 1
  else:  # channels_last
    bn_axis = 3

  x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
  x = layers.Conv2D(
      64, (7, 7),
      strides=(2, 2),
      padding='valid',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='conv1')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name='bn_conv1')(
          x)

  rm_axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]
  x = layers.Lambda(lambda x: backend.mean(x, rm_axes), name='reduce_mean')(x)
  x = layers.Dense(
      num_classes,
      kernel_initializer=initializers.RandomNormal(stddev=0.01),
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='fc1000')(
          x)

  # A softmax that is followed by the model loss must be done cannot be done
  # in float16 due to numeric issues. So we pass dtype=float32.
  x = layers.Activation('softmax', dtype='float32')(x)

  # Create model.
  return models.Model(img_input, x, name='resnet50')

