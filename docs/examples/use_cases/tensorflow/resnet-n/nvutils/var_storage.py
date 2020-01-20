import tensorflow as tf


def _fp32_trainvar_getter(getter, name, shape=None, dtype=None,
                          trainable=True, regularizer=None,
                          *args, **kwargs):

    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      trainable=trainable,
                      regularizer=regularizer if trainable and 'BatchNorm' not in name else None,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        cast_name = name + '/fp16_cast'
        try:
            cast_variable = tf.get_default_graph().get_tensor_by_name(
                cast_name + ':0')
        except KeyError:
            cast_variable = tf.cast(variable, dtype, name=cast_name)
        cast_variable._ref = variable._ref
        variable = cast_variable
    return variable


def fp32_trainable_vars(name='fp32_vars', *args, **kwargs):
    """A varible scope with custom variable getter to convert fp16 trainable
    variables with fp32 storage followed by fp16 cast.
    """
    return tf.variable_scope(
        name, custom_getter=_fp32_trainvar_getter, *args, **kwargs)
