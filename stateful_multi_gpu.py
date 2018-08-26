from keras.layers.merge import concatenate
from keras import backend as K
from keras.layers.core import Lambda
from keras.engine.training import Model

from keras.engine.topology import Layer

import tensorflow as tf


class SliceBatch(Layer):

    def __init__(self, batch_size, num_slices, slice_index, **kwargs):
        self.batch_size = batch_size
        self.num_slices = num_slices
        self.slice_index = slice_index

        if batch_size % num_slices != 0:
            raise ValueError('The batch size must be dividable by the number of slices, '
                             'batch_size=%d, num_slices=%s' % (batch_size, num_slices))

        self.sub_batch_size = tf.constant([self.batch_size // self.num_slices])

        super(SliceBatch, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SliceBatch, self).build(input_shape)

    def call(self, data, **kwargs):
        shape = tf.shape(data)
        input_shape = shape[1:]
        size = tf.concat([self.sub_batch_size, input_shape], axis=0)
        stride = tf.concat([self.sub_batch_size, input_shape * 0], axis=0)
        start = stride * self.slice_index
        return tf.slice(data, start, size)

    def compute_output_shape(self, input_shape):
        return tuple([self.batch_size // self.num_slices]) + input_shape[1:]


def _get_available_devices():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


def _normalize_device_name(name):
    name = name.lower().replace('device:', '')
    return name


def stateful_multi_gpu(inputs_generator, model_generator, batch_size, gpus):
    """
    TODO
    Based on multi_gpu_model at keras.utils.training_utils
    """
    if K.backend() != 'tensorflow':
        raise ValueError('`multi_gpu_model` is only available '
                         'with the TensorFlow backend.')
    if gpus <= 1:
        raise ValueError('For multi-gpu usage to be effective, '
                         'call `multi_gpu_model` with `gpus >= 2`. '
                         'Received: `gpus=%d`' % gpus)

    if batch_size % gpus != 0:
        raise ValueError('The batch size must be dividable by the number of gpus, '
                         'batch_size=%d, gpus=%s' % (batch_size, gpus))

    import tensorflow as tf

    target_devices = ['/cpu:0'] + ['/gpu:%d' % i for i in range(gpus)]
    available_devices = _get_available_devices()
    available_devices = [_normalize_device_name(name) for name in available_devices]
    for device in target_devices:
        if device not in available_devices:
            raise ValueError(
                'To call `multi_gpu_model` with `gpus=%d`, '
                'we expect the following devices to be available: %s. '
                'However this machine only has: %s. '
                'Try reducing `gpus`.' % (gpus,
                                          target_devices,
                                          available_devices))

    sub_batch_size = batch_size // gpus

    full_batch_inputs = inputs_generator(batch_size)
    if not isinstance(full_batch_inputs, list):
        full_batch_inputs = [full_batch_inputs]

    model = model_generator(sub_batch_size)

    all_outputs = []
    for i in range(len(model.outputs)):
        all_outputs.append([])

    # Place a copy of the model on each GPU,
    # each getting a slice of the inputs.
    for i in range(gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('replica_%d' % i):

                inputs = []
                # Retrieve a slice of the input.
                for inp in full_batch_inputs:
                    slice_i = SliceBatch(batch_size, gpus, i)(inp)
                    inputs.append(slice_i)

                # Apply model on slice
                # (creating a model replica on the target device).
                outputs = model(inputs)
                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save the outputs for merging back together later.
                for o in range(len(outputs)):
                    all_outputs[o].append(outputs[o])

    # Merge outputs on CPU.
    with tf.device('/cpu:0'):
        merged = []
        for outputs in all_outputs:
            merged.append(concatenate(outputs,
                                      axis=0))
        return Model(full_batch_inputs, merged)