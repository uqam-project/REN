from six.moves import reduce
import tensorflow as tf


def masked_conv1d_and_max(t, weights, filters, kernel_size):
    
    # Get shape and parameters
    shape = tf.shape(t)
    ndims = t.shape.ndims
    dim1 = reduce(lambda x, y: x*y, [shape[i] for i in range(ndims - 2)])
    dim2 = shape[-2]
    dim3 = t.shape[-1]

    # Reshape weights
    weights = tf.reshape(weights, shape=[dim1, dim2, 1])
    weights = tf.to_float(weights)

    # Reshape input and apply weights
    flat_shape = [dim1, dim2, dim3]
    t = tf.reshape(t, shape=flat_shape)
    t *= weights

    # Apply convolution
    t_conv = tf.layers.conv1d(t, filters, kernel_size, padding='same')
    t_conv *= weights

    # Reduce max -- set to zero if all padded
    t_conv += (1. - weights) * tf.reduce_min(t_conv, axis=-2, keepdims=True)
    t_max = tf.reduce_max(t_conv, axis=-2)

    # Reshape the output
    final_shape = [shape[i] for i in range(ndims-2)] + [filters]
    t_max = tf.reshape(t_max, shape=final_shape)

    return t_max
