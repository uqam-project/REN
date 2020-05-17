from six.moves import reduce
import tensorflow as tf


def masked_conv1d_and_max(t, weights, filters, kernel_size):

    # Get shape and parameters
    shape = tf.shape(t)
    ndims = t.shape.ndims
    dim1 = reduce(lambda x, y: x*y, [shape[i] for i in range(ndims - 2)])#multiply the first 2 dimensions #reduce apply an operation between all ellements of a list
    dim2 = shape[-2]
    dim3 = t.shape[-1]

    # Reshape weights
    weights = tf.reshape(weights, shape=[dim1, dim2, 1])
    weights = tf.to_float(weights)

    # Reshape input and apply weights
    flat_shape = [dim1, dim2, dim3]
    t = tf.reshape(t, shape=flat_shape)
    t *= weights  #apply mask to put 0. on padding

    # Apply convolution
    t_conv = tf.layers.conv1d(t, filters, kernel_size, padding='same')
    t_conv *= weights



    # Reshape the output
    final_shape = [shape[i] for i in range(ndims-1)] + [filters]
    t_max = tf.reshape(t_conv, shape=final_shape)	#batch*timeword*filters

    return t_max
