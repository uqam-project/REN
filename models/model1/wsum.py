import tensorflow as tf

def weight_layers(name, wsum_ops, l2_coef=None,
                  use_top_only=False, do_layer_norm=False):
  
    def _l2_regularizer(weights):
        if l2_coef is not None:
            return l2_coef * tf.reduce_sum(tf.square(weights))
        else:
            return 0.0

    # Get ops for computing embeddings and mask
    wsum_embeddings = wsum_ops['wsum_embeddings']
    mask = wsum_ops['mask']

    n_layers = int(wsum_embeddings.get_shape()[1])
    dim = int(wsum_embeddings.get_shape()[3])

    with tf.control_dependencies([wsum_embeddings, mask]):
        # Cast the mask and broadcast for layer use.
        mask_float = tf.cast(mask, 'float32')
        broadcast_mask = tf.expand_dims(mask_float, axis=-1)

        def _do_ln(x):
            # do layer normalization excluding the mask
            x_masked = x * broadcast_mask
            N = tf.reduce_sum(mask_float) * dim
            mean = tf.reduce_sum(x_masked) / N
            variance = tf.reduce_sum(((x_masked - mean) * broadcast_mask)**2
                                    ) / N
            return tf.nn.batch_normalization(
                x, mean, variance, None, None, 1E-12
            )

        if use_top_only:
            layers = tf.split(wsum_embeddings, n_layers, axis=1)
            # just the top layer
            sum_pieces = tf.squeeze(layers[-1], squeeze_dims=1)
            # no regularization
            reg = 0.0
        else:
            W = tf.get_variable(
                '{}_wsum_W'.format(name),
                shape=(n_layers, ),
                initializer=tf.zeros_initializer,
                regularizer=_l2_regularizer,
                trainable=True,
            )

            # normalize the weights
            normed_weights = tf.split(
                tf.nn.softmax(W + 1.0 / n_layers), n_layers
            )
            # split embeddings layers
            layers = tf.split(wsum_embeddings, n_layers, axis=1)
    
            # compute the weighted, normalized embeddings activations
            pieces = []
            for w, t in zip(normed_weights, layers):
                if do_layer_norm:
                    pieces.append(w * _do_ln(tf.squeeze(t, squeeze_dims=1)))
                else:
                    pieces.append(w * tf.squeeze(t, squeeze_dims=1))
            sum_pieces = tf.add_n(pieces)
    
            # get the regularizer 
            reg = [
                r for r in tf.get_collection(
                                tf.GraphKeys.REGULARIZATION_LOSSES)
                if r.name.find('{}_wsum_W/'.format(name)) >= 0
            ]
            if len(reg) != 1:
                raise ValueError

        # scale the weighted sum by beta
        beta = tf.get_variable(
            '{}_wsum_beta'.format(name),
            shape=(1, ),
            initializer=tf.ones_initializer,
            regularizer=None,
            trainable=True,
        )
        weighted_wsum_layers = sum_pieces * beta

        ret = {'weighted_op': weighted_wsum_layers, 'regularization_op': reg}

    return ret
