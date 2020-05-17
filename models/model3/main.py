"""GloVe Embeddings + chars bi-LSTM + bi-LSTM + CRF"""

__author__ = "Guillaume Genthial"

import functools
import json
import logging
from pathlib import Path
import sys

import tensorflow_hub as hub

import numpy as np
import tensorflow as tf
from tf_metrics import precision, recall, f1
from masked_conv import masked_conv1d_and_max

DATADIR = '../../data/example'

# Logging
Path('results').mkdir(exist_ok=True)
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler('results/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


def parse_fn(line_words, line_tags):
    # Encode in Bytes for TF
    words = [w.encode() for w in line_words.strip().split()]
    tags = [t.encode() for t in line_tags.strip().split()]
    assert len(words) == len(tags), "Words and tags lengths don't match"

    # Chars
    chars = [[c.encode() for c in w] for w in line_words.strip().split()]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]
    return ((words, len(words)), (chars, lengths)), tags #(([b'Nadim', b'Ladki'], 2), ([[b'N', b'a', b'd', b'i', b'm'], [b'L', b'a', b'd', b'k', b'i']], [5, 5])) [b'B-PER', b'I-PER']


def generator_fn(words, tags):
    with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags)


def input_fn(words, tags, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = ((([None], ()),               # (words, nwords)(([b'Nadim', b'Ladki'], 2)
               ([None, None], [None])),    # (chars, nchars) ([[b'N', b'a', b'd', b'i', b'm'], [b'L', b'a', b'd', b'k', b'i']], [5, 5]))
              [None])                      # tags [b'B-PER', b'I-PER']
    types = (((tf.string, tf.int32),
              (tf.string, tf.int32)),
             tf.string)
    defaults = ((('<pad>', 0),
                 ('<pad>', 0)),
                'O')
    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, words, tags),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])#here i can add seed

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)#get batch size value, if not found use 20 as default
               .prefetch(1))
    #(((array([[b'Peter', b'Blackburn'],[b'Yac', b'Amirat']], dtype=object), array([2, 2], dtype=int32)), 
    #(array([[[b'P', b'e', b't', b'e', b'r', b'<pad>', b'<pad>', b'<pad>',b'<pad>'], [b'B', b'l', b'a', b'c', b'k', b'b', b'u', b'r', b'n']],
	#   [[b'Y', b'a', b'c', b'<pad>', b'<pad>', b'<pad>', b'<pad>',b'<pad>', b'<pad>'],[b'A', b'm', b'i', b'r', b'a', b't', b'<pad>', b'<pad>',b'<pad>']]], dtype=object), array([[5, 9],[3, 6]], dtype=int32))), 	
	#array([[b'B-PER', b'I-PER'],[b'B-PER', b'I-PER']], dtype=object))
    return dataset


def model_fn(features, labels, mode, params):
    # For serving features are a bit different
    if isinstance(features, dict):
        features = ((features['words'], features['nwords']),
                    (features['chars'], features['nchars']))

    # Read vocabs and inputs
    (words, nwords), (chars, nchars) = features
    
    dropout = params['dropout']
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    vocab_words = tf.contrib.lookup.index_table_from_file(
        params['words'], num_oov_buckets=params['num_oov_buckets'])
    vocab_chars = tf.contrib.lookup.index_table_from_file(
        params['chars'], num_oov_buckets=params['num_oov_buckets'])
    with Path(params['tags']).open() as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1
    with Path(params['chars']).open() as f:
        num_chars = sum(1 for _ in f) + params['num_oov_buckets']


	
    # Char Embeddings
    char_ids = vocab_chars.lookup(chars) #[[a,b][c,z]] => [[0,1][2,25]]
    variable = tf.get_variable(
        'chars_embeddings', [num_chars, params['dim_chars']], tf.float32)#dimension char embeddings [86,100]
    char_embeddings = tf.nn.embedding_lookup(variable, char_ids)#char_ids [0,1] 0 va prendre le premier vecteur (variable [0,:]), donc [[0,1][2,25]] => [[variable[0,:],variable[1,:]][variable[2,:],variable[25,:]]]
    char_embeddings = tf.layers.dropout(char_embeddings, rate=dropout,
                                        training=training)#50% de l'entrée

    # Char 1d convolution
    weights = tf.sequence_mask(nchars)
    char_embeddings = masked_conv1d_and_max(
        char_embeddings, weights, params['filters'], params['kernel_size'])    
    output = tf.layers.dropout(char_embeddings, rate=dropout, training=training)
    
    # Char LSTM
    dim_words = tf.shape(char_embeddings)[1]#max dim word (time len)(or number of chars max of a word)[nombre de phrase(batch),nombre de mots max,time len, dim char 100]
    dim_chars = tf.shape(char_embeddings)[2]#dimension de char 100 [nombre de phrase(batch),nombre de mots max,time len ,dim char 100]
    
    flat = tf.reshape(output, [-1, dim_chars, params['filters']])#[?,max len word(or time len),100]
    t = tf.transpose(flat, perm=[1, 0, 2])# time major
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['char_lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['char_lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    _, (_, output_fw) = lstm_cell_fw(t, dtype=tf.float32,
                                     sequence_length=tf.reshape(nchars, [-1]))#we take last state 
    _, (_, output_bw) = lstm_cell_bw(t, dtype=tf.float32,
                                     sequence_length=tf.reshape(nchars, [-1]))#we take last state

    output = tf.concat([output_fw, output_bw], axis=-1)#concat on the last D dimension of tensors 25+25
    
    char_embeddings = tf.reshape(output, [-1,dim_words , params['char_lstm_size']*2])
    
    # Word Embeddings
    word_ids = vocab_words.lookup(words)#[[b'Peter', b'Blackburn'],[b'Yac', b'Amirat']] => [[b'0', b'1'],[b'2', b'3']]
    glove = np.load(params['glove'])['embeddings']  # np.array glove made of vocab words (reduces list)
    variable = np.vstack([glove, [[0.] * params['dim']]])#concatenate on -1 axis, glove + [[0.]]
    variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
    word_embeddings = tf.nn.embedding_lookup(variable, word_ids)#[[b'0', b'1'],[b'2', b'3']] => [[b'variable[0]', b'variable[1]'],[b'variable[2]', b'variable[3]']] [2,2,300]

    # Concatenate Word and Char Embeddings
    embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)#concat on the last dimension axis 100+300
    embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)#50% de l'entrée
	
    # LSTM for word
    t = tf.transpose(embeddings, perm=[1, 0, 2])  # Need time-major #put the word dim as first dimension. check batch-major VS time-major
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.transpose(output, perm=[1, 0, 2])
    output = tf.layers.dropout(output, rate=dropout, training=training)
	
	
    # CRF
    logits = tf.layers.dense(output, num_tags)#nn dense input : (output of bilstm), output dimension : same shape excpet last dim will be num of tags
    crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)#variable of crf pars matrix num_tags*num_tags
    pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)#decode_tags: A [batch_size, max_seq_len] matrix, with dtype tf.int32. Contains the highest scoring tag indices.
    																	#potentials(logits): A [batch_size, max_seq_len, num_tags] tensor of unary potentials.

    if mode == tf.estimator.ModeKeys.PREDICT:#prediction
        # Predictions
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
            params['tags'])
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))#indices = tf.constant([1, 5], tf.int64) => ["lake", "UNKNOWN"]
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])#get tags index from file
        tags = vocab_tags.lookup(labels)#replace lables by thier indexes
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, tags, nwords, crf_params)#calculate log_likelihood given the real tags, return: A [batch_size] Tensor containing the log-likelihood of each example, given the sequence of tag indices.
        loss = tf.reduce_mean(-log_likelihood)#Computes the mean of elements across dimensions of a tensor. x = tf.constant([[1., 1.], [2., 2.]]) tf.reduce_mean(x)  # 1.5

        # Metrics
        weights = tf.sequence_mask(nwords)#convert the vector of size n to a matrix of bool of size n * max value in the vector v[1,2] ==> m[[true,false],[true, true]]
        metrics = {
            'acc': tf.metrics.accuracy(tags, pred_ids, weights),
            'precision': precision(tags, pred_ids, num_tags, indices, weights),#ground truth, predictions, num of tags 9, The indices of the positive classes, 
            'recall': recall(tags, pred_ids, num_tags, indices, weights),
            'f1': f1(tags, pred_ids, num_tags, indices, weights),
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])#for tensor board#tuple of (scalar float Tensor, update_op) op[1] => update_op: An operation that increments the total and count variables appropriately and whose value matches accuracy.

        if mode == tf.estimator.ModeKeys.EVAL:#Eval
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:#training
            train_op = tf.train.AdamOptimizer().minimize(
                loss, global_step=tf.train.get_or_create_global_step())#adam optimizer operation to optimize the loss, global_step: Optional Variable to increment by one after the variables have been updated.
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)


if __name__ == '__main__':
    # Params
    params = {
        'dim': 300,#dimension word embedding glove
        'dim_chars': 100,#dimension char enbedding (char lstm input)
        'dropout': 0.5,
        'num_oov_buckets': 1,#to give index for out of vocabulary
        'epochs': 25,
        'batch_size': 32,
        'filters': 50,
        'kernel_size': 3,        
        'buffer': 15000,#buffer_size: A tf.int64 scalar tf.Tensor, representing the number of elements from this dataset from which the new dataset will sample.
        'char_lstm_size': 150,#char lstm unit number (hidden state size)
        'lstm_size': 250,#word lstm unit number (hidden state size)
        'words': str(Path(DATADIR, 'vocab.words.txt')),
        'chars': str(Path(DATADIR, 'vocab.chars.txt')),
        'tags': str(Path(DATADIR, 'vocab.tags.txt')),
        'glove': str(Path(DATADIR, 'glove.npz'))
    }
    with Path('results/params.json').open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    def fwords(name):
        return str(Path(DATADIR, '{}.words.txt'.format(name)))

    def ftags(name):
        return str(Path(DATADIR, '{}.tags.txt'.format(name)))

    # Estimator, train and evaluate
    train_inpf = functools.partial(input_fn, fwords('train'), ftags('train'),
                                   params, shuffle_and_repeat=True)
    eval_inpf = functools.partial(input_fn, fwords('testa'), ftags('testa'))

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    estimator = tf.estimator.Estimator(model_fn, 'results/model', cfg, params)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    hook = tf.estimator.experimental.stop_if_higher_hook(
        estimator, 'f1', 500, min_steps=8000, run_every_secs=120)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Write predictions to file
    def write_predictions(name):
        Path('results/score').mkdir(parents=True, exist_ok=True)
        with Path('results/score/{}.preds.txt'.format(name)).open('wb') as f:
            test_inpf = functools.partial(input_fn, fwords(name), ftags(name))
            golds_gen = generator_fn(fwords(name), ftags(name))#ground truth ((words, numwords), (chars, numchars)), tags
            preds_gen = estimator.predict(test_inpf)#prediction : preds_ids,tags
            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), (_, _)), tags = golds
                for word, tag, tag_pred in zip(words, tags, preds['tags']):
                    f.write(b' '.join([word, tag, tag_pred]) + b'\n')#WORD, REAL TAG, PREDICTED TAG
                f.write(b'\n')

    for name in ['train', 'testa', 'testb']:
        write_predictions(name)#PREDICT THE 3 DATASET
