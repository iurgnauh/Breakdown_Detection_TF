import os
import math
import random
import argparse
import numpy as np
import tensorflow as tf

from preprocess import merge_json_files
from transform_utils import encode_dataset, flatten, iter_data, find_trainable_variables, get_ema_vars, convert_gradient_to_tensor, shape_list, ResultLogger, assign_to_gpu, average_grads, make_path

from train import read_data, get_pretrain_embed


def gelu(x):
    return 0.5*x*(1+tf.tanh(math.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))

def swish(x):
    return x*tf.nn.sigmoid(x)


def _norm(x, g=None, b=None, e=1e-5, axis=[1]):
    u = tf.reduce_mean(x, axis=axis, keep_dims=True)
    s = tf.reduce_mean(tf.square(x-u), axis=axis, keep_dims=True)
    x = (x - u) * tf.rsqrt(s + e)
    if g is not None and b is not None:
        x = x*g + b
    return x

def norm(x, scope, axis=[-1]):
    with tf.variable_scope(scope):
        n_state = shape_list(x)[-1]
        g = tf.get_variable("g", [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable("b", [n_state], initializer=tf.constant_initializer(0))
        g, b = get_ema_vars(g, b)
        return _norm(x, g, b, axis=axis)

def dropout(x, pdrop, train):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, 1-pdrop)
    return x

def mask_attn_weights(w):
    n = shape_list(w)[-1]
    b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
    b = tf.reshape(b, [1, 1, n, n])
    w = w*b + -1e9*(1-b)
    return w

def _attn(q, k, v, train=False, scale=False):
    w = tf.matmul(q, k)

    if scale:
        n_state = shape_list(v)[-1]
        w = w*tf.rsqrt(tf.cast(n_state, tf.float32))

    w = mask_attn_weights(w)
    w = tf.nn.softmax(w)

    w = dropout(w, attn_pdrop, train)

    a = tf.matmul(w, v)
    return a

def split_states(x, n):
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1]+[n, m//n]
    return tf.reshape(x, new_x_shape)

def merge_states(x):
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2]+[np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)

def split_heads(x, n, k=False):
    if k:
        return tf.transpose(split_states(x, n), [0, 2, 3, 1])
    else:
        return tf.transpose(split_states(x, n), [0, 2, 1, 3])

def merge_heads(x):
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))

def conv1d(x, scope, nf, rf, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), pad='VALID', train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [rf, nx, nf], initializer=w_init)
        b = tf.get_variable("b", [nf], initializer=b_init)
        if rf == 1: #faster 1x1 conv
            c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, shape_list(x)[:-1]+[nf])
        else: #was used to train LM
            c = tf.nn.conv1d(x, w, stride=1, padding=pad)+b
        return c

def attn(x, scope, n_state, n_head, train=False, scale=False):
    assert n_state%n_head==0
    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3, 1, train=train)
        q, k, v = tf.split(c, 3, 2)
        q = split_heads(q, n_head)
        k = split_heads(k, n_head, k=True)
        v = split_heads(v, n_head)
        a = _attn(q, k, v, train=train, scale=scale)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, 1, train=train)
        a = dropout(a, resid_pdrop, train)
        return a

def mlp(x, scope, n_state, train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        act = tf.nn.relu
        h = act(conv1d(x, 'c_fc', n_state, 1, train=train))
        h2 = conv1d(h, 'c_proj', nx, 1, train=train)
        h2 = dropout(h2, resid_pdrop, train)
        return h2

def block(x, scope, train=False, scale=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        a = attn(x, 'attn', nx, n_head, train=train, scale=scale)
        n = norm(x+a, 'ln_1')
        m = mlp(n, 'mlp', nx*4, train=train)
        h = norm(n+m, 'ln_2')
        return h

def model(X, Y, mask_label, max_length, length_list, vocab_size, n_layer, num_classes, embed_size, is_training):
    with tf.variable_scope('model'):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.AdamOptimizer()
        we = tf.get_variable("we", [vocab_size+max_length, embed_size], initializer=tf.random_normal_initializer(stddev=0.02))
        we = dropout(we, embd_pdrop, is_training)

        X = tf.reshape(X, [-1, max_length, 2])
        mask_label = tf.reshape(mask_label, [-1])

        h = tf.reduce_sum(tf.nn.embedding_lookup(we, X), 2)  # batch * max_length * embed_size
        for layer in range(n_layer):
            h = block(h, 'h%d'%layer, train=is_training, scale=True)

        clf_h = tf.reshape(h, [-1, embed_size])
        clf_h = tf.gather(clf_h, tf.range(tf.shape(X)[0]) * max_length + (length_list - 1))  # batch * embed_size
        with tf.variable_scope("output"):
            w_predict = tf.get_variable("w_predict", [embed_size, num_classes], initializer=tf.random_normal_initializer(stddev=0.02))
            b_predict = tf.get_variable("b_predict", [num_classes], initializer=tf.constant_initializer(0))
        logits = tf.matmul(clf_h, w_predict) + b_predict
        clf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits)  # batch_size
        clf_loss = tf.reduce_mean(clf_loss * mask_label)
        predictions = tf.argmax(logits, axis=1)  # batch

        train_op = optimizer.minimize(clf_loss, global_step=global_step)
        return logits, predictions, clf_loss, train_op


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--n_gpu', type=int, default=4)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)


    parser.add_argument('--word_thre', type=int, default=None, help='Word frequency <= word_thre will be filtered')
    # can use glove_small.txt for test, which is a small glove file with only 1000 lines # glove.42B.300d.txt
    parser.add_argument('--word_embed_file', type=str, default='glove_small.txt', help='Pretrained word embedding')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes: O, T, X')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--decay_steps', type=int, default=1000, help='decay steps')
    parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate')
    parser.add_argument('--embed_size', type=int, default=300, help='Embedding Size')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.5, help='Dropout keep probability, 1 is all keep')

    args = parser.parse_args()
    print(args)
    globals().update(args.__dict__)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    data_dir = 'data'
    data_file = os.path.join(data_dir, 'data.txt')
    merge_json_files(data_dir)
    data, vocab_dict = read_data(data_file, args)
    vocab_size = len(vocab_dict)
    print('Training data and dictionary has been generated from {0}'.format(data_file))

    is_training = True
    max_seq_len = max(max(len(utter) for utter in itData[0]) for itData in data)

    word2vec_file = os.path.join(data_dir, args.word_embed_file)
    embed_matrix = get_pretrain_embed(vocab_dict.idx2word, word2vec_file, args.embed_size)
    # textRNN = TextRNN(args.num_classes, args.learning_rate, args.decay_steps, args.decay_rate,
    # sequence_length, vocab_size, args.embed_size, is_training)

    X_train = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len, 2])
    Y_train = tf.placeholder(dtype=tf.int32, shape=[None])
    mask = tf.placeholder(dtype=tf.float32, shape=[None])
    sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])

    logits, prediction, clf_loss, train_op = model(X_train, Y_train, mask, max_seq_len, sequence_length, vocab_size, args.n_layer, args.num_classes, args.embed_size, is_training=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for itData in data:
            utter_list, label_list = itData
            seq_len_list = [len(x) for x in utter_list]
            mask_list = [0 if x == -1 else 1 for x in label_list]
            label_list = [0 if x == -1 else x for x in
                          label_list]  # change the label -1 back to 0 because the mask already record those positions
            input_x = []
            for i, utter in enumerate(utter_list):
                if len(utter) < max_seq_len:
                    utter_list[i] = utter + [vocab_dict.word2idx['<PAD>']] * (max_seq_len - len(utter))
                word_index = np.reshape(utter_list[i], [max_seq_len, 1])
                word_pos = np.reshape(range(vocab_size, vocab_size + max_seq_len), [max_seq_len, 1])
                input_x.append(np.concatenate([word_index, word_pos], -1))
            input_x = np.asarray(input_x, dtype=np.int)
            input_y = np.asarray(label_list, dtype=np.int)
            mask_list = np.asarray(mask_list, dtype=np.float)

            logi, pred, loss, _ = sess.run([logits, prediction, clf_loss, train_op], feed_dict={X_train: input_x, Y_train: input_y, mask: mask_list, sequence_length: seq_len_list})
            predict_res = [input_y[i] == pred[i] for i in range(len(mask_list)) if mask_list[i] == 1]
            acc = sum(predict_res) / len(predict_res)
            print("loss: {}, acc: {}".format(loss, acc))
