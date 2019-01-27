import os
import math
import random
import argparse
import numpy as np
import tensorflow as tf
import json
import pickle

from preprocess import merge_json_files, get_string_from_turn
from transform_utils import encode_dataset, flatten, iter_data, find_trainable_variables, get_ema_vars, convert_gradient_to_tensor, shape_list, ResultLogger, assign_to_gpu, average_grads, make_path

from train import read_data, process_utter_label_list, utter_to_idx_list, get_pretrain_embed
from text_utils import TextEncoder
from models import TextRNN
from utils import get_max_index, softmax


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

def model(X, max_length, vocab_size, n_layer, embed_size, is_training):
    with tf.variable_scope('transformer_model'):
        we = tf.get_variable("we", [vocab_size+max_length, embed_size], initializer=tf.random_normal_initializer(stddev=0.02))
        we = dropout(we, embd_pdrop, is_training)

        X = tf.reshape(X, [-1, max_length, 2])
        # mask_label = tf.reshape(mask_label, [-1])

        h = tf.reduce_sum(tf.nn.embedding_lookup(we, X), 2)  # batch * max_length * embed_size
        h_concat = [h]
        for layer in range(n_layer):
            h = block(h, 'h%d'%layer, train=is_training, scale=True)
            h_concat.append(h)

        h_concat = tf.concat(h_concat, -1)

        w_clf = tf.get_variable("w_clf", [embed_size, 1], initializer=tf.random_normal_initializer(stddev=0.02))
        b_clf = tf.get_variable("b_clf", [1], initializer=tf.random_normal_initializer(stddev=0.02))

        # lm_h = tf.reshape(h[:, :-1], [-1, n_embd])
        # lm_logits = tf.matmul(lm_h, we, transpose_b=True)
        # lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits, labels=tf.reshape(X[:, 1:, 0], [-1]))
        # lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1]-1])
        # lm_losses = tf.reduce_sum(lm_losses*M[:, 1:], 1)/tf.reduce_sum(M[:, 1:], 1)

        # clf_h = tf.reshape(h, [-1, n_embd])
        # pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), 1), tf.int32)
        # clf_h = tf.gather(clf_h, tf.range(shape_list(X)[0], dtype=tf.int32)*n_ctx+pool_idx)
        #
        # clf_h = tf.reshape(clf_h, [-1, 2, n_embd])
        # if train and clf_pdrop > 0:
        #     shape = shape_list(clf_h)
        #     shape[1] = 1
        #     clf_h = tf.nn.dropout(clf_h, 1-clf_pdrop, shape)
        # clf_h = tf.reshape(clf_h, [-1, n_embd])
        # clf_logits = clf(clf_h, 1, train=train)
        # clf_logits = tf.reshape(clf_logits, [-1, 2])
        #
        # clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_logits, labels=Y)
        return h_concat


def test():
    # test_dir = 'eval/test_jsons'
    test_file_list = []
    for filename in os.listdir(test_dir):
        if filename.endswith('.json'):
            test_file_list.append(filename)
    print('There are {0} json files in test dir {1}'.format(len(test_file_list), test_dir))

    # Count unk rate in the test data
    unk_idx = vocab_dict.word2idx['<UNK>']
    pad_idx = vocab_dict.word2idx['<PAD>']
    unk_count = 0
    all_count = 0

    for test_file in test_file_list:
        out_filename = test_file.replace('log', 'labels')
        # out_dir = 'eval/transformer-lstm'
        f_out = open(os.path.join(out_dir, out_filename), 'w', encoding='utf8')
        content = json.load(open(os.path.join(test_dir, test_file), 'r', encoding='utf8'))
        turns = content['turns']
        turn_index_list = []
        utter_list = []
        label_list = []
        res_json = dict()
        res_json['dialogue-id'] = content['dialogue-id']
        res_json['turns'] = []
        count_order = ['O', 'T', 'X']
        type2idx = {v: i for i, v in enumerate(count_order)}
        for turn in turns:
            turn_index_list.append(turn['turn-index'])
            temp_string = get_string_from_turn(turn, count_order)
            temp_string = temp_string.split('\t')
            utter_list.append(utter_to_idx_list(temp_string[0], vocab_dict))
            if len(temp_string) < 3:
                label_list.append(-1)
            else:
                category_score = [temp_string[x] for x in [1, 2, 3]]
                label_list.append(get_max_index(category_score))

        seq_len_list, mask_list, input_x, input_y = process_utter_label_list(utter_list, label_list, n_ctx,
                                                                             vocab_dict)

        input_x = np.asarray(input_x, dtype=np.int)
        input_y = np.asarray(input_y, dtype=np.int)
        mask_list = np.asarray(mask_list, dtype=np.float)

        logits = sess.run([textRNN.logits], feed_dict={#X_train: input_x,
                                                       textRNN.input_x: input_x,
                                                       textRNN.input_y: input_y,
                                                       textRNN.mask_label: mask_list,
                                                       textRNN.seq_len_list: seq_len_list,
                                                       textRNN.Embed_placeholder: embed_matrix})

        logits = logits[0]  # logits is [ndarray] because the sess.run([])
        for iMask, mask_value in enumerate(mask_list):
            if mask_value == 0:
                continue
            predict_res = dict()
            predict_res['turn-index'] = turn_index_list[iMask]
            temp_score = logits[iMask]
            temp_score = softmax(temp_score)
            breakdown = count_order[get_max_index(temp_score)]
            predict_res['labels'] = [{'breakdown': breakdown, 'prob-O': temp_score[type2idx['O']],
                                      'prob-T': temp_score[type2idx['T']], 'prob-X': temp_score[type2idx['X']]}]
            res_json['turns'].append(predict_res)
        json.dump(res_json, f_out, indent=2, sort_keys=True)
        f_out.close()

        for utter in utter_list:
            unk_count += len([1 for word_idx in utter if word_idx == unk_idx])
            all_count += len([1 for word_idx in utter if word_idx != pad_idx])

    # sess.close()
    print('The UNK rate is {0}/{1} = {2} in test files'.format(unk_count, all_count, unk_count / all_count))
    print('All result json files have been saved in {0}'.format(out_dir))


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
    parser.add_argument('--seed', type=int, default=9)
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

    parser.add_argument('--word_thre', type=int, default=2, help='Word frequency <= word_thre will be filtered')
    # can use glove_small.txt for test, which is a small glove file with only 1000 lines
    parser.add_argument('--word_embed_file', type=str, default='glove_small.txt', help='Pretrained word embedding')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes: O, T, X')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--decay_steps', type=int, default=1, help='decay steps')
    parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate')
    parser.add_argument('--embed_size', type=int, default=300, help='Embedding Size of word')
    parser.add_argument('--hidden_size', type=int, default=50, help='Hidden size in RNN')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.5, help='Dropout keep probability, 1 is all keep')
    parser.add_argument('--learn_embed_matrix', action='store_true', help='Force to recalculate the embed matrix')

    parser.add_argument('--steps', type=int, default=50, help='Training steps')
    parser.add_argument('--save_interval', type=int, default=5, help='interval to save model to disk')
    parser.add_argument('--print_interval', type=int, default=5, help='interval to save model to disk')
    parser.add_argument('--verbose', action='store_true', help='Print verbose info for each train step')

    args = parser.parse_args()
    # print(args)
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

    text_encoder = TextEncoder(encoder_path, bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)
    print("n_vocab", n_vocab)

    is_training = True
    # max_seq_len = max(max(len(utter) for utter in itData[0]) for itData in data)


    n_ctx = max(max(len(utter) for utter in itData[0]) for itData in data)
    # word2vec_file = os.path.join(data_dir, args.word_embed_file)
    # embed_matrix = get_pretrain_embed(vocab_dict.idx2word, word2vec_file, args.embed_size)
    # textRNN = TextRNN(args.num_classes, args.learning_rate, args.decay_steps, args.decay_rate,
    # sequence_length, vocab_size, args.embed_size, is_training)


    # Get embedding matrix
    model_dir = 'model'
    embed_matrix_file = os.path.join(model_dir, 'embed_matrix.pkl')
    if not args.learn_embed_matrix and os.path.isfile(embed_matrix_file):
        embed_matrix = pickle.load(open(embed_matrix_file, 'rb'))
        print('embed_matrix has been loaded from {0}'.format(embed_matrix_file))
    else:
        word2vec_file = os.path.join(data_dir, args.word_embed_file)
        embed_matrix = get_pretrain_embed(vocab_dict.idx2word, word2vec_file, args.embed_size)
        with open(embed_matrix_file, 'wb') as f:
            pickle.dump(embed_matrix, f)
        print('embed_matrix has been saved to {0}'.format(embed_matrix_file))

    X_train = tf.placeholder(dtype=tf.int32, shape=[None, n_ctx, 2])
    # Y_train = tf.placeholder(dtype=tf.int32, shape=[None])
    # mask = tf.placeholder(dtype=tf.float32, shape=[None])
    # sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])

    hidden_embedding = model(X_train, n_ctx, n_vocab, args.n_layer, args.n_embd, is_training)
    with tf.variable_scope("transform_embedding_size"):
        w_embed = tf.get_variable("w_embed", [n_embd, args.embed_size], initializer=tf.random_normal_initializer(stddev=0.02))
        b_embed = tf.get_variable("b_embed", [args.embed_size], initializer=tf.constant_initializer(0))
    # new_embedding = tf.matmul(hidden_embedding, w_embed) + b_embed
    input_embedding = tf.reshape(tf.matmul(tf.reshape(hidden_embedding, [-1, n_embd]), w_embed) + b_embed, [-1, n_ctx, n_layer + 1, args.embed_size])

    textRNN = TextRNN(args.num_classes, args.learning_rate, args.decay_steps, args.decay_rate,
                      n_ctx, vocab_size, args.embed_size, args.hidden_size, args.dropout_keep_prob, is_training, tran_layer=n_layer+1, attention=False, input_embedding=input_embedding)


    params = find_trainable_variables('transformer_model')
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())

    shapes = json.load(open('model/params_shapes.json'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load('model/params_{}.npy'.format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
    init_params[0] = init_params[0][:n_ctx]
    init_params[0] = np.concatenate(
        [init_params[1], init_params[0]], 0)
    del init_params[1]

    if n_transfer == -1:
        n_transfer = 0
    else:
        n_transfer = 1 + n_transfer * 12
    sess.run([p.assign(ip) for p, ip in zip(params[:n_transfer], init_params[:n_transfer])])

    saver = tf.train.Saver()
    for epoch in range(args.n_iter):
        loss_sum = 0
        acc_sum = 0
        for itData in data:
            utter_list, label_list = itData
            seq_len_list, mask_list, input_x, input_y = process_utter_label_list(utter_list, label_list, n_ctx,
                                                                                 vocab_dict)

            input_x = np.asarray(input_x, dtype=np.int)
            input_y = np.asarray(input_y, dtype=np.int)
            mask_list = np.asarray(mask_list, dtype=np.float)

            loss, pred, _ = sess.run([textRNN.loss_val, textRNN.predictions, textRNN.train_op], feed_dict={#X_train: input_x,
                                                                                                textRNN.input_x: input_x,
                                                                                                textRNN.input_y: input_y,
                                                                                                textRNN.mask_label: mask_list,
                                                                                                textRNN.seq_len_list: seq_len_list,
                                                                                                textRNN.Embed_placeholder: embed_matrix
                                                                                                # textRNN.dropout_keep_prob: args.dropout_keep_prob,
                                                                                                # textRNN.Embed_placeholder: embed_matrix
                                                                                                })
            predict_res = [input_y[i] == pred[i] for i in range(len(mask_list)) if mask_list[i] == 1]
            acc = sum(predict_res) / len(predict_res)
            # print("loss: {}, acc: {}".format(loss, acc))
            loss_sum += loss
            acc_sum += acc
        loss_avg = loss_sum / len(data)
        acc_avg = acc_sum / len(data)
        print("Epoch {}: loss {}, acc {}".format(epoch, loss_avg, acc_avg))
        saver.save(sess, 'noattention-model/model.ckpt', global_step=epoch)

        test_dir = 'eval/test_jsons'
        out_dir = 'eval/noattention/epoch{}'.format(epoch)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        test()

