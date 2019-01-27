import os
import argparse
import random
import pickle
import json
from collections import Counter
import numpy as np
import tensorflow as tf

from preprocess import merge_json_files
from models import TextRNN
from utils import get_max_index


class Dictionary:
    def __init__(self):
        self.idx2word = []
        self.word2idx = dict()
        self.add_word('<PAD>')
        self.add_word('<UNK>')

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)

    def __len__(self):
        return len(self.idx2word)


def build_dict(data_file, threshold=None):
    # Threshold word by frequency
    word_count = Counter()
    with open(data_file, 'r', encoding='utf8') as f:
        f.readline()
        for line in f:
            line = line.strip()
            if line == '':
                continue
            utter = line.split('\t')[0]
            for word in utter.split():
                word_count[word] += 1
    vocab_dict = Dictionary()
    if threshold is not None:
        for word in word_count:
            if word_count[word] > threshold:
                vocab_dict.add_word(word)
    else:
        for word in word_count:
            vocab_dict.add_word(word)
    print('After the threshold of word frequency, the vocab size is {0} -> {1}'.format(len(word_count), len(vocab_dict)))
    return vocab_dict


def utter_to_idx_list(utterance: str, vocab_dict: Dictionary):
    word_list = utterance.split()
    idx_list = [vocab_dict.word2idx[word] if word in vocab_dict.word2idx else vocab_dict.word2idx['<UNK>'] for word in word_list]
    return idx_list


def read_data(data_file, args):
    vocab_dict = build_dict(data_file, args.word_thre)
    data = []
    with open(data_file, 'r', encoding='utf8') as f:
        f.readline()  # Omit the first line
        utter_list = []
        label_list = []
        for line in f:
            line = line.strip()
            if line == '':
                # The batch is already divided by block, which means each file is a batch
                data.append((utter_list, label_list))
                utter_list = []
                label_list = []
            else:
                line = line.split('\t')
                utter_list.append(utter_to_idx_list(line[0], vocab_dict))
                if len(line) < 3:
                    # If label is -1, it means the utterance is made by user, which don't have label
                    label_list.append(-1)
                else:
                    # Find the label with highest score
                    category_score = [line[x] for x in [1, 2, 3]]
                    label_list.append(get_max_index(category_score))
    # Count the number of unk in data
    count_unk = 0
    count_all = 0
    for utter_list, label_list in data:
        for idx_list in utter_list:
            for idx in idx_list:
                count_all += 1
                if idx == vocab_dict.word2idx['<UNK>']:
                    count_unk += 1
    print('The <UNK> rate is {0}/{1}'.format(count_unk, count_all))
    random.shuffle(data)
    return data, vocab_dict


def read_word2vec_file(word2vec_file):
    word2vec = dict()
    with open(word2vec_file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split()
            word = line[0]
            vec = line[1:]
            word2vec[word] = list(map(float, vec))
    return word2vec


def get_pretrain_embed(idx2word, word2vec_file, dim):
    word2vec = read_word2vec_file(word2vec_file)
    word_num = len(idx2word)
    embed_matrix = np.zeros((word_num, dim), dtype=np.float32)
    hit_count = 0
    unk_in_dict = 'unk' in word2vec
    for i, word in enumerate(idx2word):
        if word == '<UNK>' and unk_in_dict:
            embed_matrix[i] = np.asarray(word2vec['unk'])
        if word in word2vec:
            hit_count += 1
            embed_matrix[i] = np.asarray(word2vec[word])
    print('The hit rate of pretrained embedding is {0}/{1} = {2}'.format(hit_count, word_num, hit_count/word_num))
    return embed_matrix


def process_utter_label_list(utter_list, label_list, max_seq_len, vocab_dict):
    seq_len_list = [len(x) for x in utter_list]
    mask_list = [0 if x == -1 else 1 for x in label_list]
    label_list = [0 if x == -1 else x for x in
                  label_list]  # change the label -1 back to 0 because the mask already record those positions
    for i, utter in enumerate(utter_list):
        if len(utter) < max_seq_len:
            utter_list[i] = utter + [vocab_dict.word2idx['<PAD>']] * (max_seq_len - len(utter))
    input_x = np.asarray(utter_list)  # [None, self.sequence_length]
    input_y = np.asarray(label_list)  # np.zeros((batch_size),dtype=np.int32) #[None, self.sequence_length]
    return seq_len_list, mask_list, input_x, input_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=9, help='Random seed for reproducibility')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory')
    parser.add_argument('--model_dir', type=str, default='model', help='model directory')

    parser.add_argument('--word_thre', type=int, default=2, help='Word frequency <= word_thre will be filtered')
    # can use glove_small.txt for test, which is a small glove file with only 1000 lines
    parser.add_argument('--word_embed_file', type=str, default='glove.42B.300d.txt', help='Pretrained word embedding')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes: O, T, X')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--decay_steps', type=int, default=10, help='decay steps')
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    data_dir = args.data_dir
    model_dir = args.model_dir
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    data_file = os.path.join(data_dir, 'data.txt')
    merge_json_files(data_dir)
    data, vocab_dict = read_data(data_file, args)
    vocab_size = len(vocab_dict)
    print('Training data and dictionary has been generated from {0}'.format(data_file))
    vocab_dict_file = os.path.join(model_dir, 'vocab_dict.pkl')
    with open(vocab_dict_file, 'wb') as f:
        pickle.dump(vocab_dict, f)
    print('vocab_dict has been saved to {0}'.format(vocab_dict_file))

    is_training = True
    max_seq_len = 0
    print('There are {0} data list for training'.format(len(data)))
    for itData in data:
        utter_list, label_list = itData
        seq_len_list = [len(x) for x in utter_list]
        temp_max_seq_len = max(seq_len_list)
        max_seq_len = max(max_seq_len, temp_max_seq_len)
    sequence_length = max_seq_len

    # Get embedding matrix
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

    # Build main model textRNN
    model_option = {'num_classes': args.num_classes, 'learning_rate': args.learning_rate,
                    'decay_steps': args.decay_steps, 'decay_rate': args.decay_rate, 'sequence_length': sequence_length,
                    'vocab_size': vocab_size, 'embed_size': args.embed_size, 'hidden_size': args.hidden_size,
                    'dropout_keep_prob': args.dropout_keep_prob, 'is_training': is_training}
    textRNN = TextRNN(**model_option)
    model_option_file = os.path.join(model_dir, 'model_option.json')
    with open(model_option_file, 'w', encoding='utf8') as f:
        json.dump(model_option, f, indent=2)
    print('Model options has been saved to {0}'.format(model_option_file))

    # Train the model, the data contains [(utter_list, label_list), ...]
    # Each utter_list and label_list corresponds to a block with 20 or 21 turns
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(os.path.join(model_dir, 'tfboard'), sess.graph)
        sess.run(tf.global_variables_initializer())
        for step in range(1, args.steps + 1):
            acc_sum = 0
            loss_sum = 0
            for itData in data:
                utter_list, label_list = itData
                seq_len_list, mask_list, input_x, input_y = process_utter_label_list(utter_list, label_list, max_seq_len, vocab_dict)
                _, loss, predict, summary, _ = sess.run(
                    [textRNN.embedding_init, textRNN.loss_val, textRNN.predictions, textRNN.merged, textRNN.train_op],
                    feed_dict={textRNN.input_x: input_x, textRNN.input_y: input_y, textRNN.mask_label: mask_list,
                               textRNN.seq_len_list: seq_len_list, textRNN.Embed_placeholder: embed_matrix})
                train_writer.add_summary(summary, step)
                predict_res = [input_y[i] == predict[i] for i in range(len(mask_list)) if mask_list[i] == 1]
                acc = sum(predict_res) / len(predict_res)
                acc_sum += acc
                loss_sum += loss
                if args.verbose:
                    print("loss:", loss, "acc:", acc, "label:", input_y, "prediction:", predict)
            avg_loss = loss_sum / len(data)
            avg_acc = acc_sum / len(data)
            if step % args.print_interval == 0:
                print('Epoch {0} | Loss: {1} | Acc: {2}'.format(step, avg_loss, avg_acc))
            if step % args.save_interval == 0:
                save_path = saver.save(sess, os.path.join(model_dir, 'model.ckpt'), global_step=step)
                print("Model saved in path: {0}".format(save_path))


if __name__ == '__main__':
    main()
