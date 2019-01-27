import argparse
import os
import random
import json
import pickle
import numpy as np
import tensorflow as tf

from utils import get_max_index, softmax
from preprocess import get_string_from_turn
from train import process_utter_label_list, utter_to_idx_list, Dictionary
from models import TextRNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=9, help='Random seed for reproducibility')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory')
    parser.add_argument('--model_dir', type=str, default='model', help='model directory')
    parser.add_argument('--test_dir', type=str, default=os.path.join('eval', 'test_jsons'),
                        help='Word frequency <= word_thre will be filtered')
    parser.add_argument('--out_dir', type=str, default='textRNN',
                        help='The output file will be saved in eval/out_dir')
    parser.add_argument('--model_file', type=str, default='model.ckpt-10',
                        help='The filename of model, such as model.ckpt-10')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    out_dir = os.path.join('eval', args.out_dir)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # Load vocabulary dictionary
    with open(os.path.join(args.model_dir, 'vocab_dict.pkl'), 'rb') as f:
        vocab_dict = pickle.load(f)
    print('vocab_dict has been loaded from {0}'.format(os.path.join('model', 'vocab_dict.pkl')))

    # Load pre-trained embedding matrix
    embed_matrix_file = os.path.join(args.model_dir, 'embed_matrix.pkl')
    embed_matrix = pickle.load(open(embed_matrix_file, 'rb'))
    print('embed_matrix has been loaded from {0}'.format(embed_matrix_file))

    # Load model
    model_option_file = os.path.join(args.model_dir, 'model_option.json')
    with open(model_option_file, 'r', encoding='utf8') as f:
        model_option = json.load(f)
    print('Model options has been loaded from {0}'.format(model_option_file))
    model_option['is_training'] = False
    max_seq_len = model_option['sequence_length']
    textRNN = TextRNN(**model_option)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, os.path.join(args.model_dir, args.model_file))
    print("Model restored from disk")

    # Iterate over all json files in test dir
    test_file_list = []
    for filename in os.listdir(args.test_dir):
        if filename.endswith('.json'):
            test_file_list.append(filename)
    print('There are {0} json files in test dir {1}'.format(len(test_file_list), args.test_dir))

    # Count unk rate in the test data
    unk_idx = vocab_dict.word2idx['<UNK>']
    pad_idx = vocab_dict.word2idx['<PAD>']
    unk_count = 0
    all_count = 0

    for test_file in test_file_list:
        out_filename = test_file.replace('log', 'labels')
        f_out = open(os.path.join(out_dir, out_filename), 'w', encoding='utf8')
        content = json.load(open(os.path.join(args.test_dir, test_file), 'r', encoding='utf8'))
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
        # max_seq_len = max([len(x) for x in utter_list])
        seq_len_list, mask_list, input_x, input_y = process_utter_label_list(utter_list, label_list, max_seq_len,
                                                                             vocab_dict)
        logits = sess.run([textRNN.embedding_init, textRNN.logits], feed_dict={textRNN.input_x: input_x, textRNN.input_y: input_y,
                                                       textRNN.mask_label: mask_list,
                                                       textRNN.seq_len_list: seq_len_list,
                                                       textRNN.Embed_placeholder: embed_matrix})
        logits = logits[1]  # logits is [ndarray] because the sess.run([])
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

    sess.close()
    print('The UNK rate is {0}/{1} = {2} in test files'.format(unk_count, all_count, unk_count / all_count))
    print('All result json files have been saved in {0}'.format(out_dir))


if __name__ == '__main__':
    main()

