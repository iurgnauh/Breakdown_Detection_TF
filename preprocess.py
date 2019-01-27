import os
import json

from utils import normalizeString


def get_json_file_list(data_dir):
    file_list = []
    for _, dirs, _ in os.walk(data_dir):
        for subdir in dirs:
            for root, _, datafiles in os.walk(os.path.join(data_dir, subdir)):
                for file in datafiles:
                    if file.endswith(".json"):
                        file_list.append(os.path.join(root, file))
    return file_list


def get_string_from_turn(turn, count_order):
    # Get string in the format of 'utterance' or 'utterance \t O_num \t T_num \t X_num'
    if turn['speaker'] == 'U':
        return normalizeString(turn['utterance'])
    else:
        assert turn['speaker'] == 'S', \
            'The type of speaker must be U or S, not {0}'.format(turn['speaker'])
        count_annot = {'O': 0, 'T': 0, 'X': 0}
        for annotation in turn['annotations']:
            count_annot[annotation['breakdown']] = count_annot[annotation['breakdown']] + 1
        count_str = '\t'.join([str(count_annot[x]) for x in count_order])
        normalized_str = normalizeString(turn['utterance'])
        # The utterance of system may be some symbols filtered by normalization, which leads to empty string
        if normalized_str == '':
            normalized_str = '<none>'
        return normalized_str + '\t' + count_str


def merge_json_files(data_dir):
    out_file = os.path.join('data', 'data.txt')
    f_out = open(out_file, 'w', encoding='utf8')
    count_order = ['O', 'T', 'X']
    f_out.write('utterance' + '\t' + '\t'.join(count_order) + '\t# O: Not a breakdown; T: Possible breakdown; '
                                                              'X: Breakdown; Use the blank line to split. '
                                                              'Please ignore the first row\n')
    json_file_list = get_json_file_list(data_dir)
    for json_file in json_file_list:
        content = json.load(open(json_file, 'r', encoding='utf8'))
        turns = content['turns']
        for turn in turns:
            f_out.write(get_string_from_turn(turn, count_order) + '\n')
        f_out.write('\n')
    f_out.close()
    print('Merge {2} json files in {0} and save it to {1}'.format(data_dir, out_file, len(json_file_list)))


def main():
    merge_json_files('data')


if __name__ == '__main__':
    main()
