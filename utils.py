import math
import re
import unicodedata
import matplotlib
matplotlib.use('Agg')


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    # Lowercase, trim, and remove non-letter characters
    s = re.sub(r"([0-9]+)", r" <number> ", s)
    s = re.sub(r"[^a-zA-Z<>']+", r" ", s)
    s = unicodeToAscii(s.lower())
    # s = s.replace("it's", 'it is').replace("don't", "do not").replace("can't", "can not").replace("i'm", "i am")
    # s = s.replace("didn't", 'did not').replace("that's", "that is")
    s = " ".join(s.split())
    return s


def get_max_index(input_list):
    max_v = max(input_list)
    for i, v in enumerate(input_list):
        if v == max_v:
            return i


def softmax(scores):
    max_score = max(scores)
    scores = [score - max_score for score in scores]
    sum_exp = sum([math.exp(score) for score in scores])
    res = [math.exp(score) / sum_exp for score in scores]
    return res
