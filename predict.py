import os
import sys
import glob
import codecs
import random
import anago

from os.path import join, dirname, abspath
from string import punctuation
from seqeval.metrics import classification_report

punc = list(punctuation)


def splitting_dataset(files, cap, min_length, max_length):
    """
    Splits data into list of tuples of tokens and labels
    Splits on '.:' and also the sentence length should be in between
    max_length and min_length

    Parameters
    ----------
    files : list
        File paths
    cap : float
        how many files to split
    min_length : int
        minimum length of the sentence
    max_length : int
        maximum length of the sentence

    Returns
    -------
    data : list
        List of tuples, where each tuple contains tokens and labels.

    Examples
    ---------
    file_ = ['/home/tmp/sample.txt']  # file contains - "John has cancer."
    data = splitting_dataset(file_)

    print(data)
    >>>[(['John', 'has', 'cancer', '.'], ['O', 'O', 'B-Diseases', 'O'])]

    """
    cap = int(cap * len(files))
    xraw, yraw = [], []  # will contain the words and labels

    for f in files[:cap]:
        sentences = []
        labels = []
        for line in codecs.open(f, 'r', "utf-8"):
            try:
                word = line.split(" ")[0].strip()
                tag = line.split(" ")[1].strip('\n')
                sentences.append(word)
                labels.append(tag)

                # Break at max length, or add logic to break at punctuation
                if len(sentences) == max_length or any([punc in word for punc in ".:"]) and len(sentences) > min_length:
                    # Append to training data
                    xraw.append(sentences)
                    yraw.append(labels)
                    sentences = []
                    labels = []

            except StandardError:  # error for irrelevant blank spaces
                print('Something')

        # Last sentence
        if len(sentences) != 0:
            xraw.append(sentences)
            yraw.append(labels)

    assert len(xraw) == len(yraw)
    # convert into tuples of sentences and labels
    # data = [(i, j) for i, j in zip(xraw, yraw)]
    # print('Length of data: ', len(data))
    #
    # random.shuffle(data)

    return xraw, yraw


if __name__ == '__main__':
    sys.path.append((dirname(abspath(__file__))))
    from utils import get_config

    config = get_config()
    MODEL_DIR = config.get(u'train', 'model_dir')
    DATA_DIR = config.get(u'test', 'data_dir')
    capacity = config.getfloat(u'split', 'capacity')
    min_len = config.getint(u'split', 'min')
    max_len = config.getint(u'split', 'max')

    test_data = glob.glob(join(DATA_DIR, '*.conll'))
    x_test, y_test = splitting_dataset(test_data, capacity, min_len, max_len)
    print(len(x_test), len(y_test))
    print(x_test[:2], y_test[:2])

    model = anago.Sequence.load(join(MODEL_DIR, 'weights.h5'), join(MODEL_DIR, 'params.json'),
                                join(MODEL_DIR, 'preprocessor_file'))
    # print(model.score(x_test, y_test))
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))

    with codecs.open('output.txt', 'w', 'utf-8') as f:
        for x, y, z in zip(x_test, y_pred, y_test):
            for tok, lab, true_lab in zip(x, y, z):
                f.write('{}\t{}\t{}\n'.format(tok, lab, true_lab))
            f.write('\n')

    from sklearn.metrics import classification_report
    y_pred = [w.split('-')[1] if w is not 'O' else w for tok in y_pred for w in tok]
    y_test = [w.split('-')[1] if w is not 'O' else w for tok in y_test for w in tok]

    new_y_pred = []
    new_y_test = []
    for i, j in zip(y_test, y_pred):
        if i is 'O' and j is 'O':
            continue
        else:
            new_y_test.append(i)
            new_y_pred.append(j)

    print(classification_report(new_y_test, new_y_pred))
