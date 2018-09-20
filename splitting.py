"""
This script splits data into sentences and save them in required spacy format.
Also, if split is for training purpose, validation sentences and tags are also
stored in separate files.
"""

import sys
import os
import glob
import codecs
import random
import argparse

from os.path import dirname, abspath, join
from string import punctuation

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
    data = [(i, j) for i, j in zip(xraw, yraw)]
    print('Length of data: ', len(data))

    random.shuffle(data)

    return data


def preprocessing_data(data, filepath):
    """
    Writes data into file in a spacy required format.

    Parameters
    ----------
    data : list
        List of tuples of tokens and labels
    filepath : str
        file path of the output file.

    """
    with codecs.open(filepath, 'w', 'utf-8') as f:
        for token_list, label_list in data:
            for token, tag in zip(token_list, label_list):
                f.write('{}\t{}\n'.format(token, tag))
            f.write('\n')


if __name__ == '__main__':

    # custom imports
    sys.path.append(dirname(dirname(abspath(__file__))))
    from utils import get_config

    random.seed(999)
    config = get_config()

    # splitting for training or testing
    input_folder = config.get(u'split', 'input_folder')
    # Set file paths for dictionaries
    data_dir = config.get(u'split', 'output_folder')
    # number of files
    capacity = config.getfloat(u'split', 'capacity')
    min_len = config.getint(u'split', 'min')
    max_len = config.getint(u'split', 'max')

    # Ensure path
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # Total input files
    print('File path: ', input_folder)

    train_files = glob.glob(join(input_folder, 'train', '*.conll'))
    validation_files = glob.glob(join(input_folder, 'validation', '*.conll'))

    print('Splitting started....')

    # For training data
    data = splitting_dataset(train_files, capacity, min_len, max_len)
    print("Processing %d files" % len(train_files))
    print("Training Data: %d" % (len(data)))
    # change conll format and writes it into file
    preprocessing_data(data, join(data_dir, 'train.txt'))

    # For validation data
    data = splitting_dataset(validation_files, capacity, min_len, max_len)
    print("Processing %d files" % len(validation_files))
    print("Validation Data: %d" % len(data))

    # change conll format and writes it into file
    preprocessing_data(data, join(data_dir, 'valid.txt'))
