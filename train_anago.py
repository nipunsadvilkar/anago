import anago
import codecs
import numpy as np
import os
import sys

from os.path import dirname, abspath, join
from anago.utils import load_data_and_labels


def load_vectors(file):
    """Loads vectors in numpy array.
    Args:
        file (str): a path to a vector file.
    Return:
        dict: a dict of numpy arrays.
    """
    model = {}
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            line = line.split(' ')
            word = line[0]
            vector = np.array([float(val) for val in line[1:]])
            model[word] = vector

    return model


if __name__ == '__main__':
    sys.path.append((dirname(abspath(__file__))))
    from utils import get_config

    config = get_config()
    MODEL_DIR = config.get(u'train', 'model_dir')
    train_path = config.get(u'train', 'train_data')
    valid_path = config.get(u'train', 'validation_data')
    EMBEDDING_PATH = config.get(u'train', 'vectors')
    epoch = config.getint(u'train', 'iterations')

    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
        print('Model Directory Created')

    print('Loading data...')
    x_train, y_train = load_data_and_labels(train_path)
    x_valid, y_valid = load_data_and_labels(valid_path)
    print(len(x_train), 'train sequences')
    print(len(x_valid), 'valid sequences')

    print('Loading embeddings...')
    embeddings = load_vectors(EMBEDDING_PATH)

    # Use pre-trained word embeddings
    model = anago.Sequence(word_embedding_dim=200, char_embedding_dim=50, dropout=0.2, embeddings=embeddings)
    model.fit(x_train, y_train, x_valid, y_valid, epochs=epoch)

    print('Saving the model...')
    model.save(join(MODEL_DIR, 'weights.h5'), join(MODEL_DIR, 'params.json'), join(MODEL_DIR, 'preprocessor_file'))
