__author__ = 'dracz'

import math
import cPickle, gzip
import theano
import numpy as np

from dae import DenoisingAutoencoder

corruption_rate = 0.3
learning_rate = 0.1

n_epochs = 15
batch_size = 20

tile_shape = (28, 28)

n_visible = (tile_shape[0] * tile_shape[1])
n_hidden = 500

dw = int(math.floor(math.sqrt(n_hidden)))
display_shape = (dw, dw)


def load_data(file="../../data/mnist.pkl.gz"):
    f = gzip.open(file, 'rb')
    data = cPickle.load(f)
    f.close()
    return data


if __name__ == '__main__':

    datasets = load_data()
    train_set_x = np.asarray(datasets[0][0], dtype=theano.config.floatX)

    da = DenoisingAutoencoder(n_visible, n_hidden)

    da.train(theano.shared(train_set_x, borrow=True),
             corruption_rate=corruption_rate,
             learning_rate=learning_rate,
             n_epochs=n_epochs,
             batch_size=batch_size)

    da.render_filters(tile_shape, display_shape).show()