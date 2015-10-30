
import cPickle, gzip
import theano
import numpy as np

from nn import render_filters, reconstruct
from dae import DenoisingAutoencoder


tile_shape = (28, 28)  # the shape of the input images
mnist_data_path = "../../data/mnist.pkl.gz"


def load_data(fn=mnist_data_path):
    """load mnist data from the specified path
    :return: [[train_set_x, train_set_y], [valid_set_x, valid_set_y], [test_set_x, test_set_y]]
    """
    f = gzip.open(fn, 'rb')
    data = cPickle.load(f)
    f.close()
    return data


def train_dae(x, n_hidden=484, corruption_rate=0.3,
              learning_rate=0.1, n_epochs=5, batch_size=20):
    """
    Train a denoising autoencoder for mnist digits
    :return: A trained DenoisingAutoencoder for mnist digits
    """

    n_visible = (tile_shape[0] * tile_shape[1])

    da = DenoisingAutoencoder(n_visible, n_hidden)

    da.train(theano.shared(x, borrow=True),
             corruption_rate=corruption_rate,
             learning_rate=learning_rate,
             n_epochs=n_epochs,
             batch_size=batch_size)

    return da


def main():
    """
    Train a denoising autoencoder for mnist digits using various corruption rates.
    Display and save an image of the learned filters.
    """

    datasets = load_data()
    train_set_x = np.asarray(datasets[0][0], dtype=theano.config.floatX)
    test_set_x = np.asarray(datasets[2][0], dtype=theano.config.floatX)

    rates = [0.3]
    n_reconstruct = 20

    for rate in rates:
        ae = train_dae(train_set_x, corruption_rate=rate)
        fn = "img/mnist_filters_{}.png".format(rate)
        render_filters(ae.w1.get_value(borrow=True), tile_shape, show=True, image_file=fn)
        reconstruct(ae, test_set_x, tile_shape, n_reconstruct)


if __name__ == '__main__':
    main()
