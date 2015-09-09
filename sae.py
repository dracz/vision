__author__ = 'dracz'

import numpy
import theano
import timeit

import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from PIL import Image
from utils import tile_raster_images

default_n_visible = 16 * 16
default_n_hidden = 10000

default_corruption_rate = 0.3
default_learning_rate = 0.01

default_n_epochs = 1000
default_batch_size = 500


class SparseAutoencoder(object):
    """ A sparse autoencoder """

    def __init__(self, n_visible=default_n_visible, n_hidden=default_n_hidden, wv=None, wh=None, bh=None, bv=None):

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        numpy_rng = numpy.random.RandomState(123)
        theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not wh:
            wh = init_weights(n_visible, n_hidden, numpy_rng)

        if not bv:
            zv = numpy.zeros(n_visible, dtype=theano.config.floatX)
            bv = theano.shared(value=zv, borrow=True)

        if not bh:
            zh = numpy.zeros(n_hidden, dtype=theano.config.floatX)
            bh = theano.shared(value=zh, name='b', borrow=True)

        self.w = w  # weights
        self.b = bh  # bias of hidden units
        self.b_prime = bv  # bias of visible units
        self.w_prime = self.w.T  # tied weights, W_prime = W transpose
        self.theano_rng = theano_rng

        self.x = T.dmatrix(name='input')
        self.params = [self.w, self.b, self.b_prime]

    def get_corrupted_input(self, data, corruption_level):
        """corrupt the image """
        return self.theano_rng.binomial(size=data.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * data

    def get_hidden_values(self, data):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(data, self.w) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input from values of the hidden layer"""
        return T.nnet.sigmoid(T.dot(hidden, self.w_prime) + self.b_prime)

    def get_cost_updates(self, corruption_rate, learning_rate):
        """ This function computes the cost and the updates for one training step"""
        tilde_x = self.get_corrupted_input(self.x, corruption_rate)

        y = T.nnet.sigmoid(T.dot(tilde_x, self.w) + self.b)
        z = T.nnet.sigmoid(T.dot(y, self.w_prime) + self.b_prime)

        l = square_error(self.x, z)
        # l = cross_entropy(self.x, z)

        cost = T.mean(l)
        grads = T.grad(cost, self.params)

        updates = [
            (param, param - learning_rate * grad) for param, grad in zip(self.params, grads)
            ]
        return cost, updates

    def train(self, train_set,
              batch_size=default_batch_size,
              corruption_rate=default_corruption_rate,
              learning_rate=default_learning_rate,
              n_epochs=default_n_epochs):

        n_training = train_set.get_value(borrow=True).shape[0]
        n_train_batches = n_training / batch_size

        index = T.lscalar()  # index into mini-batch

        cost, updates = self.get_cost_updates(
            corruption_rate=corruption_rate,
            learning_rate=learning_rate
        )

        train_da = theano.function(
            [index],
            cost,
            updates=updates,
            givens={self.x: get_batch(train_set, index, batch_size)}
        )

        print("Starting to train using {} examples and {} batches of {}..."
              .format(n_training, n_train_batches, batch_size))

        t1 = timeit.default_timer()

        for epoch in xrange(n_epochs):
            c = []

            for batch_index in xrange(n_train_batches):
                c.append(train_da(batch_index))

            print('Epoch: {}, Cost: {}'.format(epoch, numpy.mean(c)))

        t2 = timeit.default_timer()

        training_time = (t2 - t1)
        print('Training took {:.2f}'.format(training_time))

    def render_filters(self, img_shape, tile_shape):
        return Image.fromarray(tile_raster_images(
            X=self.w.get_value(borrow=True).T,
            img_shape=img_shape, tile_shape=tile_shape,
            tile_spacing=(1, 1)))


def init_weights(n_visible, n_hidden, rng):
    dist = rng.uniform(low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                       high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                       size=(n_visible, n_hidden))
    w = numpy.asarray(dist, dtype=theano.config.floatX)
    return theano.shared(value=w, name='w', borrow=True)


def get_batch(data, index, batch_size):
    return data[index * batch_size: (index + 1) * batch_size]


def cross_entropy(x, z):
    return - T.sum(x * T.log(z) + (1 - x) * T.log(1 - z), axis=1)


def square_error(x, z):
    return T.square(x - z)
