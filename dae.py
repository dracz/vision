
import timeit
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

from nn import *


rng_seed = 456  # seed for random number generation
rng_n = 2**30   # large number to use for creating stream of rand vals


class DenoisingAutoencoder(object):
    """
    Denoising Auto-Encoder class based on:
    http://deeplearning.net/tutorial/dA.html
    """

    def __init__(self, n_visible=16*16, n_hidden=200, x=None, w1=None, b1=None, b2=None):

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        numpy_rng = np.random.RandomState(rng_seed)
        theano_rng = RandomStreams(numpy_rng.randint(rng_n))

        self.theano_rng = theano_rng

        if not x:
            x = T.dmatrix(name='input')

        self.x = x

        self.w1 = init_weights(w1, n_visible, n_hidden, numpy_rng)
        self.w2 = self.w1.T

        self.b1 = init_bias(b1, n_hidden, "b1")
        self.b2 = init_bias(b2, n_visible, "b2")

        self.params = [self.w1, self.b1, self.b2]

        print("Initialized network with {} input units and {} hidden units".format(n_visible, n_hidden))

    def get_corrupted_input(self, data, corruption_level=0.3):
        """corrupt the image """
        return self.theano_rng.binomial(size=data.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * data

    def get_hidden_values(self, data):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(data, self.w1) + self.b1)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input from values of the hidden layer"""
        return T.nnet.sigmoid(T.dot(hidden, self.w2) + self.b2)

    def get_cost_updates(self, corruption_rate=0.3, learning_rate=0.1):
        """ This function computes the cost and the updates for one training step"""
        tilde_x = self.get_corrupted_input(self.x, corruption_rate)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        #l = square_error(self.x, z)
        l = cross_entropy(self.x, z)
        cost = T.mean(l)
        grads = T.grad(cost, self.params)

        updates = [
            (param, param - learning_rate * grad) for param, grad in zip(self.params, grads)
            ]
        return cost, updates

    def train(self, data, batch_size=20, corruption_rate=0.3,
              learning_rate=0.1, n_epochs=15, stop_diff=None):
        """train the autoencoder. data is a theano shared ndarray with examples in rows"""

        n_training = data.get_value(borrow=True).shape[0]
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
            givens={self.x: get_batch(data, index, batch_size)}
        )

        print("Starting to train using {} examples, {} epochs, and {} batches of {}..."
              .format(n_training, n_epochs, n_train_batches, batch_size))

        print("Corruption rate = {}, learning rate = {}"
              .format(corruption_rate, learning_rate))

        t1 = timeit.default_timer()

        last_cost = None

        for epoch in range(n_epochs):
            c = []

            for batch_index in range(n_train_batches):
                c.append(train_da(batch_index))

            avg_cost = np.mean(c)

            diff = None if last_cost is None else (avg_cost - last_cost) / last_cost
            ds = "" if diff is None else " ({:.4%})".format(diff)
            print('Epoch: {}, Cost: {}{}'.format(epoch, avg_cost, ds))

            last_cost = avg_cost

            if stop_diff is not None and diff is not None and np.abs(diff) < np.abs(stop_diff):
                print("diff abs({}) < abs({}), stopping".format(diff, stop_diff))
                break

        t2 = timeit.default_timer()

        training_time = (t2 - t1)
        print('Training took {:.2f}'.format(training_time))


def train(data, n_visible=16*16, n_hidden=200, batch_size=20,
          learning_rate=0.1, n_epochs=5, beta=3.0, sparsity=0.0,
          weight_decay=0.0, stop_diff=None, corruption_rate=0.3):
    """train a new autoencoder"""
    da = DenoisingAutoencoder(n_visible=n_visible, n_hidden=n_hidden)
    da.train(data, batch_size=batch_size, corruption_rate=corruption_rate,
             learning_rate=learning_rate, n_epochs=n_epochs, stop_diff=stop_diff)
    return da


