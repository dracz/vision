
import timeit
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from nn import *


class SparseAutoencoder(object):
    """ A sparse autoencoder """

    def __init__(self, n_visible=400, n_hidden=100, w1=None, w2=None, b1=None, b2=None):

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        numpy_rng = numpy.random.RandomState(123)

        self.w1 = init_weights(w1, n_visible, n_hidden, numpy_rng, name="w1")
        self.w2 = init_weights(w2, n_hidden, n_visible, numpy_rng, name="w2")

        self.b1 = init_bias(b1, n_hidden, "b1")
        self.b2 = init_bias(b2, n_visible, "b2")

        self.theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.x = T.dmatrix(name='input')
        self.params = [self.w1, self.w2, self.b1, self.b2]

        self.L1 = (abs(self.w1).sum() + abs(self.w2).sum())
        self.L2 = ((self.w1 ** 2).sum() + (self.w2 ** 2).sum())

    def get_cost_updates(self, beta, sparsity, weight_decay, learning_rate):
        """ Compute the cost and the parameter updates for one training step """

        a1 = self.x  # activation of first layer is the input
        a2 = T.nnet.sigmoid(T.dot(a1, self.w1) + self.b1)  # activation of hidden layer
        a3 = T.nnet.sigmoid(T.dot(a2, self.w2) + self.b2)  # activation of output layer (reconstruction of input)

        e = T.mean(cross_entropy(a1, a3))

        wd = weight_decay * self.L2  # L2 regularization
        sp = beta * T.sum(kl(sparsity, a2.mean(axis=0)))  # sparseness penalty

        cost = e + wd # + sp

        grads = T.grad(cost, self.params)

        updates = [
            (param, param - learning_rate * grad) for param, grad in zip(self.params, grads)
            ]
        return cost, updates

    def train(self, data, batch_size=20, beta=3.0, sparsity=0.0, weight_decay=0.0, learning_rate=0.1, n_epochs=15):
        """
        Train the model
        :param data: A theano shared variable of ndarray of shape [n_examples, n_features]
        :param batch_size: Mini-batch size
        :param sparsity: Sparsity parameter
        :param learning_rate: Learning rate
        :param n_epochs: Number of training epochs
        """
        n_training = data.get_value(borrow=True).shape[0]
        n_train_batches = n_training / batch_size

        index = T.lscalar()  # index into mini-batch

        cost, updates = self.get_cost_updates(
            beta=beta,
            sparsity=sparsity,
            weight_decay=weight_decay,
            learning_rate=learning_rate
        )

        train_da = theano.function(
            [index],
            cost,
            updates=updates,
            givens={self.x: get_batch(data, index, batch_size)}
        )

        print("Starting to train using {} examples and {} batches of {}..."
              .format(n_training, n_train_batches, batch_size))

        t1 = timeit.default_timer()
        last_cost = None

        for epoch in xrange(n_epochs):
            c = []

            for batch_index in range(n_train_batches):
                c.append(train_da(batch_index))

            avg_cost = np.mean(c)

            diff = None if last_cost is None else (avg_cost - last_cost) / last_cost
            ds = "" if diff is None else " ({:.4%})".format(diff)
            print('Epoch: {}, Cost: {}{}'.format(epoch, avg_cost, ds))

            last_cost = avg_cost

        t2 = timeit.default_timer()

        training_time = (t2 - t1)
        print('Training took {:.2f}'.format(training_time))


def train(data, n_visible=16*16, n_hidden=200, batch_size=20,
          learning_rate=0.1, n_epochs=5, beta=3.0, sparsity=0.0,
          weight_decay=0.0, stop_diff=None, corruption_rate=None):
    """train a new autoencoder"""
    ae = SparseAutoencoder(n_visible=n_visible, n_hidden=n_hidden)
    ae.train(data, batch_size=batch_size, learning_rate=learning_rate,
             n_epochs=n_epochs, beta=beta, sparsity=sparsity, weight_decay=weight_decay)
    return ae


