
import timeit
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class SparseAutoencoder(object):
    """ A sparse autoencoder """

    def __init__(self, n_in=400, n_hidden=100, w=None, w_prime=None, b=None, b_prime=None):

        self.n_visible = n_in
        self.n_hidden = n_hidden

        numpy_rng = numpy.random.RandomState(123)

        self.w = weights(w, n_in, n_hidden, numpy_rng, "w")

        self.w_prime = weights(w_prime, n_hidden, n_in, numpy_rng, "w_prime")

        self.b = zeros(n_in, "b") if b_prime is None else b_prime
        self.b_prime = zeros(n_hidden, "b_prime") if b is None else b

        self.theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.x = T.dmatrix(name='input')
        self.params = [self.w, self.w_prime, self.b, self.b_prime]

    def get_cost_updates(self, sparsity, learning_rate):
        """ Compute the cost and the parameter updates for one training step """

        # activation of hidden layer
        a = T.nnet.sigmoid(T.dot(self.x, self.w) + self.b)

        # output, ie reconstruction of input
        z = T.nnet.sigmoid(T.dot(a, self.w_prime) + self.b_prime)

        print(a.shape)
        rho = a.mean()

        l = square_error(self.x, z)

        cost = T.mean(l)
        grads = T.grad(cost, self.params)

        updates = [
            (param, param - learning_rate * grad) for param, grad in zip(self.params, grads)
            ]
        return cost, updates

    def train(self, train_set, batch_size=100, sparsity=0.05, learning_rate=0.1, n_epochs=15):

        n_training = train_set.get_value(borrow=True).shape[0]
        n_train_batches = n_training / batch_size

        index = T.lscalar()  # index into mini-batch

        cost, updates = self.get_cost_updates(
            sparsity=sparsity,
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


def zeros(n, name):
    zh = numpy.zeros(n, dtype=theano.config.floatX)
    return theano.shared(value=zh, name=name, borrow=True)


def weights(data, n_from, n_to, rng, name="w"):
    if data is not None: return data
    dist = rng.uniform(low=-4 * numpy.sqrt(6. / (n_to + n_from)),
                       high=4 * numpy.sqrt(6. / (n_to + n_from)),
                       size=(n_from, n_to))
    w = numpy.asarray(dist, dtype=theano.config.floatX)
    return theano.shared(value=w, name=name, borrow=True)


def get_batch(data, index, batch_size):
    return data[index * batch_size: (index + 1) * batch_size]


def cross_entropy(x, z):
    return - T.sum(x * T.log(z) + (1 - x) * T.log(1 - z), axis=1)


def square_error(x, z):
    return T.square(x - z)


