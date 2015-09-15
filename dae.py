
import timeit
from theano.tensor.shared_randomstreams import RandomStreams
from nn import *


rng_seed = 456  # seed for random number generation
rng_n = 2**30   # large number to use for creating stream of rand vals


class DenoisingAutoencoder(object):
    """
    Denoising Auto-Encoder class based on:
    http://deeplearning.net/tutorial/dA.html
    """

    def __init__(self, n_visible=16*16, n_hidden=200,
                 x=None, w=None, bh=None, bv=None):

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        numpy_rng = np.random.RandomState(rng_seed)
        theano_rng = RandomStreams(numpy_rng.randint(rng_n))

        if not x:
            x = T.dmatrix(name='input')

        if not w:
            w = init_weights(n_visible, n_hidden, numpy_rng)

        if not bv:
            zv = np.zeros(n_visible, dtype=theano.config.floatX)
            bv = theano.shared(value=zv, borrow=True)

        if not bh:
            zh = np.zeros(n_hidden, dtype=theano.config.floatX)
            bh = theano.shared(value=zh, name='b', borrow=True)

        self.w = w  # weights
        self.b = bh  # bias of hidden units
        self.b_prime = bv  # bias of visible units
        self.w_prime = self.w.T  # tied weights, W_prime = W transpose
        self.theano_rng = theano_rng

        self.x = x
        self.params = [self.w, self.b, self.b_prime]
        print("Initialized network with {} input units and {} hidden units".format(n_visible, n_hidden))

    def get_corrupted_input(self, data, corruption_level=0.3):
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

    def get_cost_updates(self, corruption_rate=0.3, learning_rate=0.1):
        """ This function computes the cost and the updates for one training step"""
        tilde_x = self.get_corrupted_input(self.x, corruption_rate)

        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        l = cross_entropy(self.x, z)

        cost = T.mean(l)
        grads = T.grad(cost, self.params)

        updates = [
            (param, param - learning_rate * grad) for param, grad in zip(self.params, grads)
            ]
        return cost, updates

    def train(self, train_set, batch_size=20, corruption_rate=0.3,
              learning_rate=0.1, n_epochs=15, stop_diff=None):

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

        print("Starting to train using {} examples, {} epochs, and {} batches of {}..."
              .format(n_training, n_epochs, n_train_batches, batch_size))

        print("corruption rate = {}, learning rate = {}"
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



