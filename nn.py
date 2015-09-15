
import numpy as np
import itertools
from PIL import Image
import theano
import theano.tensor as T


""" Various utility functions for neural networks """


def render_filters(w, sh, image_file=None, axis=0, sp=1, show=False):
    """
    Visualize the weight matrix of a single layer by rendering a image of
    the inputs that maximally activate each hidden unit.

    :param w: Weights matrix for one hidden layer
    :param sh: The shape of the input image to the visible layer
    :param image_file: The file to write output image to, if any
    :param axis: 0 if weights for each unit are in columns of `w`, 1 otherwise
    :return: PIL Image instance
    """

    weights = w if axis == 0 else w.T
    n_visible, n_hidden = weights.shape

    rc = int(np.floor(np.sqrt(n_hidden)))

    out_shape = (rc * sh[0] + (rc+1) * sp, rc * sh[1] + (rc+1) * sp)

    img = Image.new("L", out_shape, "black")

    for i, (r, c) in enumerate(itertools.product(range(rc), range(rc))):
        w_i = weights[:, i]
        v = w_i/np.sqrt(np.square(w_i).sum())
        v = scale_interval(v, max_val=255)

        x_img = Image.fromarray(v.reshape(sh))
        img.paste(x_img.copy(), (sp + c*(sh[0] + sp), sp + r*(sh[1] + sp)))

    if image_file is not None:
        print("saving {}...".format(image_file))
        img.save(image_file)

    if show:
        img.show()

    return img


def init_weights(n_visible, n_hidden, rng, shared=True):
    """
    Randomly initialize weights to small values
    :param n_visible: Number of input units
    :param n_hidden: Number of hidden units
    :param rng: Theano RandomStreams
    :return: ndarray of floats as theano shared variable
    """
    dist = rng.uniform(low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                       high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                       size=(n_visible, n_hidden))
    w = np.asarray(dist, dtype=theano.config.floatX)
    if not shared:
        return w
    return theano.shared(value=w, name='w', borrow=True)


def get_batch(data, index, batch_size):
    return data[index * batch_size: (index + 1) * batch_size]


def cross_entropy(x, z):
    return - T.sum(x * T.log(z) + (1 - x) * T.log(1 - z), axis=1)


def square_error(x, z):
    return T.square(x - z)


def scale_interval(nda, min_val=0, max_val=1, eps=1e-8):
    """ Scale all vals in the ndarray nda to be in range [min_val, max_val] """
    x = nda.copy()
    x_max, x_min = x.max(), x.min()
    return (1.0 * (x - x_min) * (max_val - min_val) / (x_max - x_min + eps)) + min_val

