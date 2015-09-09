
from lfw import lfwc_paths, sample_patches
from images import scale_input, load_images, load_matrix
from dae import DenoisingAutoencoder

import theano

__author__ = 'dracz'


def train_patches(n_samples=100000,
                  tile_shape=(9, 9),
                  n_hidden=64,
                  corruption_rate=0.3,
                  learning_rate=0.1,
                  n_epochs=15,
                  batch_size=20):
    patches = scale_input(sample_patches(lfwc_paths(), tile_shape, n_samples))

    n_visible = tile_shape[0] * tile_shape[1]

    da = DenoisingAutoencoder(n_visible, n_hidden)

    da.train(theano.shared(patches, borrow=True),
             corruption_rate=corruption_rate,
             learning_rate=learning_rate,
             n_epochs=n_epochs,
             batch_size=batch_size)

    da.render_filters(tile_shape).show()


def train_whole_image(tile_shape=(64, 64),
                      n_hidden=100,
                      corruption_rate=0.3,
                      learning_rate=0.1,
                      n_epochs=15,
                      batch_size=20):
    patches = scale_input(load_images(lfwc_paths()))
    n_visible = tile_shape[0] * tile_shape[1]

    da = DenoisingAutoencoder(n_visible, n_hidden)
    da.train(theano.shared(patches, borrow=True),
             corruption_rate=corruption_rate,
             learning_rate=learning_rate,
             n_epochs=n_epochs,
             batch_size=batch_size)

    da.render_filters(tile_shape).show()

# train_whole_image()
# train_patches()
