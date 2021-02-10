import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from GaussianProcessSampler import GaussianProcess
import tensorflow_probability as tfp


# will write class such that it takes in a single [xc, yc] and [xt]
# to train train on each batch
# then get new batch and retrain
class ConditionalNeuralProcess(keras.Model):
    def __init__(self, node_size_list):
        super(ConditionalNeuralProcess, self).__init__()
        self._encoder_num_nodes = node_size_list[0]
        self._decoder_num_nodes = node_size_list[1]
        self._encoder = Encoder(self._encoder_num_nodes)
        self._decoder = Decoder(self._decoder_num_nodes)

    def call(self, inputs):
        representation = self._encoder(inputs)

        means, stds = self._decoder(representation, inputs)

        return means, stds


class Encoder(keras.layers.Layer):
    """
    The Encoder which is to be shared across all context points.
    Instantiate with list of target number of nodes per layer.
    """
    def __init__(self, num_nodes):
        super(Encoder, self).__init__()
        self._num_nodes = num_nodes

    def call(self, inputs):
        """
        Inputs is tuple (x_context, y_context, x_data), each shape [batch_size * num_points]
        Need to reshape inputs to be pairs of [x_context, y_context],
        Pass through NN,
        Compute a representation by aggregating the outputs.
        """

        # process data for encoder
        x_context, y_context = inputs[0], inputs[1]

        # grab shapes
        batch_size, num_context_points = x_context.shape

        # reshape to [batch_size * num_points * 1]
        x_context = tf.reshape(x_context, (batch_size, num_context_points, 1))
        y_context = tf.reshape(y_context, (batch_size, num_context_points, 1))

        # concatenate to form inputs [x_contect, y_context], overall shape [batch_size, num_context, 2]
        encoder_input = tf.concat([x_context, y_context], axis=-1)

        # build NN and pass inputs through
        # keras will give a vector of size [batch_size, num_context, final_num_nodes]
        # go up to penultimate layer
        for i, node_size in enumerate(self._num_nodes[:-1]):
            encoder_input = Dense(node_size, activation='relu')(encoder_input)
        # last layer do not use activation function
        encoder_input = Dense(self._num_nodes[-1], activation=None)(encoder_input)

        # now compute representation vector, average across context points, so axis 1
        representation = tf.reduce_mean(encoder_input, axis=1)

        return representation


class Decoder(keras.layers.Layer):
    """
    The Decoder to be shared amongst targets.
    Instantiate with list of target number of nodes per layer.
    For 1D regression need final layer to have 2 units
    """
    def __init__(self, num_nodes):
        super(Decoder, self).__init__()
        self._num_nodes = num_nodes

    def call(self, representation, inputs):
        """
        Takes in computed representation vector from encoder and x data to decode targets
        """
        # grab x_data
        x_data = inputs[-1]

        # need to concatenate representation to data, so each data set looks like [x_T, representation]

        # need to reshape x_data
        # grab shapes
        batch_size, num_context_points = x_data.shape

        # reshape to [batch_size * num_points * 1]
        x_data = tf.reshape(x_data, (batch_size, num_context_points, 1))

        # concatenate representation to inputs
        decoder_input = tf.concat([x_data, representation], axis=-1)

        # build NN and pass inputs
        for i, node_size in enumerate(self._num_nodes)[:-1]:
            decoder_input = Dense(node_size, activation='relu')(decoder_input)
        decoder_input = Dense(self._num_nodes[-1])(decoder_input)

        # split into 2 tensors, one with means and one with stds
        # floor the variance to avoid pathological solutions
        means, log_stds = tf.split(decoder_input, 2, axis=-1)
        stds = 0.1 + 0.9 * tf.nn.softplus(log_stds)

        return means, stds


def loss_func(means, stds, targets):
    dist = tfp.distributions.MultivariateNormalDiag(loc=means, scale_diag=stds)
    log_prob = dist.log_prob(targets)
    return -tf.reduce_mean(log_prob)


def training_data_generator(batch_size = 1, testing=False):
    while True:
        gp = GaussianProcess(batch_size=batch_size, max_num_context=10, testing=testing)
        data = gp.generate_curves()
        yield data


if __name__ == "__main__":
    # gps
    gp_test = GaussianProcess(2, 10, testing=False)
    data = gp_test.generate_curves()
    # model output sizes
    output_sizes = [[128,128,128,128], [128,128,2]]
    model = ConditionalNeuralProcess(node_size_list=output_sizes)
    model(inputs=data.Inputs)