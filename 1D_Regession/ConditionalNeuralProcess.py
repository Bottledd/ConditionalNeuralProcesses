import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras import Model
from GaussianProcessSampler import GaussianProcess
import tensorflow_probability as tfp


# will write class such that it takes in a single [xc, yc] and [xt]
# to train train on each batch
# then get new batch and retrain
class ConditionalNeuralProcess(Model):
    def __init__(self, layer_width):
        super(ConditionalNeuralProcess, self).__init__()
        self._encoder = Encoder(layer_width)
        self._decoder = Decoder(layer_width)
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, inputs):
        representation = self._encoder(inputs)
        means, stds = self._decoder(representation, inputs)

        return means, stds

    def loss_func(self, means, stds, targets):
        # want distribution of all target points
        dist = tfp.distributions.MultivariateNormalDiag(loc=means, scale_diag=stds)
        log_prob = dist.log_prob(targets)
        return -tf.reduce_mean(log_prob)

    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            means, stds = self(inputs)
            loss = self.loss_func(means, stds, targets)
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss


class Encoder(Layer):
    """
    The Encoder which is to be shared across all context points.
    Instantiate with list of target number of nodes per layer.
    """
    def __init__(self, layer_width):
        super(Encoder, self).__init__()
        self.layer1 = Dense(layer_width, activation='relu')
        self.layer2 = Dense(layer_width, activation='relu')
        self.layer3 = Dense(layer_width, activation='relu')
        self.layer4 = Dense(layer_width, activation=None)

    def call(self, inputs):
        """
        Inputs is tuple (x_context, y_context, x_data), each shape [batch_size , num_points, dimension]
        Need to reshape inputs to be pairs of [x_context, y_context],
        Pass through NN,
        Compute a representation by aggregating the outputs.
        """

        # process data for encoder
        x_context, y_context = inputs[0], inputs[1]

        # # grab shapes
        # batch_size, num_context_points = x_context.shape
        #
        # # reshape to [batch_size, num_points, 1]
        # x_context = tf.reshape(x_context, (batch_size, num_context_points, 1))
        # y_context = tf.reshape(y_context, (batch_size, num_context_points, 1))

        # concatenate to form inputs [x_context, y_context], overall shape [batch_size, num_context, 2]
        encoder_input = tf.concat([x_context, y_context], axis=-1)

        h = self.layer1(encoder_input)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)

        # now compute representation vector, average across context points, so axis 1
        representation = tf.reduce_mean(h, axis=1)

        return representation


class Decoder(keras.layers.Layer):
    """
    The Decoder to be shared amongst targets.
    Instantiate with list of target number of nodes per layer.
    For 1D regression need final layer to have 2 units
    """
    def __init__(self, layer_width):
        super(Decoder, self).__init__()
        self.layer1 = Dense(layer_width, activation='relu')
        self.layer2 = Dense(layer_width, activation='relu')
        # final layer into mean and log stds
        self.layer3 = Dense(2, activation=None)

    def call(self, representation, inputs):
        """
        Takes in computed representation vector from encoder and x data to decode targets
        """
        # grab x_data
        x_data = inputs[-1]

        # need to concatenate representation to data, so each data set looks like [x_T, representation]

        # # need to reshape x_data
        # # grab shapes
        # batch_size, num_context_points = x_data.shape
        #
        # # reshape to [batch_size * num_points * 1]
        # x_data = tf.reshape(x_data, (batch_size, num_context_points, 1))

        # reshape representation vector and repeat it
        representation = tf.repeat(tf.expand_dims(representation, axis=1), x_data.shape[1], axis=1)
        # concatenate representation to inputs
        decoder_input = tf.keras.layers.concatenate([x_data, representation], axis=-1)

        h = self.layer1(decoder_input)
        h = self.layer2(h)
        h = self.layer3(h)

        # split into 2 tensors, one with means and one with stds
        # floor the variance to avoid pathological solutions
        means, log_stds = tf.split(h, 2, axis=-1)
        stds = 0.01 + 0.99 * tf.nn.softplus(log_stds)
        #stds = tf.nn.softplus(log_stds)


        return means, stds


if __name__ == "__main__":
    # gps
    gp_test = GaussianProcess(100, 10, testing=False)
    data = gp_test.generate_curves()
    # model output sizes
    enc = Encoder(128)
    reps = enc((data.Inputs[0], data.Inputs[1]))
    masked_x = np.zeros(shape=data.Inputs[0].shape)
    masked_x[1:] = data.Inputs[0][1:]
    masked_y = np.zeros(shape=data.Inputs[1].shape)
    masked_y[1:] = data.Inputs[1][1:]
    masked_data = masked_x, masked_y
    reps_masked = enc((masked_x, masked_y))
    assert np.all(reps.numpy()[2, :] == reps_masked.numpy()[2, :])

    dec = Decoder(128)
    means, stds = dec(reps, data.Inputs)

    model = ConditionalNeuralProcess(layer_width=128)
    model(inputs=data.Inputs)
    model.summary()