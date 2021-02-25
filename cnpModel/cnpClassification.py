import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Layer, Conv2D, Flatten, MaxPool2D
from tensorflow.keras import Model
from Utils.omniglotDataSet import Omniglot
from tqdm import tqdm


class ClassificationCNP(Model):
    def __init__(self, encoder_layer_widths, decoder_layer_widths, num_classes):
        super(ClassificationCNP, self).__init__()
        self._encoder = Encoder(encoder_layer_widths)
        self._decoder = Decoder(decoder_layer_widths)
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        # equivalent to using log likelihood from paper
        self._loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self._num_classes = num_classes

    def call(self, encoder_data, decoder_data):
        """Classification task outputs softmax logits for each class label"""

        # representation which is aggregated across classes then concatenated
        encoder_output = self._encoder(encoder_data)

        # logits for softmax for each class
        logits = self._decoder(encoder_output, decoder_data)

        return logits

    def train_step(self, encoder_data, decoder_data, accuracy_metric):
        _, labels = decoder_data
        with tf.GradientTape() as tape:
            logits = self(encoder_data, decoder_data)
            loss = self._loss_func(labels, logits)
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        accuracy_metric.update_state(labels, logits)

        return loss


class Encoder(Layer):
    """
    The Encoder which is to be shared across all context points.
    Instantiate with list of target number of nodes per layer.
    """
    def __init__(self, encoder_layer_widths):
        super(Encoder, self).__init__()
        # add the convolutional layers
        self.h = []
        for layer_width in encoder_layer_widths[:-1]:
            self.h.append(Conv2D(layer_width, 8, activation='relu'))
        # no activation for the final layer
        self.h.append(Conv2D(layer_width, 8, activation=None))
        self.h.append(Flatten())

    def h_func(self, x):
        for layer in self.h:
            x = layer(x)

        return x

    def call(self, data):
        """
        Inputs is tuple (x_context, y_context, x_data), each shape [batch_size , num_points, dimension]
        Need to reshape inputs to be pairs of [x_context, y_context],
        Pass through NN,
        Compute a representation by aggregating the outputs.
        """

        # process data for encoder
        images, labels = data

        # pass image through h_func
        encoder_output = self.h_func(images)

        # now compute representation vector, average over images with same classes
        representation = tf.math.unsorted_segment_mean(
            encoder_output,
            tf.reshape(labels, (-1,)),
            labels.shape[0],
        )

        # flatten representation into single vector, effectively concatenates
        return tf.reshape(representation, (-1,))


class Decoder(keras.layers.Layer):
    """
    The Decoder to be shared amongst targets.
    Instantiate with list of target number of nodes per layer.
    For 1D regression need final layer to have 2 units
    """
    def __init__(self, decoder_layer_widths):
        super(Decoder, self).__init__()
        # add the hidden layers
        self.g = []
        for layer_width in decoder_layer_widths[:-1]:
            self.g.append(Dense(layer_width, activation='relu'))

        # no activation for the final layer
        self.g.append(Dense(decoder_layer_widths[-1], activation=None))

    def g_func(self, x):

        for layer in self.g:
            x = layer(x)

        return x

    def call(self, representation, data):
        """
        Takes in computed representation vector from encoder and x data to decode targets
        """

        # test data
        images, labels = data
        batch_size = images.shape[0]

        assert batch_size == labels.shape[0]

        # reshape representation vector and repeat it
        representation = tf.repeat(tf.expand_dims(representation, axis=0), images.shape[0], axis=0)

        # flatten image into vector (MAYBE THIS SHOULD CHANGE)
        images = tf.reshape(images, (batch_size, -1))

        # concatenate representation to inputs
        decoder_input = tf.concat([images, representation], axis=-1)

        logits = self.g_func(decoder_input)

        # SHOULD I INSTEAD USE tf.split TO GET THE LOGITS?

        return logits


if __name__ == "__main__":
    dummy_data = Omniglot()
    num_classes = 5
    shots = 1

    # first three layers are Conv2D and width means num filters, final layer is Dense
    encoder_widths = [32, 64, 128]

    # need final width to be divisible by num classes
    decoder_widths = [64, 64, num_classes]
    cnp = ClassificationCNP(encoder_widths, decoder_widths, num_classes)
    acc_metric = keras.metrics.SparseCategoricalAccuracy()
    loss = []
    for epoch in tqdm(range(10000)):

        loop_train, loop_test = dummy_data.get_mini_dataset(shots=shots, num_classes=num_classes)
        for step, (batchX, batchy) in enumerate(zip(loop_train, loop_test)):
            loss.append(cnp.train_step(batchX, batchy, acc_metric))

        training_accuracy = acc_metric.result()
        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}: training accuracy is {training_accuracy}, loss is {np.mean(loss[-100:])}")
            acc_metric.reset_states()


