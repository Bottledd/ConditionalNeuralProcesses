import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Layer, Conv2D, Flatten, MaxPool2D
from tensorflow.keras import Model
from Utils.omniglotDataSet import Omniglot
from tqdm import tqdm
import tensorflow_probability as tfp


class ClassificationCNP(Model):
    def __init__(self, cnn_layer_widths, decoder_layer_widths):
        super(ClassificationCNP, self).__init__()
        self._encoder = Encoder(cnn_layer_widths)
        self._decoder = Decoder(cnn_layer_widths, decoder_layer_widths)
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-4)

    def call(self, encoder_data, decoder_data):
        """Classification task outputs softmax logits for each class label"""

        # representation which is aggregated across classes then concatenated
        encoder_output = self._encoder(encoder_data)

        # logits for softmax for each class
        logits = self._decoder(encoder_output, decoder_data)

        return logits

    def _loss_func(self, labels, logits):
        dist = tfp.distributions.Categorical(logits=logits)
        preds = np.argmax(logits, axis=1)
        correct = np.argmax(logits, axis=1) == labels.numpy().flatten()
        accuracy = np.sum(correct) / preds.shape[0]
        each_loss = dist.log_prob(tf.squeeze(labels))
        mean_loss = tf.reduce_mean(each_loss)

        return -mean_loss, accuracy

    def train_step(self, encoder_data, decoder_data):
        _, labels = decoder_data
        with tf.GradientTape() as tape:
            logits = self(encoder_data, decoder_data)
            loss, accuracy = self._loss_func(labels, logits)
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss, accuracy


class Encoder(Layer):
    """
    The Encoder which is to be shared across all context points.
    Instantiate with list of target number of nodes per layer.
    """
    def __init__(self, cnn_layer_widths):
        super(Encoder, self).__init__()
        # first create a CNN to pass raw image through
        self.cnn = []
        for i, layer_width in enumerate(cnn_layer_widths[:-1]):
            # last two layers add strides
            self.cnn.append(Conv2D(layer_width, 3, activation='relu'))

        # no activation for the final layer
        self.cnn.append(Conv2D(cnn_layer_widths[-1], 3, strides=(2, 2), activation=None))
        self.cnn.append(MaxPool2D())
        self.cnn.append(Flatten())
        self.cnn.append(Dense(1))

    def h_func(self, x):
        for layer in self.cnn:
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
            tf.unique(tf.reshape(labels, (-1,)))[0].shape[0]
        )

        # flatten representation into single vector, effectively concatenates
        return representation


class Decoder(keras.layers.Layer):
    """
    The Decoder to be shared amongst targets.
    Instantiate with list of target number of nodes per layer.
    For 1D regression need final layer to have 2 units
    """
    def __init__(self, cnn_layer_widths, decoder_layer_widths):
        super(Decoder, self).__init__()
        # first create a CNN to pass raw image through
        self.cnn = []
        for i, layer_width in enumerate(cnn_layer_widths[:-1]):
            self.cnn.append(Conv2D(layer_width, 3, activation='relu'))
        # no activation for the final layer
        self.cnn.append(Conv2D(cnn_layer_widths[-1], 3, strides=(2, 2), activation=None))
        self.cnn.append(MaxPool2D())
        self.cnn.append(Flatten())
        self.cnn.append(Dense(1))

        # This seems to not work well
        # # create MLP
        # self.g = []
        # for i, layer_width in enumerate(decoder_layer_widths[:-1]):
        #     self.g.append(Dense(layer_width, activation='relu'))
        # # final layer outputs logits
        # self.g.append(Dense(decoder_layer_widths[-1], activation=None))

    def decoder_cnn(self, x):
        for layer in self.cnn:
            x = layer(x)

        return x

    # def g_func(self, x):
    #     for layer in self.g:
    #         x = layer(x)
    #
    #     return x

    def call(self, representation, data):
        """
        Takes in computed representation vector from encoder and x data to decode targets
        """
        # test data
        images, labels = data
        batch_size = images.shape[0]

        assert batch_size == labels.shape[0]

        # pass images through CNN
        embedded_images = self.decoder_cnn(images)

        # combine representation and embedded images with dot product
        logits = tf.tensordot(embedded_images, tf.transpose(representation), axes=1)

        # pass through MLP
        # logits = self.g_func(logits)

        return logits


if __name__ == "__main__":
    dummy_data = Omniglot()
    num_classes = 5
    shots = 1

    # first three layers are Conv2D and width means num filters
    # use this CNN for both the encoder and decoder
    cnn_widths = [32, 64, 128]

    # in decoder pass images through a 1 layer MLP
    decoder_widths = [128, 128, 64, num_classes]
    cnp = ClassificationCNP(cnn_layer_widths=cnn_widths, decoder_layer_widths=decoder_widths, num_classes=num_classes)

    for epoch in tqdm(range(500)):
        loop_train, loop_test = dummy_data.get_mini_dataset(shots=shots, num_classes=num_classes)
        loss_list = []
        accuracy_list = []
        for step, (batchX, batchy) in enumerate(zip(loop_train, loop_test)):
            loss, accuracy = cnp.train_step(batchX, batchy)
            loss_list.append(loss)
            accuracy_list.append(accuracy)

        print(f"Epoch {epoch}: Loss {np.mean(loss_list)},   Accuracy {np.round(100 * np.mean(accuracy_list), decimals=1)},"
              f"    One-shot Accuracy {100 * accuracy_list[0]}")

    big_acc = []
    for test_epoch in range(100):
        test_context, test_target = dummy_data.get_mini_dataset(shots=shots, num_classes=num_classes, testing=True)

        accuracies = []
        for step, (batchX, batchy) in enumerate(zip(test_context, test_target)):
            logits = cnp(batchX, batchy)
            loss, accuracy = cnp._loss_func(batchy[1], logits)
            accuracies.append(accuracy)
            big_acc.append(accuracy)
        print(f"Mini accuracy is {np.mean(accuracies)}")
