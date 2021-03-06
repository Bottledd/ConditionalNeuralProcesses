import os
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from Utils.mnistProcessor import process_images, format_context_points_image
from cnpModel.ConditionalNeuralProcesses import ConditionalNeuralProcess


def test_cnp(cnp, test_data, context_points=28*28, convolutional=False):
    image = next(iter(test_data.take(1)))[0].numpy().reshape(1, 28, 28)

    # grab a random image from the test set
    # image = test_data[np.random.randint(0, test_data.shape[0]+1)].reshape(1, 28, 28)

    # process image
    processed = process_images(image, context_points=context_points, convolutional=convolutional)

    # evaluate cnp on image
    means, stds = cnp(processed.Inputs)

    # reshape for plotting
    predictive_mean = tf.reshape(means, (28, 28))
    predictive_stds = tf.reshape(stds, (28, 28))
    if not convolutional:
        context_image = format_context_points_image(processed.Inputs)
    else:
        context_image = processed.Inputs[1][0]

    # plot stuff
    plt.figure('context')
    plt.imshow(context_image)
    plt.title('Context')
    plt.tight_layout()
    plt.show()

    plt.figure('means')
    plt.imshow(predictive_mean, cmap='gray')
    plt.title('Predictive Mean')
    plt.tight_layout()
    plt.show()

    plt.figure('stds')
    plt.imshow(predictive_stds, cmap='gray')
    plt.title('Predictive Std')
    plt.tight_layout()
    plt.show()

    plt.figure('actual')
    plt.imshow(processed.Targets.reshape(28, 28), cmap='gray')
    plt.title('Actual')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    os.chdir("..")
    # load corrupted data set
    data, info = tfds.load(
        "mnist_corrupted",
        split='test',
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    load = True
    attention = False
    convolutional = False
    iterations = 200000
    batching = 24
    attention_params = {}
    convolutional_params = {}
    encoder_layer_widths = []
    decoder_layer_widths = []
    if attention:
        loading_path = os.path.join(os.getcwd(), "saved_models/MNIST/ATTNCNP_400k_24B/")
        saving_path = os.path.join(os.getcwd(), "saved_models/MNIST/ATTNCNP_400k_24B/")
        # saving_path = os.path.join(os.getcwd(), f"saved_models/MNIST/ATTNCNP_{int(iterations / 1000)}k_{batching}B/")
        encoder_layer_widths = [128, 128]
        decoder_layer_widths = [64, 64, 64, 64, 2]
        attention_params = {"embedding_layer_width": 128, "num_heads": 8, "num_self_attention_blocks": 2}
    elif convolutional:
        # parameters for convolutional
        kernel_size_encoder = 9
        kernel_size_decoder = 5
        convolutional_params = {"number_filters": 128, "kernel_size_encoder": 9, "kernel_size_decoder": 5,
                                "number_residual_blocks": 4, "convolutions_per_block": 1, "output_channels": 1}
        loading_path = os.path.join(os.getcwd(), "saved_models/MNIST/CONVCNP_100k_64B/")
        saving_path = os.path.join(os.getcwd(), f"saved_models/MNIST/CONVCNP_{int(iterations / 1000)}k_{batching}B/")
    else:
        loading_path = os.path.join(os.getcwd(), "saved_models/MNIST/CNP_100k_64B/")
        saving_path = os.path.join(os.getcwd(), f"saved_models/MNIST/CNP_{int(iterations / 1000)}k_{batching}B/")

        encoder_layer_widths = [128, 128, 128]
        decoder_layer_widths = [128, 128, 128, 128, 2]

    # define the model
    cnp = ConditionalNeuralProcess(encoder_layer_widths, decoder_layer_widths, attention,
                                   attention_params=attention_params,
                                   convolutional=convolutional, convolutional_params=convolutional_params,
                                   learning_rate=1e-4)
    cnp.load_weights(loading_path)
    test_cnp(cnp, data, convolutional=convolutional)
