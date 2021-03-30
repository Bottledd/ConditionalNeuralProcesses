
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import time
from Utils.mnistProcessor import process_images, format_context_points_image
from cnpModel.ConditionalNeuralProcesses import ConditionalNeuralProcess
import os

def train(cnp, data, batch_size=64, max_iters=500000, convolutional=False):
    """
    Train with batches of size 30 since 60000 images total, 4000 iterations
    Randomly sample number of context  points
    """
    # tf.config.run_functions_eagerly(True)
    loss = []
    start = time.perf_counter()

    for i in tqdm(range(1, max_iters + 1)):
        # choice = np.random.choice([5, 10, 100, 250, 500], replace=True)
        num_context = np.random.randint(2, 784 * 0.3)
        # grab random image batch
        rand_batch = np.random.choice(np.arange(data.shape[0]), replace=False, size=batch_size)
        batch = data[rand_batch, :]

        data_train = process_images(batch, context_points=num_context, convolutional=convolutional)

        # process current batch
        loss.append(cnp.train_step(data_train.Inputs, data_train.Targets))
        if i % 10000 == 0:
            print(f'The running avg loss at iteration {i} is: {np.mean(loss[-10000:])}')

    end = time.perf_counter()
    return cnp, loss, end - start


def test_cnp(cnp, test_data, context_points=28*28, convolutional=False):
    # grab a random image from the test set
    image = test_data[np.random.randint(0, test_data.shape[0] + 1)].reshape(1, 28, 28)

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
    (train_data, _), (test_data, _) = tf.keras.datasets.mnist.load_data(
        path='mnist.npz'
    )
    load = True
    save = False
    training = False
    test = True
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
        #saving_path = os.path.join(os.getcwd(), f"saved_models/MNIST/ATTNCNP_{int(iterations / 1000)}k_{batching}B/")
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
    if load:
        cnp.load_weights(loading_path)
    if training:
        cnp, loss, total_runtime = train(cnp, train_data, max_iters=iterations, batch_size=batching,
                                         convolutional=convolutional)
        print(total_runtime)
        file_names = f"output/MNIST/training_loss/loss_{'Conv' if convolutional else 'Atten' if attention else ''}CNP"
        with open(file_names + ".txt", 'w') as file:
            for listitem in loss:
                file.write('%s\n' % listitem)
        avg_loss = pd.Series(loss).rolling(window=1000).mean().iloc[1000 - 1:].values
        plt.figure('loss')
        plt.plot(avg_loss)
        plt.savefig(file_names + '.eps')
    if save:
        cnp.save_weights(saving_path, overwrite=False)
    if test:
        test_cnp(cnp, test_data, convolutional=convolutional)
