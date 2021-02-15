import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import time
from Utils.imageProcessor import process_images, format_context_points_image
from cnpModel.ConditionalNeuralProcess import ConditionalNeuralProcess


def train(cnp, data, batch_size=64, max_iters=500000):
    """
    Train with batches of size 30 since 60000 images total, 4000 iterations
    Randomly sample number of context  points
    """
    # tf.config.run_functions_eagerly(True)
    loss = []
    start = time.perf_counter()

    for i in tqdm(range(1, max_iters+1)):
        #choice = np.random.choice([5, 10, 100, 250, 500], replace=True)
        num_context = np.random.randint(2, 784)
        # grab random image batch
        rand_batch = np.random.choice(np.arange(data.shape[0]), replace=False, size=batch_size)
        batch = data[rand_batch, :]

        data_train = process_images(batch, context_points=num_context)

        # process current batch
        loss.append(cnp.train_step(data_train.Inputs, data_train.Targets))
        if i % 10000 == 0:
            print(f'The running avg loss at iteration {i} is: {np.mean(loss[-10000:])}')

        # every 1000 iterations try new max contexts with big batch size to avoid overfitting
        # if i % 1000 == 0:
        #     data_val = generate_gp_samples(gp_train, gen_new_gp=True)
        #     val_loss = cnp.train_step(data_val.Inputs, data_val.Targets)
        #     print(f"Running avg (1000) loss at iteration {i} is: {np.mean(loss[-1000:])}")
        #     print(f"Validation Loss at iteration {i} is: {val_loss}")

        # # early stopping
        # if i > 2000:
        #     # check any progress actually being made
        #     # (just to make computationally less expensive)
        #     if np.mean(loss[-2000:-1000]) - np.mean(loss[-1000:]) < 0:
        #         break

    end = time.perf_counter()
    return cnp, loss, start-end


def test_cnp(cnp, test_data, context_points=100):
    # grab a random image from the test set
    image = test_data[np.random.randint(0, test_data.shape[0]+1)].reshape(1, 28, 28)

    # process image
    processed = process_images(image, context_points=context_points)

    # evaluate cnp on image
    means, stds = cnp(processed.Inputs)

    # reshape for plotting
    predictive_mean = tf.reshape(means, (28, 28))
    predictive_stds = tf.reshape(stds, (28, 28))

    # plot stuff
    plt.figure('context')
    image = format_context_points_image(processed.Inputs)
    plt.imshow(image)
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
    save = True
    training = True
    test = False
    loading_path = os.path.join(os.getcwd(), "saved_models/ImageNET/2021_02_15-07_25_31_PM/")
    saving_path = os.path.join(os.getcwd(), "saved_models/ImageNET/")
    cnp = ConditionalNeuralProcess(128)
    if load:
        cnp.load_weights(loading_path)
    if training:
        cnp, loss, total_runtime = train(cnp, train_data)
        print(total_runtime)
        avg_loss = pd.Series(loss).rolling(window=100).mean().iloc[100 - 1:].values
        plt.figure('loss')
        plt.plot(avg_loss)
        plt.show()
    if save:
        current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        cnp.save_weights("saved_models/ImageNET/" + current_time + "/", overwrite=False)

    if test:
        test_cnp(cnp, test_data)