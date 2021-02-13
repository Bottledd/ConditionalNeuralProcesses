import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cnpModel.ConditionalNeuralProcess import ConditionalNeuralProcess
import tensorflow as tf
import os
import time
from datetime import datetime
from Utils.imageProcessor import process_images
from tqdm import tqdm


def train(cnp, data, max_iters=25000):
    """
    Train with batches of size 30 since 60000 images total, 4000 iterations
    Randomly sample number of context  points
    """
    # tf.config.run_functions_eagerly(True)
    loss = []
    start = time.perf_counter()

    for i in range(1, max_iters+1):
        #choice = np.random.choice([5, 10, 100, 250, 500], replace=True)
        num_context = np.random.randint(3, 784)
        # grab image batch
        batch = data[(i-1)*64 % 60000: i*64 % 60000, :]
        data_train = process_images(batch, context_points=num_context)

        # process current batch
        loss.append(cnp.train_step(data_train.Inputs, data_train.Targets))
        if i % 1 == 0:
            print(f'The loss at iteration {i} is: {loss[i-1]}')

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
    return cnp, loss, end-start


def test_cnp(cnp, test_data):
    # grab a random image from the test set
    image = test_data[np.random.randint(0, test_data.shape[0]+1)].reshape(1,28,28)

    # process image
    processed = process_images(image, context_points=781)

    # evaluate cnp on image
    means , stds = cnp(processed.Inputs)

    # reshape for plotting
    predictive_mean = tf.reshape(means, (28,28))
    predictive_stds = tf.reshape(stds, (28, 28))

    # plot stuff
    plt.figure('means')
    plt.matshow(predictive_mean)
    plt.title('Predictive Mean')
    plt.show()

    plt.figure('stds')
    plt.matshow(predictive_stds)
    plt.title('Predictive Std')
    plt.show()

    plt.figure('actual')
    plt.matshow(processed.Targets.reshape(28,28))
    plt.title('Actual')
    plt.show()


if __name__ == "__main__":
    (train_data, _), (test_data, _) = tf.keras.datasets.mnist.load_data(
        path='mnist.npz'
    )
    load = False
    save = True
    training = True
    test = True
    loading_path = os.path.join(os.getcwd(), "saved_models/ImageNET/2021_02_13-09_52_23_AM/")
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