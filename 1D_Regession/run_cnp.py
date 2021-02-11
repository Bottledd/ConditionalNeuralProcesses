import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
from GaussianProcessSampler import GaussianProcess
from ConditionalNeuralProcess import ConditionalNeuralProcess
import tensorflow as tf
import os
import time


def train(cnp, max_iters=100000):
    # tf.config.run_functions_eagerly(True)
    gp_train = GaussianProcess(64, 10, testing=False)
    loss = []
    start = time.perf_counter()
    for i in range(1, max_iters):
        # TODO: Change this to a generate_samples function so we can apply it to image data
        data_train = gp_train.generate_curves()
        loss.append(cnp.train_step(data_train.Inputs, data_train.Targets))
        if i == 1 or i % 1000 == 0:
            print(f"Running avg (50) loss at iteration {i} is: {np.mean(loss[i-1000:])}")
        # if i > 5000:
        #     # check any progress actually being made
        #     # (just to make computationally less expensive)
        #     if np.mean(loss[-100:-50]) - np.mean(loss[-50:]) < 0:
        #         break
    end = time.perf_counter()
    return cnp, loss, end-start


if __name__ == "__main__":
    load = True
    save = True
    training = True
    saving_path = os.path.join(os.getcwd(), "saved_models/low_var")
    loading_path = os.path.join(os.getcwd(), "saved_models/")
    cnp = ConditionalNeuralProcess(128)
    if load:
        cnp.load_weights(loading_path)
    if training:
        cnp, loss, total_runtime = train(cnp)
        print(total_runtime)
        avg_loss = pd.Series(loss).rolling(window=1000).mean().iloc[1000 - 1:].values
        plt.figure('loss')
        plt.plot(avg_loss)
        plt.show()
    if save:
        cnp.save_weights(saving_path)

    gp = GaussianProcess(1, 10, testing=True)
    data = gp.generate_curves()
    means, stds = cnp(data.Inputs)
    gp.plot_fit(data.Inputs, data.Targets, means.numpy(), stds.numpy())
