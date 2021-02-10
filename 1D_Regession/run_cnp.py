import numpy as np
import collections
import matplotlib.pyplot as plt
from GaussianProcessSampler import GaussianProcess
from ConditionalNeuralProcess import ConditionalNeuralProcess
import tensorflow as tf


def train(max_iters=int(1e5)):
    # tf.config.run_functions_eagerly(True)
    gp_train = GaussianProcess(64, 10, testing=False)
    cnp = ConditionalNeuralProcess(128)
    loss = []
    for i in range(1, max_iters):
        data_train = gp_train.generate_curves()
        loss.append(cnp.train_step(data_train.Inputs, data_train.Targets))
        if i == 1 or i % 10 == 0:
            print(f"Loss at iteration {i} is: {loss[i-1]}")

    return cnp, loss


def test(cnp, data):
    gp_test = GaussianProcess(1, 10, testing=True)
    data_test = gp_test.generate_curves()
    means, stds = cnp(data.Inputs, data.Targets)
    return means, stds


if __name__ == "__main__":

    cnp, loss = train()
    plt.plot(loss)
    plt.show()

    gp = GaussianProcess(1, 10, testing=True)
    data = gp.generate_curves()
    means, stds = cnp(data.Inputs)
    gp.plot_fit(data.Inputs, data.Targets, means.numpy(), stds.numpy())
