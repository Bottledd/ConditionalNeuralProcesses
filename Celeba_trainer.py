import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cnpModel.ConditionalNeuralProcess import ConditionalNeuralProcess
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import time
from datetime import datetime
from Utils.celebaProcessor import process_images
from tqdm import tqdm

def data_generator(directory, type_data, batch_size=64, target_size=(28,28)):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    generator = datagen.flow_from_directory(directory,batch_size=batch_size, target_size=target_size, shuffle = True, classes=[type_data])
    while True:
        batch = next(generator)
        yield batch


def train(cnp, train_data, batch_size=64, max_iters=50000):
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
        # generate a batch
        batch = next(train_data)[0]

        data_train = process_images(batch, context_points=num_context)

        # process current batch
        loss.append(cnp.train_step(data_train.Inputs, data_train.Targets))
        if i % 1000 == 0:
            print(f'The running avg loss at iteration {i} is: {np.mean(loss[-1000:])}')

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
    batch = next(test_data)[0]
    img_shape = np.array(batch).shape[1:]

    # process image
    processed = process_images(batch, context_points=40)

    # evaluate cnp on image
    means , stds = cnp(processed.Inputs)

    # reshape for plotting
    predictive_mean = tf.reshape(means, img_shape)
    predictive_stds = tf.reshape(stds, img_shape)

    # plot stuff
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
    plt.imshow(processed.Targets.reshape(img_shape[0],img_shape[1],img_shape[2]), cmap='gray')
    plt.title('Actual')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    load = False
    save = False
    training = True
    test = True
    loading_path = os.path.join(os.getcwd(), "saved_models/Celeba/2021_02_14-12_45_56_PM/")
    saving_path = os.path.join(os.getcwd(), "saved_models/Celeba/")
    cnp = ConditionalNeuralProcess(128,3)

    #make a generator for the data
    train_data = data_generator('data/celeba', 'train', batch_size = 64)
    test_data = data_generator('data/celeba', 'test', batch_size = 1)
    if load:
        cnp.load_weights(loading_path)
    if training:
        cnp, loss, total_runtime = train(cnp, train_data, max_iters=1000)
        print(total_runtime)
        avg_loss = pd.Series(loss).rolling(window=100).mean().iloc[100 - 1:].values
        plt.figure('loss')
        plt.plot(avg_loss)
        plt.show()
    if save:
        current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        cnp.save_weights("saved_models/Celeba/" + current_time + "/", overwrite=False)

    if test:
        test_cnp(cnp, test_data)