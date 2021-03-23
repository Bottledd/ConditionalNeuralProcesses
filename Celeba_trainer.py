import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cnpModel.ConditionalNeuralProcesses import ConditionalNeuralProcess
import tensorflow as tf
import os
import time
from Utils.celebaProcessor import process_images, format_context_points_image
from tqdm import tqdm


def data_generator(directory, type_data, batch_size=64, target_size=(28, 28)):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    generator = datagen.flow_from_directory(directory, batch_size=batch_size,
                                            target_size=target_size, shuffle=True, classes=[type_data])
    while True:
        batch = next(generator)
        yield batch


def train(cnp, train_data, batch_size=64, max_iters=50000, convolutional=False):
    """
    Train with batches of size 30 since 60000 images total, 4000 iterations
    Randomly sample number of context  points
    """
    # tf.config.run_functions_eagerly(True)
    loss = []
    start = time.perf_counter()

    for i in tqdm(range(1, max_iters + 1)):
        # generate a batch
        batch = next(train_data)[0]
        img_shape = np.array(batch).shape[1:]
        # get context between 2 and 30%
        num_context = np.random.randint(2, int(img_shape[0] ** 2 * 0.3))
        data_train = process_images(batch, context_points=num_context, convolutional=convolutional)
        # process current batch
        loss.append(cnp.train_step(data_train.Inputs, data_train.Targets))
        if i % 1000 == 0:
            print(f'The running avg loss at iteration {i} is: {np.mean(loss[-1000:])}')

    end = time.perf_counter()
    return cnp, loss, end - start


def test_cnp(cnp, test_data, context_ratio=0.2, convolutional=False, attention=False):
    # grab a random image from the test set
    batch = next(test_data)[0]
    img_shape = np.array(batch).shape[1:]

    # process image
    processed = process_images(batch, context_points=int(context_ratio * img_shape[0] ** 2),
                               convolutional=convolutional)

    # evaluate cnp on image
    means, stds = cnp(processed.Inputs)

    # reshape for plotting
    predictive_mean = tf.reshape(means, img_shape)
    predictive_stds = tf.reshape(stds, img_shape)
    if not (convolutional):
        context_image = format_context_points_image(processed.Inputs)
    else:
        context_image = processed.Inputs[1][0]

    print(context_image.shape)
    # plot stuff
    plt.figure('context')
    plt.imshow(context_image, cmap='gray')
    plt.title('Context')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"Figures/CelebA/{'Conv' if convolutional else 'Atten' if attention else ''}CNP/context.svg")

    # plot stuff
    plt.figure('means')
    plt.imshow(predictive_mean, cmap='gray')
    plt.title('Predictive Mean')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"Figures/CelebA/{'Conv' if convolutional else 'Atten' if attention else ''}CNP/means.svg")

    plt.figure('stds')
    plt.imshow(predictive_stds, cmap='gray')
    plt.title('Predictive Std')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"Figures/CelebA/{'Conv' if convolutional else 'Atten' if attention else ''}CNP/stds.svg")

    plt.figure('actual')
    plt.imshow(processed.Targets.reshape(img_shape[0], img_shape[1], img_shape[2]), cmap='gray')
    plt.title('Actual')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"Figures/CelebA/{'Conv' if convolutional else 'Atten' if attention else ''}CNP/actual.svg")


if __name__ == "__main__":
    load = False
    save = False
    training = True
    test = True
    attention = False
    convolutional = True
    assert attention + convolutional != 2, "Can't have both worlds!"

    iterations = 3
    batching = 24

    attention_params = {}
    convolutional_params = {}
    encoder_layer_widths = []
    decoder_layer_widths = []
    if attention:
        loading_path = os.path.join(os.getcwd(), "saved_models/CelebA/attention_100kiterations_batch8/")
        saving_path = os.path.join(os.getcwd(),
                                   f"saved_models/CelebA/ATTNCNP{(iterations // 1000) * batching}_samples/")
        encoder_layer_widths = [128, 128]
        decoder_layer_widths = [64, 64, 64, 64, 6]
        attention_params = {"embedding_layer_width": 128, "num_heads": 8, "num_self_attention_blocks": 2}
    elif convolutional:
        # parameters for convolutional
        kernel_size_encoder = 9
        kernel_size_decoder = 5
        convolutional_params = {"number_filters": 128, "kernel_size_encoder": 9, "kernel_size_decoder": 5,
                                "number_residual_blocks": 4, "convolutions_per_block": 1, "output_channels": 3}
        loading_path = os.path.join(os.getcwd(), "saved_models/CelebA/CONVCNP_100k_64B/")
        saving_path = os.path.join(os.getcwd(),
                                   f"saved_models/CelebA/CONVCNP_{(iterations // 1000) * batching}_samples/")
    else:
        loading_path = os.path.join(os.getcwd(), "saved_models/CelebA/CNP_100k_64B/")
        saving_path = os.path.join(os.getcwd(), f"saved_models/CelebA/CNP_{(iterations // 1000) * batching}_samples/")

        encoder_layer_widths = [128, 128, 128]
        decoder_layer_widths = [128, 128, 128, 128, 6]

    # define the model
    cnp = ConditionalNeuralProcess(encoder_layer_widths, decoder_layer_widths, attention,
                                   attention_params=attention_params,
                                   convolutional=convolutional, convolutional_params=convolutional_params,
                                   learning_rate=1e-4)

    # make a generator for the data
    target_size = (32, 32)
    train_data = data_generator('DataSets/CelebA', 'train', batch_size=batching, target_size=target_size)
    test_data = data_generator('DataSets/CelebA', 'test', batch_size=1, target_size=target_size)

    if load:
        cnp.load_weights(loading_path)
    if training:
        cnp, loss, total_runtime = train(cnp, train_data, max_iters=iterations, batch_size=batching,
                                         convolutional=convolutional)
        print(total_runtime)
        file_names = f"output/CelebA/training_loss/loss_{'Conv' if convolutional else 'Atten' if attention else ''}CNP"
        with open(file_names + ".txt", 'w') as file:
            for listitem in loss:
                file.write('%s\n' % listitem)
        avg_loss = pd.Series(loss).rolling(window=1000).mean().iloc[1000 - 1:].values
        plt.figure('loss')
        plt.plot(avg_loss)
        plt.savefig(file_names + '.eps', bbox_inches='tight')
    if save:
        cnp.save_weights(saving_path, overwrite=False)
    if test:
        test_cnp(cnp, test_data, convolutional=convolutional, attention=attention)
