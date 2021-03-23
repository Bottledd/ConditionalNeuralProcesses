from tensorflow.keras.preprocessing.image import img_to_array
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from Utils.doubleMNIST import process_images, format_context_points_image
from cnpModel.ConditionalNeuralProcesses import ConditionalNeuralProcess
import os
from PIL import Image
import numpy as np


def fix_images(direct="Figures/DoubleMNist/"):
    target_size = (128, 128)
    for img_name in os.listdir(direct):
        if img_name[-4:] == ".png":
            im = Image.open(direct + img_name)
            im = im.resize(target_size, resample=Image.BOX)
            im.save(direct + img_name)
    print("resizing done !")


def test_cnp(cnp, test_data, context_points=10, convolutional=False, type=None):
    # grab a random image from the test set
    image = test_data.reshape(1, test_data.shape[0], test_data.shape[1])
    # context_points = int(context_points * (test_data.shape[0]**2))

    # process image
    processed = process_images(image, context_points=context_points, convolutional=False)
    if convolutional:
        processed = process_images(image, context_points=context_points, convolutional=convolutional,
                               pre_defined=True, pre_defined_values=processed.Inputs[0])

    # evaluate cnp on image
    means, stds = cnp(processed.Inputs)

    # reshape for plotting
    predictive_mean = tf.reshape(means, (image.shape[-2], image.shape[-1]))
    predictive_stds = tf.reshape(stds, (image.shape[-2], image.shape[-1]))
    if not convolutional:
        context_image = format_context_points_image(processed.Inputs)
    else:
        context_image = format_context_points_image(processed.Inputs, convolution=True)
        # context_image = processed.Inputs[1][0]

    # plot stuff
    plt.figure('context')
    plt.imshow(context_image)
    plt.title('Context')
    plt.tight_layout()
    im = Image.fromarray(context_image.reshape((64,64,3)).astype(np.uint8))
    im.save(f"Figures/DoubleMNist/{type}_context.png")
    plt.savefig(f"Figures/DoubleMNist/{type}_context.jpg", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure('means')
    plt.imshow(predictive_mean, cmap='gray')
    plt.title('Predictive Mean')
    plt.tight_layout()
    im = Image.fromarray(np.array(np.clip(predictive_mean, a_min=0, a_max=1) * 255).reshape(64, 64).astype(np.uint8))
    im.save(f"Figures/DoubleMNist/{type}_mean.png")
    plt.savefig(f"Figures/DoubleMNist/{type}_means.jpg", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure('stds')
    plt.imshow(predictive_stds, cmap='gray')
    plt.title('Predictive Std')
    plt.tight_layout()
    im = Image.fromarray(np.array(np.clip(predictive_stds,a_min=0, a_max=1) * 255).reshape(64, 64).astype(np.uint8))
    im.save(f"Figures/DoubleMNist/{type}_stds.png")
    plt.savefig(f"Figures/DoubleMNist/{type}_stds.jpg", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure('actual')
    plt.imshow(processed.Targets.reshape((image.shape[-2], image.shape[-1])), cmap='gray')
    im = Image.fromarray(np.array(processed.Targets.reshape((image.shape[-2], image.shape[-1])) * 255).reshape(64, 64).astype(np.uint8))
    im.save(f"Figures/DoubleMNist/{type}_actual.png")
    plt.title('Actual')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    img = Image.open("DataSets/DoubleMNIST/3_48.png")
    new_img = img_to_array(img)
    load = True
    test = True
    attention = True
    convolutional = False
    attention_params = {}
    convolutional_params = {}
    encoder_layer_widths = []
    decoder_layer_widths = []
    if attention:
        loading_path = os.path.join(os.getcwd(), "saved_models/MNIST/ATTNCNP_400K_24B/")
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
    else:
        loading_path = os.path.join(os.getcwd(), "saved_models/MNIST/CNP_100k_64B/")

        encoder_layer_widths = [128, 128, 128]
        decoder_layer_widths = [128, 128, 128, 128, 2]

    # define the model
    cnp = ConditionalNeuralProcess(encoder_layer_widths, decoder_layer_widths, attention,
                                   attention_params=attention_params,
                                   convolutional=convolutional, convolutional_params=convolutional_params)
    if load:
        cnp.load_weights(loading_path)
    if test:
        test_cnp(cnp, new_img, context_points=int(64**2 * 0.1), convolutional=convolutional,
                 type="convCNP" if convolutional else "AttnCNP" if attention else "CNP")
        fix_images()
