import os
import numpy as np
from tqdm import tqdm
from Utils.omniglotDataSet import Omniglot
from cnpModel.cnpClassification import ClassificationCNP
import tensorflow as tf
import random

def train_model(model, dataset, shots, num_classes, epochs):
    batch_size = 4*num_classes
    compliant_batch_size = batch_size - (batch_size % 8)
    big_loss = []
    for epoch in tqdm(range(epochs)):
        loop_train, loop_test = dataset.get_mini_dataset(shots=shots, num_classes=num_classes, batch_size=compliant_batch_size)
        loss_list = []
        accuracy_list = []
        for step, (batchX, batchy) in enumerate(zip(loop_train, loop_test)):
            # encoder_shuffle = np.arange(19)
            # np.random.shuffle(encoder_shuffle)
            # decoder_shuffle = np.arange(15)
            # np.random.shuffle(decoder_shuffle)
            # shuffled_x = tf.convert_to_tensor(batchX[0].numpy()[encoder_shuffle]), tf.convert_to_tensor(batchX[1].numpy()[encoder_shuffle])
            # shuffled_y = tf.convert_to_tensor(batchy[0].numpy()[decoder_shuffle]), tf.convert_to_tensor(batchy[1].numpy()[decoder_shuffle])
            # loss, accuracy = model.train_step(shuffled_x, shuffled_y)
            loss, accuracy = model.train_step(batchX, batchy)
            loss_list.append(loss)
            big_loss.append(loss)
            accuracy_list.append(accuracy)

        print(
            f"Epoch {epoch}: Loss {np.round(np.mean(loss_list), decimals=2)},\tAccuracy {np.round(100 * np.mean(accuracy_list), decimals=1)},"
            f"\tOne-shot Accuracy {np.round(100 * accuracy_list[0], decimals=2)}")
    return big_loss


def test_model(model, dataset, shots, num_classes):
    big_acc = []
    for test_epoch in range(10000):
        test_context, test_target = dataset.get_mini_dataset(shots=shots, num_classes=num_classes, testing=True)
        accuracies = []
        for step, (batchX, batchy) in enumerate(zip(test_context, test_target)):
            logits = model(batchX, batchy)
            loss, accuracy = model._loss_func(batchy[1], logits)
            accuracies.append(accuracy)
            big_acc.append(accuracy)
        print(f"Epoch {test_epoch + 1} accuracy is {np.mean(accuracies)}")
    return big_acc


if __name__ == "__main__":
    load = True
    save = False
    training = False
    test = True

    data = Omniglot()

    num_classes = 5
    shots = 5
    epochs = 100000

    loading_path = os.path.join(os.getcwd(), f"saved_models/Classifier/{num_classes}Classes_{shots}Shot_{epochs}epochs/")
    saving_path = os.path.join(os.getcwd(), f"saved_models/Classifier/{num_classes}Classes_{shots}Shot_{epochs}epochs/")

    # loading_path = os.path.join(os.getcwd(), f"saved_models/Classifier/5Classes_1Shot_50000epochs/")
    # saving_path = os.path.join(os.getcwd(), f"saved_models/Classifier/{num_classes}Classes_{shots}Shot_150000epochs/")
    # first three layers are Conv2D and width means num filters, final layer is Dense
    encoder_widths = [32, 64, 128]
    decoder_widths = [128, 128, 128, 128, 1]

    cnp = ClassificationCNP(encoder_widths, decoder_widths, num_classes, use_dot_product=False)
    # cnp = ClassificationCNP(encoder_widths, decoder_widths)

    if load:
        cnp.load_weights(loading_path)
    if training:
        loss = train_model(cnp, data, shots, num_classes, epochs=epochs)
    if save:
        cnp.save_weights(saving_path, overwrite=False)
    if test:
        accuracy = test_model(cnp, data, shots, num_classes)
        print(np.mean(accuracy))
