import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import random


class Omniglot:
    """
    Use this class to generate the few shot learning datasets for N way classification
    """
    def __init__(self):

        # load all dataset as we manually select train/test split
        data, info = tfds.load(
            "omniglot",
            split='train+test',
            shuffle_files=True,
            as_supervised=True,
            with_info=True
        )

        def format_data(image, label):
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize(image, [28, 28])
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        def build_data(dataset, random_seed=0):
            """
            :param dataset: tfds object containing full dataset
            :param random_seed: used for reproducibility
            :return: train, test data which is randomly chosen by class
            """
            AUTOTUNE = tf.data.experimental.AUTOTUNE

            # keep this the same for reproducibility
            np.random.seed(random_seed)
            # randomly sample 1200 classes to use for training
            train_classes = np.random.choice(np.arange(1600), size=1200, replace=False)
            test_classes = np.setdiff1d(np.arange(1600), train_classes)

            # make into tensors
            train_classes = tf.constant(train_classes, dtype=tf.float32)
            test_classes = tf.constant(test_classes, dtype=tf.float32)

            # filter the dataset
            train_data = dataset.filter(lambda image, label: training_filter(label, allowed_labels=train_classes))
            test_data = dataset.filter(lambda image, label: training_filter(label, allowed_labels=test_classes))

            # rotate training set images and make them new classes
            # for labels take original class and add
            rotated_train_data = train_data.map(rotate_and_new_class, num_parallel_calls=AUTOTUNE)

            train_data = train_data.concatenate(rotated_train_data)
            train_classes = np.concatenate([train_classes, train_classes+1600])

            # combine into one dataset
            full_dataset = train_data.concatenate(test_data)

            return full_dataset, train_classes, test_classes

        def make_data_dict(dataset):
            AUTOTUNE = tf.data.experimental.AUTOTUNE
            data = {}
            for image, label in dataset.map(format_data, num_parallel_calls=AUTOTUNE):
                image = image.numpy()
                label = label.numpy()
                if label not in data:
                    data[label] = []
                data[label].append(image)

            return data

        full_dataset, self._training_classes, self._testing_classes = build_data(data)

        self._data_dict = make_data_dict(full_dataset)

    def get_mini_dataset(self, batch_size=10, shots=1, num_classes=5, testing=False):

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        # 20 - shots examples for the decoder

        decoder_labels = np.zeros(shape=(num_classes * (20 - shots), 1))
        decoder_images = np.zeros(shape=(num_classes * (20 - shots), 28, 28, 1))

        # shots examples for the encoder
        encoder_labels = np.zeros(shape=(shots * num_classes, 1))
        encoder_images = np.zeros(shape=(shots * num_classes, 28, 28, 1))

        if not testing:
            # randomly choose num_classes training labels
            labels = np.random.choice(self._training_classes, size=num_classes, replace=False)
        else:
            # randomly choose num_classes testing labels
            labels = np.random.choice(self._testing_classes, size=num_classes, replace=False)

        # loop through labels and create mini datasets
        for i, label in enumerate(labels):
            # assign 20 - shots of data for decoder, shots to encoder
            decoder_labels[i * (20 - shots): (i + 1) * (20 - shots)] = i
            encoder_labels[(np.arange(shots) * num_classes) + i] = i

            # now randomly choose 20 - shots images from this label
            labelled_images = self._data_dict[label]
            # shuffle images
            random.shuffle(labelled_images)
            decoder_images[i * (20 - shots): (i + 1) * (20 - shots)] = labelled_images[:-shots]
            encoder_images[(np.arange(shots) * num_classes) + i] = labelled_images[(20-shots):]

        encoder_dataset = tf.data.Dataset.from_tensor_slices((encoder_images.astype(np.float32),
                                                             encoder_labels.astype(np.int32)))

        decoder_dataset = tf.data.Dataset.from_tensor_slices((decoder_images.astype(np.float32),
                                                             decoder_labels.astype(np.int32)))

        # shuffle the datasets and organise with batch size
        num_steps = int((num_classes*(20 - shots)) / batch_size) + 1
        encoder_dataset = encoder_dataset.cache().repeat(num_steps).batch(num_classes * shots).prefetch(AUTOTUNE)
        decoder_dataset = decoder_dataset.cache().shuffle(num_classes*(20-shots)).batch(batch_size).prefetch(AUTOTUNE)

        return encoder_dataset, decoder_dataset


def training_filter(label, allowed_labels):
    is_allowed = tf.equal(allowed_labels, tf.cast(label, tf.float32))
    reduced = tf.reduce_sum(tf.cast(is_allowed, tf.float32))
    return tf.greater(reduced, tf.constant(0.))


def rotate_and_new_class(image, label):
    image = tf.image.rot90(image)
    label = 1600 + tf.squeeze(label)
    return image, label


if __name__ == "__main__":
    test = Omniglot()
    loop_train, loop_test = test.get_mini_dataset()



