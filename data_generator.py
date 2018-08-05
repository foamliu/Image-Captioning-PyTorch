# encoding=utf-8
import os
import pickle

import keras
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.preprocessing import sequence
from keras.utils import Sequence

from config import batch_size, max_token_length, vocab_size, train_image_folder, valid_image_folder, img_size


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage

        with tf.device("/cpu:0"):
            self.image_model = VGG16(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet',
                                     pooling=None)

        vocab = pickle.load(open('data/vocab_train.p', 'rb'))
        self.idx2word = sorted(vocab)
        self.word2idx = dict(zip(self.idx2word, range(len(vocab))))

        if usage == 'train':
            samples_path = 'data/samples_train.p'
            self.image_folder = train_image_folder
            self.image_folder = train_image_folder
        else:
            samples_path = 'data/samples_valid.p'
            self.image_folder = valid_image_folder
            self.image_folder = valid_image_folder

        samples = pickle.load(open(samples_path, 'rb'))
        self.samples = samples
        np.random.shuffle(self.samples)

    def __len__(self):
        return int(np.ceil(len(self.samples) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (len(self.samples) - i))
        batch_image_input = np.empty((length, img_size, img_size, 3), dtype=np.float32)
        batch_target = np.empty((length, vocab_size), dtype=np.int32)
        text_input = []

        for i_batch in range(length):
            sample = self.samples[i + i_batch]
            image_id = sample['image_id']
            filename = os.path.join(self.image_folder, image_id)
            img = image.load_img(filename, target_size=(img_size, img_size))
            img = image.img_to_array(img)
            batch_image_input[i_batch] = img

            text_input.append(sample['input'])
            batch_target[i_batch] = keras.utils.to_categorical(sample['output'], vocab_size)

        batch_text_input = sequence.pad_sequences(text_input, maxlen=max_token_length, padding='post')
        batch_image_input = preprocess_input(batch_image_input)
        batch_image_feature = self.image_model.predict(batch_image_input)
        return [batch_image_feature, batch_text_input], batch_target

    def on_epoch_end(self):
        np.random.shuffle(self.samples)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')
