# encoding=utf-8
import os
import pickle

import keras
import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing.image import (load_img, img_to_array)
from keras.utils import Sequence

from config import image_h, image_w, batch_size, max_token_length, vocab_size, train_image_folder, valid_image_folder, \
    cnn_type

if cnn_type == 'vgg16':
    from keras.applications.vgg16 import preprocess_input as preprocess_input
elif cnn_type == 'vgg19':
    from keras.applications.vgg19 import preprocess_input as preprocess_input
elif cnn_type == 'resnet':
    from keras.applications.resnet50 import preprocess_input as preprocess_input


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage

        vocab = pickle.load(open('data/vocab_train.p', 'rb'))
        self.idx2word = sorted(vocab)
        self.word2idx = dict(zip(self.idx2word, range(len(vocab))))

        if usage == 'train':
            samples_path = 'data/samples_train.p'
            self.image_folder = train_image_folder
        else:
            samples_path = 'data/samples_valid.p'
            self.image_folder = valid_image_folder

        samples = pickle.load(open(samples_path, 'rb'))
        self.samples = samples
        np.random.shuffle(self.samples)

    def __len__(self):
        return int(np.ceil(len(self.samples) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        batch_image_input = np.empty((batch_size, image_h, image_w, 3), dtype=np.float32)
        caption_target = np.empty((batch_size, vocab_size), dtype=np.int32)
        text_input = []

        for i_batch in range(batch_size):
            sample = self.samples[i + i_batch]
            image_id = sample['image_id']
            filename = os.path.join(self.image_folder, str(image_id) + '.jpg')
            img = load_img(filename, target_size=(image_h, image_w))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            batch_image_input[i_batch] = img_array

            text_input.append(sample['input'])
            caption_target[i_batch] = keras.utils.to_categorical(sample['output'], vocab_size)

        batch_text_input = sequence.pad_sequences(text_input, maxlen=max_token_length, padding='post')
        return [batch_image_input, batch_text_input], caption_target

    def on_epoch_end(self):
        np.random.shuffle(self.samples)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')
