# encoding=utf-8
import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing.image import (load_img, img_to_array)
from keras.utils import Sequence
from keras.utils import to_categorical

from config import image_h, image_w, batch_size, max_token_length, vocab_size, train_image_folder, valid_image_folder


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'train', 'valid', 'test-a', 'test-b'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size


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
        return len(self.samples) // batch_size

    def __getitem__(self, idx):
        i = idx * batch_size

        batch_image_input = np.empty((batch_size, image_h, image_w, 3), dtype=np.float32)
        batch_text_output = np.zeros((batch_size, max_token_length, vocab_size), dtype=np.int32)
        text_input = []

        for i_batch in range(batch_size):
            sample = self.samples[i + i_batch]
            image_id = sample['image_id']
            filename = os.path.join(self.image_folder, image_id)
            img = load_img(filename, target_size=(image_h, image_w))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            batch_image_input[i_batch] = img_array

            text_input.append(sample['input'])
            for j, idx in enumerate(sample['output']):
                batch_text_output[i_batch, j] = to_categorical(idx, vocab_size)

        batch_text_input = sequence.pad_sequences(text_input, maxlen=max_token_length, padding='post')
        return [batch_image_input, batch_text_input], batch_text_output

    def on_epoch_end(self):
        np.random.shuffle(self.samples)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')
