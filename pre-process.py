import json
import os
import pickle
import zipfile

import jieba
import keras
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import (load_img, img_to_array)
from tqdm import tqdm

from config import img_rows, img_cols
from config import start_word, stop_word, unknown_word
from config import train_annotations_filename
from config import train_folder, valid_folder, test_a_folder, test_b_folder
from config import train_image_folder, valid_image_folder, test_a_image_folder, test_b_image_folder
from config import valid_annotations_filename

image_model = ResNet50(include_top=False, weights='imagenet', pooling='avg')


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def extract(folder):
    filename = '{}.zip'.format(folder)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


def encode_images(usage):
    encoding = {}

    if usage == 'train':
        image_folder = train_image_folder
    elif usage == 'valid':
        image_folder = valid_image_folder
    elif usage == 'test_a':
        image_folder = test_a_image_folder
    else:  # usage == 'test_b':
        image_folder = test_b_image_folder

    batch_size = 256
    names = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    num_batches = int(np.ceil(len(names) / float(batch_size)))

    print('encoding {} images'.format(usage))
    for idx in tqdm(range(num_batches)):
        i = idx * batch_size
        length = min(batch_size, (len(names) - i))
        image_input = np.empty((length, img_rows, img_cols, 3))
        for i_batch in range(length):
            image_name = names[i + i_batch]
            filename = os.path.join(image_folder, image_name)
            img = load_img(filename, target_size=(img_rows, img_cols))
            img_array = img_to_array(img)
            img_array = keras.applications.resnet50.preprocess_input(img_array)
            image_input[i_batch] = img_array

        preds = image_model.predict(image_input)

        for i_batch in range(length):
            image_name = names[i + i_batch]
            encoding[image_name] = preds[i_batch]

    filename = 'data/encoded_{}_images.p'.format(usage)
    with open(filename, 'wb') as encoded_pickle:
        pickle.dump(encoding, encoded_pickle)


def build_train_vocab():
    annotations_path = os.path.join(train_folder, train_annotations_filename)

    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    print('building {} train vocab')
    vocab = set()
    for a in tqdm(annotations):
        caption = a['caption']
        for c in caption:
            seg_list = jieba.cut(c)
            for word in seg_list:
                vocab.add(word)

    vocab.add(start_word)
    vocab.add(stop_word)
    vocab.add(unknown_word)

    filename = 'data/vocab_train.p'
    with open(filename, 'wb') as encoded_pickle:
        pickle.dump(vocab, encoded_pickle)


def build_samples(usage):
    if usage == 'train':
        annotations_path = os.path.join(train_folder, train_annotations_filename)
    else:
        annotations_path = os.path.join(valid_folder, valid_annotations_filename)
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    vocab = pickle.load(open('data/vocab_train.p', 'rb'))
    idx2word = sorted(vocab)
    word2idx = dict(zip(idx2word, range(len(vocab))))

    print('building {} samples'.format(usage))
    samples = []
    for a in tqdm(annotations):
        image_id = a['image_id']
        caption = a['caption']
        for c in caption:
            seg_list = jieba.cut(c)
            input = []
            last_word = start_word
            for j, word in enumerate(seg_list):
                if word not in vocab:
                    word = unknown_word
                input.append(word2idx[last_word])
                samples.append({'image_id': image_id, 'input': list(input), 'output': word2idx[word]})
                last_word = word
            input.append(word2idx[last_word])
            samples.append({'image_id': image_id, 'input': list(input), 'output': word2idx[stop_word]})

    filename = 'data/samples_{}.p'.format(usage)
    with open(filename, 'wb') as f:
        pickle.dump(samples, f)


if __name__ == '__main__':
    # parameters
    ensure_folder('data')

    # if not os.path.isdir(train_image_folder):
    extract(train_folder)

    # if not os.path.isdir(valid_image_folder):
    extract(valid_folder)

    # if not os.path.isdir(test_a_image_folder):
    extract(test_a_folder)

    # if not os.path.isdir(test_b_image_folder):
    extract(test_b_folder)

    if not os.path.isfile('data/encoded_train_images.p'):
        encode_images('train')

    if not os.path.isfile('data/encoded_valid_images.p'):
        encode_images('valid')

    if not os.path.isfile('data/encoded_test_a_images.p'):
        encode_images('test_a')

    if not os.path.isfile('data/encoded_test_b_images.p'):
        encode_images('test_b')

    if not os.path.isfile('data/vocab_train.p'):
        build_train_vocab()

    if not os.path.isfile('data/samples_train.p'):
        build_samples('train')

    if not os.path.isfile('data/samples_valid.p'):
        build_samples('valid')
