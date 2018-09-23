import json
import pickle
import zipfile
from collections import Counter
from random import seed, choice

import h5py
import jieba
import numpy as np
from scipy.misc import imread, imresize
from tqdm import tqdm

from config import *
from utils import ensure_folder


def extract(folder):
    filename = '{}.zip'.format(folder)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


def create_input_files(split, captions_per_image=5, min_word_freq=3, output_folder=data_folder, max_len=40):
    """
    Creates input files for training, validation, and test data.
    :param json_path: path of JSON file with captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    if split == 'train':
        json_path = train_annotations_filename
        image_folder = train_image_folder
    elif split == 'valid':
        json_path = valid_annotations_filename
        image_folder = valid_image_folder

    # Read JSON
    with open(json_path, 'r') as j:
        samples = json.load(j)

    # Read image paths and captions for each image
    word_freq = Counter()

    for sample in tqdm(samples):
        captions = []
        caption = sample['caption']
        for c in caption:
            seg_list = jieba.cut(c, cut_all=True)
            # Update word frequency
            word_freq.update(seg_list)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + split + '.json'), 'w') as j:
        json.dump(word_map, j)
    #
    # # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    # seed(123)
    # for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
    #                                (val_image_paths, val_image_captions, 'VAL'),
    #                                (test_image_paths, test_image_captions, 'TEST')]:
    #
    #     with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
    #         # Make a note of the number of captions we are sampling per image
    #         h.attrs['captions_per_image'] = captions_per_image
    #
    #         # Create dataset inside HDF5 file to store images
    #         images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')
    #
    #         print("\nReading %s images and captions, storing to file...\n" % split)
    #
    #         enc_captions = []
    #         caplens = []
    #
    #         for i, path in enumerate(tqdm(impaths)):
    #
    #             # Sample captions
    #             if len(imcaps[i]) < captions_per_image:
    #                 captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
    #             else:
    #                 captions = sample(imcaps[i], k=captions_per_image)
    #
    #             # Sanity check
    #             assert len(captions) == captions_per_image
    #
    #             # Read images
    #             img = imread(impaths[i])
    #             if len(img.shape) == 2:
    #                 img = img[:, :, np.newaxis]
    #                 img = np.concatenate([img, img, img], axis=2)
    #             img = imresize(img, (256, 256))
    #             img = img.transpose(2, 0, 1)
    #             assert img.shape == (3, 256, 256)
    #             assert np.max(img) <= 255
    #
    #             # Save image to HDF5 file
    #             images[i] = img
    #
    #             for j, c in enumerate(captions):
    #                 # Encode captions
    #                 enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
    #                     word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
    #
    #                 # Find caption lengths
    #                 c_len = len(c) + 2
    #
    #                 enc_captions.append(enc_c)
    #                 caplens.append(c_len)
    #
    #         # Sanity check
    #         assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)
    #
    #         # Save encoded captions and their lengths to JSON files
    #         with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
    #             json.dump(enc_captions, j)
    #
    #         with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
    #             json.dump(caplens, j)


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
            input = [word2idx[start_word]]
            output = []
            for j, word in enumerate(seg_list):
                if word not in vocab:
                    word = unknown_word
                input.append(word2idx[word])
                output.append(word2idx[word])

            output.append(word2idx[stop_word])
            samples.append({'image_id': image_id, 'input': list(input), 'output': list(output)})

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

    create_input_files('train')

    create_input_files('valid')

    # if not os.path.isfile('data/vocab_train.p'):
    #     build_train_vocab()
    #
    # if not os.path.isfile('data/samples_train.p'):
    #     build_samples('train')
    #
    # if not os.path.isfile('data/samples_valid.p'):
    #     build_samples('valid')
