import json
import pickle
import zipfile
from collections import Counter

import jieba
from tqdm import tqdm

from config import *
from utils import ensure_folder


def extract(folder):
    filename = '{}.zip'.format(folder)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


def create_input_files(split, min_word_freq=3):
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
    elif split == 'valid':
        json_path = valid_annotations_filename
    elif split == 'test-a':
        json_path = test_a_annotations_filename
    else:
        json_path = test_b_annotations_filename

    # Read JSON
    with open(json_path, 'r') as j:
        samples = json.load(j)

    # Read image paths and captions for each image
    word_freq = Counter()

    for sample in tqdm(samples):
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

    print(len(word_map))
    print(words[:10])

    # Save word map to a JSON
    with open(os.path.join(data_folder, 'WORDMAP_' + split + '.json'), 'w') as j:
        json.dump(word_map, j)


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


if __name__ == '__main__':
    # parameters
    ensure_folder('data')

    if not os.path.isdir(train_image_folder):
        extract(train_folder)

    if not os.path.isdir(valid_image_folder):
        extract(valid_folder)

    if not os.path.isdir(test_a_image_folder):
        extract(test_a_folder)

    if not os.path.isdir(test_b_image_folder):
        extract(test_b_folder)

    create_input_files('train')
    create_input_files('valid')
    # create_input_files('test-a')
    # create_input_files('test-b')
