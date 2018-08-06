# import the necessary packages
import os
import pickle
import random

import cv2 as cv
import keras.backend as K
import numpy as np
from keras.preprocessing import sequence

from config import max_token_length, start_word, stop_word, test_a_image_folder, img_rows, img_cols
from model import build_model

if __name__ == '__main__':
    channel = 3

    model_weights_path = 'models/model.00-2.1514.hdf5'
    model = build_model()
    model.load_weights(model_weights_path)

    vocab = pickle.load(open('data/vocab_train.p', 'rb'))
    idx2word = sorted(vocab)
    word2idx = dict(zip(idx2word, range(len(vocab))))

    print(model.summary())

    encoded_test_a = pickle.load(open('data/encoded_test_a_images.p', 'rb'))

    names = [f for f in encoded_test_a.keys()]

    samples = random.sample(names, 1)

    for i in range(len(samples)):
        image_name = samples[i]
        filename = os.path.join(test_a_image_folder, image_name)
        print('Start processing image: {}'.format(filename))
        image_input = np.zeros((1, 7, 7, 512))
        image_input[0] = encoded_test_a[image_name]

        start_words = [start_word]
        alpha_list = []
        while True:
            text_input = [word2idx[i] for i in start_words]
            text_input = sequence.pad_sequences([text_input], maxlen=max_token_length, padding='post')
            preds = model.predict([image_input, text_input])
            caption = preds[0]  # [1, vocab_size]
            print('caption.shape: ' + str(caption.shape))
            alpha = preds[1]  # [1, L]
            print('alpha.shape: ' + str(alpha.shape))
            alpha = np.reshape(alpha, (7, 7))
            alpha_list.append(alpha_list)
            word_pred = idx2word[int(np.argmax(caption[0]))]
            start_words.append(word_pred)
            if word_pred == stop_word or len(start_word) > max_token_length:
                break

        original = cv.imread(filename)
        original = cv.resize(original, (img_rows, img_cols), cv.INTER_CUBIC)
        cv.imwrite('images/{}_image.png'.format(i), original)

        sentence = ' '.join(start_words[1:-1])
        print(sentence)

        for j, alpha in enumerate(alpha_list):
            alpha = alpha / np.max(alpha)
            alpha_image = (alpha * 255.).astype(np.uint8)
            alpha_image = cv.resize(alpha_image, (img_rows, img_cols), cv.INTER_CUBIC)
            kernel = np.ones((5, 5), np.float32) / 25
            image = cv.filter2D(alpha_image, -1, kernel)
            cv.imwrite('alpha_{}_{}.png'.format(i, j), alpha_image)

    K.clear_session()
