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

    model_weights_path = 'models/model.00-1.9058.hdf5'
    model = build_model()
    model.load_weights(model_weights_path)

    vocab = pickle.load(open('data/vocab_train.p', 'rb'))
    idx2word = sorted(vocab)
    word2idx = dict(zip(idx2word, range(len(vocab))))

    print(model.summary())

    encoded_test_a = pickle.load(open('data/encoded_test_a_images.p', 'rb'))

    names = [f for f in encoded_test_a.keys()]

    samples = random.sample(names, 20)

    for i in range(len(samples)):
        image_name = samples[i]
        filename = os.path.join(test_a_image_folder, image_name)
        print('Start processing image: {}'.format(filename))
        image_input = np.zeros((1, 2048))
        image_input[0] = encoded_test_a[image_name]

        start_words = [start_word]
        while True:
            text_input = [word2idx[i] for i in start_words]
            text_input = sequence.pad_sequences([text_input], maxlen=max_token_length, padding='post')
            preds = model.predict([image_input, text_input])
            # print('output.shape: ' + str(output.shape))
            word_pred = idx2word[np.argmax(preds[0])]
            start_words.append(word_pred)
            if word_pred == stop_word or len(start_word) > max_token_length:
                break

        sentence = ' '.join(start_words[1:-1])
        print(sentence)

        img = cv.imread(filename)
        img = cv.resize(img, (img_rows, img_cols), cv.INTER_CUBIC)
        if not os.path.exists('images'):
            os.makedirs('images')
        cv.imwrite('images/{}_image.png'.format(i), img)

    K.clear_session()
