import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Embedding, Dense, CuDNNLSTM, TimeDistributed, RepeatVector, Concatenate, Reshape, \
    multiply, Permute, Lambda
from keras.models import Model
from keras.utils import plot_model

from config import max_token_length, vocab_size, embedding_size, hidden_size, L, D


def generate_alphas(contexts, text_embedding):
    # tile and concatenate inputs
    contexts = Reshape([L * D])(contexts)
    # print(K.int_shape(contexts))
    text_embedding = Reshape([max_token_length * embedding_size])(text_embedding)
    # print(K.int_shape(text_embedding))
    concat_input = Concatenate(axis=-1)([contexts, text_embedding])

    # feed into MLP
    x = Dense(256, activation='relu')(concat_input)
    x = Dense(1024, activation='relu')(x)
    x = Dense(L, activation='softmax', name='alpha_output')(x)
    return x


def build_model():
    # word embedding
    text_input = Input(shape=(max_token_length,), dtype='int32')
    x = Embedding(input_dim=vocab_size, output_dim=embedding_size)(text_input)
    x = CuDNNLSTM(256, return_sequences=True)(x)
    text_embedding = TimeDistributed(Dense(embedding_size))(x)

    # image embedding
    image_input = Input(shape=(7, 7, 512))
    x = image_input
    contexts = Reshape([L, D])(x)

    alphas = generate_alphas(contexts, text_embedding)
    alpha_output = alphas
    alphas = RepeatVector(1)(alphas)
    alphas = Permute((2, 1))(alphas)
    x = multiply([contexts, alphas])
    # print(K.int_shape(x))
    x = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]), name='image_embedding')(x)
    # print(K.int_shape(x))

    # # the image I is only input once
    image_embedding = RepeatVector(1)(x)

    # language model
    x = [image_embedding, text_embedding]
    x = Concatenate(axis=1)(x)
    x = CuDNNLSTM(1024, return_sequences=True)(x)
    x = CuDNNLSTM(1024, return_sequences=False)(x)
    caption_output = Dense(vocab_size, activation='softmax', name='caption_output')(x)
    model = Model(inputs=[image_input, text_input], outputs=[caption_output, alpha_output])
    return model


if __name__ == '__main__':
    with tf.device("/cpu:0"):
        model = build_model()
    print(model.summary())
    plot_model(model, to_file='model.svg', show_layer_names=True, show_shapes=True)

    K.clear_session()
