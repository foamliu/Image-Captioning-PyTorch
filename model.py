import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Dense, LSTM, Concatenate, Embedding, RepeatVector, Bidirectional, TimeDistributed, \
    BatchNormalization
from keras.models import Model
from keras.utils import plot_model
from keras.applications.resnet50 import ResNet50
from config import hidden_size, max_token_length, img_rows, img_cols, channel
from config import vocab_size, embedding_size


def build_model():
    # word embedding
    text_input = Input(shape=(max_token_length,), dtype='int32')
    x = Embedding(input_dim=vocab_size, output_dim=embedding_size)(text_input)
    x = LSTM(256, return_sequences=True)(x)
    text_embedding = TimeDistributed(Dense(embedding_size))(x)

    # image embedding
    image_encoder = ResNet50(input_shape=(img_rows, img_cols, channel), include_top=False, weights='imagenet', pooling='avg')
    for layer in image_encoder.layers:
        layer.trainable = False
    image_input = image_encoder.layers[0].input
    x = image_encoder.layers[-1].output
    x = Dense(embedding_size, activation='relu', name='image_embedding')(x)
    # the image I is only input once
    image_embedding = RepeatVector(1)(x)

    # language model
    x = [image_embedding, text_embedding]
    x = Concatenate(axis=1)(x)
    x = Bidirectional(LSTM(hidden_size, return_sequences=False))(x)

    output = Dense(vocab_size, activation='softmax', name='output')(x)

    inputs = [image_input, text_input]
    model = Model(inputs=inputs, outputs=output)
    return model


if __name__ == '__main__':
    with tf.device("/cpu:0"):
        model = build_model()
    print(model.summary())
    plot_model(model, to_file='model.svg', show_layer_names=True, show_shapes=True)

    K.clear_session()
