import tensorflow as tf
from keras import backend as K
from keras.layers import Concatenate, Add, Multiply
from keras.layers import Input, Conv1D, CuDNNLSTM
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Dense, Activation, Permute, Lambda
from keras.layers.core import RepeatVector, Dropout, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.utils import plot_model

from config import image_h, image_w, cnn_type, dr, dr_ratio, vocab_size, max_token_length, emb_dim, z_dim, sgate, \
    lstm_dim, attlstm, batch_size, cnn_train, finetune_start_layer


def image_model(input_tensor):
    '''
    Loads specified pretrained convnet
    '''

    input_shape = (image_h, image_w, 3)

    if cnn_type == 'vgg16':
        from keras.applications.vgg16 import VGG16 as cnn
    elif cnn_type == 'vgg19':
        from keras.applications.vgg19 import VGG19 as cnn
    elif cnn_type == 'resnet':
        from keras.applications.resnet50 import ResNet50 as cnn

    base_model = cnn(weights='imagenet', include_top=False,
                     input_tensor=input_tensor, input_shape=input_shape)

    if cnn_type == 'resnet':
        model = Model(input=base_model.input, output=[base_model.layers[-2].output])
    else:
        model = base_model

    return model


def language_model(wh, dim, convfeats, prev_words):
    # for testing stage where caption is predicted word by word
    seqlen = max_token_length
    num_classes = vocab_size

    # imfeats need to be "flattened" eg 15x15x512 --> 225x512
    V = Reshape((wh * wh, dim), name='conv_feats')(convfeats)  # 225x512

    # input is the average of conv feats
    Vg = GlobalAveragePooling1D(name='Vg')(V)
    # embed average imfeats
    Vg = Dense(emb_dim, activation='relu', name='Vg_')(Vg)
    if dr:
        Vg = Dropout(dr_ratio)(Vg)

    # we keep spatial image feats to compute context vector later
    # project to z_space
    Vi = Conv1D(z_dim, 1, border_mode='same',
                activation='relu', name='Vi')(V)

    if dr:
        Vi = Dropout(dr_ratio)(Vi)
    # embed
    Vi_emb = Convolution1D(emb_dim, 1, border_mode='same',
                           activation='relu', name='Vi_emb')(Vi)

    # repeat average feat as many times as seqlen to infer output size
    x = RepeatVector(seqlen)(Vg)  # seqlen,512

    # embedding for previous words
    wemb = Embedding(num_classes, emb_dim, input_length=seqlen)
    emb = wemb(prev_words)
    emb = Activation('relu')(emb)
    if dr:
        emb = Dropout(dr_ratio)(emb)

    x = Concatenate(name='lstm_in')([x, emb])
    if dr:
        x = Dropout(dr_ratio)(x)
    if sgate:
        # lstm with two outputs
        lstm_ = CuDNNLSTM(output_dim=lstm_dim,
                          return_sequences=True, stateful=True,
                          dropout_W=dr_ratio,
                          dropout_U=dr_ratio,
                          sentinel=True, name='hs')
        h, s = lstm_(x)

    else:
        # regular lstm
        lstm_ = CuDNNLSTM(lstm_dim,
                          return_sequences=True, stateful=True,
                          dropout_W=dr_ratio,
                          dropout_U=dr_ratio,
                          sentinel=False, name='h')
        h = lstm_(x)

    num_vfeats = wh * wh
    if sgate:
        num_vfeats = num_vfeats + 1

    if attlstm:

        # embed ht vectors.
        # linear used as input to final classifier, embedded ones are used to compute attention
        h_out_linear = Convolution1D(z_dim, 1, activation='tanh', name='zh_linear', border_mode='same')(h)
        if dr:
            h_out_linear = Dropout(dr_ratio)(h_out_linear)
        h_out_embed = Convolution1D(emb_dim, 1, name='zh_embed', border_mode='same')(h_out_linear)
        # repeat all h vectors as many times as local feats in v
        z_h_embed = TimeDistributed(RepeatVector(num_vfeats))(h_out_embed)

        # repeat all image vectors as many times as timesteps (seqlen)
        # linear feats are used to apply attention, embedded feats are used to compute attention
        z_v_linear = TimeDistributed(RepeatVector(seqlen), name='z_v_linear')(Vi)
        z_v_embed = TimeDistributed(RepeatVector(seqlen), name='z_v_embed')(Vi_emb)

        z_v_linear = Permute((2, 1, 3))(z_v_linear)
        z_v_embed = Permute((2, 1, 3))(z_v_embed)

        if sgate:

            # embed sentinel vec
            # linear used as additional feat to apply attention, embedded used as add. feat to compute attention
            fake_feat = Convolution1D(z_dim, 1, activation='relu', name='zs_linear', border_mode='same')(s)
            if dr:
                fake_feat = Dropout(dr_ratio)(fake_feat)

            fake_feat_embed = Convolution1D(emb_dim, 1, name='zs_embed', border_mode='same')(fake_feat)
            # reshape for merging with visual feats
            z_s_linear = Reshape((seqlen, 1, z_dim))(fake_feat)
            z_s_embed = Reshape((seqlen, 1, emb_dim))(fake_feat_embed)

            # concat fake feature to the rest of image features
            z_v_linear = Concatenate(axis=-2)([z_v_linear, z_s_linear])
            z_v_embed = Concatenate(axis=-2)([z_v_embed, z_s_embed])

        # sum outputs from z_v and z_h
        z = Add(name='merge_v_h')([z_h_embed, z_v_embed])
        if dr:
            z = Dropout(dr_ratio)(z)
        z = TimeDistributed(Activation('tanh', name='merge_v_h_tanh'))(z)
        # compute attention values
        att = TimeDistributed(Convolution1D(1, 1, border_mode='same'), name='att')(z)

        att = Reshape((seqlen, num_vfeats), name='att_res')(att)
        # softmax activation
        att = TimeDistributed(Activation('softmax'), name='att_scores')(att)
        att = TimeDistributed(RepeatVector(z_dim), name='att_rep')(att)
        att = Permute((1, 3, 2), name='att_rep_p')(att)

        # get context vector as weighted sum of image features using att
        w_Vi = Multiply(name='vi_weighted')([att, z_v_linear])
        sumpool = Lambda(lambda x: K.sum(x, axis=-2),
                         output_shape=(z_dim,))
        c_vec = TimeDistributed(sumpool, name='c_vec')(w_Vi)
        atten_out = Add(name='mlp_in')([h_out_linear, c_vec])
        h = TimeDistributed(Dense(emb_dim, activation='tanh'))(atten_out)
        if dr:
            h = Dropout(dr_ratio, name='mlp_in_tanh_dp')(h)

    predictions = TimeDistributed(Dense(num_classes, activation='softmax'), name='out')(h)

    model = Model(input=[convfeats, prev_words], output=predictions)
    return model


def build_model():
    # for testing stage where caption is predicted word by word
    seqlen = max_token_length

    # get pretrained convnet
    in_im = Input(batch_shape=(batch_size, image_h, image_w, 3), name='image')
    convnet = image_model(in_im)

    wh = convnet.output_shape[1]  # size of conv5
    dim = convnet.output_shape[3]  # number of channels

    if not cnn_train:
        for i, layer in enumerate(convnet.layers):
            if i > finetune_start_layer:
                layer.trainable = False

    imfeats = convnet(in_im)
    convfeats = Input(batch_shape=(batch_size, wh, wh, dim))
    prev_words = Input(batch_shape=(batch_size, seqlen), name='prev_words')
    lang_model = language_model(wh, dim, convfeats, prev_words)

    out = lang_model([imfeats, prev_words])

    model = Model(inputs=[in_im, prev_words], outputs=out)

    return model


if __name__ == '__main__':
    with tf.device("/cpu:0"):
        model = build_model()
    print(model.summary())
    plot_model(model, to_file='model.svg', show_layer_names=True, show_shapes=True)

    K.clear_session()
