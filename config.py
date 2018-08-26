import os

image_h = image_w = image_size = 256
channel = 3
batch_size = 256
epochs = 10000
patience = 10
num_train_samples = 14883151
num_valid_samples = 2102270
embedding_size = 512
vocab_size = 17628
max_token_length = 40
num_image_features = 2048
hidden_size = 512
L = 49
D = 512
cnn_type = 'resnet'
emb_dim = 512
z_dim = 512
lstm_dim = 512
dr_ratio = 0.5
dr = False
sgate = False
attlstm = True

train_folder = 'data/ai_challenger_caption_train_20170902'
valid_folder = 'data/ai_challenger_caption_validation_20170910'
test_a_folder = 'data/ai_challenger_caption_test_a_20180103'
test_b_folder = 'data/ai_challenger_caption_test_b_20180103'
train_image_folder = os.path.join(train_folder, 'caption_train_images_20170902')
valid_image_folder = os.path.join(valid_folder, 'caption_validation_images_20170910')
test_a_image_folder = os.path.join(test_a_folder, 'caption_test_a_images_20180103')
test_b_image_folder = os.path.join(test_b_folder, 'caption_test_b_images_20180103')
train_annotations_filename = 'caption_train_annotations_20170902.json'
valid_annotations_filename = 'caption_validation_annotations_20170910.json'

start_word = '<start>'
stop_word = '<end>'
unknown_word = '<UNK>'
