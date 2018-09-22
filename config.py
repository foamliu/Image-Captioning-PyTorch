import os

image_h = image_w = image_size = 224
channel = 3
batch_size = 1
epochs = 10000
patience = 10
num_train_samples = 1050000
num_valid_samples = 150000
embedding_size = 512
vocab_size = 17628
max_token_length = 40
hidden_size = 512
cnn_type = 'resnet'
emb_dim = 512
z_dim = 512
lstm_dim = 512
attlstm = True
cnn_train = False
finetune_start_layer = 6

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
