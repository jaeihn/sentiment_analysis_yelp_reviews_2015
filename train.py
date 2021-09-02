import io
import argparse
import json
import time
import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as layers

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


from loader import load_yelp_reviews, print_dist


def config_parser():
    '''parses configuration from system input'''
    p = argparse.ArgumentParser()

    p.add_argument('--model', required=True) # cnn / lstm
    p.add_argument('--embedding', required=True) # zero / glove / fasttext
    p.add_argument('--embed_dim', required=True)
    p.add_argument('--freeze')

    config = p.parse_args()

    return config


config = config_parser()

filename_prefix = config.model + '_' + config.embedding + '_' + config.embed_dim + '_'

# from data exploration
lower_range=26
upper_range=80


# load training data with preprocessing
training_data = load_yelp_reviews()

print("PREPROCESSING COMPLETE")


# train-valid split
train_valid_ratio = 0.8
train_valid_split = int(len(training_data) * train_valid_ratio)

print(len(training_data), print_dist(training_data))

shuffled_training_data = training_data.loc[np.random.permutation(training_data.index)].reset_index(drop=True)

train_x = shuffled_training_data[:train_valid_split].review.values
train_y = shuffled_training_data[:train_valid_split].target.values

valid_x = shuffled_training_data[train_valid_split:].review.values
valid_y = shuffled_training_data[train_valid_split:].target.values

print("TRAIN VALID SPLIT COMPLETED")


# model parameters
embedding_dim = int(config.embed_dim)
max_length = upper_range
padding_type='post'
oov_tok = "<OOV>"
num_epochs = 10
batch_size = 128
trainable = not bool(config.freeze)


# construct tokenizer and vocabulary
tokenizer = Tokenizer(oov_token=oov_tok)
tokenizer.fit_on_texts(train_x)

vocab = tokenizer.word_index
vocab_size = len(vocab)

print("VOCABULARY BUILD COMPLETE (size=%d)" % vocab_size)


# convert text to vectors + padding
train_x = tokenizer.texts_to_sequences(train_x)
train_x = pad_sequences(train_x, maxlen=max_length, padding=padding_type)

valid_x = tokenizer.texts_to_sequences(valid_x)
valid_x = pad_sequences(valid_x, max_length, padding=padding_type)


# print timestamp to differentiate between different models
t = time.localtime()
timestamp = time.strftime('%d_%H%M', t)

tokenizer_json = tokenizer.to_json() 
with io.open('tokenizers/' + filename_prefix + 'tokenizer_' + timestamp+'.json', 'w', encoding='utf-8') as f:  
      f.write(json.dumps(tokenizer_json, ensure_ascii=False))


# loading pretrained embeddings
embedding_dict = {}

if config.embedding == 'glove':
    # GloVe embedding
    glove_path = 'data/glove.twitter.27B.' + config.embed_dim + 'd.txt' 
    
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_dict[word] = vector

elif config.embedding == 'fasttext':
    # FastText embedding
    fasttext_path = 'data/cc.en.' + config.embed_dim +'.vec'

    with open(fasttext_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_dict[word] = vector


# create embedding matrix
embedding_matrix = np.zeros((vocab_size+1, embedding_dim))
for word, i in vocab.items():
    if embedding_dict.get(word) is not None:
        embedding_matrix[i] = embedding_dict.get(word)



# model architecture

# CNN model 
if config.model == 'cnn':

    # 3 channels with different kernel size
    channel1 = layers.Input(shape=(max_length,))
    channel2 = layers.Input(shape=(max_length,))
    channel3 = layers.Input(shape=(max_length,))

    # choose word embedding method 
    if config.embedding == 'zero':
        ch1 = layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, trainable=True)(channel1)
        ch2 = layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, trainable=True)(channel2)
        ch3 = layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, trainable=True)(channel3)
    elif config.embedding == 'glove' or config.embedding == 'fasttext':
        ch1 = layers.Embedding(vocab_size+1, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=trainable)(channel1)
        ch2 = layers.Embedding(vocab_size+1, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=trainable)(channel2)
        ch3 = layers.Embedding(vocab_size+1, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=trainable)(channel3)

    ch1 = layers.Conv1D(filters=32, kernel_size=3, activation='relu')(ch1)
    ch1 = layers.Dropout(0.3)(ch1)
    ch1 = layers.MaxPool1D(pool_size=2)(ch1)
    ch1 = layers.Flatten()(ch1)
    ch1 = tf.keras.Model(inputs=channel1, outputs=ch1)

    ch2 = layers.Conv1D(filters=32, kernel_size=5, activation='relu')(ch2)
    ch2 = layers.Dropout(0.3)(ch2)
    ch2 = layers.MaxPool1D(pool_size=2)(ch2)
    ch2 = layers.Flatten()(ch2)
    ch2 = tf.keras.Model(inputs=channel2, outputs=ch2)

    ch3 = layers.Conv1D(filters=32, kernel_size=7, activation='relu')(ch3)
    ch3 = layers.Dropout(0.3)(ch3)
    ch3 = layers.MaxPool1D(pool_size=2)(ch3)
    ch3 = layers.Flatten()(ch3)
    ch3 = tf.keras.Model(inputs=channel3, outputs=ch3)

    # combine channels
    channels = layers.concatenate([ch1.output, ch2.output, ch3.output])
    combined_out = layers.Dense(256, activation='relu')(channels)
    combined_out = layers.Dense(1, activation='sigmoid')(combined_out)

    # train model
    model = tf.keras.Model(inputs=[ch1.input, ch2.input, ch3.input], outputs=combined_out)

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()

    history = model.fit([train_x, train_x, train_x], train_y, epochs=num_epochs, validation_data=([valid_x, valid_x, valid_x], valid_y), verbose=1)


# LSTM model 
elif config.model == 'lstm':

    input = layers.Input(shape=(max_length,))

    # choose embedding method 
    if config.embedding == 'zero':
        out = layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, trainable=True)(input)
    elif config.embedding == 'glove' or config.embedding == 'fasttext':
        out = layers.Embedding(vocab_size+1, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=trainable)(input)
    
    out = layers.Dropout(rate=0.5)(out)
    out = layers.Bidirectional(layers.LSTM(units=32, return_sequences=True))(out)
    out = layers.Dropout(rate=0.5)(out)
    out = layers.Bidirectional(layers.LSTM(units=32, return_sequences=False))(out)
    out = layers.Dense(1, activation='sigmoid')(out)

    # train model 
    model = tf.keras.Model(inputs=input, outputs=out)

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    
    history = model.fit(train_x, train_y, epochs=num_epochs, validation_data=(valid_x, valid_y), verbose=1)


model.save('models/' + filename_prefix + 'model_' + timestamp + '.h5', save_format='h5')

print("Training Complete")


