import argparse
import json
import os
from tensorflow import keras

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from loader import load_yelp_reviews


def config_parser():
    '''parses configuration from system input'''
    p = argparse.ArgumentParser()

    p.add_argument('--model', required=True) # cnn / lstm
    p.add_argument('--embedding', required=True) # zero / glove / fasttext
    p.add_argument('--embed_dim', required=True)

    config = p.parse_args()

    return config


config = config_parser()

filename_prefix = config.model + '_' + config.embedding + '_' + config.embed_dim + '_'


# path to model and tokenizer filez
model_files = os.listdir('models/')
tokenizer_files = os.listdir('tokenizers/')

model_path = 'models/' + sorted([file for file in model_files if file.startswith(filename_prefix)])[-1]
tokenizer_path = 'tokenizers/' + sorted([file for file in tokenizer_files if file.startswith(filename_prefix)])[-1]

print(model_path)


# from data exploration 
lower_range = 26
upper_range = 80


# model parameters 
embedding_dim = int(config.embed_dim)
max_length = upper_range
padding_type='post'
oov_tok = "<OOV>"
num_epochs = 10


# load tokenizer 
with open(tokenizer_path) as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

print('TOKENIZER LOADED')


# load test set
test_set_sliced = load_yelp_reviews(is_train=False, crop=True, len_slice=True)

print('TEST SET LOADED')


# text to vectors + add padding
test_x = tokenizer.texts_to_sequences(test_set_sliced.review.values)
test_x = pad_sequences(test_x, maxlen=max_length, padding=padding_type)
test_y = test_set_sliced.target.values

print('TEST VECTOR TRANSFORMATION COMPLETE')


# load model
model = keras.models.load_model(model_path)
model.summary()


# evaluate model and write results
if config.model =='cnn':
    results = model.evaluate([test_x, test_x, test_x], test_y)
elif config.model =='lstm':
    results = model.evaluate(test_x, test_y)

with open('data/evaluation.txt', 'a') as f:
    f.write(filename_prefix + ' | %f | %f' % (results[0], results[1]))
    
print('END OF EVALUATION')