import pandas as pd
import regex

from nltk.stem import WordNetLemmatizer
from nltk.tokenize.casual import casual_tokenize


# from data exploration
lower_range=26
upper_range=80


def crop_review(text):
    cropped_review = regex.split(r'[\.|?|!]', text)
    return cropped_review[0]


# preprocessing

def clean(text):

    text = text.lower()
    text = text.replace('\\n', '')

    url = regex.compile(r'http[\S]*')
    text = url.sub("", text)

    utf = regex.compile(r'u(\d|\\x){1}[\S]*')
    text = utf.sub("", text)

    nums_x_nums = regex.compile(r'\d[\S]*\d')
    text = nums_x_nums.sub("", text)
    
    long_word = regex.compile(r'\b\w*(\w)\1\1\w*')
    text = long_word.sub("", text)

    contraction = regex.compile('([a-z]*)(\'|\.|-)([a-z]*)')
    text = contraction.sub('\1\3', text)
    
    return text 

def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    processed_text = [lemmatizer.lemmatize(clean(token)) for token in casual_tokenize(text)]
    
    return processed_text


# data loader

def load_yelp_reviews(is_train=True, crop=True, len_slice=True):
    path = './data/'
    if is_train:
        path += 'train.csv'
    else:
        path += 'test.csv'
    dataset = pd.read_csv(path, encoding='utf-8',delimiter=',', header=None)
    dataset.columns = ['target', 'review']

    # reinitialize labels 
    dataset.replace({'target': {1: 0}}, inplace=True)
    dataset.replace({'target': {2: 1}}, inplace=True)

    if crop==True:
        dataset.review = dataset.review.apply(crop_review)

    if len_slice==True:
        dataset = dataset[(dataset.review.str.len()>=lower_range) & (dataset.review.str.len()<=upper_range)].reset_index(drop=True)       

    # preprocessing here
    dataset.review = dataset.review.apply(preprocess)

    return dataset


def print_dist(dataset):
    '''print distribution over pos/neg samples'''
    pos_samples = int(dataset.target.sum()/2)
    neg_samples = len(dataset)-pos_samples

    print(pos_samples, neg_samples)

