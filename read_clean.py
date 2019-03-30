#LOAD LIBRARIES
import pandas as pd
import numpy as np
import json
import urllib.request


#SET PATHS
train_path = 'https://raw.githubusercontent.com/asbiv/STAT6016_team_op/master/data/FinNum_training.json'
test_path = 'https://raw.githubusercontent.com/asbiv/STAT6016_team_op/master/data/FinNum_test.json'
dev_path = 'https://raw.githubusercontent.com/asbiv/STAT6016_team_op/master/data/FinNum_dev.json'


#READ IN TRAIN AND TEST
#Train
with urllib.request.urlopen(train_path) as url:
    train_raw = json.loads(url.read().decode())

#Test
with urllib.request.urlopen(test_path) as url:
    test_raw = json.loads(url.read().decode())

#As dfs
train_df = pd.read_json(train_path)
test_df = pd.read_json(test_path)


#REBUILT DATA
#SET PATHS
train_rebuild = 'https://raw.githubusercontent.com/asbiv/STAT6016_team_op/master/data/FinNum_training_rebuilded.json'
test_rebuild = 'https://raw.githubusercontent.com/asbiv/STAT6016_team_op/master/data/FinNum_test_rebuilded.json'
dev_rebuild = 'https://raw.githubusercontent.com/asbiv/STAT6016_team_op/master/data/FinNum_dev_rebuilded.json'

#Ignore dev for now
train_rebuild_df = pd.read_json(train_rebuild)
test_rebuild_df = pd.read_json(test_rebuild)

#199 NANs in train
train_df = train_rebuild_df.dropna().reset_index()
test_df = test_rebuild_df.dropna().reset_index()

train_df.columns
train_df['idx'].head()
train_df['id'].head()
#PREPROCESSING


#REMOVE STOP WORDS
import nltk
from nltk.corpus import stopwords
import string
nltk.download('stopwords')

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via']

#Remove stop words
def remove_stopwords(s):
     stopset = set(stopwords.words('english'))
     cleanup = " ".join(filter(lambda word: word not in stopset, s.split()))
     return cleanup

train_rm_stop = train_df['tweet'].map(lambda x: remove_stopwords(x))


# SUBSTITUTE ALL DIGITS WITH THE LETTER D
train_digits = train_rm_stop.map(lambda x: re.sub('\d', 'D', x))


#TOLKENIZATION OF TWITTER ARTIFACTS
#Building off this: https://marcobonzanini.com/2015/03/09/mining-twitter-data-with-python-part-2/
import re
nltk.download('punkt')
from nltk.tokenize import word_tokenize

#Keep twitter elements in place
regex_str = [
    r'<[^>]+>', # HTML tags
    r'(?:\$[\w_]+)', # Ticker $
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    return tokens

#Map preprocess
train_token = train_rm_stop.map(lambda x: preprocess(x))
train_token_d = train_digits.map(lambda x: preprocess(x))


#CURRENT STATUS...
# print(train_df['tweet'][1])
# print(train_token[1])


# LEMMATIZATION
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatize(s):
    s = lemmatizer.lemmatize(s, pos='n')
    s = lemmatizer.lemmatize(s, pos='v')
    return s

def lemma_loop(s):
    # Note: Train_token is a series of lists; lists function differently than
    # series. So we convert to series prior to mapping.
    return pd.Series(s).map(lambda x: lemmatize(x))

# Map lemmatizing functions
train_lemma = train_token.map(lambda x: lemma_loop(x))
train_lemma_d = train_token_d.map(lambda x: lemma_loop(x))

# Summarize lengths of tweet documents
doc_lengths = list()
train_lemma.map(lambda x: doc_lengths.append(len(x)))
doc_lengths[1]
print('Minimum length of lemmatized tweet: ',min(doc_lengths),'\n','Average: ',sum(doc_lengths)/len(doc_lengths),'\n','Maximum: ',max(doc_lengths))

print(len(train_lemma[1]))


# KEYWORDS AND RULES
keys = {"key_p": ["%","percent","pc","pct"],
        "key_r": ["up","down","decline","increase","growth","gain","lose","+","-"],
        "key_m": ["january","jan","february","feb","march","mar","april","apr","may","june","jun","july","jul","august","aug","september","sept","sep","october","oct","november","nov","december","dec"],
        "key_i": ["ma","dma","sma","ema","rsi","ichimoku"],
        "key_d": ["day","week","month","year","mos","yrs"],
        "key_t": ["second","sec","minute","min","mins","hour","hr","hrs","p.m.","pm","a.m."]}

keynames = ["key_p","key_r","key_m","key_i","key_d","key_t", "maturity","date"]

key_vars = pd.DataFrame([], columns=keynames)

nltk.pos_tag(train_lemma[1])
nltk.pos_tag(train_lemma_d[1])

def key_loop(s):
    global key_vars
    key_counts = pd.DataFrame(np.zeros((1,6)),columns=keynames)
    for token in s:
        keywords(token, key_counts)
    key_vars = key_vars.append(key_counts.sum(), ignore_index=True)
    # return pd.DataFrame(key_counts.sum())

def keywords(token, df):
    for keylist in keys:
        if token in keys[keylist] and df[keylist][0] < 1:
            df[keylist][0] = 1
    return df

train_lemma.map(lambda x: key_loop(x))


# N-GRAM BAG OF WORDS
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(2,2))

def vectorize(s):
    s_untokenized = s.map(' '.join)
    return vectorizer.fit_transform(s_untokenized).toarray()

# Map bag of words function, join with lemmatized text
train_bow = pd.DataFrame(vectorize(train_lemma))
# train_bow.shape
train_bow['text'] = train_bow.index.map(train_lemma)

# print(train_bow.shape)

# TF-IDF 
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(ngram_range=(2,2))

def tf_idf(s):
    s_untokenized = s.map(' '.join)
    return tfidf_vectorizer.fit_transform(s_untokenized).toarray()

train_tfidf = pd.DataFrame(tf_idf(train_lemma))
# print(train_tfidf.shape)
train_tfidf['text'] = train_tfidf.index.map(train_lemma)
# print(train_tfidf.shape)