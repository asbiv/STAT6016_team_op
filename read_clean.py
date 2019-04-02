#LOAD LIBRARIES
import pandas as pd
import numpy as np
import re


#SET PATHS
'''
import json
import urllib.request

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
'''


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


#PREPROCESSING
# EXPAND TWEETS WITH >1 TARGET

# Note: I recognize this could be broken up into different functions, but I'm tired so I'll just comment it a lot instead
def dupe(s):
    # Specify global objects if you're going to alter them in a function
    global hold_df
    global to_drop
    global j
    # If there is more than one target...
    if len(s['target_num']) > 1:
        # Create a temporary empty dataframe...
        tmpdf = pd.DataFrame(columns=list(train_df.columns))
        for i in range(0,len(s['target_num'])):
            # Capture the relevant information for each target...
            singlet = pd.DataFrame([[s['index'], s['category'][i], s['id'], s['idx'], s['subcategory'][i], s['target_num'][i], s['tweet']]], columns=list(train_df.columns))
            tmpdf = tmpdf.append(singlet, ignore_index=True)
        # Build a list of rows to drop, and add the records to our dataframe
        to_drop.append(j)
        hold_df = hold_df.append(tmpdf, ignore_index=True)
    j += 1
    # It takes a long time, so it's nice to know how far along you are
    if j % 100 == 0:
        print(j, " of ", train_df.shape[0], " records completed.")

# Don't want to mess up the main dataset, so creating a duplicate and initializing a counter for our function
hold_df = pd.DataFrame([], columns = list(train_df.columns))
to_drop = list()
j = 0

dupe_df = train_df.copy()
dupe_df.apply(dupe, axis=1)

# Drop records, pull out strings from lists for future application, and append expanded versions of records
dupe_df = dupe_df.drop(to_drop)
dupe_df['target_num'] = dupe_df['target_num'].map(lambda x: x[0])
dupe_df['category'] = dupe_df['category'].map(lambda x: x[0])
dupe_df['subcategory'] = dupe_df['subcategory'].map(lambda x: x[0])
dupe_df = dupe_df.append(hold_df).reset_index(drop=True)

# Do the same to the test data
def dupe_test_f(s):
    # Specify global objects if you're going to alter them in a function
    global hold_df_test
    global to_drop_test
    global j
    # If there is more than one target...
    if len(s['target_num']) > 1:
        # Create a temporary empty dataframe...
        tmpdf = pd.DataFrame(columns=list(test_df.columns))
        for i in range(0,len(s['target_num'])):
            # Capture the relevant information for each target...
            singlet = pd.DataFrame([[s['index'], s['id'], s['idx'], s['target_num'][i], s['tweet']]], columns=list(test_df.columns))
            tmpdf = tmpdf.append(singlet, ignore_index=True)
        # Build a list of rows to drop, and add the records to our dataframe
        to_drop_test.append(j)
        hold_df_test = hold_df_test.append(tmpdf, ignore_index=True)
    j += 1
    # It takes a long time, so it's nice to know how far along you are
    if j % 100 == 0:
        print(j, " of ", test_df.shape[0], " records completed.")

hold_df_test = pd.DataFrame([], columns = list(test_df.columns))
to_drop_test = list()
j = 0
dupe_test = test_df.copy()
dupe_test.apply(dupe_test_f, axis=1)

dupe_test = dupe_test.drop(to_drop_test)
dupe_test['target_num'] = dupe_test['target_num'].map(lambda x: x[0])
dupe_test = dupe_test.append(hold_df_test).reset_index(drop=True)

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

#train_rm_stop = train_df['tweet'].map(lambda x: remove_stopwords(x))
#Active df below
train_rm_stop_dupe = dupe_df['tweet'].map(lambda x: remove_stopwords(x))
test_rm_stop_dupe = dupe_test['tweet'].map(lambda x: remove_stopwords(x))


##MAKE SUBSTITUTIONS
train_d = train_rm_stop_dupe.copy()
test_d = test_rm_stop_dupe.copy()

#Remove emojis
def de_emoji(s):
    return s.encode('ascii', 'ignore').decode('ascii')

#De-emoji
train_d = train_d.map(lambda x: de_emoji(x))
test_d = test_d.map(lambda x: de_emoji(x))

#ARTIFACT REPLACEMENT
#Urls --> URL NOTE: URL moved to first because contains the later conversions
train_d = train_d.map(lambda x: re.sub('http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', 'URL', x))
#Digits --> D
train_d = train_d.map(lambda x: re.sub('\d', 'D', x))
#User ID --> ID
train_d = train_d.map(lambda x: re.sub('@[\w_]+', 'ID', x))
#Cashtag --> TICKER
train_d = train_d.map(lambda x: re.sub('\$[\w_]+', 'TICKER', x))

#CONVERT ALL TO LOWERCASE
train_d = train_d.map(lambda x: x.lower())

#TEST ARTIFACT REPLACEMENT
#Urls --> URL NOTE: URL moved to first because contains the later conversions
test_d = test_d.map(lambda x: re.sub('http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', 'URL', x))
#Digits --> D
test_d = test_d.map(lambda x: re.sub('\d', 'D', x))
#User ID --> ID
test_d = test_d.map(lambda x: re.sub('@[\w_]+', 'ID', x))
#Cashtag --> TICKER
test_d = test_d.map(lambda x: re.sub('\$[\w_]+', 'TICKER', x))

#CONVERT ALL TO LOWERCASE
test_d = test_d.map(lambda x: x.lower())


#TOKENIZATION OF TWITTER ARTIFACTS
#Building off this: https://marcobonzanini.com/2015/03/09/mining-twitter-data-with-python-part-2/
nltk.download('punkt')
from nltk.tokenize import word_tokenize

#Keep twitter elements in place
regex_str = [
    r'<[^>]+>', # HTML tags
    r'(?:\$[\w_]+)', # Cashtag
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

def find_twitter_tokens(s, lowercase=False):
    tokens = tokenize(s)
    return tokens

#Map preprocess
train_token = train_rm_stop_dupe.map(lambda x: find_twitter_tokens(x))
test_token = test_rm_stop_dupe.map(lambda x: find_twitter_tokens(x))
#train_token_d = train_d.map(lambda x: find_twitter_tokens(x))
test_token_d = test_d.map(lambda x: find_twitter_tokens(x))

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
test_lemma = test_token.map(lambda x: lemma_loop(x))
#train_lemma_d = train_token_d.map(lambda x: lemma_loop(x))
test_lemma_d = test_token_d.map(lambda x: lemma_loop(x))

# Summarize lengths of tweet documents
doc_lengths = list()
train_lemma.map(lambda x: doc_lengths.append(len(x)))
print('Minimum length of lemmatized tweet: ',min(doc_lengths),'\n','Average: ',sum(doc_lengths)/len(doc_lengths),'\n','Maximum: ',max(doc_lengths))


# POS_TAGGING
nltk.download('averaged_perceptron_tagger')
train_pos = train_lemma.map(lambda x: nltk.pos_tag(x))
train_pos.head()
test_pos = test_lemma.map(lambda x: nltk.pos_tag(x))


# INDEX LOCATION OF TARGET

# Initialize counter and index list
k = 0
target_index = list()
def index_target(s):
    # Call out global variables
    global k
    global target_index
    # Take only numbers before periods and commas (this was necessary because of inconsistencies in tokenization)
    snum_int = s.map(lambda x: x.split('.',1)[0].split(',',1)[0])
    # Remove all non-digit characters
    snum = list(snum_int.map(lambda x: re.sub("[^0-9]","",x)))
    # Define what we're looking for, and then do the same for it
    tgt_raw = dupe_df.iloc[k]['target_num']
    tgt_int = tgt_raw.split('.',1)[0].split(',',1)[0]
    tgt = re.sub("[^0-9]","",tgt_int)
    # Sadly, gave up on the last 23 errors and just said if you don't find it, put -1 instead
    if tgt in snum:
        target_index.append(snum.index(tgt))
    else:
        target_index.append(-1)
    k += 1

# --> Outputs target_index
train_lemma.map(lambda x: index_target(x))
len(target_index)

# One-hot encode all target locations
tgt_loc = pd.get_dummies(pd.Series(target_index))
tgt_loc

# Again, :( 23 errors
target_index.count(-1)


# ENCODE ALL CHARACTERS

# Runs through all the basic unicode characters, creates a range, assigns 0's for everything except for the index of the character
def onehot_char(char):
    return [1 if i==ord(char) else 0 for i in range(32,126)]

# Simply maps each string in a tweet to onehot_char above, returns a list of lists
def onehot_charvec(s):
    global m
    if m % 100 == 0:
        print(m)
    m += 1
    return [onehot_char(c) for c in list(s)]

# Run to encode characters, check 'em out
m = 0
char_enc_list = train_d.map(lambda x: onehot_charvec(''.join(x)))
len(char_enc_list[0])

#Add lists of 0s to make sublists len 183
import copy
char_enc_list_padded = char_enc_list.copy()

#WARNING: Long run time (~5 seconds)
def pad_char_enc(l):
    list_of_zeros = [0] * 94
    for i in range(len(l)):
        n = 183 - len(l[i])
        for j in range(n):
            l[i].append(list_of_zeros)
    return(l)

char_enc_list_padded = pad_char_enc(char_enc_list_padded)


#RECREATE CHAR_VEC FOR INPUT_MAT
# This is very complicated, but all it's doing is mapping each lemmatized tweet, 
# concatenating the strings, mapping those, and collecting the unique characters (and sorting that list)
all_char = sorted(list(set(train_lemma.map(lambda x: ''.join(set(''.join(x)))).str.cat(sep=''))))
# Initialize character encoding vector
char_vec = pd.DataFrame(0, index=np.arange(dupe_df.shape[0]), columns=all_char)
l = 0

def char_enc1(s):
    global char_vec
    global l
    char_vec.iloc[l][s] += 1
    
def char_enc(s):
    global l
    list(map(char_enc1, s))
    if l % 100 == 0:
        print(l)
    l += 1
# Map each tweet to encode it --> Outputs char_vec
#WARNING: This takes a pretty long time, probably 1m+
train_lemma.map(lambda x: char_enc(''.join(x)))
char_vec.sum()


# Find lengths of character vectors, max
lenlist = list(map(len, char_enc_list))
max(lenlist)

# KEYWORDS AND RULES
keys = {"key_p": ["%","percent","pc","pct"],
        "key_r": ["up","down","decline","increase","growth","gain","lose","+","-"],
        "key_m": ["january","jan","february","feb","march","mar","april","apr","may","june","jun","july","jul","august","aug","september","sept","sep","october","oct","november","nov","december","dec"],
        "key_i": ["ma","dma","sma","ema","rsi","ichimoku"],
        "key_d": ["day","week","month","year","mos","yrs"],
        "key_t": ["second","sec","minute","min","mins","hour","hr","hrs","p.m.","pm","a.m."],
        "key_callput": ["call","put"]}

keynames = ["key_p","key_r","key_m","key_i","key_d","key_t", "key_callput"]

key_vars = pd.DataFrame([], columns=keynames)

# Function to check if token is number
def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def key_loop(s):
    global key_vars
    key_counts = pd.DataFrame(np.zeros((1,7)),columns=keynames)
    for keylist in keys:
        if len(list(set(s) & set(keys[keylist]))) > 0:
            key_counts[keylist] = 1
    key_vars = key_vars.append(key_counts.sum(), ignore_index=True)

# --> Outputs key_vars
train_lemma.map(lambda x: key_loop(x))


#TODO
##BUILD INPUT MATRIX
#key_vars, char_vec, tgt_loc
X_mat = pd.concat([key_vars, char_vec, tgt_loc], axis=1)
y_train = dupe_df['category'].map(lambda x: x[0] if type(x) == list else x)
input_mat = pd.concat([y_train, X_mat], axis=1)

#input_mat.to_csv('data/input_mat.csv', index=False)


###HOLD FOR NOW
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