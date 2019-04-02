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
# NOTE: Ignore this
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

##MAKE SUBSTITUTIONS
train_copy = train_rm_stop_dupe.copy()

#Remove emojis
def de_emoji(s):
    return s.encode('ascii', 'ignore').decode('ascii')

#De-emoji
train_emo = train_copy.map(lambda x: de_emoji(x))

#ARTIFACT REPLACEMENT
#Urls --> URL NOTE: URL moved to first because contains the later conversions
train_url = train_emo.map(lambda x: re.sub('http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', 'URL', x))
#User ID --> ID
train_id = train_url.map(lambda x: re.sub('@[\w_]+', 'ID', x))
#Cashtag --> TICKER
train_cash = train_id.map(lambda x: re.sub('\$[^0-9.][\w_]+', 'TICKER', x))

#CONVERT ALL TO LOWERCASE
train_lower = train_cash.map(lambda x: x.lower())

#Digits --> D - Note: moving this to bottom for target locating below
train_d = train_lower.map(lambda x: re.sub('\d', 'D', x))


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


# ENCODE ALL CHARACTERS

# Runs through all the basic unicode characters, creates a range, assigns 0's for everything except for the index of the character
def onehot_char(char):
    return [1 if i==ord(char) else 0 for i in range(32,126)]

# Simply maps each string in a tweet to onehot_char above, returns a list of lists
def onehot_charvec(s):
    global m
    if m % 100 == 0:
        print("One-hot encoding characters on row",m)
    m += 1
    return [onehot_char(c) for c in list(s)]

# Run to encode characters, check 'em out
m = 0
char_enc_list = train_lower.map(lambda x: onehot_charvec(''.join(x)))

# Max size of character list = 165
char_size = max(char_enc_list.map(len))
char_size

#Add lists of 0s to make sublists len of char_size
import copy
char_enc_list_padded = char_enc_list.copy()

#WARNING: Long run time (~5 seconds)
def pad_char_enc(l):
    list_of_zeros = [0] * 94
    for i in range(len(l)):
        n = char_size - len(l[i])
        for j in range(n):
            l[i].append(list_of_zeros)
    return(l)

char_enc_list_padded = pad_char_enc(char_enc_list_padded)
char_enc_list_padded[0]

# INDEX LOCATION OF TARGET

# INDEX TOKEN LOCATION OF TARGET
# NOTE: Ignore this token indexing - use the character index below

def index_target(s):
    # Call out global variables
    i = list(train_lemma.index(s))
    # Take only numbers before periods and commas (this was necessary because of inconsistencies in tokenization)
    snum_int = s.map(lambda x: x.split('.',1)[0].split(',',1)[0])
    # Remove all non-digit characters
    snum = list(snum_int.map(lambda x: re.sub("[^0-9]","",x)))
    # Define what we're looking for, and then do the same for it
    tgt_raw = dupe_df.iloc[i]['target_num']
    tgt_int = tgt_raw.split('.',1)[0].split(',',1)[0]
    tgt = re.sub("[^0-9]","",tgt_int)
    # Sadly, gave up on the last 23 errors and just said if you don't find it, put -1 instead
    if tgt in snum:
        return snum.index(tgt)
    else:
        return -1
    
list(train_lemma).index(train_lemma[0])
# --> Outputs target_index
target_index = list(train_lemma.map(lambda x: index_target(x)))
len(target_index)

# One-hot encode all target locations
tgt_loc = pd.get_dummies(pd.Series(target_index))

# Again, :( 23 errors
target_index.count(-1)

# INDEX CHARACTER LOCATION OF TARGET
def index_target(itarget):
    # Set index to pull target from original df
    i = list(dupe_df['itarget']).index(itarget)
    target = dupe_df.iloc[i]['target_num']
    # Pull tweet and make any non-digit characters spaces for consistency
    twtnum = re.sub("[^0-9]"," ",train_lower[i])
    tgt = re.sub("[^0-9]"," ",target)
    if i % 500 == 0:
        print("Indexing characters at row ", i)
    # Return the string index of the target in the tweet
    return twtnum.find(tgt)

# Create a semi-unique id
dupe_df['itarget'] = dupe_df['index'].map(str) + dupe_df['target_num']

# --> Outputs tgt_loc
target_char_index = list(dupe_df['itarget'].map(index_target))

# 51 values that aren't being found :(
target_char_index.count(-1)
len(target_char_index)

# 168 values with a non-unique itarget :(
len(dupe_df) - dupe_df['itarget'].nunique()

# One-hot encode target locations
tgt_loc_char = pd.get_dummies(pd.Series(target_char_index))

# Add in extra columns for values that don't exist
numlist = list(range(-1, char_size))
collist = list(tgt_loc_char.columns)
colpad = list(np.setdiff1d(numlist, collist))

for x in colpad:
    tgt_loc_char[x] = 0


# STACK EVERYTHING FOR CNN INPUT

# Stack character encoding lists of lists into individual dataframes and transpose them
# NOTE: The numbering in the print statements is weird here - disregard it, 
# it's purely an aesthetic issue, and a result of us restacking the same tweets multiple times for multiple targets.
# Also takes quite a long time to run
def stack_char_list(s):
    i = list(char_enc_list_padded).index(s)
    tmpdf = pd.DataFrame.from_records(s)
    if i % 100 == 0:
        print("Stacking character encodings at element",i)
    return tmpdf.transpose()

char_enc_stack = char_enc_list_padded.map(stack_char_list)

# Stack rule dataframe into char_size columns for CNN input
n = 0
def stack_rules(s):
    global n
    if n % 100 == 0:
        print("Stacking rules at element",n)
    n += 1
    return pd.concat([s] * char_size, axis=1)

# Takes a little bit of time!
rule_stack = key_vars.apply(stack_rules, axis=1)

# Create final dataframe
o = 0
tgt_loc_char.columns = list(range(0,165))

# Select corresponding elements, make sure column names align, and stack them
def final_stack(s):
    global o
    s.columns = list(range(0,165))
    loc = tgt_loc_char.iloc[o]
    rule = rule_stack[o]
    rule.columns = list(range(0,165))
    if o % 100 == 0:
        print("Generating final stack for element",o)
    o += 1
    return pd.concat([s.append(loc),rule])

final = char_enc_stack.map(final_stack)

#TODO
#Flatten to produce char_vec
#slist =[]
#for x in char_enc_list_padded:
#    slist.extend(x)



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
print(key_vars.head())


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