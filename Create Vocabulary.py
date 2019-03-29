from keras.preprocessing import text
from keras.utils import np_utils
from keras.preprocessing import sequence
import re

tokenizer = text.Tokenizer()
#train_token_D = re.sub('\d', 'D', train_token[:3873])
tokenizer.fit_on_texts(train_token[:3873])
word2id = tokenizer.word_index

word2id['PAD'] = 0
id2word = {v:k for k, v in word2id.items()}
#wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for train_token]

vocab_size = len(word2id)
embed_size = 250
window_size = 2 # context window size

print('Vocabulary Size:', vocab_size)
print('Vocabulary Sample:', list(word2id.items())[:10])
