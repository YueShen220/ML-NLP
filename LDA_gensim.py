# Load text
lines=[]
with open("./data/full_texts.txt",'r', encoding = 'utf-8') as f:
    for line in f:
        lines.append(line)
f.close()

# Preprocessing the raw text (use the NLTK and gensim libraries)

# Loading gensim and nltk libraries

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(400)
import nltk
nltk.download('wordnet')
import pandas as pd
stemmer = SnowballStemmer("english")

# Functions for preprocess text
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result

# Perform preprocessing
processed_lines=[]
for line in lines:
    processed_lines.append(preprocess(line))

dictionary = gensim.corpora.Dictionary(processed_lines)
bow_corpus = [dictionary.doc2bow(line) for line in processed_lines]

# Train and save the LDA model
lda_model = gensim.models.LdaModel(bow_corpus, 
                                   num_topics = 10, 
                                   id2word = dictionary,                                    
                                   passes = 50)
import pickle
pickle.dump(lda_model,open("./data/lda_model_gensim.pkl",'wb'))

for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")                 
