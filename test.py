import pickle
import LDA_sklearn.py

# Load sklearn lda model
pickle_file = open("./data/lda_model_sklearn.pkl", "rb")
lda_model_skl = []
while True:
    try:
        lda_model_skl.append(pickle.load(pickle_file))
    except EOFError:
        break
pickle_file.close()

# Load text
data=[]   # a list of string(sentences)
with open("./data/full_texts.txt",'r', encoding = 'utf-8') as f:
    for line in f:
        data.append(line)
f.close()

# from sentences to words
data_words = list(sent_to_words(data))
# lemmatization of data (only show words of none, adj, verb, adv)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
# create doc-word matrix
vectorizer = CountVectorizer(analyzer='word',       
                             min_df=10,                        # minimum reqd occurences of a word 
                             stop_words='english',             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             # max_features=50000,             # max number of uniq words
                            )

data_vectorized = vectorizer.fit_transform(data_lemmatized)
# output predictions
lda_output = lda_model_skl.fit_transform(data_vectorized)
print(lda_output)