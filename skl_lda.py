# Remove twitter-specific from documents (use some other packages)

# Show full-text about document contents

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
import pandas as pd
import preprocessor as p

p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.NUMBER)

# Load documents (sentences)
documents = [] # a list of string (sentences)
with open("./data/full_texts.txt",'r',encoding='utf-8') as f:
    for line in f:
        documents.append(p.clean(line))
documents.close()

# Preprocess document words and vectorize documents (to show words and their counts) using countvectorizer
vectorizer = CountVectorizer(max_df=0.9, #truncate words that occur more than 90% of all documents (sentences)
                             min_df=10,   #truncate words that occur only in 1-10 documents
                             stop_words='english', #truncate meaningless stopwords
                             lowercase=True,  # all lowercase
                             token_pattern='[a-zA-Z0-9]{3,}', # num char > 3
                             max_features=50000) # only allow max number of unique words
matrix = vectorizer.fit_transform(documents) # returns all "(document_index, feature_index) count"

# Build LDA model
lda = LatentDirichletAllocation(n_components=10, # 10 topics
                                max_iter= 10, # max learning iterations
                                learning_method='online', # less accurate at start but can process big data
                                batch_size=128, # number of documents trained in each iteration
                                random_state=100, # random state
                                n_jobs=-1) # use all available CPUs
# Train and save LDA model
lda.fit(matrix)
pickle.dump(lda,open('./data/lda.pkl','wb'))

# Load LDA model from disk and show topic assignemnts along with probs
loaded_lda = pickle.load(open('./data/lda.pkl','rb'))
lda_output = loaded_lda.transform(matrix)

# Output topic-keywords matrix based on p(word|topic); 20 words per topic
def show_topics(vectorizer,lda_model,n_words):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer, loaded_lda,n_words=20)
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word'+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic'+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords.to_csv('./data/topic_keywords.csv')
print(df_topic_keywords)

# Output doc-topic matrix for all docs and all 10 topics
topic_names = ['Topic'+str(i) for i in range(loaded_lda.n_components)]
doc_names = ['Doc'+str(i) for i in range(len(documents))]
df_doc_topic = pd.DataFrame(np.round(lda_output,2),columns=topic_names,index=doc_names)
df_doc_topic.to_csv('./data/doc_topic.csv')
print(df_doc_topic)

# Output topic-doc matrix based on p(topic|doc); 5 docs per topic
topic_docs = []
for i in range(loaded_lda.n_components):
    topic_list = df_doc_topic['Topic'+str(i)]
    idx  = np.argsort(-topic_list)
    docs_per_topic = []
    for j in range(5):
        doc = 'Doc'+str(idx[j])
        full_text = docs[idx[j]]
        docs_per_topic.append(doc)
    topic_docs.append(docs_per_topic)

df_topic_docs = pd.DataFrame(topic_docs)
df_topic_docs.columns = ['Max'+str(i+1) for i in range(df_topic_docs.shape[1])]
df_topic_docs.index = ['Topic'+str(i) for i in range(df_topic_docs.shape[0])]
df_topic_docs.to_csv('./data/topic_docs.csv')
print(df_topic_docs)