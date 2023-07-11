#%%_________________________BERTopic ALGORITHM 
import nltk
import gensim
import gensim.corpora as corpora
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from gensim.models.coherencemodel import CoherenceModel
from sklearn.datasets import fetch_20newsgroups
from bertopic.representation import MaximalMarginalRelevance
from bertopic.representation import KeyBERTInspired
import numpy as np
from umap import UMAP
from bertopic import BERTopic
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import string 
#%%__________________________PREPROCESSING
df=pd.read_excel('Database.xlsx')
df=df.loc[df['Type']!='Review']
df=df.dropna(subset=['Abstracts'])
T=list(df.Abstracts)
#%%
def remove(list):
    remove_digits = str.maketrans('', '', string.digits)
    list = [i.translate(remove_digits) for i in list]
    return list
for i in range(0,len(T)): T[i]=remove(T[i])

documents=pd.DataFrame(T,columns=['text'])

filtered_text = []
lemmatizer = WordNetLemmatizer()

for w in T:
    tokens=word_tokenize(w)
    filtered_text.append(lemmatizer.lemmatize(w))
print(filtered_text[:1])

#%%__________________________ALGORITHM INITIALIZATION
# Step 2.1 - Extract embeddings
UCI=[]
U_MASS=[]
#%%

embedding_model = SentenceTransformer("all-mpnet-base-v2")
#%%
for x in range(20,25,4):
    print(str(x))
    # Step 2.2 - Reduce dimensionality
    def rescale(x, inplace=False):
        """ Rescale an embedding so optimization will not have convergence issues.
        """
        if not inplace:
            x = np.array(x, copy=True)
    
        x /= np.std(x[:, 0]) * 10000
    
        return x
    
    
    # Initialize and rescale PCA embeddings
    embeddings = embedding_model.encode(filtered_text, show_progress_bar=True)
    
    pca_embeddings = rescale(PCA(n_components=5).fit_transform(embeddings))
    
    # Start UMAP from PCA embeddings
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        init=pca_embeddings,
    )
    # Step 2.3 - Cluster reduced embeddings
    kmeans_model = KMeans(n_clusters=x)
    hdbscan_model = kmeans_model
    #HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    # Step 2.4 - Tokenize topics
    from nltk.corpus import stopwords
    stopwords = list(stopwords.words('english')) + ['doi', 'crossref', 'author', 'ieee', 'http', 'et al','https', 'org', 'dx', 'com','org', 'pp.', '2019', 'Proceedings', 'conference']
    vectorizer_model = CountVectorizer(ngram_range= (1,3),stop_words=stopwords)
    # Step 2.5 - Create topic representation
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    representation_model = KeyBERTInspired()
    representation_model = MaximalMarginalRelevance(diversity=0.2)
    topic_model = BERTopic(
      embedding_model=embedding_model,    # Step 1 - Extract embeddings
      umap_model=umap_model,              # Step 2 - Reduce dimensionality
      hdbscan_model=hdbscan_model,        # Step 3 - Cluster reduced embeddings
      vectorizer_model=vectorizer_model,  # Step 4 - Tokenize topics
      ctfidf_model=ctfidf_model,          # Step 5 - Extract topic words
      representation_model=representation_model,
      nr_topics=x, verbose=True           # Step 6 - Diversify topic words
    )
#________________________TRAINING
    topics, probabilities = topic_model.fit_transform(filtered_text)
    try:
        fig=topic_model.visualize_topics()
        fig.write_html(str(x)+'Topic_dist.html')
    except:
        print('Error 1')
    try:
        fig1=topic_model.visualize_barchart(top_n_topics=x,n_words=10)
        fig1.write_html(str(x)+'Bar.html')
    except:
        print('Error 2')
    # try: 
    #     fig2=topic_model.visualize_hierarchy()
    #     fig2.write_html(str(x)+'Hier.html')
    # except:
    #     print('Error 3')
    try:
        fig3 = topic_model.visualize_heatmap()
        fig3.write_html(str(x)+'heatmap.html')
    except:
        print('Error 4')
#______________________EVALUATION________________________________________________
    documents = pd.DataFrame({"Document": filtered_text,
                              "ID": range(len(filtered_text)),
                              "Topic": topics})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)
    
    # Extract vectorizer and analyzer from BERTopic
    vectorizer = topic_model.vectorizer_model
    analyzer = vectorizer.build_analyzer()
    
    # Extract features for Topic Coherence evaluation
    words = vectorizer_model.get_feature_names()
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [[words for words, _ in topic_model.get_topic(topic)] 
                   for topic in range(len(set(topics)))]
    coherence_model = CoherenceModel(topics=topic_words, 
                                     texts=tokens, 
                                     corpus=corpus,
                                     dictionary=dictionary, 
                                     coherence='c_uci')
    coherence = coherence_model.get_coherence()
    UCI.append(coherence)
    #print(coherence)
    #C-umass: this measure takes into consideration the document co-occurrence counts, one-preceding segmentation, and a logarithmic conditional probability as a confirmation measure
    coherence_model1 = CoherenceModel(topics=topic_words, 
                                     texts=tokens, 
                                     corpus=corpus,
                                     dictionary=dictionary, 
                                     coherence='u_mass')
    coherence1 = coherence_model1.get_coherence()
    #print(coherence1)
    U_MASS.append(coherence1)
#%%____________________VISUALIZE TOPIC PER TIME_____________________________-

freq_df=doc52['Topic'].value_counts()
freq_df=freq_df.reset_index()
freq_df.columns = ['Topic','Count']
freq_df = freq_df.loc[freq_df.Topic != -1, :]
selected_topics = sorted(freq_df.Topic.to_list())

if model52.custom_labels_ is not None and custom_labels:
    topic_names = {key: model52.custom_labels_[key + model52._outliers] for key, _ in model52.topic_labels_.items()}
else:
    topic_names = {key: value[:40] + "..." if len(value) > 40 else value
                   for key, value in model52.topic_labels_.items()}
topics_over_time["Name"] = topics_over_time.Topic.map(topic_names)
data = topics_over_time.loc[topics_over_time.Topic.isin(selected_topics), :].sort_values(["Topic", "Timestamp"])
    
timestamps=timestamps.reset_index(drop=True)
timestamps=pd.DataFrame(timestamps)
timestamps[timestamps.Date<2000]=2000
timestamps[timestamps.Date>2023]=2023

topics_over_time = topic_model.topics_over_time(documents.Document, list(timestamps.Date))
topic_model.visualize_topics_over_time(topics_over_time)
fig = topic_model.visualize_topics_over_time(topics_over_time)
fig.write_html("time.html")
#%%_________________TREND ANALYSIS

common_items = set(y['Document']) & set(doc52['Document'])
x=doc52[doc52['Document'].isin(common_items)].Date
y['Date']=x.values

#%%
b=[]
for i in range(0,len(T)):
    try:
        a = mk.pre_whitening_modification_test(T[i].values)
        b.append(list(a))
    except: print('NA')

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(T[4].values)
ax.plot(T[4].index, trend_line)

#%% T1-topic modelling process 
#Data preprocessing
import os
os.chdir('/Users/thanosbach/Desktop/ReviewAIandSGs/BerTopic Results/Level1')
Topics=doc52.Topic.unique()
for i in Topics:
    globals()[f'df_{i}']=doc52[doc52.Topic==i]
    globals()[f'df_{i}'].to_excel(f'doc_{i}.xlsx')

#%%BERTopic
for i in Topics:
    print(str(i))
    x=globals()[f'df_{i}']
    documents=pd.DataFrame(x.Document,columns=['text'])
    filtered_text = []
    lemmatizer = WordNetLemmatizer()
    T=list(x.Document)
    for w in T:
        tokens=word_tokenize(w)
        filtered_text.append(lemmatizer.lemmatize(w))
    print(filtered_text[:1])
    for y in range(4,5,4):
        print(str(y))
        # Step 2.2 - Reduce dimensionality
        def rescale(x, inplace=False):
            """ Rescale an embedding so optimization will not have convergence issues.
            """
            if not inplace:
                x = np.array(x, copy=True)
        
            x /= np.std(x[:, 0]) * 10000
        
            return x
        # Initialize and rescale PCA embeddings
        embeddings = embedding_model.encode(filtered_text, show_progress_bar=True)
        pca_embeddings = rescale(PCA(n_components=5).fit_transform(embeddings))
        # Start UMAP from PCA embeddings
        umap_model = UMAP( n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", init=pca_embeddings,)
        # Step 2.3 - Cluster reduced embeddings
        kmeans_model = KMeans(n_clusters=y)
        hdbscan_model = kmeans_model
        #HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        # Step 2.4 - Tokenize topics
        from nltk.corpus import stopwords
        stopwords = list(stopwords.words('english')) + ['doi', 'crossref', 'author', 'ieee', 'http', 'et al','https', 'org', 'dx', 'com','org', 'pp.', '2019', 'Proceedings', 'conference']
        vectorizer_model = CountVectorizer(ngram_range= (1,3),stop_words=stopwords)
        # Step 2.5 - Create topic representation
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        representation_model = KeyBERTInspired()
        representation_model = MaximalMarginalRelevance(diversity=0.2)
        topic_model = BERTopic(
          embedding_model=embedding_model,    # Step 1 - Extract embeddings
          umap_model=umap_model,              # Step 2 - Reduce dimensionality
          hdbscan_model=hdbscan_model,        # Step 3 - Cluster reduced embeddings
          vectorizer_model=vectorizer_model,  # Step 4 - Tokenize topics
          ctfidf_model=ctfidf_model,          # Step 5 - Extract topic words
          representation_model=representation_model,
          nr_topics=y, verbose=True           # Step 6 - Diversify topic words
        )
        topics, probabilities = topic_model.fit_transform(filtered_text)
        try:
            fig=topic_model.visualize_topics()
            fig.write_html(str(i)+'_Topic_dist.html')
            print('Done')
        except:
            print('Error 1')
        try:
            fig1=topic_model.visualize_barchart(top_n_topics=y,n_words=10)
            fig1.write_html(str(i)+'_Bar.html')
            print('Done')
        except:
            print('Error 2')
        # try: 
        #     fig2=topic_model.visualize_hierarchy()
        #     fig2.write_html(str(x)+'Hier.html')
        # except:
        #     print('Error 3')
        try:
            fig3 = topic_model.visualize_heatmap()
            fig3.write_html(str(i)+'_heatmap.html')
            print('Done')
        except:
            print('Error 4')
        documents = pd.DataFrame({"Document": filtered_text,
                                  "ID": range(len(filtered_text)),
                                  "Topic": topics})
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)
        # Extract vectorizer and analyzer from BERTopic
        vectorizer = topic_model.vectorizer_model
        analyzer = vectorizer.build_analyzer()
        # Extract features for Topic Coherence evaluation
        words = vectorizer_model.get_feature_names()
        tokens = [analyzer(doc) for doc in cleaned_docs]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        topic_words = [[words for words, _ in topic_model.get_topic(topic)] 
                       for topic in range(len(set(topics)))]
        pd.DataFrame(topic_words).to_excel(f'Topics_T1_5_{i}.xlsx')
        # coherence_model = CoherenceModel(topics=topic_words, 
        #                                  texts=tokens, 
        #                                  corpus=corpus,
        #                                  dictionary=dictionary, 
        #                                  coherence='c_uci')
        # coherence = coherence_model.get_coherence()
        # UCI.append(coherence)
        # print(coherence)
        # #C-umass: this measure takes into consideration the document co-occurrence counts, one-preceding segmentation, and a logarithmic conditional probability as a confirmation measure
        # coherence_model1 = CoherenceModel(topics=topic_words, 
        #                                  texts=tokens, 
        #                                  corpus=corpus,
        #                                  dictionary=dictionary, 
        #                                  coherence='u_mass')
        # coherence1 = coherence_model1.get_coherence()
        # print(coherence1)
        # U_MASS.append(coherence1)
        common_items = set(documents['Document']) & set(x['Document'])
        q=x[x['Document'].isin(common_items)].Date
        documents['Date']=q.values
        documents.to_excel(f'doc_T1_5_{i}.xlsx')
#%% Emerging all excel topics files in one file
import os
import pandas as pd
import natsort
cwd = os.path.abspath('/Users/thanosbach/Desktop/ReviewAIandSGs/BerTopic Results/Level1') #If i leave it empty and have files in /Viktor it works but I have the desired excel files in /excel 

#print(cwd)
files = os.listdir(cwd)  
a=[]
for file in files:
   if file.startswith('Topics_T1_5_'): a.append(file)
a=natsort.natsorted(a)
excl_list = []
for file in a: excl_list.append(pd.read_excel(file))

excl_merged = pd.DataFrame()
 
for excl_file in excl_list:
    excl_merged = excl_merged.append(excl_file, ignore_index=True)
excl_merged.to_excel('T1_4_topics.xlsx')
#%% Get frequency of T1-topics

#print(cwd)
files = os.listdir(cwd)  
a=[]
for file in files:
   if file.startswith('doc_T1_5_'): a.append(file)
a=natsort.natsorted(a)
F=[]
for file in a:
    x=pd.read_excel(file)
    F.append(pd.DataFrame(x['Topic'].value_counts()).sort_index().Topic.values)
F=np.array(F)
F=F.reshape(-1,1)
pd.DataFrame(F).to_excel('F.xlsx')