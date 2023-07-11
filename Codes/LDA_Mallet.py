import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import nltk 
nltk.download('stopwords')
import pyLDAvis
import pyLDAvis.gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
#%%
df=pd.read_excel('Review_database.xlsx')
plt.figure(figsize=(8,4))
sns.countplot(x='Date', data=df);
data = list(df.Abstracts)
data = [x for x in data if str(x) != 'nan']
bigram = gensim.models.Phrases(data, min_count=20, threshold=100)
trigram = gensim.models.Phrases(bigram[data], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
#%%
#____________________________PREPROCESSING____________________________________
#______________________________________________________________________________
#Extract review papers descriptive analytics 

Rev=df.loc[df['Type']=='Review']
Rev['Date'].value_counts().sort_index().plot(kind='bar')
Rev.reset_index(drop=True)
doi_list = pd.DataFrame(Rev.Doi.values,columns=['DOIs'])
#Make txt file to extract pdfs from schihub
with open('input.txt', 'w') as fout:
    fout.writelines(line+str('\n') for line in doi_list.DOIs)
#%%
#______________________________________________________________________________

# only need tagger, no need for parser and named entity recognizer, for faster implementation
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# get stopwords from nltk library
stop_words = nltk.corpus.stopwords.words('english')

def process_words(texts, stop_words=stop_words, allowed_tags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    
    """Convert a document into a list of lowercase tokens, build bigrams-trigrams, implement lemmatization"""
    
    # remove stopwords, short tokens and letter accents 
    texts = [[word for word in simple_preprocess(str(doc), deacc=True, min_len=3) if word not in stop_words] for doc in texts]
    
    # bi-gram and tri-gram implementation
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    
    texts_out = []
    
    # implement lemmatization and filter out unwanted part of speech tags
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_tags])
    
    # remove stopwords and short tokens again after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc), deacc=True, min_len=3) if word not in stop_words] for doc in texts_out]    
    
    return texts_out

data_ready = process_words(data)
#______________________________________________________________________________
#_________________________Words frequency______________________________________
id2word = corpora.Dictionary(data_ready)
print('Total Vocabulary Size:', len(id2word))
corpus = [id2word.doc2bow(text) for text in data_ready]
#creating a dictionary and then convert it to a dataframe that shows each word in the corpus and its frequency:
dict_corpus = {}
for i in range(len(corpus)):
  for idx, freq in corpus[i]:
    if id2word[idx] in dict_corpus:
      dict_corpus[id2word[idx]] += freq
    else:
       dict_corpus[id2word[idx]] = freq
dict_df = pd.DataFrame.from_dict(dict_corpus, orient='index', columns=['freq'])
#words frequency histogram
plt.figure(figsize=(8,6))
sns.displot(dict_df['freq'], bins=100);
dict_df.sort_values('freq', ascending=False).head(10)
#Filtering out externals (high-frequency and low-frequency words in documents)
extension = dict_df[dict_df.freq>4000].index.tolist()
# add high frequency words to stop words list
stop_words.extend(extension)
# rerun the process_words function
data_ready = process_words(data)
# recreate Dictionary
id2word = corpora.Dictionary(data_ready)
print('Total Vocabulary Size:', len(id2word))
# Filter out words that occur less than 10 documents, or more than 95% of the documents.
id2word.filter_extremes(no_below=3, no_above=0.95)
print('Total Vocabulary Size:', len(id2word))
# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_ready]
#______________________________________________________________________________
#%%––––––––––––––––––Topic modelling- DLA
import little_mallet_wrapper as lmw
mallet_path='/Users/thanosbach/mallet-2.0.8/bin/mallet'
#______________________________________________________________________________
#%%First approach
output_directory_path = '/Users/thanosbach/Desktop/ReviewAIandSGs'
num_topics=50
for i in range(0,len(data_ready)):data_ready[i]=' '.join(data_ready[i])

lmw.print_dataset_stats(data_ready)
topic_keys, topic_distributions = lmw.quick_train_topic_model(mallet_path, output_directory_path, num_topics, data_ready)
assert(len(topic_distributions) == len(data_ready))
for i, t in enumerate(topic_keys):
    print(i, '\t', ' '.join(t[:10]))
    
for p, d in lmw.get_top_docs(data_ready, topic_distributions, topic_index=0, n=3):
    print(round(p, 4), d)
    print()
#______________________________________________________________________________
#______________________________________________________________________________
#%%Second approach
from pprint import pprint
ldamallet = LdaMallet(mallet_path, corpus=corpus, num_topics=10, id2word=id2word,iterations=15)
pprint(ldamallet.show_topics(formatted=False))
# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_ready, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('Coherence Score: ', coherence_ldamallet)
tm_results = ldamallet[corpus]
#most dominant topic of each document
corpus_topics = [sorted(topics, key=lambda record: -record[1])[0] for topics in tm_results]
#most probable words for the given topicid
topics = [[(term, round(wt, 3)) for term, wt in ldamallet.show_topic(n, topn=20)] for n in range(0, ldamallet.num_topics)]
topics_df = pd.DataFrame([[term for term, wt in topic] for topic in topics], columns = ['Term'+str(i) for i in range(1, 21)], index=['Topic '+str(t) for t in range(1, ldamallet.num_topics+1)]).T
topics_df.head()
pd.set_option('display.max_colwidth', -1)
topics_df = pd.DataFrame([', '.join([term for term, wt in topic]) for topic in topics], columns = ['Terms per Topic'], index=['Topic'+str(t) for t in range(1, ldamallet.num_topics+1)] )
topics_df
#______________________________________________________________________________
#%%Optimal topic number selection
# display a progress meter
from tqdm import tqdm

def topic_model_coherence_generator(corpus, texts, dictionary, start_topic_count, end_topic_count, step, cpus):
  models = []
  uci = []
  u=[]
  v=[]
  for topic_nums in tqdm(range(start_topic_count, end_topic_count+1, step)):
      mallet_lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=dictionary,
                                               num_topics=topic_nums, 
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
   # mallet_lda_model = LdaMallet(mallet_path=mallet_path, corpus=corpus, num_topics=topic_nums, id2word=dictionary)
      c_uci_coherence = CoherenceModel(model=mallet_lda_model, corpus=corpus, texts=texts, dictionary=dictionary, coherence='c_uci')
      c_uci_score = c_uci_coherence.get_coherence()
      uci.append(c_uci_score)
      u_coherence = CoherenceModel(model=mallet_lda_model, corpus=corpus, texts=texts, dictionary=dictionary, coherence='u_mass')
      u_score = u_coherence.get_coherence()
      u.append(u_score)
      models.append(mallet_lda_model)
      c_v_coherence = CoherenceModel(model=mallet_lda_model, corpus=corpus, texts=texts, dictionary=dictionary, coherence='c_v')
      c_v_score = c_v_coherence.get_coherence()
      v.append(c_v_score)
  return models, uci, u, v


#Run multiple times to reduce the variation caused by the random sampling initialization
lda_models=[]
cv=[]
u_mass=[]
vv=[]
run=2
for i in range(0,run):
    l, c, u, v = topic_model_coherence_generator(corpus=corpus, texts=data_ready, dictionary=id2word, start_topic_count=10, end_topic_count=24, step=2, cpus=-1)
    lda_models.append(l)
    cv.append(c)
    u_mass.append(u)
    x_ax = range(10, 26, 2)
    y_ax = cv[i]
    plt.figure(figsize=(12, 6))
    plt.plot(x_ax, y_ax, label = run)
    xl = plt.xlabel('Number of Topics')
    yl = plt.ylabel('c_uci')
    plt.rcParams['figure.facecolor'] = 'white'
    x_ax = range(10, 26, 2)
    y_ax = u_mass[i]
    plt.figure(figsize=(12, 6))
    plt.plot(x_ax, y_ax, label = run)
    xl = plt.xlabel('Number of Topics')
    yl = plt.ylabel('u_mass')
    plt.rcParams['figure.facecolor'] = 'white'
    vv.append(v)
    x_ax = range(10, 26, 2)
    y_ax = vv[i]
    plt.figure(figsize=(12, 6))
    plt.plot(x_ax, y_ax, label = run)
    xl = plt.xlabel('Number of Topics')
    yl = plt.ylabel('c_v')
    plt.rcParams['figure.facecolor'] = 'white'
plt.show()
#%%#––––––––––––––EVALUATE TOPIC DISTRIBUTION SIMILARITY
import seaborn as sns
from gensim.matutils import hellinger
import matplotlib.pylab as plt

def parse_topic_string(topic):
    # takes the string returned by model.show_topics()
    # split on strings to get topics and the probabilities
    topic = topic.split('+')
    # list to store topic bows
    topic_bow = []
    for word in topic:
        # split probability and word
        prob, word = word.split('*')
        # get rid of spaces and quote marks
        word = word.replace(" ","").replace('"', '')
        # convert to word_type
        word = model.id2word.doc2bow([word])[0][0]
        topic_bow.append((word, float(prob)))
    return topic_bow
#x=model.show_topics(num_topics=len(model.get_topics()))



M=pd.DataFrame()
for z in range(0,len(lda_models)):
    for q in range(0,len(lda_models[z])):
        model=lda_models[z][q]
        x=model.show_topics(num_topics=len(model.get_topics()))
        y=[]
        for i in range(0,len(model.get_topics())):
            x[i] = parse_topic_string(x[i][1])

        for i in range(0,len(model.get_topics())):
            [y.append(1-hellinger(x[i],x[j])) for j in range(0,len(model.get_topics()))]
        M.loc[z,q]=pd.DataFrame(y).mean()[0]


def to_matrix(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]
print('Similarity Topic Index (1-Hellinger):'+ str(pd.DataFrame(y).mean()[0]))
np=to_matrix(y,len(model.get_topics()))
ax = sns.heatmap(np, linewidth=0.5)
plt.show()

M=pd.DataFrame()
for z in range(0,len(lda_models)):
    for q in range(0,len(lda_models[z])):
        print(q)
        mallet_lda_model=lda_models[z][q]
        c_v_coherence = CoherenceModel(model=mallet_lda_model, corpus=corpus, texts=data_ready, dictionary=id2word, coherence='c_v')
        c_v_score = c_v_coherence.get_coherence()
        M.loc[z,q]=c_v_score



#%%_______________Visualization
import wordcloud
from wordcloud import WordCloud

# initiate wordcloud object
wc = WordCloud(background_color="white", colormap="Dark2", max_font_size=150, random_state=42)

# set the figure size
plt.rcParams['figure.figsize'] = [20, 15]

# Create subplots for each topic
for i in range(10):

    wc.generate(text=topics_df["Terms per Topic"][i])
    
    plt.subplot(5, 4, i+1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(topics_df.index[i])

plt.show()
#Visualization with pyLDAvis
from gensim.models.ldamodel import LdaModel

def convertldaMalletToldaGen(mallet_model):
    model_gensim = LdaModel(
        id2word=mallet_model.id2word, num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha) 
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim

ldagensim = convertldaMalletToldaGen(ldamallet)

import pyLDAvis.gensim as gensimvis
vis_data = gensimvis.prepare(ldagensim, corpus, id2word, sort_topics=False)
pyLDAvis.show(vis_data)
#______________________________________________________________________________
#Dominant topic for each Document
corpus_topic_df = pd.DataFrame()
corpus_topic_df['Title'] = df.Title
corpus_topic_df['Dominant Topic'] = [item[0]+1 for item in corpus_topics]
corpus_topic_df['Contribution %'] = [round(item[1]*100, 2) for item in corpus_topics]
corpus_topic_df['Topic Terms'] = [topics_df.iloc[t[0]]['Terms per Topic'] for t in corpus_topics]
corpus_topic_df.head()
#Dominant papers per topic
dominant_topic_df = corpus_topic_df.groupby('Dominant Topic').agg(
                                  Doc_Count = ('Dominant Topic', np.size),
                                  Total_Docs_Perc = ('Dominant Topic', np.size)).reset_index()

dominant_topic_df['Total_Docs_Perc'] = dominant_topic_df['Total_Docs_Perc'].apply(lambda row: round((row*100) / len(corpus), 2))

dominant_topic_df
#______________________________________________________________________________
#Most dominant paper per topic
corpus_topic_df.groupby('Dominant Topic').apply(lambda topic_set: (topic_set.sort_values(by=['Contribution %'], ascending=False).iloc[0])).reset_index(drop=True)
#______________________________________________________________________________
#Topic weight per document & trend analysis
df_weights = pd.DataFrame.from_records([{v: k for v, k in row} for row in tm_results])
df_weights.columns = ['Topic ' + str(i) for i in range(1,51)]
df_weights
df_weights['Date']=df.Date
df_weights.groupby('Date').mean()
#Dominant topic per document #Better trend analysis
df_weights['Dominant'] = df_weights.drop('Date', axis=1).idxmax(axis=1)
df_weights.head()
#Percentage of dominant topics in a year
df_dominance = df_weights.groupby('Date')['Dominant'].value_counts(normalize=True).unstack()
df_dominance