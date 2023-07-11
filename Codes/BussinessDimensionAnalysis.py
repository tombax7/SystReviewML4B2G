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
#%%
Orig=[]
Mod=[]
df=pd.read_excel('doc52.xlsx')
#___________________________________________
#___________________________________________
#%%
#Non-intrusive load monitoring
a=df.loc[df['Topic']==4]['Date']
concat=pd.DataFrame(a,columns=['Date'])
T=concat.groupby(['Date']).size()
T=T.drop(labels=2023)
years = T.index.unique()
missing_years = [y for y in range(2000, 2022+1) if y not in years]
for i in range(0,len(missing_years)): T[missing_years[i]]=0
T=T.sort_index()
Orig.append(mk.original_test(T.values))
Mod.append(mk.hamed_rao_modification_test(T.values))
print('NILM treding original:', mk.original_test(T.values))
print('NILM treding modified:', mk.hamed_rao_modification_test(T.values))
#___________________________________________
#Fault prevention, diagnosis and protection 
a=df.loc[df['Topic']==3]['Date']
b=df.loc[df['Topic']==26]['Date']
c=df.loc[df['Topic']==36]['Date']
concat=pd.concat([a,b,c])
concat=concat.reset_index(drop=True)
concat=pd.DataFrame(concat,columns=['Date'])
T=concat.groupby(['Date']).size()
T=T.drop(labels=2023)
years = T.index.unique()
missing_years = [y for y in range(2000, 2022+1) if y not in years]
for i in range(0,len(missing_years)): T[missing_years[i]]=0
T=T.sort_index()
Orig.append(mk.original_test(T.values))
Mod.append(mk.hamed_rao_modification_test(T.values))
print('Fault prevention, diagnosis and protection original:', mk.original_test(T.values))
print('Fault prevention, diagnosis and protection treding modified:', mk.hamed_rao_modification_test(T.values))
#___________________________________________
#Non-technical losses
a=df.loc[df['Topic']==2]['Date']
concat=pd.DataFrame(a,columns=['Date'])
T=concat.groupby(['Date']).size()
T=T.drop(labels=2023)
years = T.index.unique()
missing_years = [y for y in range(2000, 2022+1) if y not in years]
for i in range(0,len(missing_years)): T[missing_years[i]]=0
T=T.sort_index()
Orig.append(mk.original_test(T.values))
Mod.append(mk.hamed_rao_modification_test(T.values))
print('Non-technical losses treding original:', mk.original_test(T.values))
print('Non-technical losses treding modified:', mk.hamed_rao_modification_test(T.values))
#___________________________________________
#Restoration
a=df.loc[df['Topic']==9]['Date']
b=df.loc[df['Topic']==44]['Date']
concat=pd.concat([a,b])
concat=concat.reset_index(drop=True)
concat=pd.DataFrame(concat,columns=['Date'])
T=concat.groupby(['Date']).size()
T=T.drop(labels=2023)
years = T.index.unique()
missing_years = [y for y in range(2000, 2022+1) if y not in years]
for i in range(0,len(missing_years)): T[missing_years[i]]=0
T=T.sort_index()
Orig.append(mk.original_test(T.values))
Mod.append(mk.hamed_rao_modification_test(T.values))
print('Restorationoriginal:', mk.original_test(T.values))
print('Restoration treding modified:', mk.hamed_rao_modification_test(T.values))
#___________________________________________
#Energy and flexibility trading
a=df.loc[df['Topic']==7]['Date']
b=df.loc[df['Topic']==30]['Date']
c=df.loc[df['Topic']==41]['Date']
concat=pd.concat([a,b,c])
concat=concat.reset_index(drop=True)
concat=pd.DataFrame(concat,columns=['Date'])
T=concat.groupby(['Date']).size()
T=T.drop(labels=2023)
years = T.index.unique()
missing_years = [y for y in range(2000, 2022+1) if y not in years]
for i in range(0,len(missing_years)): T[missing_years[i]]=0
T=T.sort_index()
Orig.append(mk.original_test(T.values))
Mod.append(mk.hamed_rao_modification_test(T.values))
print('Energy and flexibility original:', mk.original_test(T.values))
print('Energy and flexibility treding modified:', mk.hamed_rao_modification_test(T.values))
#___________________________________________
#Forecasting
a=df.loc[df['Topic']==5]['Date']
b=df.loc[df['Topic']==22]['Date']
c=df.loc[df['Topic']==32]['Date']
concat=pd.concat([a,b,c])
concat=concat.reset_index(drop=True)
concat=pd.DataFrame(concat,columns=['Date'])
T=concat.groupby(['Date']).size()
T=T.drop(labels=2023)
years = T.index.unique()
missing_years = [y for y in range(2000, 2022+1) if y not in years]
for i in range(0,len(missing_years)): T[missing_years[i]]=0
T=T.sort_index()
Orig.append(mk.original_test(T.values))
Mod.append(mk.hamed_rao_modification_test(T.values))
print('Forecasting original:', mk.original_test(T.values))
print('Forecasting treding modified:', mk.hamed_rao_modification_test(T.values))
#___________________________________________
#Storage and EV analytics
a=df.loc[df['Topic']==10]['Date']
b=df.loc[df['Topic']==19]['Date']
concat=pd.concat([a,b])
concat=concat.reset_index(drop=True)
concat=pd.DataFrame(concat,columns=['Date'])
T=concat.groupby(['Date']).size()
T=T.drop(labels=2023)
years = T.index.unique()
missing_years = [y for y in range(2000, 2022+1) if y not in years]
for i in range(0,len(missing_years)): T[missing_years[i]]=0
T=T.sort_index()
Orig.append(mk.original_test(T.values))
Mod.append(mk.hamed_rao_modification_test(T.values))
print('Storage and EV analytics original:', mk.original_test(T.values))
print('Storage and EV analytics treding modified:', mk.hamed_rao_modification_test(T.values))
#___________________________________________
#Grid planning
a=df.loc[df['Topic']==11]['Date']
concat=pd.DataFrame(a,columns=['Date'])
T=concat.groupby(['Date']).size()
T=T.drop(labels=2023)
years = T.index.unique()
missing_years = [y for y in range(2000, 2022+1) if y not in years]
for i in range(0,len(missing_years)): T[missing_years[i]]=0
T=T.sort_index()
Orig.append(mk.original_test(T.values))
Mod.append(mk.hamed_rao_modification_test(T.values))
print('Grid planning treding original:', mk.original_test(T.values))
print('Grid planning treding modified:', mk.hamed_rao_modification_test(T.values))
#___________________________________________
#Cybersecurity
a=df.loc[df['Topic']==14]['Date']
b=df.loc[df['Topic']==34]['Date']
concat=pd.concat([a,b])
concat=concat.reset_index(drop=True)
concat=pd.DataFrame(concat,columns=['Date'])
T=concat.groupby(['Date']).size()
T=T.drop(labels=2023)
years = T.index.unique()
missing_years = [y for y in range(2000, 2022+1) if y not in years]
for i in range(0,len(missing_years)): T[missing_years[i]]=0
T=T.sort_index()
Orig.append(mk.original_test(T.values))
Mod.append(mk.hamed_rao_modification_test(T.values))
print('Cybersecurity original:', mk.original_test(T.values))
print('Cybersecurity treding modified:', mk.hamed_rao_modification_test(T.values))
#___________________________________________
#Stability analysis
a=df.loc[df['Topic']==15]['Date']
concat=pd.DataFrame(a,columns=['Date'])
T=concat.groupby(['Date']).size()
T=T.drop(labels=2023)
years = T.index.unique()
missing_years = [y for y in range(2000, 2022+1) if y not in years]
for i in range(0,len(missing_years)): T[missing_years[i]]=0
T=T.sort_index()
Orig.append(mk.original_test(T.values))
Mod.append(mk.hamed_rao_modification_test(T.values))
print('Stability analysis treding original:', mk.original_test(T.values))
print('Stability analysis treding modified:', mk.hamed_rao_modification_test(T.values))
#___________________________________________
#Building-level events detection
a=df.loc[df['Topic']==16]['Date']
concat=pd.DataFrame(a,columns=['Date'])
T=concat.groupby(['Date']).size()
T=T.drop(labels=2023)
years = T.index.unique()
missing_years = [y for y in range(2000, 2022+1) if y not in years]
for i in range(0,len(missing_years)): T[missing_years[i]]=0
T=T.sort_index()
Orig.append(mk.original_test(T.values))
Mod.append(mk.hamed_rao_modification_test(T.values))
print('Building-level events detection treding original:', mk.original_test(T.values))
print('Building-level events detection treding modified:', mk.hamed_rao_modification_test(T.values))
#___________________________________________
#Load profiling                             
a=df.loc[df['Topic']==39]['Date']
concat=pd.DataFrame(a,columns=['Date'])
T=concat.groupby(['Date']).size()
T=T.drop(labels=2023)
years = T.index.unique()
missing_years = [y for y in range(2000, 2022+1) if y not in years]
for i in range(0,len(missing_years)): T[missing_years[i]]=0
T=T.sort_index()
Orig.append(mk.original_test(T.values))
Mod.append(mk.hamed_rao_modification_test(T.values))
print('Load profiling treding original:', mk.original_test(T.values))
print('Load profiling treding modified:', mk.hamed_rao_modification_test(T.values))
#___________________________________________
#Optimal power flow                         
a=df.loc[df['Topic']==46]['Date']
concat=pd.DataFrame(a,columns=['Date'])
T=concat.groupby(['Date']).size()
T=T.drop(labels=2023)
years = T.index.unique()
missing_years = [y for y in range(2000, 2022+1) if y not in years]
for i in range(0,len(missing_years)): T[missing_years[i]]=0
T=T.sort_index()
Orig.append(mk.original_test(T.values))
Mod.append(mk.hamed_rao_modification_test(T.values))
print('OPF original:', mk.original_test(T.values))
print('OPF modified:', mk.hamed_rao_modification_test(T.values))
#___________________________________________
#Energy management and control
a=df.loc[df['Topic']==0]['Date']
b=df.loc[df['Topic']==12]['Date']
c=df.loc[df['Topic']==20]['Date']
a1=df.loc[df['Topic']==25]['Date']
b1=df.loc[df['Topic']==28]['Date']
c1=df.loc[df['Topic']==42]['Date']
a2=df.loc[df['Topic']==45]['Date']
concat=pd.concat([a,b,c,a1,b1,c1,a2])
concat=concat.reset_index(drop=True)
concat=pd.DataFrame(concat,columns=['Date'])
T=concat.groupby(['Date']).size()
T=T.drop(labels=2023)
years = T.index.unique()
missing_years = [y for y in range(2000, 2022+1) if y not in years]
for i in range(0,len(missing_years)): T[missing_years[i]]=0
T=T.sort_index()
Orig.append(mk.original_test(T.values))
Mod.append(mk.hamed_rao_modification_test(T.values))
print('Energy M and control original:', mk.original_test(T.values))
print('Energy M and control treding modified:', mk.hamed_rao_modification_test(T.values))
#___________________________________________
#Grid variables estimation
a=df.loc[df['Topic']==29]['Date']
concat=pd.DataFrame(a,columns=['Date'])
T=concat.groupby(['Date']).size()
T=T.drop(labels=2023)
years = T.index.unique()
missing_years = [y for y in range(2000, 2022+1) if y not in years]
for i in range(0,len(missing_years)): T[missing_years[i]]=0
T=T.sort_index()
Orig.append(mk.original_test(T.values))
Mod.append(mk.hamed_rao_modification_test(T.values))
print('Grid variables estimation original:', mk.original_test(T.values))
print('Grid variables estimation modified:', mk.hamed_rao_modification_test(T.values))