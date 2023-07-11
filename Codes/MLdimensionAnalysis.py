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
import pymannkendall as mk
import os
import natsort
Orig=[]
Mod=[]
#print(cwd)
cwd='/Users/thanosbach/Desktop/ReviewAIandSGs/Services'
files = os.listdir(cwd)  
a=[]
for file in files:
   if file.startswith('T_'): a.append(file)
a=natsort.natsorted(a)
#%% ML AREA ANALYSIS
M=[]
for pp in range(0,len(a)):
    print(a[pp])
    #df=pd.read_excel('doc52.xlsx')
    df=pd.read_excel(a[pp])
    # RL_word = ['reinforcement learning', 'deep q learning', 'multi-agent', 'deep deterministic','actor critic','sarsa agent','policy gradient agent','monte carlo tree search','q-learning']
    # SL_word=['supervised','support vector machine','ensemble','linear regression','nearest','naive bayes','classification','regression','random forest','trees','neural network','multilayer perceptron','ensemble','boosting','bagging','logistic regression','k-nearest','convolutional', 'long short-term memory','lstm','cnn','feedforward','gradient boosting','xgboost','lightgbm','vision transformers','vision transformer','transformer networks','transformer neural','gated recurrent','graph neural']
    # UL_word=['unsupervised','k-means','k-medoids','k means','hidden markov','clustering','autoencoder','boltzmann','principal component analysis','dimensionality reduction','dbscan','density estimation','gaussian mixture model','birch','singular value decomposition','self-organizing maps','t-SNE','generative adversarial networks','gaussian mixture models']
    # SS_word=['semi-supervised','semi supervised','self-training','self training']
    # AL_word=['active learning']
    # TL_word=['transfer learning','domain adaptation','multi-task learning','multi task learning']
    EX_word=['explainable','explainability','interpretability','interpretable']
    # FL_word=['federated learning','federated','fedavg','federated averaging','fedsgd']
    # SP_word=['fourier','wavelet','filtering','spectral','signal smoothing','signal compression']
    # TinyML_word=['resource-constrained','resource constrained','efficient inference','quantization','pruning','compression','sparse representations','low-rank approximation','edge inference']
    # DL_word=['ann','deep learning','convolutional', 'recurrent neural','long short-term memory', 'autoencoder','neural networks','deep neural','neural network','multilayer perceptron','boltzmann machines','attention mechanism','vision transformers','vision transformer','transformer networks','transformer neural','gated recurrent','graph neural','deep deterministic']
    # RG_word=['linear regression','logistic','lasso','ridge regression']
    # TR_word=['gradient boosting','xgboost','lightgbm','decision trees','random forest']
    # CL_word=['k means,k-means', 'k-medoids','k-nearest neighbors', 'principal component analysis','hierarchical clustering', 'dbscan','gaussian mixture models']
    # EN_word=['bagging','boosting','stacking']
    # BY_word=['bayesian','gaussian','markov chain monte carlo','latent dirichlet allocation','hidden markov models']
    RL_word=['feedforward neural','deep feedforward','multi-layer perceptrons','multi layer perceptrons','multi-layer perceptron','multi layer perceptron']
    SL_word=['recurrent neural','long short-term memory','lstm','rnn','gated recurrent','grus']
    UL_word=['convolutional neural', 'cnn','convolution']
    SS_word=['attention mechanism','vision transformers','vision transformer','transformer networks','transformer neural']
    AL_word=['graph neural','graph attention','graph convolutional','gated graph sequence']
    TL_word=['linear regression','logistic','lasso','ridge regression']
    #EX_word=['support vector','svm']
    FL_word=['value-based method','q-learning','q learning','deep q','q-networks','q networks','sarsa']
    SP_word=['policy-based methods','policy gradient', 'proximal policy optimization', 'trust region policy','trpo']
    TinyML_word= ['actor critic','advantage actor-critic','a2c', 'asynchronous advantage','a3c','soft actor-critic','sac']
    DL_word=['reinforced model predictive','rl-mpc','monte carlo tree search','mcts','dyna-q']
    RG_word=['experience replay','deterministic policy gradient']
    TR_word=['autoencoder','autoencoders']
    CL_word=['generative adversarial','gans']
    EN_word=['random forest']
    BY_word=['arima']
    SL=[];SL_dates=[]
    UL=[];UL_dates=[]
    SS=[];SS_dates=[]
    RL=[];RL_dates=[]
    AL=[];AL_dates=[]
    TL=[];TL_dates=[]
    EX=[];EX_dates=[]
    FL=[];FL_dates=[]
    SP=[];SP_dates=[]
    TinyML=[];TinyML_dates=[]
    DL=[];DL_dates=[]
    RG=[];RG_dates=[]
    TR=[];TR_dates=[]
    CL=[];CL_dates=[]
    EN=[];EN_dates=[]
    BY=[];BY_dates=[]
    df['Document']=df['Document'].apply(str.lower)
    for line in df.Document:
        line=line.lower()
        df[df['Document']==line].Date
        if any(word in line for word in SL_word):
            SL.append(line)
            SL_dates.append(df[df['Document']==line].Date.values)
        if any(word in line for word in UL_word):
            UL.append(line)
            UL_dates.append(df[df['Document']==line].Date.values)
        if any(word in line for word in SS_word):
            SS.append(line)
            SS_dates.append(df[df['Document']==line].Date.values)
        if any(word in line for word in RL_word):
            RL.append(line)
            RL_dates.append(df[df['Document']==line].Date.values)
        if any(word in line for word in AL_word):
            AL.append(line)
            AL_dates.append(df[df['Document']==line].Date.values)
        if any(word in line for word in TL_word):
            TL.append(line)
            TL_dates.append(df[df['Document']==line].Date.values)
        if any(word in line for word in EX_word):
            EX.append(line)
            EX_dates.append(df[df['Document']==line].Date.values)
        if any(word in line for word in FL_word):
            FL.append(line)
            FL_dates.append(df[df['Document']==line].Date.values)
        if any(word in line for word in SP_word):
            SP.append(line)
            SP_dates.append(df[df['Document']==line].Date.values)
        if any(word in line for word in TinyML_word):
            TinyML.append(line)
            TinyML_dates.append(df[df['Document']==line].Date.values)
        if any(word in line for word in DL_word):
            DL.append(line)
            DL_dates.append(df[df['Document']==line].Date.values)
        if any(word in line for word in RG_word):
            RG.append(line)
            RG_dates.append(df[df['Document']==line].Date.values)
        if any(word in line for word in TR_word):
            TR.append(line)
            TR_dates.append(df[df['Document']==line].Date.values)
        if any(word in line for word in CL_word):
            CL.append(line)
            CL_dates.append(df[df['Document']==line].Date.values)
        if any(word in line for word in EN_word):
            EN.append(line)
            EN_dates.append(df[df['Document']==line].Date.values)
        if any(word in line for word in BY_word):
            BY.append(line)
            BY_dates.append(df[df['Document']==line].Date.values)
    print('Supervised learning:',len(SL))
    print('Unsupervised learning:',len(UL))
    print('Semi-supervised learning:',len(SS))
    print('Reinforcement learning:',len(RL))
    print('Active learning:',len(AL))
    print('Transfer learning:',len(TL))
    print('Explainable learning:',len(EX))
    print('Federated learning:',len(FL))
    print('Signal processing methods:',len(SP))
    print('TinyML methods:',len(TinyML))
    print('Deep learning:',len(DL))
    print('Regression learning:',len(RG))
    print('Trees learning:',len(TR))
    print('Clustering learning:',len(CL))
    print('Enemble processing methods:',len(EN))
    print('Bayessian methods:',len(BY))
    M.append([len(SL),len(UL),len(SS),len(RL),len(AL),len(TL),len(EX),len(FL),len(SP),len(TinyML),len(DL),len(RG),len(TR),len(CL),len(EN),len(BY)])
    SL_D=[];UL_D=[];SS_D=[];RL_D=[];AL_D=[];TL_D=[];EX_D=[];FL_D=[];SP_D=[];TinyML_D=[];DL_D=[];RG_D=[];TR_D=[];CL_D=[];EN_D=[];BY_D=[]
    for i in range(0,len(SL_dates)): SL_D.append(SL_dates[i][0])
    for i in range(0,len(UL_dates)): UL_D.append(UL_dates[i][0])
    for i in range(0,len(SS_dates)): SS_D.append(SS_dates[i][0])
    for i in range(0,len(RL_dates)): RL_D.append(RL_dates[i][0])
    for i in range(0,len(AL_dates)): AL_D.append(AL_dates[i][0])
    for i in range(0,len(TL_dates)): TL_D.append(TL_dates[i][0])
    for i in range(0,len(EX_dates)): EX_D.append(EX_dates[i][0])
    for i in range(0,len(FL_dates)): FL_D.append(FL_dates[i][0])
    for i in range(0,len(SP_dates)): SP_D.append(SP_dates[i][0])
    for i in range(0,len(TinyML_dates)): TinyML_D.append(TinyML_dates[i][0])
    for i in range(0,len(DL_dates)): DL_D.append(DL_dates[i][0])
    for i in range(0,len(RG_dates)): RG_D.append(RG_dates[i][0])
    for i in range(0,len(TR_dates)): TR_D.append(TR_dates[i][0])
    for i in range(0,len(CL_dates)): CL_D.append(CL_dates[i][0])
    for i in range(0,len(EN_dates)): EN_D.append(EN_dates[i][0])
    for i in range(0,len(BY_dates)): BY_D.append(BY_dates[i][0])
    SL_D=pd.DataFrame(SL_D,columns=['Dates'])
    UL_D=pd.DataFrame(UL_D,columns=['Dates'])
    SS_D=pd.DataFrame(SS_D,columns=['Dates'])
    RL_D=pd.DataFrame(RL_D,columns=['Dates'])
    AL_D=pd.DataFrame(AL_D,columns=['Dates'])
    TL_D=pd.DataFrame(TL_D,columns=['Dates'])
    EX_D=pd.DataFrame(EX_D,columns=['Dates'])
    FL_D=pd.DataFrame(FL_D,columns=['Dates'])
    SP_D=pd.DataFrame(SP_D,columns=['Dates'])
    TinyML_D=pd.DataFrame(TinyML_D,columns=['Dates'])
    DL_D=pd.DataFrame(DL_D,columns=['Dates'])
    RG_D=pd.DataFrame(RG_D,columns=['Dates'])
    TR_D=pd.DataFrame(TR_D,columns=['Dates'])
    CL_D=pd.DataFrame(CL_D,columns=['Dates'])
    EN_D=pd.DataFrame(EN_D,columns=['Dates'])
    BY_D=pd.DataFrame(BY_D,columns=['Dates'])
    SL_D=SL_D.groupby(['Dates']).size()
    try: 
        SL_D=SL_D.drop(labels=2023)
    except: 
        print('ok')
    years = SL_D.index.unique()
    missing_years = [y for y in range(2000, 2022+1) if y not in years]
    try:
        for i in range(0,len(missing_years)): SL_D[missing_years[i]]=0
        SL_D=SL_D.sort_index()
        Orig.append(mk.original_test(SL_D.values))
        Mod.append(mk.hamed_rao_modification_test(SL_D.values))
        print('Supervised treding original:', mk.original_test(SL_D.values))
        print('Supervised treding modified:', mk.hamed_rao_modification_test(SL_D.values))
    except:
        Orig.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
        Mod.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
    UL_D=UL_D.groupby(['Dates']).size()
    try: 
        UL_D=UL_D.drop(labels=2023)
    except: 
        print('ok')
    years = UL_D.index.unique()
    missing_years = [y for y in range(2000, 2022+1) if y not in years]
    try:
        for i in range(0,len(missing_years)): UL_D[missing_years[i]]=0
        UL_D=UL_D.sort_index()
        Orig.append(mk.original_test(UL_D.values))
        Mod.append(mk.hamed_rao_modification_test(UL_D.values))
        print('Unsupervised treding original:', mk.original_test(UL_D.values))
        print('Unsupervised treding modified:', mk.hamed_rao_modification_test(UL_D.values))
    except:
        Orig.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
        Mod.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
    SS_D=SS_D.groupby(['Dates']).size()
    try: 
        SS_D=SS_D.drop(labels=2023)
    except: 
        print('ok')
    years = SS_D.index.unique()
    missing_years = [y for y in range(2000, 2022+1) if y not in years]
    try:
        for i in range(0,len(missing_years)): SS_D[missing_years[i]]=0
        SS_D=SS_D.sort_index()
        Orig.append(mk.original_test(SS_D.values))
        Mod.append(mk.hamed_rao_modification_test(SS_D.values))
        print('Semi-supervised treding original:', mk.original_test(SS_D.values))
        print('Semi-supervised treding modified:', mk.hamed_rao_modification_test(SS_D.values))
    except:
        Orig.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
        Mod.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
    RL_D=RL_D.groupby(['Dates']).size()
    try:
        RL_D=RL_D.drop(labels=2023)
    except: 
        print('ok')
    years = RL_D.index.unique()
    missing_years = [y for y in range(2000, 2022+1) if y not in years]
    try:
        for i in range(0,len(missing_years)): RL_D[missing_years[i]]=0
        RL_D=RL_D.sort_index()
        Orig.append(mk.original_test(RL_D.values))
        Mod.append(mk.hamed_rao_modification_test(RL_D.values))
        print('Reinforcement treding original:', mk.original_test(RL_D.values))
        print('Reinforcement treding modified:', mk.hamed_rao_modification_test(RL_D.values))
    except:
        Orig.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
        Mod.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
    AL_D=AL_D.groupby(['Dates']).size()
    try:
        AL_D=AL_D.drop(labels=2023)
    except: 
        print('ok')
    years = AL_D.index.unique()
    missing_years = [y for y in range(2000, 2022+1) if y not in years]
    try:
        for i in range(0,len(missing_years)): AL_D[missing_years[i]]=0
        AL_D=AL_D.sort_index()
        Orig.append(mk.original_test(AL_D.values))
        Mod.append(mk.hamed_rao_modification_test(AL_D.values))
        print('Active learning treding original:', mk.original_test(AL_D.values))
        print('Active learning treding modified:', mk.hamed_rao_modification_test(AL_D.values))
    except:
        Orig.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
        Mod.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
    TL_D=TL_D.groupby(['Dates']).size()
    try:
        TL_D=TL_D.drop(labels=2023)
    except: 
        print('ok')
    years = TL_D.index.unique()
    missing_years = [y for y in range(2000, 2022+1) if y not in years]
    try: 
        for i in range(0,len(missing_years)): TL_D[missing_years[i]]=0
        TL_D=TL_D.sort_index()
        Orig.append(mk.original_test(TL_D.values))
        Mod.append(mk.hamed_rao_modification_test(TL_D.values))
        print('Transfer learning treding original:', mk.original_test(TL_D.values))
        print('Transfer learning treding modified:', mk.hamed_rao_modification_test(TL_D.values))
    except:
        Orig.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
        Mod.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
    EX_D=EX_D.groupby(['Dates']).size()
    try:
        EX_D=EX_D.drop(labels=2023)
    except: 
        print('ok')
    years = EX_D.index.unique()
    missing_years = [y for y in range(2000, 2022+1) if y not in years]
    try:
        for i in range(0,len(missing_years)): EX_D[missing_years[i]]=0
        EX_D=EX_D.sort_index()
        Orig.append(mk.original_test(EX_D.values))
        Mod.append(mk.hamed_rao_modification_test(EX_D.values))
        print('Explainable treding original:', mk.original_test(EX_D.values))
        print('Explainable treding modified:', mk.hamed_rao_modification_test(EX_D.values))
    except:
        Orig.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
        Mod.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
    FL_D=FL_D.groupby(['Dates']).size()
    try:
        FL_D=FL_D.drop(labels=2023)
    except: 
        print('ok')
    years = FL_D.index.unique()
    missing_years = [y for y in range(2000, 2022+1) if y not in years]
    try:
        for i in range(0,len(missing_years)): FL_D[missing_years[i]]=0
        FL_D=FL_D.sort_index()
        Orig.append(mk.original_test(FL_D.values))
        Mod.append(mk.hamed_rao_modification_test(FL_D.values))
        print('Federated learning treding original:', mk.original_test(FL_D.values))
        print('Federated learning treding modified:', mk.hamed_rao_modification_test(FL_D.values))
    except:
        Orig.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
        Mod.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
    SP_D=SP_D.groupby(['Dates']).size()
    try:
        SP_D=SP_D.drop(labels=2023)
    except: 
        print('ok')
    years = SP_D.index.unique()
    missing_years = [y for y in range(2000, 2022+1) if y not in years]
    try:
        for i in range(0,len(missing_years)): SP_D[missing_years[i]]=0
        SP_D=SP_D.sort_index()
        Orig.append(mk.original_test(SP_D.values))
        Mod.append(mk.hamed_rao_modification_test(SP_D.values))
        print('Signal Processing treding original:', mk.original_test(SP_D.values))
        print('Signal Processing treding modified:', mk.hamed_rao_modification_test(SP_D.values))
    except:
        Orig.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
        Mod.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
    TinyML_D=TinyML_D.groupby(['Dates']).size()
    try:
        TinyML_D=TinyML_D.drop(labels=2023)
    except: 
        print('ok')
    years = TinyML_D.index.unique()
    missing_years = [y for y in range(2000, 2022+1) if y not in years]
    try:
        for i in range(0,len(missing_years)): TinyML_D[missing_years[i]]=0
        TinyML_D=TinyML_D.sort_index()
        Orig.append(mk.original_test(TinyML_D.values))
        Mod.append(mk.hamed_rao_modification_test(TinyML_D.values))
        print('TinyML treding original:', mk.original_test(TinyML_D.values))
        print('TinyML treding modified:', mk.hamed_rao_modification_test(TinyML_D.values))
    except:
        Orig.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
        Mod.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
    DL_D=DL_D.groupby(['Dates']).size()
    try:
        DL_D=DL_D.drop(labels=2023)
    except: 
        print('ok')
    years = DL_D.index.unique()
    missing_years = [y for y in range(2000, 2022+1) if y not in years]
    try:
        for i in range(0,len(missing_years)): DL_D[missing_years[i]]=0
        DL_D=DL_D.sort_index()
        Orig.append(mk.original_test(DL_D.values))
        Mod.append(mk.hamed_rao_modification_test(DL_D.values))
        print('DEEP learning treding original:', mk.original_test(DL_D.values))
        print('DEEP learning treding modified:', mk.hamed_rao_modification_test(DL_D.values))
    except:
        Orig.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
        Mod.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
    RG_D=RG_D.groupby(['Dates']).size()
    try:
        RG_D=RG_D.drop(labels=2023)
    except: 
        print('ok')
    years = RG_D.index.unique()
    missing_years = [y for y in range(2000, 2022+1) if y not in years]
    try: 
        for i in range(0,len(missing_years)): RG_D[missing_years[i]]=0
        RG_D=RG_D.sort_index()
        Orig.append(mk.original_test(RG_D.values))
        Mod.append(mk.hamed_rao_modification_test(RG_D.values))
        print('REGRESSION treding original:', mk.original_test(RG_D.values))
        print('regression treding modified:', mk.hamed_rao_modification_test(RG_D.values))
    except:
        Orig.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
        Mod.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
    TR_D=TR_D.groupby(['Dates']).size()
    try:
        TR_D=TR_D.drop(labels=2023)
    except: 
        print('ok')
    years = TR_D.index.unique()
    missing_years = [y for y in range(2000, 2022+1) if y not in years]
    try:
        for i in range(0,len(missing_years)): TR_D[missing_years[i]]=0
        TR_D=TR_D.sort_index()
        Orig.append(mk.original_test(TR_D.values))
        Mod.append(mk.hamed_rao_modification_test(TR_D.values))
        print('Trees original:', mk.original_test(TR_D.values))
        print('Trees modified:', mk.hamed_rao_modification_test(TR_D.values))
    except:
        Orig.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
        Mod.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
    CL_D=CL_D.groupby(['Dates']).size()
    try:
        CL_D=CL_D.drop(labels=2023)
    except: 
        print('ok')
    years = CL_D.index.unique()
    missing_years = [y for y in range(2000, 2022+1) if y not in years]
    try:
        for i in range(0,len(missing_years)): CL_D[missing_years[i]]=0
        CL_D=CL_D.sort_index()
        Orig.append(mk.original_test(CL_D.values))
        Mod.append(mk.hamed_rao_modification_test(CL_D.values))
        print('Clustering treding original:', mk.original_test(CL_D.values))
        print('Clustering treding modified:', mk.hamed_rao_modification_test(CL_D.values))
    except:
        Orig.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
        Mod.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
    EN_D=EN_D.groupby(['Dates']).size()
    try:
        EN_D=EN_D.drop(labels=2023)
    except: 
        print('ok')
    years = EN_D.index.unique()
    missing_years = [y for y in range(2000, 2022+1) if y not in years]
    try:
        for i in range(0,len(missing_years)): EN_D[missing_years[i]]=0
        EN_D=EN_D.sort_index()
        Orig.append(mk.original_test(EN_D.values))
        Mod.append(mk.hamed_rao_modification_test(EN_D.values))
        print('Ensemble treding original:', mk.original_test(EN_D.values))
        print('Ensemble treding modified:', mk.hamed_rao_modification_test(EN_D.values))
    except:
        Orig.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
        Mod.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
    BY_D=BY_D.groupby(['Dates']).size()
    try:
        BY_D=BY_D.drop(labels=2023)
    except: 
        print('ok')
    years = BY_D.index.unique()
    missing_years = [y for y in range(2000, 2022+1) if y not in years]
    try:
        for i in range(0,len(missing_years)): BY_D[missing_years[i]]=0
        BY_D=BY_D.sort_index()
        Orig.append(mk.original_test(BY_D.values))
        Mod.append(mk.hamed_rao_modification_test(BY_D.values))
        print('BY treding original:', mk.original_test(BY_D.values))
        print('BY treding modified:', mk.hamed_rao_modification_test(BY_D.values))
    except:
        Orig.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
        Mod.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
pd.DataFrame(Mod).to_excel('ModifiedTrend1.xlsx')
pd.DataFrame(Orig).to_excel('OriginalTrend1.xlsx')
pd.DataFrame(M).to_excel('FrequencyPerService1.xlsx')
