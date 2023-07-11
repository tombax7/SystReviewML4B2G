#%%________________________________________SCOPUS
#this function sends a request and returns the total articles,
#the starting position of the first article, and the metadata of each #article.
import requests
import json
from pybliometrics.scopus.utils import config
base_url = 'http://api.elsevier.com/content/search/scopus?'
# search in title, abstract, and key
scope    = 'TITLE-ABS-KEY'
# formulating the query structure
terms1   = '({machine learning} OR {deep learning} OR {federated learning} OR {data-driven} OR {reinforcement learning} OR {end-to-end learning} OR {supervised learning} OR {unsupervised learning} OR {learning-based} OR {neural networks})'
terms2   = '({distribution grids} OR {distributed energy resources} OR {energy management system} OR {energy flexibility of buildings} OR {flexible buildings} OR {energy flexibility} OR {buildings flexibility} OR {flexibility markets} OR {multi-energy systems} OR {integrated energy system} OR {demand response} OR {demand side management} OR {demand-side management} OR {local energy} OR {energy communities} OR {load management} OR {distribution system} OR {smart buildings} OR {local energy market} OR {smart meter} OR {power flow})'
terms3   = '({water} OR {powertrain})' 
terms    = '({} AND {} AND NOT {})'.format(terms1, terms2, terms3)
# formulating the query structure

# insert your personal key (it is free and available on https://dev.elsevier.com/)
apiKey   = '&apiKey=XXXXXXXXXXXXXXXXXXX' 
date     = '&date=2000-2015'
# it is the maximum number of results per query for a free account
count    = '&count=25' 
sort     = '&sort=relevance-count' #citedby-count
view     = '&view=standard'
L=[]
def search_scopus(url):
    
    res = requests.get(url)
    if res.status_code ==200:
        content  = json.loads(res.content)['search-results']
        total    = content['opensearch:totalResults']
        start    = content['opensearch:startIndex']
        metadata = content['entry']
        return int(total), int(start), metadata
    
    else:
        error = json.loads(res.content)['service-error']['status']          
        print(res.status_code, error['statusText'])
# list of all subjects in Scopus database
subjects = ['BUSI', 'CENG', 'CHEM', 'COMP', 'DECI', 'ECON', 'ENER', 'ENGI', 'ENVI', 'MATE','HEAL', 'MATH','PHYS','MULT','SOCI']
for sub in subjects:
    start_index  = 0
    while True:   
        
        # starting index of the results for display
        # starting index refers to number position of not pages
        
        
        start    = '&start={}'.format(start_index) 
        subj     = '&subj={}'.format(sub)
        query    = 'query=' + scope + terms + date + start + count +  sort + subj + apiKey + view +'&subscribed=true'
        url  = base_url + query
        # total results per subject, starting index of first result in 
        #each query and data
        try: 
            total, start_index, metadata = search_scopus(url)
        except:
            break
        # save metadata now in SQL (not shown here)
        L.append(metadata)
        print(total)
        # check how many results need to be retrieved
        remain = total - start_index - len(metadata)
        print(remain)
        if remain>0:
            start_index+=25 # to search next 25 results
        else:
            break # breaking from while loop
#%%________________________________________Scopus data parser
import pandas as pd
concat_list = [j for i in L for j in i]
df=pd.DataFrame(columns=['Title', 'Type','Doi', 'Url','Publication', 'Date'])
x=[]
y=[]
q=[]
z=[]
w=[]
aa=[]
for count, value in enumerate(concat_list):
    print(count)
    try: 
        x.append(value['dc:title'])
    except: 
        x.append(0)
    try:
        q.append(value['subtypeDescription'])
    except: 
        q.append(0)
    try:
        y.append(value['prism:doi'])
    except: 
        y.append(0)
    try:
        aa.append(value['prism:url'])
    except:
        aa.append(0)
    try:
        z.append(value['prism:publicationName'])
    except: 
        z.append(0)
    try:
        w.append(value['prism:coverDate'][:4])
    except:
        w.append(0)
df['Title']=x
df['Type']=q
df['Doi']=y
df['Publication']=z
df['Date']=w
df['Url']=aa
df=df.loc[df['Doi']!=0]
df = df.drop_duplicates('Doi')
#%%________________________________________Scopus abstract retrieval
from pybliometrics.scopus import AbstractRetrieval
import pandas as pd
import os
os.chdir('/Users/thanosbach/Desktop/ReviewAIandSGs')
df=pd.read_excel('Scholar_1.xlsx')
abst=[]
ref=[]
for i in range(0,len(df)):
    try:
        ab = AbstractRetrieval(df.iloc[i].Doi,view="FULL", refresh=True,subscriber=True)
        x=ab.abstract
        y=ab.refcount
        print('Round:'+str(i)+'   Length:'+str(len(x)))
    except: 
        x=0
        y=0
    abst.append(x)
    ref.append(y)

df['Abstracts']=abst
df['References']=ref
#%%
for i in range(0,len(df)):
    if df.iloc[i].Abstracts==0:
        print(i)
        try:
            ab = AbstractRetrieval(df.iloc[i].Doi,view="FULL", refresh=True,subscriber=True)
            df['Abstracts'].iloc[i]=ab.abstract
            df['References'].iloc[i]=ab.refcount
        except: 
            df['Abstracts'].iloc[i]=0
            df['References'].iloc[i]=0
#%%________________________________________FROM TITLE TO DOI
from habanero import Crossref
import pandas as pd
import os
cr = Crossref()
os.chdir('/Users/thanosbach/Desktop/ReviewAIandSGs')
df=pd.read_excel('Scopus_a.xlsx')
doi=[]
for i in range(0,len(df)):
    try: 
        result=cr.works(query=df.iloc[i].Title)
        a=result['message']['items'][0]['DOI']
        print(str(i)+'  OK')
    except:
        a=0
        print(str(i)+'  notfind')
    doi.append(a)

df['Doi']=doi

#%%
for i in range(0,len(df)):
    if df.iloc[i].Abstracts==0:
        try: 
            result=cr.works(query=df.iloc[i].Title)
            a=result['message']['items'][1]['DOI']
            print(str(i)+'  OK')
            df.iloc[i].Doi=a
        except:
            a=0
            print(str(i)+'  notfind')

