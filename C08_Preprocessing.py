import pyprind
import pandas as pd
import os
import numpy as np


# ----------------Preparing csv File From Folder files---------------


basepath="/home/ghanshyam/Machine Learning/aclImdb"

label={'pos':1,'neg':0}
pbar=pyprind.ProgBar(50000)

df=pd.DataFrame()
for s in ('test','train'):
    for l in ('pos','neg'):
        path=os.path.join(basepath,s,l)
        for file in os.listdir(path):
            with open(os.path.join(path,file),'r',encoding='utf-8') as infile:
                txt=infile.read()
            df=df.append([[txt,label[l]]],ignore_index=True)
            pbar.update()

df.columns=['review','sentiment']


np.random.seed(0)
df=df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv',index=False,encoding='utf-8')

'''


df = pd.read_csv("/home/ghanshyam/Machine Learning/movie_data.csv")
# print(df.head(5))


 
 --------------TF-IDF Wording-----------------


from sklearn.feature_extraction.text import CountVectorizer
count=CountVectorizer()
docs=np.array(['My Name is Shyam','Hello I am in B.Tech','I am Shyam and studying B.Tech'])
bag=count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())

from sklearn.feature_extraction.text import TfidfTransformer
tfid=TfidfTransformer(use_idf=True,norm='l2',smooth_idf=True)
np.set_printoptions(precision=2)
print(tfid.fit_transform(count.fit_transform(docs)).toarray())

'''

'''
print(df.loc[0,'review'][-50:])

import re
def preprocessing(text):
    text=re.sub('<[^>]*>','',text)
    emotions=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text=(re.sub('[\W]+',' ',text.lower())+''.join(emotions).replace('-',''))
    return text
print(preprocessing(df.loc[0,'review'][-50:]))
print(preprocessing("<\a>This :) is :( a text :-)!"))
df['review']=df['review'].apply(preprocessing)
# print(df['review'])

def tokanizer(text):
    return text.split()
print(tokanizer('Runner like running and thus they run .'))


from nltk.stem.porter import PorterStemmer
porter=PorterStemmer()
def tokanizer_nltk(text):
    return [porter.stem(word) for word in text.split()]
print(tokanizer_nltk('Runner like running and thus they run .'))


from nltk.corpus import stopwords
stop=stopwords.words('english')
a=[w for w in tokanizer_nltk('Runner like running and thus they runs a lot .') if w not in stop]
print(a)'''