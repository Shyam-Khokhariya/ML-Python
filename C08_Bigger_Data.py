import numpy as np
import re
from nltk.corpus import stopwords
stop=stopwords.words("english")
def tokenizer(text):
    text=re.sub('<[^>]*>','',text)
    emotions=re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)",text.lower())
    text=re.sub('[\W]+',' ',text.lower())+' '.join(emotions).replace('-','')
    tokenized=[w for w in text.split() if w not in stop]
    return tokenized
def stream_doc(path):
    with open(path,'r',encoding='utf-8') as csv:
        next(csv)
        for line in csv:
            text,label=line[:-3],int(line[-2])
            # if int(line[-3])==1:
            #     label=int(line[-3])
                # print('hellllllllllllllllllllllllllllllllllllllllllllll')
            #     label=1
            # if label!=1 and label!=0:
            # print(text)
            # print(label)
            yield text,label
# print(next(stream_doc('/home/ghanshyam/Machine Learning/movie_data.csv')))

def get_minibatch(doc_stream,size):
    docs,y=[],[]
    try:
        for _ in range(size):
            text,label=next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None,None
    return docs,y


from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
vect=HashingVectorizer(decode_error='ignore',n_features=2**21,preprocessor=None,tokenizer=tokenizer)
clf=SGDClassifier(loss='log',random_state=1,n_iter=1)
doc_stream=stream_doc(path="/home/ghanshyam/Machine Learning/movie_data.csv")

import pandas as pd
df=pd.read_csv("/home/ghanshyam/Machine Learning/movie_data.csv",converters={'sentiment':int})
df.to_csv('/home/ghanshyam/Machine Learning/movie_data.csv',index=False,encoding='utf-8')

y=df.loc[:,'sentiment'].values
classes=np.unique(y)

import pyprind
pbar=pyprind.ProgBar(45)
for _ in range(45):
    X_train,y_train=get_minibatch(doc_stream,size=1000)
    if not X_train:
        break
    X_train=vect.transform(X_train)
    clf.partial_fit(X_train,y_train,classes=classes)
    pbar.update()
X_test,y_test=get_minibatch(doc_stream,size=5000)
X_test=vect.transform(X_test)
print("Accuracy:%.3f"%clf.score(X_test,y_test))