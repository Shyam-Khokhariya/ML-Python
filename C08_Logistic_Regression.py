import numpy as np
import pandas as pd

df=pd.read_csv("/home/ghanshyam/Machine Learning/movie_data.csv",encoding='utf-8')

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def tokenizer(text):
    return text.split()
from nltk.corpus import stopwords
stop=stopwords.words('english')
from nltk.stem.porter import PorterStemmer
porter=PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
import re
def preprocessor(text):
    text=re.sub('<[^>]*>','',text)
    emotions=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text=(re.sub('[\W]+',' ',text.lower())+' '.join(emotions).replace('-',''))
    return text
df['review']=df['review'].apply(preprocessor)

X_train=df.loc[:25000,'review'].values
y_train=df.loc[:25000,'sentiment'].values
X_test=df.loc[25000:,'review'].values
y_test=df.loc[25000:,'sentiment'].values


tfidf=TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None)

param_grid=[{'vect__ngram_range':[(1,1)],
             'vect__stop_words':[stop,None],
             'vect__tokenizer':[tokenizer,tokenizer_porter],
             'clf__penalty':['l1','l2'],
             'clf__C':[1.0,10.0,100.0]},
            {'vect__ngram_range':[(1,1)],
             'vect__stop_words':[stop,None],
             'vect__tokenizer':[tokenizer,tokenizer_porter],
             'vect__use_idf':[False],
             'vect__norm':[None],
             'clf__penalty':['l1','l2'],
             'clf__C':[1.0,10.0,100.0]}]

lr_tfidf=Pipeline([('vect',tfidf),
                   ('clf',LogisticRegression(random_state=0))])

gs_lr_tfidf=GridSearchCV(lr_tfidf,param_grid,scoring='accuracy',cv=5,
                         verbose=1,n_jobs=1)
gs_lr_tfidf.fit(X_train,y_train)

print("Tfidf Best Parameters : %s"%gs_lr_tfidf.best_params_)
print("CV Accuracy : %.5f"%gs_lr_tfidf.best_score_)
clf=gs_lr_tfidf.best_estimator_
print("Test Accuracy : %.5f"%clf.score(X_test,y_test))