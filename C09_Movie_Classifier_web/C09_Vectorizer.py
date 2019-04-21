from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle

cur_dir=os.path.dirname(__file__)
stop=pickle.load(open(os.path.join(cur_dir,'pkl_object','stopwords.pkl'),'rb'))

def tokenizer(text):
    text=re.sub('<[^>]*>','',text)
    emotions=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text.lower())
    text=re.sub('[\W]+','',text.lower())+' '.join(emotions).replace('-','')
    tokenized=[w for w in text.split() if w not in stop]
    return tokenized

vect=HashingVectorizer(decode_error='ignore',n_features=2**21,preprocessor=None,tokenizer=tokenizer)

from C09_Movie_Classifier_Web.C09_Vectorizer import vect
clf=pickle.load(open(os.path.join('pkl_object','classifier.pkl'),'rb'))
import numpy as np
label={0:'negative',1:'positive'}
ex=['I love this movie.']
X=vect.transform(ex)
print('Prediction : %s\nProbability : %.2f%%'%(label[clf.predict(X)[0]],np.max(clf.predict_proba(X))*100))