import pandas as pd
import numpy as np
df= pd.read_csv("/home/ghanshyam/Machine Learning/movie_data.csv",encoding="utf-8")

from sklearn.feature_extraction.text import CountVectorizer
count=CountVectorizer(stop_words='english',max_df=.1,max_features=5000)
X=count.fit_transform(df['review'].values)

from sklearn.decomposition import LatentDirichletAllocation
lda=LatentDirichletAllocation(n_topics=10,random_state=123,learning_method='batch')
X_topic=lda.fit_transform(X)
print(X_topic)

print(lda.components_.shape)
n_top_words=5
feature_names=count.get_feature_names()
for topic_idx,topic in enumerate(lda.components_):
    print("Topic %d"%(topic_idx+1))
    print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words:-1]]))


horror=X_topic[:,5].argsort()[::-1]
for iter_idx,movie in enumerate(horror[:3]):
    print("Horror Movie %d"%(iter_idx+1))
    print(df['review'][movie][:300],'...')