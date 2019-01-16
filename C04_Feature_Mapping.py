import pandas as pd
import numpy as np
df=pd.DataFrame([

    ['Green',10.0,'M','Class1'],
    ['Blue',15,'S','Class2'],
    ['Yellow',14.5,'XL','Class3'],
    ['Pink',12.2,'L','Class4']
    ])
df.columns=('Color','Price','Size','ClassLable')
# print(df)
size_mapping={
    'M':2,
    'S':1,
    'L':3,
    'XL':4
}
df['Size']=df['Size'].map(size_mapping)
# inv_mapping={v:k for k,v in size_mapping.items()}
# df['Size']=df['Size'].map(inv_mapping)
# print(df)

class_mapping={lable:index for index,lable in enumerate(np.unique(df['ClassLable']))}
df['ClassLable']=df['ClassLable'].map(class_mapping)
inv_class={v:k for k,v in class_mapping.items()}
df['ClassLable']=df['ClassLable'].map(inv_class)
# print(df)


from sklearn.preprocessing import LabelEncoder
class_LE=LabelEncoder()
y=class_LE.fit_transform(df['ClassLable'].values)
y1=class_LE.inverse_transform(y)
# print(y)
# print(y1)


X=df[['Color','Size','Price']].values
color_LE=LabelEncoder()
X[:,0]=color_LE.fit_transform(X[:,0])
print(X)

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features=[0])
p=ohe.fit_transform(X).toarray()
print(p)

print(pd.get_dummies(df[['Price','Color','Size']],drop_first=True))