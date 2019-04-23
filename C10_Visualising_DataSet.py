'''############## Preprocessing Data Set ##############3

import csv
with open("/home/ghanshyam/Machine Learning/housing.data.txt") as f:
    lst=[line.split() for line in f]
    # print(lst)
for l in lst:
    with open("/home/ghanshyam/Machine Learning/Housing.csv",'a') as csvFile:
        writer=csv.writer(csvFile)
        writer.writerow(l)
csvFile.close()

'''




import pandas as pd

df=pd.read_csv("/home/ghanshyam/Machine Learning/Housing.csv")


df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']


import seaborn as sns
import matplotlib.pyplot as plt
cols=['LSTAT','INDUS','NOX','RM','MEDV']
sns.pairplot(df[cols],size=2.5)
plt.tight_layout(pad=3.0)
plt.show()
# plt.pause(1)
# plt.close()


import numpy as np
cm=np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.3f',annot_kws={'size':15},yticklabels=cols,xticklabels=cols)
plt.show()