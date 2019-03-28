import C08_Bigger_Data as C08BD
from nltk.corpus import stopwords

import pickle
import os
dest=os.path.join('movieclassifier','pkl_object')
if not os.path.join(dest):
    os.makedirs(dest)

pickle.dump(C08BD.stop,open(os.path.join(dest,'stopwords.pkl'),'wb'),protocol=4)
pickle.dump(C08BD.clf,open(os.path.join(dest,'classifier.pkl'),'wb'),protocol=4)


