import pickle

import numpy as np
from matplotlib import pyplot
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from numpy import mean, nan
from numpy import std

## Correlation
import seaborn as sns
import matplotlib.pyplot as plt

# manual nested cross-validation for random forest on a classification dataset
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# function to convert sequence strings into k-mer words, default size = 6 (hexamer words)
from xgboost import plot_tree, plot_importance

#kmers function
def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

  ##preprocessing
#######################################################
gata3_data=pd.read_table("gata3_er_test")
#print(gata3_data.head(5))


##replace seuence column with kmers words
gata3_data['words'] = gata3_data.apply(lambda x: getKmers(x['sequence']), axis=1)
gata3_data = gata3_data.drop('sequence', axis=1)

human_texts = list(gata3_data['words'])
for item in range(len(human_texts)):
    human_texts[item] = ' '.join(human_texts[item])


from sklearn.feature_extraction.text import CountVectorizer

loaded_vectorizer = pickle.load(open('vectorizer_gataless.pickle', 'rb'))
X = loaded_vectorizer.transform(human_texts)

print(X.shape)
model = xgb.XGBClassifier()
model.load_model("xgb_model_gataless.json")

count=0
y_pred=model.predict(X)
print(y_pred)
for result in y_pred:
    if(result==1): count = count +1
    print(str(result))
print(count)
