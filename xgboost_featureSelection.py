import xgboost as xgb
import pandas as pd

def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

#read preprocessed table, see github repository
gata3_data=pd.read_table("final")

gata3_data['words'] = gata3_data.apply(lambda x: getKmers(x['sequence']), axis=1)
gata3_data = gata3_data.drop('sequence', axis=1)

print(gata3_data['words'].values[0])
#196 hexamer (size of the sequence is 201)
#37622 rows of sequences
human_texts = list(gata3_data['words'])

for item in range(len(human_texts)):
    human_texts[item] = ' '.join(human_texts[item])

#print(human_texts[0])
y_data = gata3_data.iloc[:, 0].values

# Creating the Bag of Words model using CountVectorizer()
# This is equivalent to k-mer counting
# The n-gram size of 4 was previously determined by testing
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(human_texts)


seed = 7
test_size = 0.3
# Splitting the human dataset into the training set and test set
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y_data,
                                                    test_size=test_size,
                                                    random_state=seed)

params={'min_child_weight': 7,
        'max_depth': 10, 'learning_rate': 0.25,
        'gamma': 0.0, 'colsample_bytree': 0.7}

#save this shit
classifier = xgb.XGBClassifier(**params)

classifier.fit(X_train,y_train,verbose=True ,
               early_stopping_rounds=20,eval_metric='aucpr',
               eval_set=[(X_test,y_test)])


f = open("prediction_score.txt", "w")
f.write(str(classifier.feature_importances_))
f.close()

# feature importance
print(classifier.feature_importances_)
# plot
pyplot.bar(range(len(classifier.feature_importances_)), classifier.feature_importances_)
pyplot.show()
