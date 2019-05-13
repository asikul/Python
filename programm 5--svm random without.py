
import pandas as pd  
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
csv = 'm8869_final_null_removed.csv'
df = pd.read_csv(csv) 
reviews=df['text'].values.tolist()    
labels=df['target'].values.tolist()    
labels = list(map(str, labels))
for i in range(len(labels)):
    if labels[i]=='0': labels[i]='positive';
    else:labels[i]='negative';
 

reviews_tokens = [review.split() for review in reviews]
from sklearn.preprocessing import MultiLabelBinarizer
onehot_enc = MultiLabelBinarizer()
onehot_enc.fit(reviews_tokens)
MultiLabelBinarizer(classes=None, sparse_output=False)




from sklearn.cross_validation import train_test_split
SEED = 2000

X_train,X_test,y_train,y_test = train_test_split(reviews_tokens, labels, test_size=.25, random_state=SEED)

from sklearn.svm import LinearSVC
lsvm =LinearSVC(C=.01, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
lsvm.fit(onehot_enc.transform(X_train), y_train)
#input_matrix=onehot_enc.transform(X_train)
print(lsvm.score(onehot_enc.transform(X_train), y_train))
 
score = lsvm.score(onehot_enc.transform(X_test), y_test) 
print(score)   