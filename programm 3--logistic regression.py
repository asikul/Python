import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
csv = 'final_null_removed.csv'
my_df = pd.read_csv(csv)
my_df.head()
my_df.columns  = ['text','target']
my_df[my_df.target==0].head(10)
my_df.dropna(inplace=True)
my_df.reset_index(drop=True,inplace=True)
my_df.info()
x=my_df.text
y=my_df.target

# a=int(len(my_df)*(2/3))
#x_train=my_df[1,1]
#x_test=my_df[len(my_df)*(2/3):len(my_df),1]
#y_train=my_df[1:len(my_df)*(2/3),2]
#y_test=my_df[len(my_df)*(2/3):len(my_df),2]

from sklearn.cross_validation import train_test_split
SEED = 2000

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=.25, random_state=SEED)

from sklearn.feature_extraction.text import TfidfVectorizer
tvec1 = TfidfVectorizer(max_features=20000,ngram_range=(1,2))
tvec1.fit(x_train)
##
d=tvec1.get_feature_names()
tvec1.get_params()
x_train_tfidf = tvec1.transform(x_train)
c=x_train_tfidf.toarray()
#x_validation_tfidf = tvec1.transform(x_validation).toarray()

x_test_tfidf = tvec1.transform(x_test).toarray()
from sklearn.linear_model import LogisticRegression
#clf = LogisticRegression()
clf =LogisticRegression(C=.1, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=10000, multi_class='ovr', n_jobs=1, penalty='l2', random_state=None, solver='liblinear', tol=0.0001,verbose=0, warm_start=False)
clf.fit(x_train_tfidf, y_train);
print('train accuracy : ')
print(clf.score(x_train_tfidf, y_train))

print('test accuracy : ')
print(clf.score(x_test_tfidf, y_test))
# I will first start by loading required dependencies. In order to run Keras with TensorFlow backend, you need to install both TensorFlow and Keras.
predict=clf.predict(x_test_tfidf)
