# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 08:20:15 2018


@author: Md.Asikul Islam
"""


import pandas as pd  
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
csv = 'm8869_final_null_removed.csv'
df = pd.read_csv(csv) 

df_freq = pd.read_csv('normcdf_freq.csv', index_col=0, encoding='utf_8') 

reviews=df['text'].values.tolist()    
labels=df['target'].values.tolist()    
labels = list(map(str, labels))
for i in range(len(labels)):
    if labels[i]=='0': labels[i]='positive';
    else:labels[i]='negative';
 

reviews_tokens = [review.split() for review in reviews]
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
mlb.fit(reviews_tokens)
MultiLabelBinarizer(classes=None, sparse_output=False)
input_matrix_X=mlb.transform(reviews_tokens)
mlb.get_params()

#######################################  mappiing 0 to 9 ###################################
#wordno=0
#map_matrix_X=input_matrix_X
#
#for j in range(2,2):
#    for i in range(0,0):
#        if(input_matrix_X[j][i]==1):
#            if  reviews_tokens[j][wordno] in  term_freq_df2.index :
#                value_n=input_matrix_X[j][i]*10*term_freq_df2[ 'neg_normcdf_hmean' ][reviews_tokens[j][ wordno]]
#                value_p=input_matrix_X[j][i]*10*term_freq_df2[ 'pos_normcdf_hmean' ][reviews_tokens[j][ wordno]]
#                map_matrix_X[j][i]=max(value_p,value_n)
#            wordno=wordno+1
#    wordno=0
    
    
wordno=0
i=0
j=0
map_matrix_X=input_matrix_X

for j in range(0,9961):
    for i in range(0,12976):
        if(input_matrix_X[j][i]==1):
            
        
            if reviews_tokens[j][wordno] in  df_freq .index:
                #print(wordno)
            
                value_n=input_matrix_X[j][i]*10*df_freq [ 'neg_normcdf_hmean' ][reviews_tokens[j][ wordno]]
                value_p=input_matrix_X[j][i]*10*df_freq [ 'pos_normcdf_hmean' ][reviews_tokens[j][ wordno]]
                map_matrix_X[j][i]=min(value_p,value_n)
                #print(map_matrix_X[j][i])
            wordno=wordno+1
    wordno=0
       
###############   mappiing 0 to 1  ############ 
#
#for j in range(0,9961):
#    for i in range(0,12976):
#        if(map_matrix_X[j][i]<=5):
#            map_matrix_X[j][i]=0
#            
#        else: 
#            map_matrix_X[j][i]=1
         
##############################################################################   


from sklearn.cross_validation import train_test_split
input_matrix_X_train,input_matrix_X_test,y_train,y_test = train_test_split(map_matrix_X, labels, test_size=.25, random_state=None)


from sklearn.svm import LinearSVC
lsvm =LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=10000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.01,
     verbose=0)

lsvm.fit(input_matrix_X_train, y_train)
print(lsvm.score(input_matrix_X_train, y_train))



     
score = lsvm.score(input_matrix_X_test, y_test) 
print(score)   


