import pandas as pd  
import matplotlib.pyplot as plt
import numpy as np

csv = 'clean_tweet_1.csv'
my_df = pd.read_csv(csv)
my_df.head()
my_df.columns  = ['text','target']

my_df. head()
my_df. info()

my_df[ my_df. isnull() . any(axis=1) ] . head()
np. sum(my_df. isnull() . any(axis=1) )
my_df. isnull() . any(axis=0)   
my_df. dropna(inplace=True)
my_df. reset_index(drop=True, inplace=True)
my_df. info()

                                                                 #wordcloud                
neg_tweets = my_df[ my_df. target == 0]
neg_string = [ ]
for t in neg_tweets.text:
    neg_string.append(t)
neg_string = pd. Series(neg_string) . str. cat(sep=' ' )


                    # count_vectorizer
from sklearn.feature_extraction.text import CountVectorizer
cvec = CountVectorizer()
cvec. fit(my_df.text)
len(cvec. get_feature_names())
neg_doc_matrix = cvec. transform(my_df[ my_df. target == 0].text)
pos_doc_matrix = cvec. transform(my_df[ my_df. target == 4].text)
neg_tf = np. sum(neg_doc_matrix, axis=0)
pos_tf = np. sum(pos_doc_matrix, axis=0)
neg = np. squeeze(np. asarray(neg_tf))
pos = np. squeeze(np. asarray(pos_tf))
term_freq_df = pd. DataFrame([ neg, pos], columns=cvec. get_feature_names()).transpose()
document_matrix = cvec. transform(my_df. text)
my_df[ my_df. target == 0] . tail()



neg_batches = np. linspace(0, 4980, 100) . astype(int)
i=0
neg_tf = []

while i < len(neg_batches) -1:
         batch_result = np. sum(document_matrix[ neg_batches[ i]: neg_batches[ i+1]] . toarray(), axis=0)
         neg_tf. append(batch_result)
         if (i % 10 == 0) | (i == len(neg_batches) -2):
             print (neg_batches[ i+1], "entries' term freuquency calculated")
         i += 1
         
my_df. tail()


pos_batches = np. linspace(4980, 9962, 100) . astype(int)
i=0
pos_tf = []
while i < len(pos_batches) -1:
     batch_result = np. sum(document_matrix[ pos_batches[ i]: pos_batches[ i+1]] . toarray(), axis=0)
     pos_tf. append(batch_result)
     if (i % 10 == 0) | (i == len(pos_batches) -2):
         print (pos_batches[ i+1], "entries' term freuquency calculated")
     i += 1
neg = np. sum(neg_tf, axis=0)
pos = np. sum(pos_tf, axis=0)
term_freq_df = pd. DataFrame([ neg, pos], columns=cvec. get_feature_names()) . transpose()
term_freq_df. head()
term_freq_df. columns = [ 'negative' , 'positive' ]
term_freq_df[ 'total' ] = term_freq_df[ 'negative' ] + term_freq_df[ 'positive' ]
term_freq_df. sort_values(by='total' , ascending=False) . iloc[: 10]

                 #part 3
                 
#term_freq_df. columns = [ 'negative' , 'positive' ]
#term_freq_df[ 'total' ] = term_freq_df[ 'negative' ] + term_freq_df[ 'positive' ]
#term_freq_df. sort_values(by='total' , ascending=False) . iloc[: 10]
                 
                 
len(term_freq_df)
term_freq_df. to_csv('term_freq_df.csv' , encoding='utf-8' )

#Zipf's Law\
y_pos = np. arange(500)
plt. figure(figsize=(10, 8))
s = 1
expected_zipf = [ term_freq_df. sort_values(by='total' , ascending=False)[ 'total' ][ 0] /(i+1) **s for i in y_pos
]
plt. bar(y_pos, term_freq_df. sort_values(by='total' , ascending=False)[ 'total' ][: 500], align='center' , alpha=0.5)
plt. plot(y_pos, expected_zipf, color='r' , linestyle='--' , linewidth=2, alpha=0.5)
plt. ylabel('Frequency' )
plt. title('Top 500 tokens in tweets' )

from pylab import *
counts = term_freq_df. total
tokens = term_freq_df. index
ranks = arange(1, len(counts) +1)
indices = argsort(-counts)
frequencies = counts[ indices]
plt. figure(figsize=(8, 6))
plt. ylim(1, 10**6)
plt. xlim(1, 10**6)
loglog(ranks, frequencies, marker=".")
plt. plot([ 1, frequencies[ 0]],[ frequencies[ 0], 1], color='r' )
title("Zipf plot for tweets tokens" )
xlabel("Frequency rank of token" )
ylabel("Absolute frequency of token" )
grid(True)
for n in list(logspace(-0.5, log10(len(counts) -2), 25) . astype(int)):
 dummy = text(ranks[ n], frequencies[ n], " " + tokens[ indices[ n]],
 verticalalignment="bottom",
 horizontalalignment="left")
 
term_freq_df = pd. read_csv('term_freq_df.csv' , index_col=0, encoding='utf_8' )
from sklearn.feature_extraction.text import CountVectorizer
cvec = CountVectorizer(stop_words='english' , max_features=10000)
cvec. fit(my_df. text)
document_matrix = cvec. transform(my_df. text)
neg_batches = np. linspace(0, 4980, 10) . astype(int)
i=0
neg_tf = []
while i < len(neg_batches) -1:
 batch_result = np. sum(document_matrix[ neg_batches[ i]: neg_batches[ i+1]] . toarray(), axis=0)
 neg_tf. append(batch_result)
 print( neg_batches[ i+1], "entries' term freuquency calculated")
 i += 1
 
pos_batches = np. linspace(4980, 9962, 10) . astype(int)
i=0
pos_tf = []
while i < len(pos_batches) -1:
 batch_result = np. sum(document_matrix[ pos_batches[ i]: pos_batches[ i+1]] . toarray(), axis=0)
 pos_tf. append(batch_result)
 print( pos_batches[ i+1], "entries' term freuquency calculated")
 i += 1
 
neg = np. sum(neg_tf, axis=0)
pos = np. sum(pos_tf, axis=0)
term_freq_df2 = pd. DataFrame([ neg, pos], columns=cvec. get_feature_names()) . transpose()
term_freq_df2. columns = [ 'negative' , 'positive' ]
term_freq_df2[ 'total' ] = term_freq_df2[ 'negative' ] + term_freq_df2[ 'positive' ]
term_freq_df2. sort_values(by='total' , ascending=False) . iloc[: 10]
y_pos = np. arange(50)
plt. figure(figsize=(12, 10))
plt. bar(y_pos, term_freq_df2. sort_values(by='negative' , ascending=False)[ 'negative' ][: 50], align='center' ,
alpha=0.5)
plt. xticks(y_pos, term_freq_df2. sort_values(by='negative' , ascending=False)[ 'negative' ][: 50] . index, rotation='vertical' )
plt. ylabel('Frequency' )
plt. xlabel('Top 50 negative tokens' )
plt. title('Top 50 tokens in negative tweets' )
y_pos = np. arange(50)
plt. figure(figsize=(12, 10))
plt. bar(y_pos, term_freq_df2. sort_values(by='positive' , ascending=False)[ 'positive' ][: 50], align='center' ,
alpha=0.5)
plt. xticks(y_pos, term_freq_df2. sort_values(by='positive' , ascending=False)[ 'positive' ][: 50] . index, rotation='vertical' )
plt. ylabel('Frequency' )
plt. xlabel('Top 50 positive tokens' )
plt. title('Top 50 tokens in positive tweets' )
import seaborn as sns
plt. figure(figsize=(8, 6))
ax = sns. regplot(x="negative", y="positive", fit_reg=False, scatter_kws={'alpha' : 0.5}, data=term_freq_df2)
plt. ylabel('Positive Frequency' )
plt. xlabel('Negative Frequency' )
plt. title('Negative Frequency vs Positive Frequency' )
term_freq_df2[ 'pos_rate' ] = term_freq_df2[ 'positive' ] * 1. /term_freq_df2[ 'total' ]
term_freq_df2. sort_values(by='pos_rate' , ascending=False) . iloc[: 10]
term_freq_df2[ 'pos_freq_pct' ] = term_freq_df2[ 'positive' ] * 1. /term_freq_df2[ 'positive' ] . sum()
term_freq_df2. sort_values(by='pos_freq_pct' , ascending=False) . iloc[: 10]

from scipy.stats import hmean
term_freq_df2[ 'pos_hmean' ] = term_freq_df2. apply(lambda x: (hmean([ x[ 'pos_rate' ], x[ 'pos_freq_pct' ]])
 if x[ 'pos_rate' ] > 0 and x[ 'pos_freq_pct' ] > 0
 else  0),axis=1) 
term_freq_df2. sort_values(by='pos_hmean' , ascending=False) . iloc[: 10]



from scipy.stats import norm
def normcdf(x):
 return norm. cdf(x, x. mean(), x. std())
term_freq_df2[ 'pos_rate_normcdf' ] = normcdf(term_freq_df2[ 'pos_rate' ])
term_freq_df2[ 'pos_freq_pct_normcdf' ] = normcdf(term_freq_df2[ 'pos_freq_pct' ])
term_freq_df2[ 'pos_normcdf_hmean' ] = hmean([ term_freq_df2[ 'pos_rate_normcdf' ], term_freq_df2[ 'pos_freq_pct_normcdf' ]])
term_freq_df2. sort_values(by='pos_normcdf_hmean' , ascending=False) . iloc[: 10]
term_freq_df2[ 'neg_rate' ] = term_freq_df2[ 'negative' ] * 1. /term_freq_df2[ 'total' ]
term_freq_df2[ 'neg_freq_pct' ] = term_freq_df2[ 'negative' ] * 1. /term_freq_df2[ 'negative' ] . sum()
term_freq_df2[ 'neg_hmean' ] = term_freq_df2. apply(lambda x: (hmean([ x[ 'neg_rate' ], x[ 'neg_freq_pct' ]])
 if x[ 'neg_rate' ] > 0 and x[ 'neg_freq_pct' ] > 0
 else 0), axis=1) 
term_freq_df2[ 'neg_rate_normcdf' ] = normcdf(term_freq_df2[ 'neg_rate' ])
term_freq_df2[ 'neg_freq_pct_normcdf' ] = normcdf(term_freq_df2[ 'neg_freq_pct' ])
term_freq_df2[ 'neg_normcdf_hmean' ] = hmean([ term_freq_df2[ 'neg_rate_normcdf' ], term_freq_df2[ 'neg_freq_pct_normcdf' ]])
term_freq_df2. sort_values(by='neg_normcdf_hmean' , ascending=False) . iloc[: 10]

plt. figure(figsize=(8, 6))
ax = sns. regplot(x="neg_hmean", y="pos_hmean", fit_reg=False, scatter_kws={'alpha' : 0.5}, data=term_freq_df2)
plt. ylabel('Positive Rate and Frequency Harmonic Mean' )
plt. xlabel('Negative Rate and Frequency Harmonic Mean' )
plt. title('neg_hmean vs pos_hmean' )
plt. figure(figsize=(8, 6))
ax = sns. regplot(x="neg_normcdf_hmean" , y="pos_normcdf_hmean" , fit_reg=False, scatter_kws={'alpha' : 0.5}, data=term_freq_df2)
plt. ylabel('Positive Rate and Frequency CDF Harmonic Mean' )
plt. xlabel('Negative Rate and Frequency CDF Harmonic Mean' )
plt. title('neg_normcdf_hmean vs pos_normcdf_hmean' )


# add 10.08.2018 from https://github.com/tthustla/twitter_sentiment_analysis_part5/blob/master/Capstone_part4-Copy3.ipynb  in [49]
#pos_hmean = term_freq_df2.pos_normcdf_hmean
#pos_hmean['wtf']
#y_val_predicted_proba = []
#for t in x_validation:
#    hmean_scores = [pos_hmean[w] for w in t.split() if w in pos_hmean.index]
#    if len(hmean_scores) > 0:
#        prob_score = np.mean(hmean_scores)
#    else:
#        prob_score = np.random.random()
#    y_val_predicted_proba.append(prob_score)
#pred = [1 if t > 0.56 else 0 for t in y_val_predicted_proba]
#from sklearn.metrics import accuracy_score
#accuracy_score(y_validation,pred)
# added 10.08.2018 from 

#from bokeh.plotting import figure
#from bokeh.io import output_notebook, show
#from bokeh.models import LinearColorMapper
#output_notebook()
#color_mapper = LinearColorMapper(palette='Inferno256' , low=min(term_freq_df2. pos_normcdf_hmean), high=max(term_freq_df2. pos_normcdf_hmean))
#p = figure(x_axis_label='neg_normcdf_hmean' , y_axis_label='pos_normcdf_hmean' )
#p. circle('neg_normcdf_hmean' , 'pos_normcdf_hmean' , size=5, alpha=0.3, source=term_freq_df2, color={'field' : 'pos_normcdf_hmean' , 'transform' : color_mapper})
#from bokeh.models import HoverTool
#hover = HoverTool(tooltips=[('token' , '@index' )])
#p. add_tools(hover)
#show(p)



                     #PART 4
 
