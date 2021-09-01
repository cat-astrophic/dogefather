# This script performs the time series analyses on the cleaned twitter data and generates wordclouds

# Importing required modules

import pandas as pd
import numpy as np
from pylab import rcParams
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# Specifying your username -- you will need to update filepaths according to your setup

username = ''

# Reading in the data

doge = pd.read_csv('C:/Users/' + username + '/Documents/Data/dogefather/dogecoin.csv')
s1snl = pd.read_csv('C:/Users/' + username + '/Documents/Data/dogefather/sentiment_snl.csv')
s2snl = pd.read_csv('C:/Users/' + username + '/Documents/Data/dogefather/sentiment2_snl.csv')
s1doge = pd.read_csv('C:/Users/' + username + '/Documents/Data/dogefather/sentiment_doge.csv')
s2doge = pd.read_csv('C:/Users/' + username + '/Documents/Data/dogefather/sentiment2_doge.csv')

# Merging the data sets

newcol = [str(doge.Day[i]) + '-' + str(doge.Hour[i]) + '-' + str(doge.Minute[i]) for i in range(len(doge))]
newcol = pd.Series(newcol, name = 'idx')
doge = pd.concat([doge, newcol], axis = 1)

window1 = list(s1snl.Time)
window2 = list(s2snl.Time)

doge1 = doge[doge.idx.isin(window1) == True]
doge2 = doge[doge.idx.isin(window2) == True]


snlcols = ['Time', 'Day', 'Hour', 'Minute', 'Tweets_SNL', 'Sentiment_SNL', 'Positive_SNL',
           'Negative_SNL', 'Sentiment_Total_SNL', 'Positive_Total_SNL', 'Negative_Total_SNL']
dogecols = ['Time', 'Day', 'Hour', 'Minute', 'Tweets_doge', 'Sentiment_doge', 'Positive_doge',
            'Negative_doge', 'Sentiment_Total_doge', 'Positive_Total_doge', 'Negative_Total_doge']

s1snl.columns = snlcols
s2snl.columns = snlcols

s1doge.columns = dogecols
s2doge.columns = dogecols

doge1 = pd.merge(doge1, s1snl)
doge2 = pd.merge(doge2, s2snl)

doge1 = pd.merge(doge1, s1doge)
doge2 = pd.merge(doge2, s2doge)

# Creating ln prices and volumes

ln_close1 = [np.log(p) for p in doge1.close]
ln_close2 = [np.log(p) for p in doge2.close]

ln_close1 = pd.Series(ln_close1, name = 'Ln_Price')
ln_close2 = pd.Series(ln_close2, name = 'Ln_Price')

ln_vol1 = [np.log(v) for v in doge1.Volume]
ln_vol2 = [np.log(v) for v in doge2.Volume]

ln_vol1 = pd.Series(ln_vol1, name = 'Ln_Trading_Volume')
ln_vol2 = pd.Series(ln_vol2, name = 'Ln_Trading_Volume')

doge1 = pd.concat([doge1, ln_close1, ln_vol1], axis = 1)
doge2 = pd.concat([doge2, ln_close2, ln_vol2], axis = 1)

# VARs

data1 = doge1[['close', 'Positive_Total_SNL', 'Negative_Total_SNL']]
data1 = data1.diff().dropna()
data1.columns = ['Price', 'Positive_Sentiment', 'Negative_Sentiment']
model1 = VAR(data1)
model1.select_order(15)
results1 = model1.fit(maxlags = 15, ic = 'aic')
results1.summary()
results1.plot_acorr()
lag_order = results1.k_ar
results1.forecast(data1.values[-lag_order:], 5)
results1.plot_forecast(30)
irf1 = results1.irf(15)
irf1.plot(orth = False)
irf1.plot_cum_effects(orth = False)
print(results1.test_causality('Price', ['Negative_Sentiment'], kind = 'f'))

data2 = doge2[['close', 'Positive_Total_SNL', 'Negative_Total_SNL']]
data2 = data2.diff().dropna()
data2.columns = ['Price', 'Positive_Sentiment', 'Negative_Sentiment']
data2.Positive_Sentiment = data2.Positive_Sentiment / 1000 # Normalizing
data2.Negative_Sentiment = data2.Negative_Sentiment / 1000 # Normalizing
model2 = VAR(data2)
model2.select_order(15)
results2 = model2.fit(maxlags = 15, ic = 'aic')
results2.summary()
results2.plot_acorr(15)
lag_order = results2.k_ar
results2.forecast(data2.values[-lag_order:], 5)
results2.plot_forecast(30)
irf2 = results2.irf(16)
irf2.plot(orth = False, subplot_params = {'fontsize': 10})
irf2.plot_cum_effects(orth = False, subplot_params = {'fontsize': 10})
print(results2.test_causality('Price', ['Negative_Sentiment'], kind = 'f'))

# Full Granger Causality Table

print(results2.test_causality('Price', ['Positive_Sentiment'], kind = 'f'))
print(results2.test_causality('Price', ['Negative_Sentiment'], kind = 'f'))

print(results2.test_causality('Positive_Sentiment', ['Price'], kind = 'f'))
print(results2.test_causality('Positive_Sentiment', ['Negative_Sentiment'], kind = 'f'))

print(results2.test_causality('Negative_Sentiment', ['Price'], kind = 'f'))
print(results2.test_causality('Negative_Sentiment', ['Positive_Sentiment'], kind = 'f'))

# Stationarity Tests via Augmented Dickey-Fuller

for x in ['Price', 'Positive_Sentiment', 'Negative_Sentiment']:
    
    adfres = adfuller(data2[x])
    print('ADF Results for: ' + x + '\n')
    print('ADF Statistic: %f\n' % adfres[0])
    print('p-value: %f\n' % adfres[1])
    
    for key, value in adfres[4].items():
        
        print('\t%s: %.3f\n' % (key, value))

# Checking to see if the VAR is stable

A = results2.coefs # VAR results
M = np.zeros((45,45)) # Initializing the matrix whoes eigenvalues we need

for a in range(15): # Adding values from A into M
    
    M[0][a*3:a*3+3] = A[a][0]
    M[1][a*3:a*3+3] = A[a][1]
    M[2][a*3:a*3+3] = A[a][2]

for i in range(3,45): # Adding the I3 matrices to M
    
    M[i][i-3] = 1

# Get the eignvalues of M

eigs = np.linalg.eig(M)

# Test for stability

if max(abs(eigs[0])) < 1:
    
    print('The VAR process is stable!')
    
else:
    
    print('The VAR process is not stable!')

# Creating an image of the roots

x = [e.real for e in eigs[0]] # real part of the eigenvalues
y = [e.imag for e in eigs[0]] # imaginary part of the eigenvalues

plt.figure(figsize = (6,6))
plt.scatter(x, y, color = 'black', linewidth = 2)
t = np.linspace(0, np.pi*2, 1000)
plt.plot(np.cos(t), np.sin(t), linewidth = 1, color = 'red')
plt.title('Roots of the VAR Characteristic Polynomial')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.savefig('C:/Users/' + username + '/Documents/Data/dogefather/unit_roots.eps')

# Creating a time series plot for each of the three time series with two scales

rcParams['figure.figsize'] = 8.5, 8.5
cm = plt.get_cmap('gist_rainbow')

plt.figure(0)
fig, ax1 = plt.subplots()
basis = [i for i in range(-60,150)]
plt.plot(basis, doge2.close, label = 'Price of dogecoin', color = 'k')
plt.ylabel('USD')
plt.xlabel('Time (Minutes Relative to Start of Episode)')
plt.vlines(x = 0, ymin = 0, ymax = .75)
plt.vlines(x = 90, ymin = 0, ymax = .75)
ax2 = ax1.twinx()
plt.plot(basis, doge2.Positive_Total_SNL, label = 'Positive Sentiment', color = cm(180))
plt.plot(basis, doge2.Negative_Total_SNL, label = 'Negative Sentiment', color = cm(0))
plt.title("A Comparison of the price of dogecoin and public perception of Musk's performance", loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.ylabel('Aggregate Sentiment')
fig.tight_layout()
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc = 0)
plt.savefig('C:/Users/' + username + '/Documents/Data/dogefather/time_series_plot.eps')

# Creating a word cloud for the harvested tweets

td = pd.read_csv('C:/Users/' + username + '/Documents/Data/dogefather/raw_twitter_data.csv')

# Remove duplicate tweets

td = td.drop_duplicates(subset = 'id', keep = 'first').reset_index(drop = True)

# Sentiment analysis

# Tokenizing the twitter data and removing stopwords

tweets = [str(tweet) for tweet in td.tweet]
stop_words = set(stopwords.words('english'))
clean_tweets = []

for tweet in tweets:
    
    word_tokens = word_tokenize(tweet)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    
    for w in word_tokens:
        
        if w not in stop_words:
            
            filtered_sentence.append(w)
    
    clean_tweets.append(filtered_sentence)

twitters = []

for i in range(len(clean_tweets)):
        
    for j in range(len(clean_tweets[i])):
        
        twitters.append(clean_tweets[i][j])
        
# Manually created list of keys from twitters (manually inspected list and dropped some non-legible keys)

keys = ['Elon Musk', 'doge coin', 'elonmusk', 'elonmusk nbcsnl', 'doge', 'solana_daily blockfolio',
'solana solana_daily', 'dogecoin', 'nbcsnl elonmusk', 'think', 'Airdrop solana', 'one', 'giveaway tothemoon',
'token crt', 'money giveaway', 'crt farmerscarrot', 'farmerscarrot money', 'discord token', 'solana discord',
'lol', 'look', 'blockfolio Airdrop', 'right', 'elonmusk Doge', 'last night', 'SNL', 'know', 'Saturday Night',
'need', 'guy', 'Doge moon', 'buy doge', 'dogetothemoon', 'BTC ETH', 'crypto', 'Musk SNL', 'elonmusk dogecoin','much',
'say','well', 'come', 'today', 'want', 'even', 'show', 'MileyCyrus', 'shit', 'dogecoin elonmusk', 'doge elonmusk',
'mean', 'love', 'man', 'thought', 'thing', 'see', 'Shiba Inu', 'real', 'everyone', 'make', 'watch SNL', 'made',
'doge shib', 'better', 'watching SNL', 'something', 'still', 'Good project', 'tweet', 'Now', 'SNL skit', 'hope',
'Ye', 'take', 'bro', 'maybe', 'joke', 'Shibtoken', 'soon', 'NFT Lottery', 'Animated NFT', 'Musk Animated']

# Generating the wordcloud

wordcloud = WordCloud(max_font_size = 50, max_words = 100, colormap = 'magma', background_color = 'white').generate(' '.join(keys))
plt.figure()
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.savefig('C:/Users/' + username + '/Documents/Data/dogefather/dogecloud.eps')

# Creating LaTeX results tables for the VAR

# Get values

P = round(results2.pvalues,4)
T = round(results2.tvalues,4)
S = round(results2.stderr,4)
C = round(results2.params,4)
V = results2.exog_names

# Make tables

dogtab = []
postab = []
negtab = []

for i in range(len(V)):
    
    # Create strings for each row
    
    sd = V[i] + ' & ' + str(C['Price'][i]) + ' & ' + str(S['Price'][i]) + ' & ' + str(T['Price'][i]) + ' & ' + str(P['Price'][i]) + '\\\\'
    sp = V[i] + ' & ' + str(C['Positive_Sentiment'][i]) + ' & ' + str(S['Positive_Sentiment'][i]) + ' & ' + str(T['Positive_Sentiment'][i]) + ' & ' + str(P['Positive_Sentiment'][i]) + '\\\\'
    sn = V[i] + ' & ' + str(C['Negative_Sentiment'][i]) + ' & ' + str(S['Negative_Sentiment'][i]) + ' & ' + str(T['Negative_Sentiment'][i]) + ' & ' + str(P['Negative_Sentiment'][i]) + '\\\\'
    
    dogtab.append(sd)
    postab.append(sp)
    negtab.append(sn)

tabouts = ['dogtab', 'postab', 'negtab']
tabs = [dogtab, postab, negtab]

for tab in tabs:
    
    with open('C:/Users/' + username + '/Documents/Data/dogefather/' + tabouts[tabs.index(tab)] + '.txt', 'w') as f:
        
        for row in tab:
            
            f.write("%s\n" % row)
            
    f.close()

