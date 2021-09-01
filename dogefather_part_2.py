# This script performs the sentiment analysis on the cleaned twitter data

# Importing required modules

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer

# Read in the raw data file -- input your useername / change filepath as needed

username = ''
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

# Separating the data set into two data sets - one for SNL and one for $doge

snl_keys = set(['#SNL', '#snl', '#dogefather', '#Elon', '#SNLmay8', 'SNL', 'snl', 'dogefather', 'Elon', 'Musk'])
doge_keys = set(['#dogecoin', '#doge', '#dogecoinrise', '#tothemoon', '$doge', 'dogecoin', 'doge', '#doge', '#dogefather', 'dogefather'])

snl_check = [1 if len(snl_keys.intersection(tweet)) > 0 else 0 for tweet in clean_tweets]
doge_check = [1 if len(doge_keys.intersection(tweet)) > 0 else 0 for tweet in clean_tweets]

snl_check = pd.Series(snl_check, name = 'SNL_Check')
doge_check = pd.Series(doge_check, name = 'doge_Check')

td = pd.concat([td, snl_check, doge_check], axis = 1)

td_snl = td[td['SNL_Check'] == 1].reset_index(drop = True)
td_dog = td[td['doge_Check'] == 1].reset_index(drop = True)

# Re-tokenizing the new split data sets

tweets_snl = [str(tweet) for tweet in td_snl.tweet]
tweets_dog = [str(tweet) for tweet in td_dog.tweet]
stop_words = set(stopwords.words('english'))
clean_tweets_snl = []
clean_tweets_dog = []

for tweet in tweets_snl:
    
    word_tokens = word_tokenize(tweet)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    clean_tweets_snl.append(filtered_sentence)

for tweet in tweets_dog:
    
    word_tokens = word_tokenize(tweet)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    clean_tweets_dog.append(filtered_sentence)

# Lemmatizing

lemon = WordNetLemmatizer()

for t in range(len(clean_tweets_snl)):
    
    res = []
    
    for w in clean_tweets_snl[t]:
        
        res.append(lemon.lemmatize(w))
    
    clean_tweets_snl[t] = res

for t in range(len(clean_tweets_dog)):
    
    res = []
    
    for w in clean_tweets_dog[t]:
        
        res.append(lemon.lemmatize(w))
    
    clean_tweets_dog[t] = res

# Stemming

ps = PorterStemmer()

for t in range(len(clean_tweets_snl)):
    
    res = []
    
    for w in clean_tweets_snl[t]:
        
        res.append(ps.stem(w))
        
    clean_tweets_snl[t] = res

for t in range(len(clean_tweets_dog)):
    
    res = []
    
    for w in clean_tweets_dog[t]:
        
        res.append(ps.stem(w))
        
    clean_tweets_dog[t] = res

# Remove symbols

baddies = ['@', '#', '$', '%', '&', '*', ':', ';', '"', '.', ',', '/', '!',
           "'s", 'http', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

for t in range(len(clean_tweets_snl)):
    
    clean_tweets_snl[t] = [c for c in clean_tweets_snl[t] if c not in baddies]

for t in range(len(clean_tweets_dog)):
    
    clean_tweets_dog[t] = [c for c in clean_tweets_dog[t] if c not in baddies]

# The data is now prepped and ready to get all sentimental

sad = SentimentIntensityAnalyzer()

tweets_snl = [' '.join(t) for t in clean_tweets_snl]
tweets_dog = [' '.join(t) for t in clean_tweets_dog]

svals_snl = []
svals_dog = []

for t in tweets_snl:
    
    svals_snl.append(sad.polarity_scores(t))

for t in tweets_dog:
    
    svals_dog.append(sad.polarity_scores(t))

negs_snl = []
pos_snl = []

negs_dog = []
pos_dog = []

for s in svals_snl:
    
    negs_snl.append(s['neg'])
    pos_snl.append(s['pos'])

for s in svals_dog:
    
    negs_dog.append(s['neg'])
    pos_dog.append(s['pos'])

scores_snl = [pos_snl[i] - negs_snl[i] for i in range(len(pos_snl))]
scores_dog = [pos_dog[i] - negs_dog[i] for i in range(len(pos_dog))]

# Adding the sentiment analysis scores to the main dataframe

negs_snl = pd.Series(negs_snl, name = 'Negative')
pos_snl = pd.Series(pos_snl, name = 'Positive')
scores_snl = pd.Series(scores_snl, name = 'Sentiment')
td_snl = pd.concat([td_snl, scores_snl, pos_snl, negs_snl], axis = 1)

negs_dog = pd.Series(negs_dog, name = 'Negative')
pos_dog = pd.Series(pos_dog, name = 'Positive')
scores_dog = pd.Series(scores_dog, name = 'Sentiment')
td_dog = pd.concat([td_dog, scores_dog, pos_dog, negs_dog], axis = 1)

# Convert date data to more useful form

dayss = [int(x[8:10]) for x in td_snl.date]
hourss = [int(x[11:13]) for x in td_snl.date]
minss = [int(x[14:16]) for x in td_snl.date]

daysd = [int(x[8:10]) for x in td_dog.date]
hoursd = [int(x[11:13]) for x in td_dog.date]
minsd = [int(x[14:16]) for x in td_dog.date]

# Add these to the data frame

hourss = pd.Series(hourss, name = 'Hour')
minss = pd.Series(minss, name = 'Minute')
dayss = pd.Series(dayss, name = 'Day')
td_snl = pd.concat([td_snl, dayss, hourss, minss], axis = 1)

hoursd = pd.Series(hoursd, name = 'Hour')
minsd = pd.Series(minsd, name = 'Minute')
daysd = pd.Series(daysd, name = 'Day')
td_dog = pd.concat([td_dog, daysd, hoursd, minsd], axis = 1)

# Remove tweets from outside of the time frame

# The first set gives an hour window around the target window

c1 = [1 if (td_snl.Day[i] == 8) and (td_snl.Hour[i] >= 22) else 0 for i in range(len(td_snl))] # Tweets from 10 pm to midnight EDT on 5/8
c2 = [1 if (td_snl.Day[i] == 8) and (td_snl.Hour[i] == 22) and (td_snl.Minute[i] >= 30) else 0 for i in range(len(td_snl))] # Remove tweets from 10 to 10:30 pm EDT on 5/8
c3 = [1 if (td_snl.Day[i] == 8) and (td_snl.Hour[i] == 23) else 0 for i in range(len(td_snl))] # Tweets from 11 pm to midnight EDT on 5/8
c4 = [1 if (td_snl.Day[i] == 9) and (td_snl.Hour[i] < 2) else 0 for i in range(len(td_snl))] # Tweets from midnight to 2 am EDT on 5/9
c5 = [c1[i]*c2[i] for i in range(len(c1))]
c = [c3[i] + c4[i] + c5[i] for i in range(len(c1))]
c = pd.Series(c, name = 'c')
td_snl = pd.concat([td_snl,c], axis = 1)
td_snl = td_snl[td_snl['c'] == 1].reset_index(drop = True)
td_snl = td_snl.drop('c', axis = 1)

c1 = [1 if (td_dog.Day[i] == 8) and (td_dog.Hour[i] >= 22) else 0 for i in range(len(td_dog))] # Tweets from 10 pm to midnight EDT on 5/8
c2 = [1 if (td_dog.Day[i] == 8) and (td_dog.Hour[i] == 22) and (td_dog.Minute[i] >= 30) else 0 for i in range(len(td_dog))] # Remove tweets from 10 to 10:30 pm EDT on 5/8
c3 = [1 if (td_dog.Day[i] == 8) and (td_dog.Hour[i] == 23) else 0 for i in range(len(td_dog))] # Tweets from 11 pm to midnight EDT on 5/8
c4 = [1 if (td_dog.Day[i] == 9) and (td_dog.Hour[i] < 2) else 0 for i in range(len(td_dog))] # Tweets from midnight to 2 am EDT on 5/9
c5 = [c1[i]*c2[i] for i in range(len(c1))]
c = [c3[i] + c4[i] + c5[i] for i in range(len(c1))]
c = pd.Series(c, name = 'c')
td_dog = pd.concat([td_dog,c], axis = 1)
td_dog = td_dog[td_dog['c'] == 1].reset_index(drop = True)
td_dog = td_dog.drop('c', axis = 1)

# The second set is exactly the target window

cc1 = [1 if (td_snl.Day[i] == 8) and (td_snl.Hour[i] == 23) and (td_snl.Minute[i] >= 30) else 0 for i in range(len(td_snl))] # Tweets from 11:30 pm to midnight EDT on 5/8
cc2 = [1 if (td_snl.Day[i] == 9) and (td_snl.Hour[i] < 1) else 0 for i in range(len(td_snl))] # Tweets from midnight to 1 am EDT on 5/9
cc = [cc1[i] + cc2[i] for i in range(len(cc1))]
cc = pd.Series(cc, name = 'cc')
td_snl2 = pd.concat([td_snl,cc], axis = 1)
td_snl2 = td_snl2[td_snl2['cc'] == 1].reset_index(drop = True)
td_snl2  = td_snl2.drop('cc', axis = 1)

cc1 = [1 if (td_dog.Day[i] == 8) and (td_dog.Hour[i] == 23) and (td_dog.Minute[i] >= 30) else 0 for i in range(len(td_dog))] # Tweets from 11:30 pm to midnight EDT on 5/8
cc2 = [1 if (td_dog.Day[i] == 9) and (td_dog.Hour[i] < 1) else 0 for i in range(len(td_dog))] # Tweets from midnight to 1 am EDT on 5/9
cc = [cc1[i] + cc2[i] for i in range(len(cc1))]
cc = pd.Series(cc, name = 'cc')
td_dog2 = pd.concat([td_dog,cc], axis = 1)
td_dog2 = td_dog2[td_dog2['cc'] == 1].reset_index(drop = True)
td_dog2  = td_dog2.drop('cc', axis = 1)

# Creating minute level identifiers

ids1 = ['8-23-' + str(i) for i in range(30,60)]
ids2 = ['9-0-' + str(i) for i in range(60)]
ids = ids1 + ids2

# Adding these identifiers to the dataframe

newcol = [str(td_snl2.Day[i]) + '-' + str(td_snl2.Hour[i]) + '-' + str(td_snl2.Minute[i]) for i in range(len(td_snl2))]
newcol = pd.Series(newcol, name = 'idx')
td_snl2 = pd.concat([td_snl2, newcol], axis = 1)

newcol = [str(td_dog2.Day[i]) + '-' + str(td_dog2.Hour[i]) + '-' + str(td_dog2.Minute[i]) for i in range(len(td_dog2))]
newcol = pd.Series(newcol, name = 'idx')
td_dog2 = pd.concat([td_dog2, newcol], axis = 1)

# Creating the minute by minute data for the target window for the SNL data

volumes = []
score_sums = []
pos_sums = []
neg_sums = []
mean_scores = []
mean_pos = []
mean_neg = []

for idx in ids:
    
    tmp = td_snl2[td_snl2.idx == idx]
    volumes.append(len(tmp))
    score_sums.append(sum(tmp.Sentiment))
    pos_sums.append(sum(tmp.Positive))
    neg_sums.append(sum(tmp.Negative))
    mean_scores.append(sum(tmp.Sentiment) / len(tmp))
    mean_pos.append(sum(tmp.Positive) / len(tmp))
    mean_neg.append(sum(tmp.Negative) / len(tmp))

# Create a dataframe of this data and write to file

dias = [i[0] for i in ids]
horas = [i[2:i[2:].find('-')+2] for i in ids]
minutos = [i[i[2:].find('-')+3:] for i in ids]

ids = pd.Series(ids, name = 'Time')
dias = pd.Series(dias, name = 'Day')
horas = pd.Series(horas, name = 'Hour')
minutos = pd.Series(minutos, name = 'Minute')
volumes = pd.Series(volumes, name = 'Tweets')
mean_scores = pd.Series(mean_scores, name = 'Sentiment')
mean_pos = pd.Series(mean_pos, name = 'Positive')
mean_neg = pd.Series(mean_neg, name = 'Negative')
score_sums = pd.Series(score_sums, name = 'Sentiment.Total')
pos_sums = pd.Series(pos_sums, name = 'Positive.Total')
neg_sums = pd.Series(neg_sums, name = 'Negative.Total')

df1_snl = pd.concat([ids, dias, horas, minutos, volumes, mean_scores, mean_pos, mean_neg, score_sums, pos_sums, neg_sums], axis = 1)
df1_snl.to_csv('C:/Users/' + username + '/Documents/Data/dogefather/sentiment_snl.csv', index = False)

# Creating the minute by minute data for the target window for the doge data

volumes = []
score_sums = []
pos_sums = []
neg_sums = []
mean_scores = []
mean_pos = []
mean_neg = []

for idx in ids:
    
    tmp = td_dog2[td_dog2.idx == idx]
    volumes.append(len(tmp))
    score_sums.append(sum(tmp.Sentiment))
    pos_sums.append(sum(tmp.Positive))
    neg_sums.append(sum(tmp.Negative))
    mean_scores.append(sum(tmp.Sentiment) / len(tmp))
    mean_pos.append(sum(tmp.Positive) / len(tmp))
    mean_neg.append(sum(tmp.Negative) / len(tmp))

# Create a dataframe of this data and write to file

dias = [i[0] for i in ids]
horas = [i[2:i[2:].find('-')+2] for i in ids]
minutos = [i[i[2:].find('-')+3:] for i in ids]

ids = pd.Series(ids, name = 'Time')
dias = pd.Series(dias, name = 'Day')
horas = pd.Series(horas, name = 'Hour')
minutos = pd.Series(minutos, name = 'Minute')
volumes = pd.Series(volumes, name = 'Tweets')
mean_scores = pd.Series(mean_scores, name = 'Sentiment')
mean_pos = pd.Series(mean_pos, name = 'Positive')
mean_neg = pd.Series(mean_neg, name = 'Negative')
score_sums = pd.Series(score_sums, name = 'Sentiment.Total')
pos_sums = pd.Series(pos_sums, name = 'Positive.Total')
neg_sums = pd.Series(neg_sums, name = 'Negative.Total')

df1_doge = pd.concat([ids, dias, horas, minutos, volumes, mean_scores, mean_pos, mean_neg, score_sums, pos_sums, neg_sums], axis = 1)
df1_doge.to_csv('C:/Users/' + username + '/Documents/Data/dogefather/sentiment_doge.csv', index = False)

# Repeat the above for the larger window for SNL data

ids1 = ['8-22-' + str(i) for i in range(30,60)]
ids2 = ['8-23-' + str(i) for i in range(60)]
ids3 = ['9-0-' + str(i) for i in range(60)]
ids4 = ['9-1-' + str(i) for i in range(60)]
ids = ids1 + ids2 + ids3 + ids4

newcol = [str(td_snl.Day[i]) + '-' + str(td_snl.Hour[i]) + '-' + str(td_snl.Minute[i]) for i in range(len(td_snl))]
newcol = pd.Series(newcol, name = 'idx')
td_snl = pd.concat([td_snl, newcol], axis = 1)

volumes = []
score_sums = []
pos_sums = []
neg_sums = []
mean_scores = []
mean_pos = []
mean_neg = []

for idx in ids:
    
    tmp = td_snl[td_snl.idx == idx]
    volumes.append(len(tmp))
    score_sums.append(sum(tmp.Sentiment))
    pos_sums.append(sum(tmp.Positive))
    neg_sums.append(sum(tmp.Negative))
    mean_scores.append(sum(tmp.Sentiment) / len(tmp))
    mean_pos.append(sum(tmp.Positive) / len(tmp))
    mean_neg.append(sum(tmp.Negative) / len(tmp))

dias = [i[0] for i in ids]
horas = [i[2:i[2:].find('-')+2] for i in ids]
minutos = [i[i[2:].find('-')+3:] for i in ids]

ids = pd.Series(ids, name = 'Time')
dias = pd.Series(dias, name = 'Day')
horas = pd.Series(horas, name = 'Hour')
minutos = pd.Series(minutos, name = 'Minute')
volumes = pd.Series(volumes, name = 'Tweets')
mean_scores = pd.Series(mean_scores, name = 'Sentiment')
mean_pos = pd.Series(mean_pos, name = 'Positive')
mean_neg = pd.Series(mean_neg, name = 'Negative')
score_sums = pd.Series(score_sums, name = 'Sentiment.Total')
pos_sums = pd.Series(pos_sums, name = 'Positive.Total')
neg_sums = pd.Series(neg_sums, name = 'Negative.Total')

df2_snl = pd.concat([ids, dias, horas, minutos, volumes, mean_scores, mean_pos, mean_neg, score_sums, pos_sums, neg_sums], axis = 1)
df2_snl.to_csv('C:/Users/' + username + '/Documents/Data/dogefather/sentiment2_snl.csv', index = False)

# Repeat the above for the larger window for SNL data

ids1 = ['8-22-' + str(i) for i in range(30,60)]
ids2 = ['8-23-' + str(i) for i in range(60)]
ids3 = ['9-0-' + str(i) for i in range(60)]
ids4 = ['9-1-' + str(i) for i in range(60)]
ids = ids1 + ids2 + ids3 + ids4

newcol = [str(td_dog.Day[i]) + '-' + str(td_dog.Hour[i]) + '-' + str(td_dog.Minute[i]) for i in range(len(td_dog))]
newcol = pd.Series(newcol, name = 'idx')
td_dog = pd.concat([td_dog, newcol], axis = 1)

volumes = []
score_sums = []
pos_sums = []
neg_sums = []
mean_scores = []
mean_pos = []
mean_neg = []

for idx in ids:
    
    tmp = td_dog[td_dog.idx == idx]
    volumes.append(len(tmp))
    score_sums.append(sum(tmp.Sentiment))
    pos_sums.append(sum(tmp.Positive))
    neg_sums.append(sum(tmp.Negative))
    mean_scores.append(sum(tmp.Sentiment) / len(tmp))
    mean_pos.append(sum(tmp.Positive) / len(tmp))
    mean_neg.append(sum(tmp.Negative) / len(tmp))

dias = [i[0] for i in ids]
horas = [i[2:i[2:].find('-')+2] for i in ids]
minutos = [i[i[2:].find('-')+3:] for i in ids]

ids = pd.Series(ids, name = 'Time')
dias = pd.Series(dias, name = 'Day')
horas = pd.Series(horas, name = 'Hour')
minutos = pd.Series(minutos, name = 'Minute')
volumes = pd.Series(volumes, name = 'Tweets')
mean_scores = pd.Series(mean_scores, name = 'Sentiment')
mean_pos = pd.Series(mean_pos, name = 'Positive')
mean_neg = pd.Series(mean_neg, name = 'Negative')
score_sums = pd.Series(score_sums, name = 'Sentiment.Total')
pos_sums = pd.Series(pos_sums, name = 'Positive.Total')
neg_sums = pd.Series(neg_sums, name = 'Negative.Total')

df2_doge = pd.concat([ids, dias, horas, minutos, volumes, mean_scores, mean_pos, mean_neg, score_sums, pos_sums, neg_sums], axis = 1)
df2_doge.to_csv('C:/Users/' + username + '/Documents/Data/dogefather/sentiment2_doge.csv', index = False)

