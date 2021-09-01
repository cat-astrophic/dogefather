# This script uses twint to harvest relevant twitter data

# Importing required modules

import twint
import nest_asyncio
import pandas as pd

# Use nest_asyncio to permit asynchronous loops

nest_asyncio.apply()

# Keywords to search for

keywords = ['#SNL', '#snl', '#dogefather', '#dogecoin', '#doge', '#Elon',
            '#SNLmay8', '#dogecoinrise', '#tothemoon', '$doge', 'SNL', 'snl',
            'dogefather', 'dogecoin', 'doge', 'Elon', 'Musk', '#doge', 'doge']

# Initializing the main dataframe

twitter_df = pd.DataFrame()

# Using twint to get data

for k in keywords:
    
    t = twint.Config()
    t.Search = k
    t.Since = '2021-05-09'
    t.Until = '2021-05-12'
    t.Lang = 'en'
    t.Store_csv = True
    t.Pandas = True
    twint.run.Search(t)
    twint.storage.panda.save
    Tweets_df = twint.storage.panda.Tweets_df
    twitter_df = pd.concat([twitter_df, Tweets_df], axis = 0)

# Writing the complete raw data file

username = ''
twitter_df.to_csv('C:/Users/' + username + '/Documents/Data/dogefather/raw_twitter_data.csv', index = False)

