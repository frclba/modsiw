import tweepy
from textblob import TextBlob

consumer_key = 'XMjTThyPDd5ac6xtcEygq8RZD'
consumer_secret = '4TLQ0xPxpW0R3hLzezS3hIKoHC1ZZfFyvSW0zuWflZORVWq1NK'

access_token = '259877044-qD7vq4UtqHf04QwceWX9qpDcQqF8C97DDp0Iwwoy'
access_token_secret = '0e1F6XquuP30KipxhqjGaJGElK45xbii5blSvxdgtmbRZ'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Artificial Inteligence')

for tweet in public_tweets:
    print(tweet.text)
    
    #Step 4 Perform Sentiment Analysis on Tweets
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)
    print("")