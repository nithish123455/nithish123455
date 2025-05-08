import tweepy
from transformers import pipeline
import pandas as pd

# Twitter API credentials
API_KEY = 'your_api_key'
API_SECRET = 'your_api_secret'
ACCESS_TOKEN = 'your_access_token'
ACCESS_SECRET = 'your_access_secret'

# Authenticate to Twitter
auth = tweepy.OAuth1UserHandler(API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth)

# Fetch tweets based on a hashtag or keyword
def fetch_tweets(query, count=100):
    tweets = tweepy.Cursor(api.search_tweets, q=query, lang="en", tweet_mode='extended').items(count)
    tweet_list = [tweet.full_text for tweet in tweets]
    return tweet_list

# Load emotion classification model
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

# Analyze emotions in tweets
def analyze_emotions(tweets):
    results = []
    for tweet in tweets:
        emotions = emotion_analyzer(tweet)[0]
        top_emotion = max(emotions, key=lambda x: x['score'])
        results.append({
            'Tweet': tweet,
            'Emotion': top_emotion['label'],
            'Confidence': top_emotion['score']
        })
    return pd.DataFrame(results)

# Main program
if __name__ == "__main__":
    query = "mental health"
    tweets = fetch_tweets(query, count=50)
    emotion_results = analyze_emotions(tweets)
    print(emotion_results.head())
    emotion_results.to_csv("emotion_analysis_results.csv", index=False)
