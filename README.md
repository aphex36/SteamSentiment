# SteamSentiment
CS229 Project

This is the sentiment classifier we are doing for the CS229 project. I have aggregated 1000 labeled examples from 10 different games into files in json format

As of now, this can do the baseline of just taking a sentiment lexicon and aggregating scores and Binarized naive bayes with stop-word filtering

To run the baseline:

python baseline_sentiment_detector.py

To run Naive Bayes:

python NaiveBayes.py -fb .

