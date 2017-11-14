import json 
import csv
import math
import sys
import re
import numpy as np
import collections
import random
posWords = set()
negWords = set()

def determineSentiment(words):
	sentiment = 0
	for i in range(len(words)):
		tempWord = words[i].lower()
		tempWord = re.sub(r'[^\w\s]','',tempWord)
		if tempWord in posWords:
			sentiment += 1
		elif tempWord in negWords:
			sentiment -= 1
		
	if sentiment >= 0:
		return 'positive'
	else:
		return 'negative'


with open('positive_words.txt') as p:
	for x in p.readlines():
		posWords.add(x.strip('\n'))


with open('negative_words.txt') as n:
	for x in n.readlines():
		negWords.add(x.strip('\n'))

with open('all_neg_reviews.txt') as f:
    content = [x.strip('\n') for x in f.readlines()]

negScore = 0
totalNeg = len(content)-1
for i in range(len(content)-1):
	if content[i] == '':
		totalNeg -= 1
		continue
	tempJson = json.loads(content[i])
	individualWords = tempJson['review'].split()

	if(determineSentiment(individualWords) == 'negative'):
		negScore += 1

with open('all_pos_reviews.txt') as f:
    otherContent = [x.strip('\n') for x in f.readlines()]
posScore = 0
totalPos = len(otherContent)-1


for i in range(len(otherContent) - 1):
	if otherContent[i] == '':
		totalPos -= 1
		continue
	tempJson = json.loads(otherContent[i])
	individualWords = tempJson['review'].split()
	

	if(determineSentiment(individualWords) == 'positive'):
		posScore += 1

print("Total percentage: " + str(posScore + negScore) + "/" + str(totalPos + totalNeg) + " = " + str((posScore+negScore)/((1.0)*(totalPos+totalNeg))))

