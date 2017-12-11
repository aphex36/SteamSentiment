import json
import nltk
import csv
import math
import sys
import re
import numpy as np
import collections
import random
import os

#nltk.download('averaged_perceptron_tagger')
posWords = set()
negWords = set()
posDictCount = dict()
negDictCount = dict()
truePositives = 0
trueNegatives = 0
falsePositives = 0
falseNegatives = 0

def determineSentiment(words):
   sentiment = 0
   for i in range(len(words)):
    	tempWord = words[i].lower()
	tempWord = re.sub(r'[^\w\s]','',tempWord)
	if str(tempWord) in posWords:
            sentiment += 1
            if str(tempWord) not in posDictCount:
               posDictCount[str(tempWord)] = 1
            else:
               posDictCount[str(tempWord)] += 1
        elif str(tempWord) in negWords:
   	    sentiment -= 1
            if str(tempWord) not in negDictCount:
               negDictCount[str(tempWord)] = 1
            else:
               negDictCount[str(tempWord)] += 1

   if sentiment >= 0:
      return 'positive'
   else:
      return 'negative'

def printOutMaxOccuringWords(indicator):
   results = []
   refDict = negDictCount
   if indicator == 'Positive':
      refDict = posDictCount
   for key in refDict:
      results.append((key, refDict[key]))
   results.sort(key= lambda x: x[1], reverse=True)

   print(indicator)
   for i in xrange(len(results)):
      print(results[i][0] + ": " + str(results[i][1]))



with open('positive_words.txt') as p:
   for x in p.readlines():
      posWords.add(x.strip('\n'))

with open('negative_words.txt') as n:
   for x in n.readlines():
      negWords.add(x.strip('\n'))
p.close()
n.close()
allFiles = os.listdir("./pos/")
allFiles.extend(os.listdir("./neg/"))
totalScore = 0
totalRevs = 0
fileNo = 0
for fileName in allFiles:
   actualDir = ""
   isPos = True

   if fileName[-7:-4] == "neg":
      actualDir = "./neg/" + fileName
      isPos = False
   else:
      actualDir = "./pos/" + fileName
   fileNo += 1

   with open(actualDir) as f:
      content = [x.strip('\n') for x in f.readlines()]
   for i in range(len(content)-1):
      if content[i] == '':
   	  continue
      #print(content[i])
      tempJson = json.loads(content[i])
      individualWords = tempJson['review'].split()
      newStr = ""
      for wordGiven in individualWords:
          newStr += re.sub(r'[^\w\s]','',wordGiven.encode('ascii', 'ignore')) + " "


      totalRevs += 1
      if determineSentiment(individualWords) == 'negative' and not isPos:
         totalScore += 1
         trueNegatives += 1

      elif determineSentiment(individualWords) == 'positive' and isPos:
	 totalScore += 1
         truePositives += 1
      elif determineSentiment(individualWords) == 'positive' and not isPos:
         falsePositives += 1
      else:
         falseNegatives += 1

   f.close()

print("\n")
print("Test Accuracy: " + str((1.0*totalScore)/totalRevs))
print("Precision: " + str((1.0*truePositives)/(truePositives+falsePositives)))
print("Recall: " + str((1.0*truePositives)/(truePositives+falseNegatives)))
