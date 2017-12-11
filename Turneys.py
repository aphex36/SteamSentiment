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

trainAccs = []
precisionsCollected = []
recallsCollected = []
testAccs = []
def test10Fold(foldNum, trainNums, testNums):
    posWords = set()
    negWords = set()
    posDictCount = dict()
    negDictCount = dict()
    phraseDict = dict()
    reviewContextAroundPhrases = dict()
    initialCountPass = True
    def determineSentiment(words):
       sentiment = 0
       for i in range(len(words)):
        	tempWord = words[i].lower()
       tempWord = re.sub(r'[^\w\s]','',tempWord)
       if str(tempWord) in posWords:
            sentiment += 1
            if initialCountPass:
                    if str(tempWord) not in posDictCount:
                       posDictCount[str(tempWord)] = 1
                    else:
                       posDictCount[str(tempWord)] += 1
            elif str(tempWord) in negWords:
       	        sentiment -= 1
            if initialCountPass:
                if str(tempWord) not in negDictCount:
                   negDictCount[str(tempWord)] = 1
                else:
                   negDictCount[str(tempWord)] += 1

       if sentiment >= 0:
          return 1
       else:
          return -1

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


    def printOutPhrases(reviewNum, tokenizedReview):
        for i in range(len(tokenizedReview)-1):
            currPhrase = tokenizedReview[i][0] + " " + tokenizedReview[i+1][0]
            isPhrase = False
            if tokenizedReview[i][1] == 'JJ' and (tokenizedReview[i+1][1] == 'NN' or tokenizedReview[i+1][1] == 'NNS'):
                isPhrase = True
            elif (tokenizedReview[i][1] == 'RB' or tokenizedReview[i][1] == 'RBR' or tokenizedReview[i][1] == 'RBS'):
                if (i+2) < len(tokenizedReview):
                    if tokenizedReview[i+2][1] != 'NN' and tokenizedReview[i+2][1] != 'NNS':
                        isPhrase = True
            elif (tokenizedReview[i][1] == 'JJ') and (tokenizedReview[i+1][1] == 'JJ'):
                if (i+2) < len(tokenizedReview):
                    if tokenizedReview[i+2][1] != 'NN' and tokenizedReview[i+2][1] != 'NNS':
                        isPhrase = True
            elif (tokenizedReview[i][1] == 'NN' or tokenizedReview[i][1] == 'NNS' or tokenizedReview[i][1] == 'RBS') and (tokenizedReview[i+1][1] == 'JJ'):
                if (i+2) < len(tokenizedReview):
                    if tokenizedReview[i+2][1] != 'NN' and tokenizedReview[i+2][1] != 'NNS':
                        isPhrase = True
            elif (tokenizedReview[i][1] == 'RB' or tokenizedReview[i][1] == 'RBR' or tokenizedReview[i][1] == 'RBS') and (tokenizedReview[i+1][1] == 'VB' or tokenizedReview[i+1][1] == 'VBD' or tokenizedReview[i+1][1] == 'VBN' or tokenizedReview[i+1][1] == 'VBG'):
                isPhrase = True
            if isPhrase:
                if currPhrase not in phraseDict:
                    phraseDict[currPhrase] = set()
                phraseDict[currPhrase].add(reviewNum)
                if str(reviewNum) not in reviewContextAroundPhrases:
                    reviewContextAroundPhrases[str(reviewNum)] = dict()
                if currPhrase not in reviewContextAroundPhrases[str(reviewNum)]:
                    reviewContextAroundPhrases[str(reviewNum)][currPhrase] = []
                for j in range(1, 20):
                    if i - j < 0:
                        break
                    else:
                        reviewContextAroundPhrases[str(reviewNum)][currPhrase].insert(0, tokenizedReview[i - j][0])
                for j in range(1, 20):
                    if i + 1 + j >= len(tokenizedReview):
                        break
                    else:
                        reviewContextAroundPhrases[str(reviewNum)][currPhrase].append(tokenizedReview[i + 1 + j][0])
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
    revNo = 0
    for fileName in allFiles:
       if fileName[0] not in trainNums:
           fileNo += 1
           continue
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

          tempJson = json.loads(content[i])
          individualWords = tempJson['review'].split()
          newStr = ""
          determineSentiment(individualWords)
          for wordGiven in individualWords:
              newStr += re.sub(r'[^\w\s]','',wordGiven.encode('ascii', 'ignore')).lower() + " "
          printOutPhrases(totalRevs, nltk.pos_tag(nltk.word_tokenize(newStr)))
          totalRevs += 1

       f.close()

    initialCountPass = False

    def findSemanticOrientation(reviewNum, listOfWords):
        semanticOrientationAvg = 0
        numPhrasesFound = 0

        if reviewNum not in reviewContextAroundPhrases:
            return determineSentiment(listOfWords)
        for phraseInRev in reviewContextAroundPhrases[reviewNum]:
            numPhrasesFound += 1
            semanticOrientation = 0
            positiveSemOrientation = dict()
            negativeSemOrientation = dict()
            #print("On review number " + reviewNum)
            #print(reviewContextAroundPhrases[reviewNum])
            #print("Here is the phrase we are on:")
            #print(phraseInRev)
            allRevsWithPhrase = phraseDict[phraseInRev]
            #print("here is where that phrase appears")
            #print(allRevsWithPhrase)
            for numOfRev in allRevsWithPhrase:
                currSentenceContext = reviewContextAroundPhrases[str(numOfRev)][phraseInRev]
                #print("Now for review " + str(numOfRev))

                wordPos = 0
                for word in currSentenceContext:
                    if word in posDictCount:
                        #print(word + " is in posDict")
                        if word not in positiveSemOrientation:
                            positiveSemOrientation[word] = (1.0/posDictCount[word])
                        else:
                            positiveSemOrientation[word] += (1.0/posDictCount[word])
                    elif word in negDictCount:
                        if word not in negativeSemOrientation:
                            negativeSemOrientation[word] = (1.0/negDictCount[word])
                        else:
                            negativeSemOrientation[word] += (1.0/negDictCount[word])

            for word in positiveSemOrientation:
                semanticOrientation += math.log(positiveSemOrientation[word])
            for word in negativeSemOrientation:
                semanticOrientation -= math.log(negativeSemOrientation[word])
            semanticOrientationAvg += semanticOrientation

        return semanticOrientationAvg/(1.0*numPhrasesFound)

    someRevNum = 0
    totalScore = 0
    totalRevs = 0
    trainTotal = 0
    trainAccurate = 0
    fileNo = 0
    tn = 0
    tp = 0
    fp = 0
    fn = 0
    for fileName in allFiles:
       isTraining = False
       if fileName[0] not in testNums:
          isTraining = True
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
          if isTraining:
              trainTotal += 1

          tempJson = json.loads(content[i])
          individualWords = tempJson['review'].split()
          newStr = ""

          if not isTraining:
              totalRevs += 1

          if findSemanticOrientation(str(someRevNum), individualWords) < 0 and not isPos:
             if not isTraining:
                 fn += 1
          elif findSemanticOrientation(str(someRevNum), individualWords) > 0 and not isPos:
             if isTraining:
                 trainAccurate += 1
             else:
                 totalScore += 1
                 tp += 1

          elif findSemanticOrientation(str(someRevNum), individualWords) > 0 and isPos:
    	     if not isTraining:
                fp += 1
          else:
             if isTraining:
                trainAccurate += 1
             else:
                totalScore += 1
                tn += 1
          someRevNum += 1
       f.close()

    print("Fold " + str(foldNum) + " Train Accuracy : " + str((1.0*trainAccurate)/trainTotal))
    print("Fold " + str(foldNum) + " Test Accuracy : " + str((1.0*totalScore)/totalRevs))
    print("Fold " + str(foldNum) + " Precision : " + str((1.0*tp)/(tp+fp)))
    print("Fold " + str(foldNum) + " Recall : " + str((1.0*tp)/(tp+fn)))

    testAccs.append((1.0*totalScore)/totalRevs)
    precisionsCollected.append((1.0*tp)/(tp+fp))
    recallsCollected.append((1.0*tp)/(tp+fn))
    trainAccs.append(((1.0*trainAccurate)/trainTotal))
for i in range(10):
    trainIndices = ''
    testIndices = ''
    for j in range(i, i+7):
        turnedIndex = j % 10
        trainIndices += str(turnedIndex)
    for j in range(i+7, i+10):
        turnedIndex = j % 10
        testIndices += str(turnedIndex)
    test10Fold(i+1, trainIndices, testIndices)
print("Average Train Accuracy: " + str(np.mean(trainAccs)))
print("Average Test Accuracy: " + str(np.mean(testAccs)))
print("Average Precision: " + str(np.mean(precisionsCollected)))
print("Average Recall: " + str(np.mean(recallsCollected)))
