import sys
import getopt
import re
import os
import math
import collections
import sets
import json
import numpy as np
import operator
from PorterStemmer import PorterStemmer
from sklearn import svm
from datetime import datetime
class SupportVectorMachine:


  def readFileAndLoadData(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here,
     * unless you don't like the way we segment files.
    """
    contents = {}
    f = open(fileName)
    lines = [s.rstrip("\n\r") for s in f.readlines()]
    f.close()


    i = 0
    for line in lines:
      tempJson = json.loads(line)
      currDocName = fileName + "_review_" + str(i)
      self.allReviews[currDocName] = tempJson
      self.numDocs += 1

      individualWords = tempJson['review'].split()
      countWithoutStopWords = 0
      countWithStop = 0
      positionOfWord = 0
      for tempWord in individualWords:
        positionToWeight = 1.5*math.cos(2*math.pi*positionOfWord/(len(individualWords))) + 2.5
        positionOfWord += 1
        wordWithoutPunc = self.stemmer.stem(re.sub(r'[^\w\s]','', tempWord.lower()))
        if wordWithoutPunc in self.stopList:
          continue
        else:
          countWithoutStopWords += 1
          self.vocab.add(wordWithoutPunc)
          if wordWithoutPunc not in self.invertedIndex:
            self.invertedIndex[wordWithoutPunc] = dict()

          if currDocName not in self.invertedIndex[wordWithoutPunc]:
            self.invertedIndex[wordWithoutPunc][currDocName] = dict()
            self.invertedIndex[wordWithoutPunc][currDocName]["count"] = 1
            self.invertedIndex[wordWithoutPunc][currDocName]["weighting"] = positionToWeight
          else:
            self.invertedIndex[wordWithoutPunc][currDocName]["count"] += 1
            self.invertedIndex[wordWithoutPunc][currDocName]["weighting"] += positionToWeight
      for tempWord in individualWords:
        wordWithoutPunc = self.stemmer.stem(re.sub(r'[^\w\s]','', tempWord.lower()))
        if wordWithoutPunc not in self.stopList:
          self.invertedIndex[wordWithoutPunc][currDocName]["tf"] = self.invertedIndex[wordWithoutPunc][currDocName]["count"] #/ (1.0*countWithoutStopWords)
          self.invertedIndex[wordWithoutPunc][currDocName]["weighting"] = self.invertedIndex[wordWithoutPunc][currDocName]["weighting"]/(1.0*self.invertedIndex[wordWithoutPunc][currDocName]["count"])
      if str(fileName) not in self.countsByGame:
        self.countsByGame[str(fileName)] = 0
      self.countsByGame[str(fileName)] += 1
      i += 1



  def getVectorizedForm(self, jsonifiedReview, docName, matrix, rowNum):

    allReviewWords = jsonifiedReview['review'].split()
    for word in allReviewWords:
      if self.stemmer.stem(re.sub(r'[^\w\s]','', word.lower())) in self.stopList or re.sub(r'[^\w\s]','', word.lower()):
        continue
      wordWithoutPunc = self.stemmer.stem(re.sub(r'[^\w\s]','', word.lower()))
      indexOfWord = self.mapIndicesOfVector[wordWithoutPunc]
      matrix[rowNum][indexOfWord] = self.getTFIDF(wordWithoutPunc, docName)

    matrix[rowNum][matrix.shape[1]-3] = jsonifiedReview["hours"]
    matrix[rowNum][matrix.shape[1]-2] = jsonifiedReview["helpful_percent"]
    matrix[rowNum][matrix.shape[1]-1] = jsonifiedReview["funny_percent"]

  def readFileNormal(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here,
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents))
    return result

  def segmentWords(self, review):
    return review.split()

  def __init__(self):
    self.stopList = set(self.readFileNormal('data/english.stop'))
    self.invertedIndex = dict()
    self.vocab = set()
    self.mapIndicesOfVector = dict()
    self.fileNames = dict()
    self.numDocs = 0
    self.countsByGame = dict()
    self.allReviews = dict()
    self.stemmer = PorterStemmer()

  def getTF(self, word, docName):
    if word not in self.invertedIndex:
      return 0
    wordToDocs = self.invertedIndex[word]
    if docName in wordToDocs:
      return (1+math.log10(wordToDocs[docName]["tf"]))
    else:
      return 0

  def getIDF(self, word):
    wordToDocs = self.invertedIndex[word]
    countSoFar = 0
    for docName in wordToDocs:
      if wordToDocs[docName]['count'] > 0:
        countSoFar += 1

    return math.log10(self.numDocs/(1.0*countSoFar))

  def getTFIDF(self, word, docName):
    return self.getTF(word, docName)*self.getIDF(word)*self.invertedIndex[word][docName]["weighting"]

  def splitTrainAndTest(self, startIndex):
      startOfTest = (startIndex+7)%10
      firstIndex = startOfTest
      secondIndex = (startOfTest+1)%10
      thirdIndex = (startOfTest+2)%10

      dataSplit = dict()
      dataSplit['test'] = []
      dataSplit['train'] = []
      dataSplit['test_docs'] = 0
      dataSplit['train_docs'] = 0
      testIndices = [firstIndex, secondIndex, thirdIndex]
      for i in range(10):
        if i in testIndices:
           dataSplit['test'].append(self.fileNames[str(i)][0])
           dataSplit['test'].append(self.fileNames[str(i)][1])
           dataSplit['test_docs'] += self.countsByGame[self.fileNames[str(i)][0]] + self.countsByGame[self.fileNames[str(i)][1]]
        else:
           dataSplit['train'].append(self.fileNames[str(i)][0])
           dataSplit['train'].append(self.fileNames[str(i)][1])
           dataSplit['train_docs'] += self.countsByGame[self.fileNames[str(i)][0]] + self.countsByGame[self.fileNames[str(i)][1]]
      return dataSplit


  def test10Folds(self):
    numFeatures = len(self.vocab) + 3
    avgTestAccuracy = 0.0
    avgTrainAccuracy = 0.0
    avgPrecision = 0.0
    avgRecall = 0.0
    for i in range(10):

      dataSplit = self.splitTrainAndTest(i)
      X_train = np.zeros((dataSplit['train_docs'], numFeatures))
      Y_train = np.zeros(dataSplit['train_docs'])
      X_test = np.zeros((dataSplit['test_docs'], numFeatures))
      Y_actual = np.zeros(dataSplit['test_docs'])
      rowNum = 0
      for fileType in dataSplit['train']:
        j = 0
        while True:
          reviewType = fileType + "_review_" + str(j)
          if reviewType not in self.allReviews:
            break
          self.getVectorizedForm(self.allReviews[reviewType], reviewType, X_train, rowNum)
          if fileType[2] == 'p':
            Y_train[rowNum] = 1
          else:
            Y_train[rowNum] = 0
          rowNum += 1
          j += 1

      clf = svm.SVC(kernel = 'linear')

      clf.fit(X_train, Y_train)
      rowNum = 0
      for fileType in dataSplit['test']:
        j = 0
        while True:
          reviewType = fileType + "_review_" + str(j)

          if reviewType not in self.allReviews:
            break
          self.getVectorizedForm(self.allReviews[reviewType], reviewType, X_test, rowNum)
          if fileType[2] == 'p':
            Y_actual[rowNum] = 1
          else:
            Y_actual[rowNum] = 0

          rowNum += 1
          j += 1

      accuratePreds = 0
      trainPreds = 0

      Y_predicted = clf.predict(X_test)
      Y_train_pred = clf.predict(X_train)

      tp = 0
      tn = 0
      fp = 0
      fn = 0

      for k in range(Y_actual.shape[0]):
        if Y_actual[k] == Y_predicted[k]:
          accuratePreds += 1
          if Y_actual[k] == 1:
              tp += 1
          else:
              tn += 1
        else:
          if Y_actual[k] == 1 and Y_predicted[k] == 0:
              fn += 1
          else:
              fp += 1
      for k in range(Y_train.shape[0]):
        if Y_train[k] == Y_train_pred[k]:
          trainPreds += 1

      currPrecision = (tp*1.0)/(tp+fp)
      currRecall = (tp*1.0)/(tp+fn)
      avgTestAccuracy += accuratePreds/(1.0*Y_actual.shape[0])
      avgTrainAccuracy += trainPreds/(1.0*Y_train.shape[0])
      avgPrecision += currPrecision
      avgRecall += currRecall

      print("Fold " + str(i+1) + " Train Accuracy: " + str(trainPreds/(1.0*Y_train.shape[0])))
      print("Fold " + str(i+1) + " Test Accuracy: " + str(accuratePreds/(1.0*Y_actual.shape[0])))
      print("Fold " + str(i+1) + " Precision: " + str(currPrecision))
      print("Fold " + str(i+1) + " Recall: " + str(currRecall))

    print("Average train accuracy: " + str(avgTrainAccuracy*10))
    print("Average test accuracy: " + str(avgTestAccuracy*10))
    print("Average precision: " + str(avgPrecision*10))
    print("Average recall: " + str(avgRecall*10))


  def main(self):

    #Just read all files at first (to get vocab)
    posTrainFileNames = os.listdir('./pos/')
    negTrainFileNames = os.listdir('./neg/')

    fileNum = 0
    for fileName in posTrainFileNames:
      self.readFileAndLoadData('./pos/%s' % (fileName))
      self.fileNames[fileName[0]] = []
      self.fileNames[fileName[0]].append('./pos/%s' % (fileName))

    for fileName in negTrainFileNames:
      self.readFileAndLoadData('./neg/%s' % (fileName))
      self.fileNames[fileName[0]].append('./neg/%s' % (fileName))

    tempL  = sorted(list(self.vocab))
    for i in xrange(len(tempL)):
      self.mapIndicesOfVector[tempL[i]] = i

    self.test10Folds()

if __name__ == '__main__':
  someSVM = SupportVectorMachine()
  someSVM.main()
