from PorterStemmer import PorterStemmer
import sys
import getopt
import re
import os
import math
import collections
import sets
import json
import operator

tempStemmer = PorterStemmer()
class NaiveBayes:

  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test.
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []

  def __init__(self):
    """NaiveBayes initialization"""
    self.FILTER_STOP_WORDS = False
    self.BOOLEAN_NB = False
    self.BEST_MODEL = False
    self.stopList = set(self.readFileNormal('data/english.stop'))   
    self.numFolds = 10
    self.total = 0
    self.vocab = set([])
    self.punctuation = ["!","?",".",";","..."]
    self.amplifiers = ["like", "very", "super", "absolutely", "really"]
    self.negationWords = ["n't","not","no","never", "didn't", "doesn't", "don't", "hasn't", "haven't", "didnt", "doesnt", "dont", "hasnt", "havent"]
    self.vocabAdded = False
    self.countsOfClasses = collections.defaultdict(lambda:0)
    self.priorProbs = collections.defaultdict(lambda: 0)
    self.wordsInClass = collections.defaultdict(lambda: 0)
    self.conditionalProbs = collections.defaultdict(lambda: 0)
    self.wordCounter = collections.defaultdict(lambda: 0)
  #############################################################################
  # TODO TODO TODO TODO TODO
  # Implement the Multinomial Naive Bayes classifier and the Naive Bayes Classifier with
  # Boolean (Binarized) features.
  # If the BOOLEAN_NB flag is true, your methods must implement Boolean (Binarized)
  # Naive Bayes (that relies on feature presence/absence) instead of the usual algorithm
  # that relies on feature counts.
  #
  # If the BEST_MODEL flag is true, include your new features and/or heuristics that
  # you believe would be best performing on train and test sets.
  #
  # If any one of the FILTER_STOP_WORDS, BOOLEAN_NB and BEST_MODEL flags is on, the
  # other two are meant to be off. That said, if you want to include stopword removal
  # or binarization in your best model, write the code accordingl

  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    alpha = 1
    if self.FILTER_STOP_WORDS:
      words =  self.filterStopWords(words)
    elif self.BOOLEAN_NB:
      words = set(words)
    elif self.BEST_MODEL:
      encounteredNegation = False
      for i in range(0, len(words)):
          if words[i] in self.punctuation:
              encounteredNegation = False
          if encounteredNegation:
              words[i] = "NOT_"+words[i]
              continue
          if words[i] in self.negationWords:
              encounteredNegation = True
      words = list(set(words))
      originalSize = len(words)
      for i in range(0, originalSize):
          if words[i] in self.amplifiers:
              words.append(words[i])
    if self.vocabAdded == False:
        for wordGiven, className in self.wordCounter:
            self.vocab.add(wordGiven)
        self.vocabAdded = True
    self.total += 1
    self.priorProbs['pos'] = self.countsOfClasses['pos']*1.0/(self.countsOfClasses['pos'] + self.countsOfClasses['neg'])
    self.priorProbs['neg'] = self.countsOfClasses['neg']*1.0/(self.countsOfClasses['pos'] + self.countsOfClasses['neg'])
    for word in words:
        self.conditionalProbs[(word, 'pos')] = 1.0*(self.wordCounter[(word, 'pos')] + alpha)/(self.countsOfClasses['pos'] + len(self.vocab))
        self.conditionalProbs[(word, 'neg')] = 1.0*(self.wordCounter[(word, 'neg')] + 1)/(self.countsOfClasses['neg'] + len(self.vocab))
    posProb = math.log(self.priorProbs['pos'])
    negProb = math.log(self.priorProbs['neg'])
    for word in words:
        if word in self.vocab:
            posProb += math.log(self.conditionalProbs[(word, 'pos')])
            negProb += math.log(self.conditionalProbs[(word, 'neg')])
    if posProb >= negProb:
        return 'pos'
    else:
        return 'neg'


  def addExample(self, klass, words):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier
     * in the NaiveBayes class.
     * Returns nothing
    """

    if self.FILTER_STOP_WORDS:
        words = self.filterStopWords(words)
    elif self.BOOLEAN_NB:
        words = set(words)
    elif self.BEST_MODEL:
        encounteredNegation = False
        for i in range(0, len(words)):
            if words[i] in self.punctuation:
                encounteredNegation = False
            if encounteredNegation:
                words[i] = "NOT_"+words[i]
                continue
            if words[i] in self.negationWords:
                encounteredNegation = True
        words = set(words)
    for word in words:
        if word in self.amplifiers and self.BEST_MODEL:
            self.wordCounter[(word,klass)] += 2
        else:
            self.wordCounter[(word,klass)] += 1
        self.countsOfClasses[klass] += 1



  # END TODO (Modify code beyond here with caution)
  #############################################################################


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


  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here,
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    lines = [s.rstrip("\n\r") for s in f.readlines()]
    f.close()


    for line in lines:
      tempJson = json.loads(line)
      contents.append(self.segmentWords(tempJson['review']))
    return contents

  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    words = s.split()
    for i in range(len(words)):
      words[i] = tempStemmer.stem(re.sub(r'[^\w\s]','', words[i].lower()))
    return words




  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      allExamplesInFile = self.readFile('%s/pos/%s' % (trainDir, fileName))
      for exampleInFile in allExamplesInFile: 
        example = self.Example()
        example.words = exampleInFile
        example.klass = 'pos'
        split.train.append(example)

    for fileName in negTrainFileNames:
      allExamplesInFile = self.readFile('%s/neg/%s' % (trainDir, fileName))
      for exampleInFile in allExamplesInFile: 
        example = self.Example()
        example.words = exampleInFile
        example.klass = 'neg'
        split.train.append(example)

    return split

  def train(self, split):
    for example in split.train:
      words = example.words
      if self.FILTER_STOP_WORDS:
        words =  self.filterStopWords(words)
      self.addExample(example.klass, words)


  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
    splits = []
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        allExamplesInFile = self.readFile('%s/pos/%s' % (trainDir, fileName))
        for exampleInFile in allExamplesInFile: 
          example = self.Example()
          example.words = exampleInFile
          example.klass = 'pos'
          if fileName[0] in self.determineFold(fold):
            split.test.append(example)
          else:
            split.train.append(example)


      for fileName in negTrainFileNames:
        allExamplesInFile = self.readFile('%s/neg/%s' % (trainDir, fileName))
        for exampleInFile in allExamplesInFile: 
          example = self.Example()
          example.words = exampleInFile
          example.klass = 'neg'
          if fileName[0] in self.determineFold(fold):
            split.test.append(example)
          else:
            split.train.append(example)
      yield split

  def test(self, split):
    """Returns a list of labels for split.test."""
    labels = []
    for example in split.test:
      words = example.words
      if self.FILTER_STOP_WORDS:
        words =  self.filterStopWords(words)
      guess = self.classify(words)
      labels.append(guess)
    return labels

  def determineFold(self, startIndex):
   startOfTest = (startIndex+7)%10
   return (str(startOfTest)+":"+str((startOfTest+1)%10)+":"+str((startOfTest+2)%10))

  def buildSplits(self, args):
    """Builds the splits for training/testing"""
    trainData = []
    testData = []
    splits = []
    trainDir = args[0]
    if len(args) == 1:
      print '[INFO]\tPerforming %d-fold cross-validation on data set:\t%s' % (self.numFolds, trainDir)

      posTrainFileNames = os.listdir('%s/pos/' % trainDir)
      negTrainFileNames = os.listdir('%s/neg/' % trainDir)
      for fold in range(0, self.numFolds):
        split = self.TrainSplit()
        for fileName in posTrainFileNames:
          allExamplesInFile = self.readFile('%s/pos/%s' % (trainDir, fileName))
          for exampleInFile in allExamplesInFile: 
            example = self.Example()
            example.words = exampleInFile
            example.klass = 'pos'
            if fileName[0] in self.determineFold(fold):
              split.test.append(example) 
            else:
              split.train.append(example)

        for fileName in negTrainFileNames:
          allExamplesInFile = self.readFile('%s/neg/%s' % (trainDir, fileName))
          for exampleInFile in allExamplesInFile:
            example = self.Example()
            example.words = exampleInFile
            example.klass = 'neg'
            if fileName[0] in self.determineFold(fold):
              split.test.append(example)
            else:
              split.train.append(example)

        splits.append(split)
    elif len(args) == 2:
      split = self.TrainSplit()
      testDir = args[1]
      print '[INFO]\tTraining on data set:\t%s testing on data set:\t%s' % (trainDir, testDir)
      posTrainFileNames = os.listdir('%s/pos/' % trainDir)
      negTrainFileNames = os.listdir('%s/neg/' % trainDir)
      for fileName in posTrainFileNames:
        allExamplesInFile = self.readFile('%s/pos/%s' % (trainDir, fileName))
        for exampleInFile in allExamplesInFile:
          example = self.Example()
          example.words = exampleInFile
          example.klass = 'pos'
          split.train.append(example)

      for fileName in negTrainFileNames:
        allExamplesInFile = self.readFile('%s/neg/%s' % (trainDir, fileName))
        for exampleInFile in allExamplesInFile:
          example = self.Example()
          example.words = exampleInFile
          example.klass = 'neg'
          split.train.append(example)

      posTestFileNames = os.listdir('%s/pos/' % testDir)
      negTestFileNames = os.listdir('%s/neg/' % testDir)


      for fileName in posTestFileNames:
        allExamplesInFile = self.readFile('%s/pos/%s' % (testDir, fileName))
        for exampleInFile in allExamplesInFile:
          example = self.Example()
          example.words =  exampleInFile
          example.klass = 'pos'
          split.test.append(example)

      for fileName in negTestFileNames:
        allExamplesInFile = self.readFile('%s/neg/%s' % (testDir, fileName))
        for exampleInFile in allExamplesInFile:
          example = self.Example()
          example.words = exampleInFile
          example.klass = 'neg'
          split.test.append(example)

      splits.append(split)
    return splits

  def filterStopWords(self, words):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in self.stopList and word.strip() != '':
        filtered.append(word)
    return filtered

def test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL):
  nb = NaiveBayes()
  splits = nb.buildSplits(args)
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = NaiveBayes()
    classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
    classifier.BOOLEAN_NB = BOOLEAN_NB
    classifier.BEST_MODEL = BEST_MODEL
    accuracy = 0.0
    for example in split.train:
      words = example.words
      classifier.addExample(example.klass, words)

    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy)
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print '[INFO]\tAccuracy: %f' % avgAccuracy


def classifyFile(FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL, trainDir, testFilePath):
  classifier = NaiveBayes()
  classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
  classifier.BOOLEAN_NB = BOOLEAN_NB
  classifier.BEST_MODEL = BEST_MODEL
  trainSplit = classifier.trainSplit(trainDir)
  classifier.train(trainSplit)
  testFile = classifier.readFile(testFilePath)
  print classifier.classify(testFile)

def main():
  FILTER_STOP_WORDS = False
  BOOLEAN_NB = False
  BEST_MODEL = False
  (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
  if ('-f','') in options:
    FILTER_STOP_WORDS = True
  elif ('-b','') in options:
    BOOLEAN_NB = True
  elif ('-m','') in options:
    BEST_MODEL = True

  if len(args) == 2 and os.path.isfile(args[1]):
    classifyFile(FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL, args[0], args[1])
  else:
    test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL)
 
if __name__ == "__main__":
    main()
