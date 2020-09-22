#   author:     KISHAN TAILOR
#   file:       main.py
#   description:
#       this file implements the starter code for Project 1.
#
#   requirements:
#       this file assumes that the 'breast-cancer-wisconsin.data' is
#       located in the same directory
#   
#   resources used for building this starter file
#   - https://bradfieldcs.com/algos/trees/representing-a-tree/
import csv
from numpy import mean 
from math import log2

class InternalNode(object):
    # An Internal Node class that has an associated feature and criteria for splitting. 
    def __init__(self, feature, criteria): # Constructor
        self.type = type
        self.feature = feature
        self.criteria = criteria
        self.left = None
        self.right = None

    def insert_left(self, child):
        if self.left is None:
            self.left = child
        else:
            child.left = self.left
            self.left = child

    def insert_right(self, child):
        if self.right is None:
            self.right = child
        else:
            child.right = self.right
            self.right = child
    
    def get_depth(self, iter):
        # Recursively return the depth of the node.
        l_depth = self.left.get_depth(iter+1)
        r_depth = self.right.get_depth(iter+1)

        # return the highest of the two
        return max([l_depth, r_depth])


class LeafNode(object):
    # A Leaf Node class that has an associated decision.
    def __init__(self, decision): # Constructor
        self.decision = decision

    def retreiveDecision(self):
        return self.decision

    def get_depth(self, iter):
        return iter

def cleanData(self,data):
    cleanData = []
    for row in data:
      if '?' in row:
        continue
      else:
        cleanRow = [int(i) for i in row]
        cleanData.append(cleanRow)
    return cleanData

class DecisionTreeBuilder:
  '''This is a Python class named DecisionTreeBuilder.'''
  def __init__(self): # Constructor
    self.tree = None # Define a ``tree'' instance variable.

  def cleanData(self,data):
    cleanData = []
    for row in data:
      if '?' in row:
        continue
      else:
        cleanRow = [int(i) for i in row]
        cleanData.append(cleanRow)
    return cleanData

  def countBenMal(self,data):
    ben = 0
    mal = 0
    for row in data:
      if row[-1] == 2:
        ben += 1
      if row[-1] == 4:
        mal += 1
    return ben, mal 

  def findMidpoints (self, data):
    		# create lists to house mal / ben
    mal = []
    ben = []
    cleanRow = []
    cleanData = []
    # Use the constructed tree here., e.g. self.tree
    
    # Clean and split the data upon mal and ben 
    for row in data:
      if '?' in row:
	      pass
      else:
        cleanRow = [int(i) for i in row]
        if cleanRow[-1] == 2:
          ben.append(cleanRow)
        elif cleanRow[-1] == 4:
          mal.append(cleanRow)
        cleanData.append(cleanRow)

    # Find means of all columns
    rowAvgMal = mean(mal, axis=0)
    rowAvgBen = mean(ben, axis=0)

    # Find midpoint on each feature
    midpoints = []
    
    for i in range (len(rowAvgMal)):
      midpoints.append(round(((rowAvgMal[i] + rowAvgBen[i]) / 2),4))

    return midpoints[1:-1]

  def findSmallestFeature(self,data,midpoints):
    
    # Iterate through each midpoint
    smallestFeature = 99999999999
    featureindex = 0
    malT = 0
    malF = 0
    benT = 0
    benF = 0

    for i in range(len(midpoints)):
      mal = []
      ben = []
     
      # Create mal and ben lists for each feature
      for row in data:
        if row[i] > midpoints[i]:
          mal.append(row)
        else:
          ben.append(row)
    
      malAmount = len(mal)
      benAmount = len(ben)
      total = malAmount + benAmount
      malTrues = sum([1 for i in mal if i[-1] == 4])
      benTrues = sum([1 for i in ben if i[-1] == 2])
      malFalses = malAmount - malTrues
      benFalses = benAmount - benTrues
      
       # Construct H of Feature for each feature
      if (malTrues != 0 and malFalses != 0):
        left = (malAmount/total) *( (-malTrues/ malAmount) * log2(malTrues/ malAmount) - malFalses/malAmount * log2(malFalses/malAmount) )
      else:
        left = 0
      if (benTrues != 0 and benFalses != 0):
        right = (benAmount/total) *( (-benTrues/benAmount) * log2(benTrues/benAmount) - benFalses/benAmount * log2(benFalses/benAmount) )
      else:
        right = 0
      
      
      H_feature = left + right

      # Record which feature was the best
      if H_feature < smallestFeature:
        smallestFeature = H_feature 
        featureindex = i
        malT = malTrues
        benT = benTrues
        malF = malFalses
        benF = benFalses
      
      #print("Smallest Feature:",smallestFeature)

    return smallestFeature, featureindex, malT, benT, malF, benF

  def recursiveConstructHelper(self, cleanData, threshold, nodedepth, depthcounter):
    # print("RECURSION CALLED")

    benAmount, malAmount = self.countBenMal(cleanData)
    if (malAmount == 0 or benAmount == 0 or len(cleanData) == 0):
      if (malAmount == 0):
        leaf = LeafNode(2)
        return leaf
      if (benAmount == 0):
        leaf = LeafNode(4)
        return leaf

    midpoints = self.findMidpoints(cleanData)
    total = malAmount + benAmount
    
    Hs = - (malAmount/total) * log2(malAmount/total) - (benAmount/total) * log2(benAmount/total)
    
    smallestFeature,featureIndex, malTrues, benTrues, malFalses, benFalses  = self.findSmallestFeature(cleanData,midpoints)
    highestInfoGain = Hs - smallestFeature
    print("INFO GAIN:",highestInfoGain)
    print("Threshold is", threshold)

    if(highestInfoGain < threshold):
      if(malAmount > benAmount):
        leaf = LeafNode(4)
      else:
        leaf = LeafNode(2)
      return leaf

    # Base Cases
    if(highestInfoGain<threshold):
      print("BASECASE")
      print("INFO GAIN:", highestInfoGain)
      if (benFalses >= malFalses):
        leaf = LeafNode(2)
      else:
        leaf = LeafNode(4)
      if (benTrues >= malTrues):
        leaf = LeafNode(2)
      else:
        leaf = LeafNode(4)
      
      return leaf
    
    if(log2(depthcounter) >= nodedepth):
      print("DEPTH COUNTER RETURN")
      return

    root = InternalNode(featureIndex,midpoints[featureIndex])

    leftData = []
    rightData = []
    for row in cleanData:
      #
      # REMEMBER THIS IF STATEMENT CHANGE SIGN IF GO WRONG
      #
      if row[featureIndex] > midpoints[featureIndex]:
        leftData.append(row)
      else:
        rightData.append(row)

    if (len(leftData) != 0):
      tmpNodeLeft = self.recursiveConstructHelper(leftData,threshold,10,depthcounter+1)
    if (len(rightData) != 0):
      tmpNodeRight = self.recursiveConstructHelper(rightData,threshold,10,depthcounter+1)

    root.insert_left(tmpNodeLeft)
    root.insert_right(tmpNodeRight)

    return root


  def construct(self, data, threshold=.1):
    # '''
    #    This function constructs your tree with a default threshold of None. 
    #    The depth of the constructed tree is returned.
    # '''
    cleanData = self.cleanData(data)

    rootNode = self.recursiveConstructHelper(cleanData,threshold,10,1)
    self.tree = rootNode

    return self.tree.get_depth(0) # Return the depth of your constructed tree.
    
    
  def classifyHelper(self,currentNode,testRow):
    if (isinstance(currentNode,LeafNode)):
      x = currentNode.retreiveDecision()
      return x
    else:
      featureIndex = currentNode.feature
      if(testRow[featureIndex + 1] > currentNode.criteria):
        return self.classifyHelper(currentNode.left,testRow)
      else:
        return self.classifyHelper(currentNode.right,testRow)

  def classify(self, data):
    '''
       This function classifies data with your tree. 
       The predictions for the given data are returned.
    '''
    cleanData = self.cleanData(data)
    ret = []
    for row in cleanData:
      ret.append(self.classifyHelper(self.tree,row))

    # Return a list of predictions.
    return ret
  
def printResults(test_data,predictions):
  TB,FB,TM,FM = (0,0,0,0)
  # Create Items for Confusion Matrix
  for idx,prediction in enumerate(predictions):
    if(int(prediction) == int(test_data[idx][-1])):
      if (int(test_data[idx][-1]) == 2):
        TB += 1  
      else:
        TM += 1
    else:
      if (int(test_data[idx][-1]) == 2):
        FB += 1  
      else:
        FM += 1
    
  print("TB:", TB)
  print("FB:", FB)
  print("TM:", TM)
  print("FM:", FM)
  acc = (TB + TM) / (TB+TM+FB+FM)
  recall = TM / (TM+FB)
  precision = TM / (TM+FM)
  F1S = 2 * precision * recall / (precision + recall)
  print("Accuracy:", acc)
  print("Recall:", recall)
  print("Precision", precision)
  print("F1 Score", F1S)
  return acc

def main(threshold):
  # 1. Read in data from file.
  print("1. Reading File")
  with open("breast-cancer-wisconsin.data") as fp:
      reader = csv.reader(fp, delimiter=",", quotechar='"')
      
      # create a list (i.e. array) where each index is a row of the CSV file.
      all_data = [row for row in reader]
  print()

  # 2. Split the data into training and test sets.
  print("2. Separating Data")
  number_of_rows = len(all_data)
  all_data = all_data
  training_data = all_data[:500]
  test_data = all_data[500:]    

  print()

  # 3. Create an instance of the DecisionTreeBuilder class.
  print("3. Instantiating DecisionTreeBuilder")
  dtb = DecisionTreeBuilder()
  print()

  # 4. Construct the Tree.
  print("4. Constructing the Tree with Training Data")
  tree_length = dtb.construct(training_data,threshold)
  print("Tree Length: " + str(tree_length))
  print()

  # 5. Classify Test Data using the Tree.
  print("5. Classifying Test Data with the Constructed Tree")
  predictions = dtb.classify(test_data)
  print()

  # 6. Perform Data Analysis
  acc = printResults(test_data,predictions)

#Params: Threshold
main(.1)





