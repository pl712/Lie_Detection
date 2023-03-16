import os
import glob
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import helpers

truthPath = './processed_truth/'
liePath = './processed_lie/'

featuresToKeep = helpers.featuresToKeep

newFeaturesToKeep = ["gaze_0_x","gaze_0_y","gaze_0_z","gaze_angle_x", "gaze_angle_y", "AU01_r","AU04_r","AU10_r","AU12_r","AU45_r"]
#["pose_Tx","pose_Ty", "pose_Tz", "pose_Ry"]

# create a single dataset from a specified patha (must be all truth or all lie)
def createDatasetSingle(path, truth):
  df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(path+"*.csv")))).reset_index()
  helpers.addGazeDelta(df)
  helpers.addTFLabel(df, truth)
  df = helpers.filterColumn(df)

  return df

# input a truthpath and a liepath, create a dual dataset and create a train
# test split based on the testRatio
# outputs total train, train with x, train with y, test with x, and test with y
def createDatasetRF(truthPath, liePath, testRatio, byPerson = False, personlst = []):
  dfT = createDatasetSingle(truthPath, True)
  dfL = createDatasetSingle(liePath, False)
  
  dfTotal = helpers.veticalMerge(dfT, dfL, shuffle=True)
  
  if byPerson:
    Train, Test = helpers.shuffleByPerson(dfTotal, testRatio, personlst)
  else:
    Train, Test = train_test_split(dfTotal, test_size=testRatio, shuffle=False)

  Xtrain, Ytrain = Train.reset_index().drop(columns = ["Result", "Person", "index", "level_0"]), Train["Result"]
  Xtest, Ytest = Test.reset_index().drop(columns = ["Result", "Person", "index", "level_0"]), Test["Result"]
  Train = Train.reset_index().drop(columns = ["index", "Person", "level_0"])

  return Train, Xtrain, Ytrain, Xtest, Ytest

def createDatasetLSTM(truthPath, liePath, testRatio, numFrames=10, minConfidence=0.9, byPerson=False, personlst = []):
  dfT = createDatasetSingle(truthPath, True)
  dfL = createDatasetSingle(liePath, False)

  dfMap = {1:dfT, 0:dfL}

  Xtrain, Ytrain, Xtest, Ytest = [], [], [], []

  idxTotext = {0:"Lie", 1:"Truth"}

  for idx in dfMap:
    print(f'Processing {idxTotext[idx]}')
    
    if byPerson:
      Train, Test = helpers.shuffleByPerson(dfMap[idx], lst = personlst)
    elif not byPerson:
      Train, Test = helpers.shuffleByPerson(dfMap[idx], ratio = testRatio)
    
    print(f'Processing Train')
    trainGroups = Train.groupby("Person")

    for i in trainGroups.groups:
      currData = trainGroups.get_group(i).sort_index()
      bad_frames = np.where(currData["confidence"] < minConfidence)[0]
      print(f'Processing Person {i}, shape of data is {currData.shape}')
      
      blocksLst = helpers.getLSTMBlocks(bad_frames.tolist(), currData.shape[0], blockSize=numFrames, start=0)

      for i, j in tqdm(blocksLst):
        Xtrain.append(currData.iloc[i:j].reset_index().drop(columns = ["index", "confidence", "Result", "Person"]).to_numpy())
        Ytrain.append(idx)
      
    print(f'Processing Test')
    testGroups = Test.groupby("Person")
    for i in testGroups.groups:
      currData = testGroups.get_group(i).sort_index()
      bad_frames = np.where(currData["confidence"] < minConfidence)[0]
      print(f'Processing Person {i}, shape of data is {currData.shape}')
      
      blocksLst = helpers.getLSTMBlocks(bad_frames.tolist(), currData.shape[0], blockSize=numFrames, start=0)

      for i, j in tqdm(blocksLst):
        Xtest.append(currData.iloc[i:j].reset_index().drop(columns = ["index", "confidence", "Result", "Person"]).to_numpy())
        Ytest.append(idx)

  Xtrain = np.array(Xtrain)
  Ytrain = np.array(Ytrain)
  Xtest = np.array(Xtest)
  Ytest = np.array(Ytest)

  random.seed(random.randint(1, 100))

  # Create an array of indices, then shuffle it
  indices = np.arange(len(Xtrain)).astype(int)
  np.random.shuffle(indices)

  # Same order of indices for both X and Y
  Xtrain  = Xtrain[indices]
  Ytrain = Ytrain[indices]

  random.seed(random.randint(1, 100))

  # Create an array of indices, then shuffle it
  indices = np.arange(len(Xtest)).astype(int)
  np.random.shuffle(indices)

  # Same order of indices for both X and Y
  Xtest  = Xtest[indices]
  Ytest = Ytest[indices]

  return Xtrain, Ytrain, Xtest, Ytest

def preprocessing(truthPath, liePath, additionalPath=None, minConfidence = 0.9, numOfFrames = 10, byPerson = False):

  data = []
  label = []

  if not byPerson:

    for file in sorted(os.listdir(truthPath)):
      if file.endswith(".csv"):
        df = pd.read_csv(truthPath + file)
        
        bad_frame = set(np.where(df["confidence"] < minConfidence)[0])
        df = helpers.filterColumn(df, colList=newFeaturesToKeep)

        index = numOfFrames
        next_index = numOfFrames
        
        while index < len(df):
          if index not in bad_frame and index >= next_index:
            data.append((df.iloc[index-numOfFrames:index]).to_numpy())
            label.append(1)
          elif index in bad_frame:
            next_index = index + numOfFrames
          index += 1

    for file in sorted(os.listdir(liePath)):
      if file.endswith(".csv"):
        df = pd.read_csv(liePath + file)
        
        bad_frame = set(np.where(df["confidence"] < minConfidence)[0])
        df= helpers.filterColumn(df, colList=newFeaturesToKeep)

        index = numOfFrames
        next_index = numOfFrames
        
        while index < len(df):
          if index not in bad_frame and index >= next_index:
            data.append((df.iloc[index-numOfFrames:index]).to_numpy())
            label.append(0)
          elif index in bad_frame:
            next_index = index + numOfFrames
          index += 1

    if additionalPath:
      for file in sorted(os.listdir(additionalPath)):
        if file.endswith(".csv"):
          df = pd.read_csv(additionalPath + file)
          
          bad_frame = set(np.where(df["confidence"] < minConfidence)[0])
          df= helpers.filterColumn(df, colList=newFeaturesToKeep)

          index = numOfFrames
          next_index = numOfFrames
          
          while index < len(df):
            if index not in bad_frame and index >= next_index:
              data.append((df.iloc[index-numOfFrames:index]).to_numpy())
              if file.endswith("T.csv"):
                label.append(1)
              elif file.endswith("L.csv"):
                label.append(0)
            elif index in bad_frame:
              next_index = index + numOfFrames
            index += 1

    data = np.array(data)
    label = np.array(label)
    random.seed(random.randint(1, 100))

    # Create an array of indices, then shuffle it
    indices = np.arange(len(data)).astype(int)
    np.random.shuffle(indices)

    # Same order of indices for both X and Y
    data  = data[indices]
    label = label[indices]

  return data, label
