import os
import glob
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

import tensorflow_decision_forests as tfdf
from sklearn.model_selection import train_test_split

import helpers

truthPath = './processed_truth/'
liePath = './processed_lie/'

featuresToKeep = helpers.featuresToKeep

newFeaturesToKeep = ["gaze_0_x","gaze_0_y","gaze_0_z","gaze_angle_x", "gaze_angle_y", "AU01_r","AU04_r","AU10_r","AU12_r","AU45_r","pose_Tx","pose_Ty", "pose_Tz", "pose_Ry"]

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
      currData = trainGroups.get_group(i)
      bad_frames = np.where(currData["confidence"] < minConfidence)[0]
      print(f'Processing Person {i}, shape of data is {currData.shape}')

      index = numFrames
      next_index = numFrames
        
      while index < currData.shape[0]:
        if index in bad_frames:
            index += numFrames
        else:
          Xtrain.append(currData.iloc[index-numFrames:index])
          Ytrain.append(idx)
          index += 1
      
    print(f'Processing Test')
    testGroups = Test.groupby("Person")
    for i in testGroups.groups:
      currData = testGroups.get_group(i)
      bad_frames = np.where(currData["confidence"] < minConfidence)[0]
      print(f'Processing Person {i}, shape of data is {currData.shape}')

      for _ in tqdm(range(currData.shape[0])):
        
        good = True

        for i in range(numFrames, currData.shape[0]):
          for j in range(i - numFrames, i):
            if j in bad_frames:
              good = False

          if not good: 
            continue
          
          Xtest.append(currData.iloc[i - numFrames: i])
          Ytest.append(idx)

  return Xtrain, Ytrain, Xtest, Ytest

# after the model is trained, predict the video output from the path
# use tensorflow if modelName = tf, use sklearn if modelName = sk
# the modelObj is the training model object
# keepList is the list of features used to predict
# prints the possibility of lie and truth in the video
def perdictSingleVideo(path, modelName, modelObj, keepList=featuresToKeep):

  df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(path+"*.csv")))).reset_index()
  helpers.addGazeDelta(df)
  df = helpers.filterConfidence(df).reset_index().drop(columns = ["index"])

  counterLie, counterTrue = 0, 0

  if modelName == "tf":
    res = pd.DataFrame(modelObj.predict(tfdf.keras.pd_dataframe_to_tf_dataset(df)))

  elif modelName == "sk":
    res = modelObj.predict(df)
    temp = res.shape[0]
    res = pd.DataFrame(np.reshape(res, (temp, 1)))

  for i in range(res.shape[0]):
    if res.iloc[i][0] > 0.5:
      counterTrue = counterTrue + 1
    else:
      counterLie = counterLie + 1

  print("Lie Possibility: ", round(counterLie/res.shape[0] * 100, 2), "%")
  print("Truth Possibility: ", round(counterTrue/res.shape[0]* 100, 2), "%")

def LSTM_preprocessing(truthPath, liePath, minConfidence = 0.9, numOfFrames = 10, byPerson = False):

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
