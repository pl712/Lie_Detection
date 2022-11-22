import os
import glob
import random
import pandas as pd
import numpy as np

import tensorflow_decision_forests as tfdf
from sklearn.model_selection import train_test_split

import helpers

truthPath = './processed_truth/'
liePath = './processed_lie/'

featuresToKeep = helpers.featuresToKeep

# create a single dataset from a specified path (must be all truth or all lie)
def createDatasetSingle(path, truth):
  df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(path+"*.csv")))).reset_index()
  helpers.addGazeDelta(df)
  helpers.addTFLabel(df, truth)
  df = helpers.filterConfidence(df).reset_index().drop(columns = ["index"])

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

  dfMap = {dfT:1, dfL:0}

  Xtrain, Ytrain, Xtest, Ytest = [], [], [], []

  for data in dfMap:

    if byPerson:
      Train, Test = helpers.shuffleByPerson(data, lst = personlst)
    elif not byPerson:
      Train, Test = helpers.shuffleByPerson(data, ratio = testRatio)

    trainGroups = Train.groupby("Person")
    for i in trainGroups:
      currData = trainGroups.get_group(i)

      for _ in range(currData.shape[0]):
        bad_frames = np.where(currData["Confidence"] < minConfidence)[0]

        good = True

        for i in range(numFrames, currData.shape[0]):
          for j in range(i - numFrames, i):
            if j in bad_frames:
              good = False

          if not good: 
            continue
  
          Xtrain.append(currData.iloc[i - numFrames: i])
          Ytrain.append(dfMap[data])
      
    testGroups = Test.groupby("Person")
    for i in testGroups:
      currData = testGroups.get_group(i)

      for _ in range(currData.shape[0]):
        bad_frames = np.where(currData["Confidence"] < minConfidence)[0]

        good = True

        for i in range(numFrames, currData.shape[0]):
          for j in range(i - numFrames, i):
            if j in bad_frames:
              good = False

          if not good: 
            continue
          
          Xtest.append(currData.iloc[i - numFrames: i])
          Ytest.append(dfMap[data])

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









def preprocessing(folderPath, trueOrFalse, minConfidence = 0.9, numOfFrames = 10):
  csv_files = glob.glob(os.path.join(folderPath, "*.csv"))
  dropped = 0
  data = []
  label = []

  dfs = []
  for fn in csv_files:
      df = pd.read_csv(fn)
      df = helpers.filterColumn(df)
      dfs.append(df)

  full_df = pd.concat(dfs, axis=0, ignore_index=False)
  full_df = full_df[full_df["confidence"] > minConfidence]

  maxes = full_df.max(axis=0)
  maxes = np.where(maxes > 0, maxes, 1)

  for i, df in enumerate(dfs):
      if df.shape[1] != maxes.shape[0]:
          print(f"CSV file {csv_files[i]} only has {df.shape[1]} columns")
          continue

      df = df / maxes
      bad = np.where(df["confidence"] <= minConfidence)[0]
      bad = {b: 1 for b in bad}

      for i in range(numOfFrames, df.shape[0]):
          good_frame = True
          for j in range(i - numOfFrames, i):
              if j in bad:
                  good_frame = False
                  dropped += 1

          if not good_frame:
              continue

          data.append(df.iloc[i - numOfFrames: i])
          label.append(int(trueOrFalse))


  # # #perform normalization
  # total_csv = []
  # for file in csv_files:
  #   csv_file = pd.read_csv(file)
  #   csv_file = helpers.filterColumn(csv_file)

  #   if total_csv == []:
  #     total_csv = np.array(csv_file)
  #   else:
  #     # take out frames with confidence less than 0.9
  #     for i in range(len(csv_file)):
  #       if csv_file.iloc[i]["confidence"] <= minConfidence:
  #         total_csv = np.vstack((total_csv, np.array(csv_file.iloc[i])))
  # 
  # max_total = np.amax(total_csv, axis = 0)

  # for file in csv_files: 
  #   csv_file = pd.read_csv(file)
  #   csv_file = helpers.filterColumn(csv_file)
  #   for i in range(csv_file.shape[0]):
  #     for j in range(csv_file.shape[1]):
  #       if max_total[j] != 0:
  #         csv_file.iloc[i].iloc[j] = csv_file.iloc[i].iloc[j] / max_total[j]

    # for i in range(numOfFrames, len(csv_file)):
    #   good_frame = True

    #   # if any frame has previous frames with confidence below threhold, skip it 
    #   for j in range(i - numOfFrames, i):
    #     if csv_file.iloc[j]["confidence"] <= minConfidence:
    #       good_frame = False
    #       dropped += 1
    #       break

    #   # if it is a good frame, let's process it 
    #   if not good_frame:
    #     continue
    #   
    #   # append frames and labels to data and label array
    #   data.append(csv_file.iloc[i - numOfFrames:i])
    #   label.append(1) if trueOrFalse else label.append(0)

  # return data, label


def path_preprocessing(truthFolderPath, lieFolderPath, minConfidence = 0.9, numOfFrames = 10):
  truth_data, truth_label = preprocessing(truthFolderPath, True, minConfidence, numOfFrames)
  lie_data, lie_label = preprocessing(lieFolderPath, False, minConfidence, numOfFrames) 
  
  total_X = np.array(truth_data + lie_data)
  total_Y = np.array(truth_label + lie_label)
  random.seed(random.randint(1, 100))

  # Create an array of indices, then shuffle it
  indices = np.arange(len(total_X)).astype(int)
  np.random.shuffle(indices)

  # Same order of indices for both X and Y
  total_X = total_X[indices]
  total_Y = total_Y[indices]

  # # This shuffles X and Y independently
  # random.shuffle(total_X)
  # random.shuffle(total_Y)


  return total_X, total_Y
