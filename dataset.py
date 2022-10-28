import os
import glob
import random
import pandas as pd
import numpy as np

import tensorflow_decision_forests as tfdf
from sklearn.model_selection import train_test_split
import helpers

# create a single dataset from a specified path (must be all truth or all lie)
def createDatasetSingle(path, truth):
  df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(path+"*.csv")))).reset_index()
  helpers.addGazeDelta(df)
  helpers.addTFLabel(df, truth)
  df = helpers.filterConfidence(df).reset_index().drop(columns = ["index"])

  return df

# create a dual dataset, one with truth dataset and one with false dataset, 
# then shuffle them and merge them into a single dataset
# outputs total dataset, the data X, and a label Y
def createDatasetDual(truthPath, liePath):
  dfT = createDatasetSingle(truthPath, True)
  dfL = createDatasetSingle(liePath, False)
  dfTotal = helpers.veticalMerge(dfT, dfL, shuffle=True)

  X, Y = dfTotal.drop(columns = ["Result"]), dfTotal["Result"]

  return dfTotal, X, Y

# input a truthpath and a liepath, create a dual dataset and create a train
# test split based on the testRatio
# outputs total train, train with x, train with y, test with x, and test with y
def createDatasetGeneral(truthPath, liePath, testRatio):
  dfT = createDatasetSingle(truthPath, True)
  dfL = createDatasetSingle(liePath, False)

  dfTotal = helpers.veticalMerge(dfT, dfL, shuffle=True)
  Train, Test = train_test_split(dfTotal, test_size=testRatio, shuffle=False)
  Xtrain, Ytrain = Train.drop(columns = ["Result"]), Train["Result"]
  Xtest, Ytest = Test.drop(columns = ["Result"]), Test["Result"]

  return Train, Xtrain, Ytrain, Xtest, Ytest

# after the model is trained, predict the video output from the path
# use tensorflow if modelName = tf, use sklearn if modelName = sk
# the modelObj is the training model object
# keepList is the list of features used to predict
# prints the possibility of lie and truth in the video
def perdictSingleVideo(path, modelName, modelObj):

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

def preprocessing(folderPath, trueOrFalse, numOfFrames, minConfidence = 0.9):
  csv_files = glob.glob(os.path.join(folderPath, "*.csv"))
  dropped = 0
  data = []
  label = []

  # #perform normalization
  total_csv = []
  for file in csv_files:
    csv_file = pd.read_csv(file)
    csv_file = helpers.filterColumn(csv_file)

    if total_csv == []:
      total_csv = np.array(csv_file)
    else:
      # take out frames with confidence less than 0.9
      for i in range(len(csv_file)):
        if csv_file.iloc[i]["confidence"] <= minConfidence:
          total_csv = np.vstack((total_csv, np.array(csv_file.iloc[i])))
  
  max_total = np.amax(total_csv, axis = 0)

  for file in csv_files: 
    csv_file = pd.read_csv(file)
    csv_file = helpers.filterColumn(csv_file)
    for i in range(csv_file.shape[0]):
      for j in range(csv_file.shape[1]):
        if max_total[j] != 0:
          csv_file.iloc[i].iloc[j] = csv_file.iloc[i].iloc[j] / max_total[j]

    for i in range(numOfFrames, len(csv_file)):
      good_frame = True

      # if any frame has previous frames with confidence below threhold, skip it 
      for j in range(i - numOfFrames, i):
        if csv_file.iloc[j]["confidence"] <= minConfidence:
          good_frame = False
          dropped += 1
          break

      # if it is a good frame, let's process it 
      if not good_frame:
        continue
      
      # append frames and labels to data and label array
      data.append(csv_file.iloc[i - numOfFrames:i])
      label.append(1) if trueOrFalse else label.append(0)

  return data, label


def path_preprocessing(truthFolderPath, lieFolderPath, minConfidence = 0.9, numOfFrames = 10):
  truth_data, truth_label = preprocessing(truthFolderPath, True, numOfFrames, minConfidence)
  lie_data, lie_label = preprocessing(lieFolderPath, False,numOfFrames, minConfidence) 
  
  total_X = truth_data + lie_data
  total_Y = truth_label + lie_label

  random.seed(random.randint(1, 100))
  random.shuffle(total_X)
  random.shuffle(total_Y)


  return np.array(total_X), np.array(total_Y)