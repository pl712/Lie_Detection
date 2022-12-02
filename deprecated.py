import os
import glob
import random
import pandas as pd
import numpy as np

import helpers

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

      #df = df / maxes
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

  return data, label


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

  print(total_X.shape)
  print(total_Y.shape)

  return total_X, total_Y