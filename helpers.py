# write all ofthe imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sk

featuresToKeep = ["gaze_0_x","gaze_0_y","gaze_0_z","gaze_angle_x", "gaze_angle_y",
                  "dgaze_0_x", "dgaze_0_y", "dgaze_angle_y", 
                  "AU01_r","AU04_r","AU10_r","AU12_r","AU45_r", 
                  "pose_Tx", "pose_Ty", "pose_Tz", "pose_Ry", 
                  "Result",
                  "confidence"]

def displayHeatmap(df):
    plt.figure(figsize=(16, 6))
    sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')

def displayConfusion(actual, predicted):
    sk.ConfusionMatrixDisplay(sk.confusion_matrix(actual, predicted)).plot()
    print("Accuracy is ", round(sk.accuracy_score(actual, predicted) * 100, 2), "%")

def filterColumn(df, colList = featuresToKeep):
    currdf = df
    for col in currdf.columns:
        if (str(col) not in colList):
            currdf = currdf.drop(columns = [str(col)])

    return currdf

def filterConfidence(df, colList = featuresToKeep):
    currdf = df
    currdf = filterColumn(currdf, colList)

    currdf = currdf.query("confidence >= 0.9")
    return currdf.drop(columns = ["confidence"]).dropna()

def veticalMerge(df1, df2, shuffle = False):
    df = pd.concat([df1, df2]).reset_index()
    if shuffle:
        df = df.sample(frac=1)
    return df

def addTFLabel(df, TrueOrFalse):
    if TrueOrFalse:
        df["Result"] = 1
    elif not TrueOrFalse:
        df["Result"] = 0

def shuffleDF(df):
    return df.sample(frac=1)

def addGazeDelta(currCSV):
  for j in range(10, currCSV.shape[0]):
      if currCSV.iloc[[j - 10]]["confidence"].iloc[0] >= 0.8:
        currCSV.at[j, 'dgaze_0_x'] = abs(currCSV.at[j - 10, 'gaze_0_x'] - currCSV.at[j, 'gaze_0_x'])
        currCSV.at[j, 'dgaze_0_y'] = abs(currCSV.at[j - 10, 'gaze_0_y'] - currCSV.at[j, 'gaze_0_y'])
        currCSV.at[j, 'dgaze_0_z'] = abs(currCSV.at[j - 10, 'gaze_0_z'] - currCSV.at[j, 'gaze_0_z'])
        currCSV.at[j, 'dgaze_angle_x'] = abs(currCSV.at[j - 10, 'gaze_angle_x'] - currCSV.at[j, 'gaze_angle_x'])
        currCSV.at[j, 'dgaze_angle_y'] = abs(currCSV.at[j - 10, 'gaze_angle_y'] - currCSV.at[j, 'gaze_angle_y'])

  return currCSV

