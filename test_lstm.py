import pandas as pd
import numpy as np
import tensorflow_decision_forests as tfdf
import os
import helpers

featuresToKeep = ["gaze_0_x","gaze_0_y","gaze_0_z","gaze_angle_x", "gaze_angle_y",
                  "dgaze_0_x", "dgaze_0_y", "dgaze_angle_y", 
                  "AU01_r","AU04_r","AU10_r","AU12_r","AU45_r"]

#"pose_Tx", "pose_Ty", "pose_Tz", "pose_Ry"

# after the model is trained, predict the video output from the path
# use tensorflow if modelName = tf, use sklearn if modelName = sk
# the modelObj is the training model object
# keepList is the list of features used to predict
# prints the possibility of lie and truth in the video
def perdictSingleVideo(path, modelObj, numOfFrames = 10, minConfidence = 0.9):
  
  for file in os.listdir(path):
    if file.endswith('test.csv'):
        df = pd.read_csv(path + file)
        helpers.addGazeDelta(df)
        print(df)
        data = []

        index = numOfFrames
        next_index = numOfFrames
        
        bad_frame = set(np.where(df["confidence"] <= minConfidence)[0])
        helpers.filterColumn(df, colList=featuresToKeep)
        
        while index < len(df):
          if index not in bad_frame and index >= next_index:
            data.append((df.iloc[index-numOfFrames:index]).to_numpy())
          elif index in bad_frame:
            next_index = index + numOfFrames
          index += 1

  print(data)



        
            


        