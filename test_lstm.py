import pandas as pd
import numpy as np
import tensorflow_decision_forests as tfdf
import os
import helpers

featuresToKeep = ["gaze_0_x","gaze_0_y","gaze_0_z","gaze_angle_x", "gaze_angle_y", 
                  "AU01_r","AU04_r","AU10_r","AU12_r","AU45_r"]
#["pose_Tx","pose_Ty", "pose_Tz", "pose_Ry"]

#"pose_Tx", "pose_Ty", "pose_Tz", "pose_Ry"

# after the model is trained, predict the video output from the path
# use tensorflow if modelName = tf, use sklearn if modelName = sk
# the modelObj is the training model object
# keepList is the list of features used to predict
# prints the possibility of lie and truth in the video
def perdictSingleVideo(path, modelObj, label, numOfFrames = 10, minConfidence = 0.9):
  
  prediction = []
  for file in os.listdir(path):
    data = []

    try:
      if file.endswith('.csv'):

        if file.endswith('L.csv'):
          label = 0
        elif file.endswith('T.csv'):
          label = 1

        df = pd.read_csv(path + file)

        bad_frame = set(np.where(df["confidence"] <= minConfidence)[0])
        df = helpers.filterColumn(df, colList=featuresToKeep)
        
        index = numOfFrames
        next_index = numOfFrames
        
        while index < len(df):
          if index not in bad_frame and index >= next_index:
            data.append((df.iloc[index-numOfFrames:index]).to_numpy())
          elif index in bad_frame:
            next_index = index + numOfFrames
          index += 1

        video_prediction = modelObj.predict(np.array(data))
      
        accuracy = 0
        for frame in video_prediction:
          if frame > 0.5:
            frame = 1
          else:
            frame = 0
          
          if frame == label:
            accuracy += 1
        
        accuracy = accuracy / len(video_prediction)
        print(file, ' accuracy: ', accuracy, ' label: ', label)
        prediction.append(accuracy)
    except Exception as e:
      print('error in ', file, ' ', e)
      continue

  return prediction




        
            


        