import dataset
import pandas as pd

truthPath = './processed_truth/'
liePath = './processed_lie/'

print(f'Creating dataset with {truthPath} and {liePath}')

Xtrain, Ytrain, Xtest, Ytest = dataset.createDatasetLSTM(truthPath, liePath, 0.2)

print(f'Xtrain shape is {Xtrain.shape}')
print(f'Ytrain shape is {Ytrain.shape}')
print(f'Xtest shape is {Xtest.shape}')
print(f'Ytest shape is {Ytest.shape}')