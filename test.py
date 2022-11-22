import dataset

truthPath = './processed_truth/'
liePath = './processed_lie/'

Xtrain, Ytrain, Xtest, Ytest = dataset.createDatasetLSTM(truthPath, liePath, 0.2)

print(f'Xtrain shape is {len(Xtrain)}')
print(f'Ytrain shape is {len(Ytrain)}')
print(f'Xtest shape is {len(Xtest)}')
print(f'Ytest shape is {len(Ytest)}')