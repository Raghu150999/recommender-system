import pandas as pd
from utilmat import UtilMat
from cf import CollabFilter
from cur import CUR
from lf import LF
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('ratings_shuffled.csv')

# Data preparation
# Splitting into 8 : 1 : 1 (train validation test split)
l = len(df)
training_split = int(l * 0.8)
l -= training_split
validation_split = int(l * 0.5) + training_split
training_data = df.iloc[:training_split, :]
validation_data = df.iloc[training_split: validation_split].reset_index(drop=True)
test_data = df.iloc[validation_split:].reset_index(drop=True)

train_utilmat = UtilMat(training_data)
val_utilmat = UtilMat(validation_data)
test_utilmat = UtilMat(test_data)

# For creating dev data set
'''
df = df.sample(frac=1).reset_index(drop=True)
df = df.iloc[:10000]
df.to_csv('ratings_dev.csv', index=False)
'''

'''
Todos:
    Try different weight functions for weighted similarity
    Add select options functionality through command line arguments
'''


# Using latent factor model for prediction
lf = LF(n=100, learning_rate=0.01, lmbda=0.05, verbose=True)

# Training the model
try:
    lf.train(train_utilmat, iters=50, val_utilmat=val_utilmat, method='stochastic')
except BaseException as e:
    print('Error: ', e)
    lf.save('tmp')
    exit()

train_loss = lf.history['train_loss']
val_loss = lf.history['val_loss']
l = len(train_loss)

# Plotting training loss
plt.plot(np.arange(0, l), train_loss, color='red')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('Training loss curve')
plt.savefig('plots/train_loss.png', format='png')
plt.clf()

# Plotting validation loss
plt.plot(np.arange(0, l), val_loss, color='blue')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('Validation loss curve')
plt.savefig('plots/val_loss.png', format='png')
plt.clf()

# Plotting together
plt.plot(np.arange(0, l), train_loss, color='red', label='Training loss')
plt.plot(np.arange(0, l), val_loss, color='blue', label='Validation loss')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('Loss curve')
plt.legend()
plt.savefig('plots/loss.png', format='png')
plt.clf()

print('Test Loss: ', lf.calc_loss(test_utilmat, get_mae=True))

# Save the model
lf.save('m1_100')

