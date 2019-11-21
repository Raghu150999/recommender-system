import pandas as pd
from utilmat import UtilMat
from cf import CollabFilter
from cur import CUR
from lf import LF
import matplotlib.pyplot as plt
import numpy as np

# Splitting data into 8 : 1 : 1 (train : validation : test)
df = pd.read_csv('ratings.csv')
# Shuffling data
df = df.sample(frac=1).reset_index(drop=True)

# Data preparation
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

# Using collaborative filtering model
'''
cf = CollabFilter(train_utilmat)
rmse_b, rmse_u, rmse_i = cf.calc_loss(test_utilmat)
print('RMSE using baseline approach: ', rmse_b)
print('RMSE using user-user filtering: ', rmse_u)
print('RMSE using item-item filtering: ', rmse_i)
'''

# Using CUR decompositon
'''
df = pd.read_csv('ratings_dev.csv')
utilmat = UtilMat(df)
cur = CUR(utilmat, 100)
rmse, mae = cur.calc_error2()
print(rmse, mae)
'''

'''
Todos:
    Try different weight functions for weighted similarity
'''

# Using latent factor model for prediction
lf = LF(n=100, learning_rate=0.05, lmbda=0.1, verbose=True)

train_loss, val_loss = lf.train(train_utilmat, 100, val_utilmat=val_utilmat)
l = len(train_loss)
plt.plot(np.arange(0, l), train_loss, color='red')
plt.show()
if len(val_loss) > 0:
    plt.plot(np.arange(0, l), val_loss, color='blue')
    plt.show()
print(lf.calc_loss(test_utilmat))

# Save the model
lf.save('m1_100')

'''
Tuning number of latent factors:
N        Test (RMSE)             Val (RMSE)             iters
10       0.9218474399435459      0.896568172451457      60
25       0.9143556181123862      0.8906144750476012     60
30       0.9139402779404207      0.8891032522100686     60
50       0.9101132420959622      0.8878834753298361     50
100      0.9094701143684434      0.8869900381243345     40
500      0.9391975314056097      0.9046234350554568     800
'''

'''
Tuning learning rate:
n = 100
learning rate   Test (RMSE)         Val (RMSE)          iters
0.005          0.9095923523179322  0.8851655069169209   100 
0.01           0.9094701143684434  0.8869900381243345   40
'''

'''
Tuning lambda:
n = 100, learning_rate = 0.005
lambda      Test (RMSE)         Val (RMSE)          iters
0.01        0.9126156837325936  0.8902664516797075  100
0.1         0.9095923523179322  0.8851655069169209  100
1           0.9232470785332939  0.8956753581040031  30
'''
