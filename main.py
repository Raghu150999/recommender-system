import pandas as pd
from utilmat import UtilMat
from cf import CollabFilter

df = pd.read_csv('ratings_dev.csv')

# For creating dev data set
'''
df = df.sample(frac=1).reset_index(drop=True)
df = df.iloc[:10000]
df.to_csv('ratings_dev.csv', index=False)
'''
utilmat = UtilMat(df)
cf = CollabFilter(utilmat)
cf.predict_u(3511, 3702)

'''
Todos:
    Try different weight functions for weighted similarity
'''