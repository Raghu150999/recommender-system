import numpy as np
import pandas as pd
import math

class SVD:
    '''
    SVD (singular value decomposition) is used for prediction of user - movie ratings
    '''

    def __init__(self, df):
        '''
        Arguments:
            energy: 
                Amount of energy (in percentage) to be retained after decomposition and reduction.
                Higher energy requirement will lead to retaining more singular values and vice-versa
            df: training data
        '''
        # Creating the utility matrix
        MAX_USER_ID = 6040
        MAX_MOVIE_ID = 3952
        self.M = np.zeros((MAX_USER_ID + 1, MAX_MOVIE_ID + 1))
        for i in range(len(df)):
            user = df.loc[i, 'userid']
            movie = df.loc[i, 'movieid']
            rating = df.loc[i, 'rating']
            self.M[user][movie] = rating
        U, S, V = computeSVD(self.M)
        self.U = U
        self.S = S
        self.V = V
    
    def calc_error(self, energy=0.999):
        '''
        Calculates reconstruction error (RMSE and MAE)
        '''
        S = self.S
        l = len(S)
        sos = 0
        for s in S:
            sos += s ** 2
        i = 0
        retained = 0
        while (retained / sos < energy) and i < l:
            retained += S[i] ** 2
            i += 1
        print('k: ', i)
        U = self.U[:, :i]
        V = self.V[:, :i]
        S = S[:i]
        k = S.shape[0]
        sigma = np.zeros((k, k))
        for i in range(k):
            sigma[i, i] = S[i]
        R = np.dot(U, sigma).dot(V.T)
        N = R.shape[0] * R.shape[1]
        rmse = math.sqrt(np.sum((self.M - R) ** 2) / N)
        mae = np.sum(np.abs(self.M - R)) / N
        return rmse, mae

def computeSVD(M):
    U, S, V = np.linalg.svd(M, full_matrices=False)
    return U, S, V.T

def computeSVD2(M):
    '''
    Perform singular value decomposition (SVD)
    '''
    # m -> number of users
    # n -> number of movies
    # n X m * m X n
    S = np.dot(M.T, M)
    w, V = np.linalg.eig(S)
    # Extract real part of eigen values and eigen vectors
    w = w.real
    V = V.real
    # Perform argsort and reverse (for descending)
    idx = w.argsort()[::-1]
    w = w[idx]
    l = len(w)
    V = V[:, idx]
    S = np.dot(M, M.T)
    t, U = np.linalg.eig(S)
    t = t.real
    l = min(l, len(t))
    U = U.real
    idx = t.argsort()[::-1]
    t = t[idx]
    U = U[:, idx]
    U = U[:, :l]
    V = V[:, :l]
    w = w[:l]
    return U, np.sqrt(w), V

if __name__ == "__main__":
    df = pd.read_csv('ratings.csv')
    svd = SVD(df)
    rmse, mae = svd.calc_error(1.)
    print('RMSE: ', rmse)
    print('MAE: ', mae)

