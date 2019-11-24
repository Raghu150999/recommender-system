import numpy as np
from svd import computeSVD
import math
from utilmat import UtilMat
import pandas as pd

class CUR:

    def __init__(self, utilmat, r):
        '''
        Performs CUR decomposition of the given input utility matrix
        Arguments:
            utilmat: Utility matrix object <class: UtilMat>
            r: size of square matrix U in CUR decomposition
        '''
        self.r = r
        self.utilmat = utilmat
        # Probabilty of selecting columns
        pc = {}
        # Probability of selecting rows
        pr = {}
        um = utilmat.um
        ium = utilmat.ium
        # Frobenius norm
        f = 0
        for user in um:
            for movie in um[user]:
                if pr.get(user):
                    pr[user] += um[user][movie] ** 2
                else:
                    pr[user] = um[user][movie] ** 2
                if pc.get(movie):
                    pc[movie] += um[user][movie] ** 2
                else:
                    pc[movie] = um[user][movie] ** 2
                f += um[user][movie] ** 2
        sumprob = 0
        for user in pr:
            pr[user] /= f
            sumprob += pr[user]
        for movie in pc:
            pc[movie] /= f
        
        # Select r columns and r rows
        cols = []
        rows = []
        # Intersection matrix W
        w = np.zeros((r, r))
        for i in range(r):
            chance = np.random.random()
            probsum = 0
            selected_movie = -1
            for movie in pc:
                probsum += pc[movie]
                if probsum >= chance:
                    selected_movie = movie
                    break
            assert(selected_movie != -1)
            column = ium[selected_movie].copy()
            # Normalizing chosen column by dividing it with sqrt(r * prob)
            for user in column:
                column[user] /= math.sqrt(r * pc[selected_movie])
            cols.append((selected_movie, column))
            chance = np.random.random()
            probsum = 0
            selected_user = -1
            for user in pr:
                probsum += pr[user]
                if probsum >= chance:
                    selected_user = user
                    break
            assert(selected_user != -1)
            row = um[selected_user].copy()
            # Normalizing chosen row
            for movie in row:
                row[movie] /= math.sqrt(r * pr[selected_user])
            rows.append((selected_user, row))
        
        # Creating W matrix
        for i in range(r):
            for j in range(r):
                movie, column = cols[j]
                user, row = rows[i]
                val = 0
                if um[user].get(movie):
                    val = um[user][movie]
                w[i, j] = val
        
        # Perform SVD on W
        # W = X S Y.T
        X, S, Y = computeSVD(w)
        # Calculate moore-penrose inverse and square it
        # i.e. S = (S+) ^ 2 
        # S+[i] = 1 / S[i] if S[i] > 0 else 0
        s_plus_squared = np.zeros((r, r))
        for i in range(r):
            if S[i] > 0.0001:
                S[i] = 1 / S[i]
            S[i] = S[i] ** 2
            s_plus_squared[i, i] = S[i]
        # Compute U = Y (S+) ^ 2 X.T
        U = np.dot(Y, s_plus_squared).dot(X.T)
        self.C = cols
        self.R = rows
        self.U = U
    
    def calc_error2(self, energy=1.):
        r = self.r
        M = 3953
        N = 6041
        I = np.zeros((N, M))
        um = self.utilmat.um
        for user in um:
            for movie in um[user]:
                I[user, movie] = um[user][movie]
        C = np.zeros((N, r))
        R = np.zeros((r, M))
        for i, t in enumerate(self.C):
            movie, c = t
            for user in c:
                C[user, i] = c[user]
        for i, t in enumerate(self.R):
            user, r = t
            for movie in r:
                R[i, movie] = r[movie]
        Res = np.dot(C, self.U).dot(R)
        rmse = np.sum((Res - I) ** 2) / (N * M)
        mae = np.sum(np.abs(Res - I)) / (N * M)
        return rmse, mae

    def calc_error(self, energy=1.):
        # Multiply C * U
        r = self.r
        cu = np.full(r, {})
        for y, t in enumerate(self.C):
            c = t[1]
            for x in c:
                for i in range(r):
                    val = self.U[y, i] * c[x]
                    if val == 0:
                        continue
                    if cu[i].get(x):
                        cu[i][x] += val
                    else:
                        cu[i][x] = val
        # Multiply CU * R
        out = {}
        for y, c in enumerate(cu):
            for x in c:
                r = self.R[y][1]
                for k in r:
                    val = c[x] * r[k]
                    if val == 0:
                        continue
                    if out.get((x, k)):
                        out[(x, k)] += val
                    else:
                        out[(x, k)] = val
        um = self.utilmat.um
        rmse = 0
        mae = 0
        cnt = 0
        for user in um:
            for movie in um[user]:
                pred = 0
                if out.get((user, movie)):
                    pred = out[(user, movie)]
                rmse += (pred - um[user][movie]) ** 2
                mae += abs(pred - um[user][movie])
                cnt += 1
        rmse /= cnt
        rmse = math.sqrt(rmse)
        mae /= cnt
        return rmse, mae

if __name__ == "__main__":
    df = pd.read_csv('ratings.csv')
    utilmat = UtilMat(df)
    cur = CUR(utilmat, 2000)
    rmse, mae = cur.calc_error2()
    print('Reconstruction Error (RMSE, MAE): ', rmse, mae)