import math
import pandas as pd
from utilmat import UtilMat

class CollabFilter:
    '''
    Uses collaborative filtering approach to predict user ratings for recommendation systems.
    '''
    
    def __init__(self, utilmat, ftype='item', k=5):
        '''
        Arguments:
            um: Utility matrix
            ftype: Type of filtering used
                Available:
                    'item': item - item filtering
                    'user': user - user filtering
                    'baseline': item - item filtering using baseline approach
            k: Number of nearest neighbours to use
        '''
        self.utilmat = utilmat
        self.k = k
        if ftype == 'item':
            self.predict = self.predict_i
        elif ftype == 'user':
            self.predict = self.predict_u
        elif ftype == 'baseline':
            self.predict = self.predict_b
        else:
            raise ValueError('Invalid collaborative filter type')
    
    def predict_i(self, user, movie):
        '''
        Uses item - item filtering approach
        Predicts the rating given by user 'user' to movie 'movie'
        Arguments:
            user: userid
            movie: movieid
        '''
        ium = self.utilmat.ium
        um = self.utilmat.um
        bi = self.utilmat.bi
        bx = self.utilmat.bx
        mu = self.utilmat.mu
        if um.get(user) == None:
            if ium.get(movie) == None:
                return mu
            else:
                return bi[movie] + mu
        elif ium.get(movie) == None:
            return mu + bx[user]
        ix = ium[movie]
        b1 = -(bi[movie] + mu)
        scores = []
        for moviey in um[user]:
            if moviey == movie:
                continue
            iy = ium[moviey]
            b2 = -(bi[moviey] + mu)
            sxy = self.sim(ix, iy, b1, b2)
            scores.append((sxy, um[user][moviey] + b2))
        scores.sort(reverse=True)
        return self.get_rating(scores, -b1)

    def predict_u(self, user, movie):
        '''
        Uses user - user filtering to find predicted rating
        '''
        ium = self.utilmat.ium
        um = self.utilmat.um
        bx = self.utilmat.bx
        bi = self.utilmat.bi
        mu = self.utilmat.mu
        if um.get(user) == None:
            if ium.get(movie) == None:
                return mu
            else:
                return bi[movie] + mu
        elif ium.get(movie) == None:
            return mu + bx[user]
        ix = um[user]
        b1 = -(bx[user] + mu)
        scores = []
        for usery in ium[movie]:
            if usery == user:
                continue
            iy = um[usery]
            b2 = -(bx[usery] + mu)
            sxy = self.sim(ix, iy, b1, b2)
            scores.append((sxy, ium[movie][usery] + b2))
        scores.sort(reverse=True)
        return self.get_rating(scores, -b1)

    def predict_b(self, user, movie):
        '''
        Uses baseline approach in collaborative filtering to find predicted rating
        '''
        ium = self.utilmat.ium
        um = self.utilmat.um
        bx = self.utilmat.bx
        bi = self.utilmat.bi
        mu = self.utilmat.mu
        if um.get(user) == None:
            if ium.get(movie) == None:
                return mu
            else:
                return bi[movie] + mu
        elif ium.get(movie) == None:
            return mu + bx[user]
        baseline = mu + bx[user] + bi[movie]
        ix = ium[movie]
        b1 = -(bi[movie] + mu)
        scores = []
        for moviey in um[user]:
            if moviey == movie:
                continue
            iy = ium[moviey]
            b2 = -(bi[moviey] + mu)
            sxy = self.sim(ix, iy, b1, b2)
            baseliney = mu + bx[user] + bi[moviey]
            scores.append((sxy, um[user][moviey] - baseliney))
        scores.sort(reverse=True)
        return self.get_rating(scores, baseline)

    def get_rating(self, scores, base):
        '''
        Computes the rating using weighted average of top k scores
        '''
        l = min(self.k, len(scores))
        rating = 0
        den = 0
        for i in range(l):
            rating += (scores[i][0]) * scores[i][1]
            den += abs(scores[i][0])
        if den == 0:
            return self.bound(base)
        rating = rating / den
        rating += base
        return self.bound(rating)

    def sim(self, vec1, vec2, b1=0, b2=0):
        '''
        Finds cosine similarity between given two vectors
        Arguments:
            vec1: first vector
            vec2: second vector
            b1: bias for first vector
            b2: bias for second vector
        '''
        sim = 0
        # Normalization constants
        n1 = 0
        n2 = 0
        for feature in vec1:
            a = vec1[feature] + b1
            if vec2.get(feature):
                b = vec2[feature] + b2
                sim += a * b
            n1 += a * a
        for feature in vec2:
            b = vec2[feature] + b2
            n2 += b * b
        if sim == 0:
            return 0
        n1 = math.sqrt(n1)
        n2 = math.sqrt(n2)
        return sim / (n1 * n2)
    
    def bound(self, rating):
        '''
        Bounds the rating in the range [1, 5]
        '''
        return min(max(rating, 1), 5)

    def calc_loss(self, utilmat):
        '''
        Computes RMSE loss for the data given in utilmat
        '''
        um = utilmat.um
        rmse_u = 0
        rmse_i = 0
        rmse_b = 0
        mae_u = 0
        mae_i = 0
        mae_b = 0
        cnt = 0
        for user in um:
            for movie in um[user]:
                actual = um[user][movie]
                predu = self.predict_u(user, movie)
                predi = self.predict_i(user, movie)
                predb = self.predict_b(user, movie)
                rmse_u += (actual - predu) ** 2
                rmse_i += (actual - predi) ** 2
                rmse_b += (actual - predb) ** 2
                mae_u += abs(actual - predu)
                mae_i += abs(actual - predi)
                mae_b += abs(actual - predb)
                cnt += 1
        rmse_b /= cnt
        rmse_u /= cnt
        rmse_i /= cnt
        mae_b /= cnt
        mae_u /= cnt
        mae_i /= cnt
        return rmse_b, rmse_u, rmse_i, mae_b, mae_u, mae_i

if __name__ == "__main__":
    df = pd.read_csv('ratings_shuffled.csv')
    df = df.iloc[:100000]
    l = len(df)
    test_data = df.iloc[:100, :].reset_index(drop=True)
    train_data = df.iloc[100:, :].reset_index(drop=True)
    train_utilmat = UtilMat(train_data)
    test_utilmat = UtilMat(test_data)

    # Using collaborative filtering model
    cf = CollabFilter(train_utilmat)
    rmse_b, rmse_u, rmse_i, mae_b, mae_u, mae_i = cf.calc_loss(test_utilmat)
    print('Using baseline approach: ', rmse_b, mae_b)
    print('Using user-user filtering: ', rmse_u, mae_u)
    print('Using item-item filtering: ', rmse_i, mae_i)