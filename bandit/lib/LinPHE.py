"""
Created on Oct 13, 2020
Author: Jiayi Chen
"""
import numpy as np
import sys

class LinPHE_Struct:
    def __init__(self, featureDimension, lambda_, noiseScale):
        self.d = featureDimension
        self.a = 2  # integer tunable scale a > 0
        self.lambda_ = lambda_  # regularization
        self.NoiseScale = noiseScale  # variance of pseudo rewards

        self.Gt = self.lambda_ * (self.a + 1) * np.identity(n=self.d) # G_0
        self.UserTheta_t = np.zeros(self.d)
        self.history = {
            'X':[],
            'Y':[]
        }
        self.time = 0

    def updateParameters(self, x_At, Yt):
        self.history['X'].append(x_At)
        self.history['Y'].append(Yt)
        self.time += 1
        if self.time > self.d:
            # update G_t
            self.Gt = self.Gt + (self.a + 1) * np.outer(x_At, x_At)

            # generate pseudo rewards, and calculate X_l[Y_l + sum_j{Zjl}]
            tmp = np.zeros(self.time)
            PseudoRewards = np.random.normal(0, self.NoiseScale, int(self.time * self.a))
            for l in range(self.time):
                PseudoRewards_l = PseudoRewards[ l*self.a : (l+1)*self.a ]
                sum_PseudoRewards_l = np.sum(PseudoRewards_l)  # sum_j Z_lj
                # calculate X_l [Y_l + sum Z]
                tmp[l] = self.history['X'][l] * (self.history['Y'][l] + sum_PseudoRewards_l)

            # update theta_t
            self.UserTheta_t = np.dot( np.linalg.inv(self.Gt), np.sum(tmp) )
        else:
            # update Gt
            self.Gt = self.Gt + (self.a + 1) * np.outer(x_At, x_At)


    def getTheta(self):
        return self.UserTheta_t

    def getA(self):
        return self.A

    def decide(self, pool_articles):
        if self.time > self.d:
            max = float('-inf')
            articlePicked = None
            for article in pool_articles:
                estimation = np.dot(self.UserTheta_t, article.featureVector)
                if max < estimation:
                    articlePicked = article
                    max = estimation
            return articlePicked
        else:
            return pool_articles[self.d - self.time]  # select arm (K-t+1)


class LinPHE:
    def __init__(self, dimension, lambda_, noise):
        self.users = {}
        self.dimension = dimension
        self.lambda_ = lambda_
        self.noiseScale =noise
        self.CanEstimateUserPreference = True

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = LinPHE_Struct(self.dimension, self.lambda_, self.noiseScale)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.featureVector[:self.dimension], click)

    def getTheta(self, userID):
        return self.users[userID].UserTheta_t


