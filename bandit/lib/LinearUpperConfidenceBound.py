"""
Created on Oct 13, 2020
Author: Jiayi Chen
"""
import numpy as np
import sys

class LinUCB_Struct:
    def __init__(self, featureDimension, lambda_, noiseScale):
        self.d = featureDimension
        self.A = lambda_ * np.identity(n=self.d)
        self.lambda_ = lambda_ # regularization
        self.b = np.zeros(self.d)
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta_t = np.zeros(self.d)
        self.time = 0
        self.delta = 0.01
        self.R = noiseScale
        self.alpha_t = self.R * np.sqrt(self.d * np.log(1/self.delta))\
                       + np.sqrt(self.lambda_)

    def updateParameters(self, x_At, Rt):
        self.A += np.outer(x_At, x_At)
        self.b += x_At * Rt
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta_t = np.dot(self.AInv, self.b)
        self.time += 1
        self.alpha_t = self.R * np.sqrt(
                       self.d * np.log((1+self.time/(self.d *self.lambda_))/self.delta))\
                       + np.sqrt(self.lambda_)

    def getTheta(self):
        return self.UserTheta_t

    def getA(self):
        return self.A

    def decide(self, pool_articles):
        maxBD = float('-inf')
        articlePicked = None
        for article in pool_articles:
            uncertainty = self.alpha_t * np.sqrt(article.featureVector.T
                                               .dot(self.AInv)
                                               .dot(article.featureVector))
            upper_bound = np.dot(self.UserTheta_t, article.featureVector) + uncertainty
            if maxBD < upper_bound:
                articlePicked = article
                maxBD = upper_bound
        return articlePicked

class LinUCB:
    def __init__(self, dimension, lambda_, noise):
        self.users = {}
        self.dimension = dimension
        self.lambda_ = lambda_
        self.noiseScale =noise
        self.CanEstimateUserPreference = True

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = LinUCB_Struct(self.dimension, self.lambda_, self.noiseScale)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.featureVector[:self.dimension], click)

    def getTheta(self, userID):
        return self.users[userID].UserTheta_t


