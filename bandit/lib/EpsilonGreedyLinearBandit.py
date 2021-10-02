"""
Created on Oct 5, 2020
Author: Jiayi Chen
"""

import numpy as np

class EpsilonGreedyStruct:
    def __init__(self, featureDimension, lambda_, epsilon):
        self.d = featureDimension
        self.A = lambda_ * np.identity(n=self.d)
        self.lambda_ = lambda_ # regularization
        self.epsilon = epsilon
        self.b = np.zeros(self.d)
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta_t = np.zeros(self.d)
        self.time = 0

    def updateParameters(self, x_At, Rt):
        self.A += np.outer(x_At, x_At)
        self.b += x_At * Rt
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta_t = np.dot(self.AInv, self.b)
        self.time += 1

    def getTheta(self):
        return self.UserTheta_t

    def getA(self):
        return self.A

    def decide(self, pool_articles):
        if self.epsilon is None:
            explore = np.random.binomial(1, (self.time+1)**(-1.0/3))
        else:
            explore = np.random.binomial(1, self.epsilon)
        if explore == 1:
            # print("EpsilonGreedy: explore")
            articlePicked = np.random.choice(pool_articles)
        else:
            # print("EpsilonGreedy: greedy")
            maxPTA = float('-inf')
            articlePicked = None

            for article in pool_articles:
                # estimated_reward = xa * estimated_theta_t
                article_pta = np.dot(self.UserTheta_t, article.featureVector)
                # pick article with highest Prob
                if maxPTA < article_pta:
                    articlePicked = article
                    maxPTA = article_pta

        return articlePicked

class EpsilonGreedyLinearBandit:
    def __init__(self, dimension, lambda_, epsilon):
        self.users = {}
        self.dimension = dimension
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.CanEstimateUserPreference = True

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = EpsilonGreedyStruct(self.dimension, self.lambda_, self.epsilon)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.featureVector[:self.dimension], click)

    def getTheta(self, userID):
        return self.users[userID].UserTheta_t


