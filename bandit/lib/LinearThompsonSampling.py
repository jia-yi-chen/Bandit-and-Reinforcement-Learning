"""
Created on Oct 8, 2020
Author: Jiayi Chen
"""

import numpy as np
import sys

class LinTS_Struct:
    def __init__(self, featureDimension, prior_lambda_, noise, v2=0.1):
        self.d = featureDimension
        self.A = prior_lambda_ * np.identity(n=self.d)
        self.prior_lambda_ = prior_lambda_
        self.b = np.zeros(self.d)
        self.AInv = np.linalg.inv(self.A)
        self.reward_noise_sigma = noise
        self.v_sq = v2
        self.theta_mean = np.zeros(self.d)
        self.theta_cov = self.AInv
        self.time = 0

    def updateParameters(self, x_At, Rt):
        # update A, b
        self.A += np.outer(x_At, x_At)
        self.b += x_At * Rt
        self.AInv = np.linalg.inv(self.A)
        # update posterior
        self.theta_mean = np.dot(self.AInv, self.b)
        self.theta_cov = self.v_sq * self.AInv.copy()

        self.time += 1

    def updateParameters2(self, x_At, Rt):

        # update A, b
        self.A += np.outer(x_At, x_At) / (self.reward_noise_sigma**2)
        self.b += x_At * Rt/ (self.reward_noise_sigma**2)
        self.AInv = np.linalg.inv(self.A)

        # update posterior
        self.theta_mean = np.dot(self.AInv, self.b)
        self.theta_cov = self.AInv.copy()

        self.time += 1

    def decide(self, pool_articles):
        maxM = float('-inf')
        articlePicked = None
        estimated_theta = np.random.multivariate_normal(self.theta_mean,
                                                        self.theta_cov)
        for article in pool_articles:
            estimated_reward = np.dot(estimated_theta, article.featureVector)
            if maxM < estimated_reward:
                articlePicked = article
                maxM = estimated_reward
        return articlePicked

class LinTS:
    def __init__(self, dimension, prior_lambda_, noise):
        self.users = {}
        self.dimension = dimension
        self.prior_lambda_ = prior_lambda_
        self.reward_noise_sigma = noise
        self.CanEstimateUserPreference = True

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = LinTS_Struct(self.dimension, self.prior_lambda_, self.reward_noise_sigma)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.featureVector[:self.dimension], click)

    def getTheta(self, userID):
        return self.users[userID].theta_mean #????


