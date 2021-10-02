"""
Created on Oct 13, 2020
Author: Jiayi Chen
"""
import numpy as np
import sys

class PHE_Struct:
    def __init__(self, num_arm, noiseScale):
        self.d = num_arm
        self.UserArmMean = {aid: sys.float_info.max for aid in range(self.d)}
        self.V = {aid: 0.0 for aid in range(self.d)}
        self.N = {aid: 0 for aid in range(self.d)}
        self.a = 2  # integer tunable scale a > 0
        self.NoiseScale = noiseScale  # variance of pseudo rewards

    def updateParameters(self, At, Yt):
        for i in self.N.keys():
            if self.N[i] > 0:
                s = self.N[i]
                PseudoRewards = np.random.normal(0, self.NoiseScale, int(self.a * s))
                sum_PseudoRewards = np.sum(PseudoRewards)
                self.UserArmMean[i] = (self.V[i] + sum_PseudoRewards) / ((1+self.a)*s)
            else:
                self.UserArmMean[At] = sys.float_info.max
        self.V[At] += Yt
        self.N[At] += 1

    def getTheta(self):
        return self.UserArmMean

    def decide(self, pool_articles):
        max = float('-inf')
        articlePicked = None
        for article in pool_articles:
            if max < self.UserArmMean[article.id]:
                articlePicked = article
                max = self.UserArmMean[article.id]
        return articlePicked

class PHE:
    def __init__(self, num_arm, noiseScale):
        self.users = {}
        self.num_arm = num_arm
        self.noiseScale = noiseScale
        self.CanEstimateUserPreference = True

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = PHE_Struct(self.num_arm, self.noiseScale)
        return self.users[userID].decide(pool_articles)

    def updateParameters(self, At, Rt, userID):
        self.users[userID].updateParameters(At.id, Rt)

    def getTheta(self, userID):
        tmp = np.zeros(self.num_arm)
        for a in range(self.num_arm):
            tmp[a] = self.users[userID].UserArmMean[a]
        return tmp
        # return self.users[userID].UserArmMean


