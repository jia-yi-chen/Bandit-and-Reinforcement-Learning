"""
Created on Oct 9, 2020
Author: Jiayi Chen
"""
import numpy as np
import sys

class UCB_Struct:
    def __init__(self, num_arm):
        self.d = num_arm
        self.UserArmMean = {aid: 0.0 for aid in range(self.d)}
        self.N = {aid: 0 for aid in range(self.d)}
        self.delta = 0.01
        self.UCB = {aid: sys.float_info.max for aid in range(self.d)}
        self.time = 0

    def updateParameters(self, At, Rt):
        self.UserArmMean[At] = (self.UserArmMean[At]*self.N[At] + Rt) / (self.N[At]+1)
        self.N[At] += 1
        self.time += 1
        if self.N[At] != 0:
            uncertainty = np.sqrt(2 * np.log(1 / self.delta) / self.N[At])  # Bt(a)
            self.UCB[At] = self.UserArmMean[At] + uncertainty  # Qt(a)+Bt(a)
        else:
            self.UCB[At] = sys.float_info.max

    def getTheta(self):
        return self.UserArmMean

    def decide(self, pool_articles):
        maxBD = float('-inf')
        articlePicked = None
        for article in pool_articles:
            if maxBD < self.UCB[article.id]:
                articlePicked = article
                maxBD = self.UCB[article.id]
        return articlePicked

class UCB:
    def __init__(self, num_arm):
        self.users = {}
        self.num_arm = num_arm
        self.CanEstimateUserPreference = True

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = UCB_Struct(self.num_arm)
        return self.users[userID].decide(pool_articles)

    def updateParameters(self, At, Rt, userID):
        self.users[userID].updateParameters(At.id, Rt)

    def getTheta(self, userID):
        tmp = np.zeros(self.num_arm)
        for a in range(self.num_arm):
            tmp[a] = self.users[userID].UserArmMean[a]
        return tmp
        # return self.users[userID].UserArmMean


