"""
Created on Oct 9, 2020
Author: Jiayi Chen
"""
import numpy as np
import sys

class TS_Struct:
    def __init__(self, num_arm, noise, prior_dist):
        self.d = num_arm

        self.SumReward = {aid: 0.0 for aid in range(self.d)}
        self.N = {aid: 0 for aid in range(self.d)}

        self.prior_mean  = {aid: prior_dist['mu'] for aid in range(self.d)}
        self.prior_sigma = {aid: prior_dist['sigma'] for aid in range(self.d)}

        self.reward_noise_sigma = noise

        self.posterior_mean = {aid: prior_dist['mu'] for aid in range(self.d)}
        self.posterior_sigma = {aid: prior_dist['sigma'] for aid in range(self.d)}
        self.time = 0

    def updateParameters(self, At, Rt):
        self.posterior_mean[At] = 1 / (1 / self.posterior_sigma[At] ** 2 + 1 / self.reward_noise_sigma ** 2) * \
                                  ((self.posterior_mean[At] / self.posterior_sigma[At] ** 2) +
                                   (Rt / self.reward_noise_sigma ** 2))
        self.posterior_sigma[At] = np.sqrt(
                                  1 / (1 / self.posterior_sigma[At] ** 2 + 1 / self.reward_noise_sigma ** 2))
        self.time += 1

    def posterior_update(self, At, Rt):
        # update from prior
        # self.SumReward[At] += Rt
        # self.N[At] += 1
        # self.posterior_sigma[At] = np.sqrt( 1 / (1 / self.prior_sigma[At]**2 + self.N[At] / self.reward_noise_sigma**2))
        # self.posterior_mean[At] = self.posterior_sigma[At]**2 * \
        #                            ((self.prior_mean[At] / self.prior_sigma[At]**2) +
        #                             (self.SumReward[At] / self.reward_noise_sigma**2))

        # update from last posterior
        self.posterior_mean[At] = 1 / (1 / self.posterior_sigma[At] ** 2 + 1 / self.reward_noise_sigma ** 2) * \
                                  ((self.posterior_mean[At] / self.posterior_sigma[At] ** 2) +
                                   (Rt / self.reward_noise_sigma ** 2))
        self.posterior_sigma[At] = np.sqrt(
            1 / (1 / self.posterior_sigma[At] ** 2 + 1 / self.reward_noise_sigma ** 2))

    def decide(self, pool_articles):
        maxM = float('-inf')
        articlePicked = None
        for article in pool_articles:
            estimated_mean = np.random.normal(loc=self.posterior_mean[article.id],
                                              scale=self.posterior_sigma[article.id])
            if maxM < estimated_mean:
                articlePicked = article
                maxM = estimated_mean
        return articlePicked

class TS:
    def __init__(self, num_arm, noise, prior_dist):
        self.users = {}
        self.num_arm = num_arm
        self.reward_noise_sigma = noise
        self.prior = prior_dist
        self.CanEstimateUserPreference = True

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = TS_Struct(self.num_arm, self.reward_noise_sigma, self.prior)
        return self.users[userID].decide(pool_articles)

    def updateParameters(self, At, Rt, userID):
        self.users[userID].updateParameters(At.id, Rt)

    def getTheta(self, userID):
        tmp=np.zeros(self.num_arm)
        for a in range(self.num_arm):
            tmp[a]=self.users[userID].posterior_mean[a]
        return tmp


