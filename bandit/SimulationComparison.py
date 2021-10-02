"""
Created on Oct 13, 2020

Multi-arm Bandit algorithms: e-greedy, CUB, TS, PHE
Linear Bandit algorithms: e-greedy-linear, LinearCUB, LinearTS

Author: Jiayi Chen
"""

import copy
import numpy as np
from random import sample, shuffle
import datetime
import os.path
import matplotlib.pyplot as plt
import argparse
# local address to save simulated users, simulated articles, and results
from conf import sim_files_folder, save_address
from util_functions import featureUniform, gaussianFeature
from Articles import ArticleManager
from Users import UserManager

from lib.EpsilonGreedyLinearBandit import EpsilonGreedyLinearBandit
from lib.LinearUpperConfidenceBound import LinUCB
from lib.LinearThompsonSampling import LinTS
from lib.EpsilonGreedyMultiArmedBandit import EpsilonGreedyMultiArmedBandit
from lib.UpperConfidenceBound import UCB
from lib.ThompsonSampling import TS


class simulateOnlineData(object):
    def __init__(self, context_dimension, testing_iterations, plot, articles,
                 users, noise=lambda: 0, signature='', NoiseScale=0.0, poolArticleSize=None):

        self.simulation_signature = signature

        self.context_dimension = context_dimension
        self.testing_iterations = testing_iterations
        self.batchSize = 1500

        self.plot = plot

        self.noise = noise

        self.NoiseScale = NoiseScale  # lambda,  random sample a value from N(0,1) with scale 0.1

        self.articles = articles
        self.users = users

        if poolArticleSize is None:
            self.poolArticleSize = len(self.articles)
        else:
            self.poolArticleSize = poolArticleSize

    def getTheta(self):
        Theta = np.zeros(shape=(self.context_dimension, len(self.users)))
        for i in range(len(self.users)):
            Theta.T[i] = self.users[i].theta
        return Theta

    def batchRecord(self, iter_):
        print("Iteration %d" % iter_, " Elapsed time", datetime.datetime.now() - self.startTime)

    def getReward(self, user, pickedArticle):
        return np.dot(user.theta, pickedArticle.featureVector)

    def GetOptimalReward(self, user, articlePool):

        maxReward = float('-inf')
        maxx = None
        for x in articlePool:
            reward = self.getReward(user, x)  # r = x * real_theta
            if reward > maxReward:
                maxReward = reward
                maxx = x
        return maxReward, maxx

    def getL2Diff(self, x, y):
        return np.linalg.norm(x - y)  # L2 norm

    def regulateArticlePool(self):
        # Randomly generate articles
        self.articlePool = sample(self.articles, self.poolArticleSize)

    def runAlgorithms(self, algorithms):
        self.startTime = datetime.datetime.now()
        timeRun = self.startTime.strftime('_%m_%d_%H_%M')
        filenameWriteRegret = os.path.join(save_address, 'AccRegret' + timeRun + '.csv')
        filenameWritePara = os.path.join(save_address, 'ParameterEstimation' + timeRun + '.csv')

        tim_ = []
        BatchCumlateRegret = {}
        AlgRegret = {}
        ThetaDiffList = {}
        ThetaDiff = {}

        # Initialization
        userSize = len(self.users)
        for alg_name, alg in algorithms.items():
            AlgRegret[alg_name] = []
            BatchCumlateRegret[alg_name] = []
            if alg.CanEstimateUserPreference:
                ThetaDiffList[alg_name] = []

        with open(filenameWriteRegret, 'w') as f:
            f.write('Time(Iteration)')
            f.write(',' + ','.join([str(alg_name) for alg_name in algorithms.keys()]))
            f.write('\n')

        with open(filenameWritePara, 'w') as f:
            f.write('Time(Iteration)')
            f.write(',' + ','.join([str(alg_name) + 'Theta' for alg_name in ThetaDiffList.keys()]))
            f.write('\n')

        for iter_ in range(self.testing_iterations):
            # prepare to record theta estimation error
            for alg_name, alg in algorithms.items():
                if alg.CanEstimateUserPreference:
                    ThetaDiff[alg_name] = 0

            for u in self.users:
                # print("=====\nUser=", u.id)
                self.regulateArticlePool()

                OptimalReward, OptimalArticle = self.GetOptimalReward(u, self.articlePool)
                noise = self.noise()  # nt ~ N(0,1) with scale 0.1
                OptimalReward += noise

                for alg_name, alg in algorithms.items():

                    # select At by algorithm
                    pickedArticle = alg.decide(self.articlePool, u.id)

                    # apply At to get real reward Rt ~ N( x_At * theta0_u , 1 )
                    reward = self.getReward(u, pickedArticle) + noise

                    alg.updateParameters(pickedArticle, reward, u.id)

                    # if alg_name=='LinearUpperConfidenceBound':
                    #     print("Rt=", reward, "At=", pickedArticle.id)
                    #     print("Real theta =", u.theta)
                    #     print("Estimated theta at t", alg.users[u.id].UserTheta_t)
                    # if alg_name=='LinearThompsonSampling':
                    #     print("User=", u.id, "At=", pickedArticle.id)
                    #     print("Real theta =", u.theta)
                    #     print("Estimated theta at t", alg.users[u.id].theta_mean)
                    #     print("Estimated theta at t", np.linalg.norm(alg.users[u.id].theta_cov))


                    # pseudo regret, since noise is canceled out
                    regret = OptimalReward - reward
                    AlgRegret[alg_name].append(regret)

                    # update parameter estimation record
                    if alg.CanEstimateUserPreference:
                        ThetaDiff[alg_name] += self.getL2Diff(u.theta, alg.getTheta(u.id))

            for alg_name, alg in algorithms.items():
                if alg.CanEstimateUserPreference:
                    ThetaDiffList[alg_name] += [ThetaDiff[alg_name] / userSize]

            if iter_ % self.batchSize == 0:
                self.batchRecord(iter_)
                tim_.append(iter_)
                for alg_name in algorithms.keys():
                    BatchCumlateRegret[alg_name].append(sum(AlgRegret[alg_name]) / userSize)

                with open(filenameWriteRegret, 'a+') as f:
                    f.write(str(iter_))
                    f.write(',' + ','.join([str(BatchCumlateRegret[alg_name][-1]) for alg_name in algorithms.keys()]))
                    f.write('\n')
                with open(filenameWritePara, 'a+') as f:
                    f.write(str(iter_))
                    f.write(',' + ','.join([str(ThetaDiffList[alg_name][-1]) for alg_name in ThetaDiffList.keys()]))
                    f.write('\n')
        finalRegret = {}
        for alg_name in algorithms.keys():
            finalRegret[alg_name] = BatchCumlateRegret[alg_name][:-1]
        return finalRegret, tim_, BatchCumlateRegret, ThetaDiffList



        # return finalRegret


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--contextdim', type=int, help='Set dimension of context features.')
    parser.add_argument('--subsetsize', type=int, help='Set dimension of context features.')
    parser.add_argument('--actionset', type=str, help='Set dimension of context features.')
    parser.add_argument('--noise', type=float, help='Set noise.')
    parser.add_argument('--K', type=int, help='Set noise.')
    args = parser.parse_args()


    K=20
    if args.K:
        K=args.K

    ## Environment Settings ##
    context_dimension = K  # feature vector / theta dimension

    if args.actionset:
        actionset = args.actionset
    else:
        actionset = "basis_vector"  # "basis_vector" or "random"

    actionset = "basis_vector"

    testing_iterations =100000  # 200000
    NoiseScale = 0.1  # standard deviation of Gaussian noise
    if args.noise:
        NoiseScale = args.noise
    n_articles = K  # fullset size
    n_users = 15  # 10
    poolArticleSize = None

    print("n_articles=", n_articles)
    print("poolArticleSize=", poolArticleSize)
    print("context_dimension=", context_dimension)

    if actionset == "basis_vector":
        n_articles = context_dimension  # there can be at most context_dimension number of basis vectors

    ## Set Up Simulation ##
    UM = UserManager(context_dimension, n_users, thetaFunc=gaussianFeature, argv={'l2_limit': 1})
    users = UM.simulateThetafromUsers()
    AM = ArticleManager(context_dimension, n_articles=n_articles, argv={'l2_limit': 1})
    articles = AM.simulateArticlePool(actionset)
    print(articles[1].featureVector)

    simExperiment = simulateOnlineData(context_dimension=context_dimension,
                                       testing_iterations=testing_iterations,
                                       plot=True,
                                       articles=articles,  # feature vectors
                                       users=users,
                                       noise=lambda: np.random.normal(scale=NoiseScale),
                                       signature=AM.signature,
                                       NoiseScale=NoiseScale,
                                       poolArticleSize=poolArticleSize)

    ## Initiate Bandit Algorithms ##
    algorithms = {}

    algorithms['basic_EpsilonGreedyLinearBandit'] = EpsilonGreedyLinearBandit(dimension=context_dimension, lambda_=0.1, epsilon=None)
    algorithms['basic_LinearUpperConfidenceBound'] = LinUCB(dimension=context_dimension, lambda_=2, noise=NoiseScale)
    algorithms['basic_LinearThompsonSampling'] = LinTS(dimension=context_dimension, prior_lambda_=1, noise=NoiseScale)
    # algorithms['EpsilonGreedyMultiArmedBandit'] = EpsilonGreedyMultiArmedBandit(num_arm=n_articles, epsilon=None)
    # algorithms['UpperConfidenceBound'] = UCB(num_arm=n_articles)
    # algorithms['ThompsonSampling'] = TS(num_arm=n_articles, noise=NoiseScale, prior_dist={'mu': 0.0, 'sigma': 1.0})

    ## Run Simulation ##
    print("Starting for ", simExperiment.simulation_signature)
    Y1, X1, Y2, Y3 = simExperiment.runAlgorithms(algorithms)



    # Linear performance on multi-armed setting
    actionset = "random"


    ## Set Up Simulation ##
    # UM = UserManager(context_dimension, n_users, thetaFunc=gaussianFeature, argv={'l2_limit': 1})
    # users = UM.simulateThetafromUsers()
    AM = ArticleManager(context_dimension, n_articles=n_articles, argv={'l2_limit': 1})
    articles = AM.simulateArticlePool(actionset)
    print(articles[1].featureVector)

    simExperiment2 = simulateOnlineData(context_dimension=context_dimension,
                                       testing_iterations=testing_iterations,
                                       plot=True,
                                       articles=articles,  # feature vectors
                                       users=users,
                                       noise=lambda: np.random.normal(scale=NoiseScale),
                                       signature=AM.signature,
                                       NoiseScale=NoiseScale,
                                       poolArticleSize=poolArticleSize)

    ## Initiate Bandit Algorithms ##
    algorithms2 = {}

    algorithms2['random_EpsilonGreedyLinearBandit'] = EpsilonGreedyLinearBandit(dimension=context_dimension, lambda_=0.1, epsilon=None)
    algorithms2['random_LinearUpperConfidenceBound'] = LinUCB(dimension=context_dimension, lambda_=0.1, noise=NoiseScale)
    algorithms2['random_LinearThompsonSampling'] = LinTS(dimension=context_dimension, prior_lambda_=1, noise=NoiseScale)
    # algorithms['EpsilonGreedyMultiArmedBandit'] = EpsilonGreedyMultiArmedBandit(num_arm=n_articles, epsilon=None)
    # algorithms['UpperConfidenceBound'] = UCB(num_arm=n_articles)
    # algorithms['ThompsonSampling'] = TS(num_arm=n_articles, noise=NoiseScale, prior_dist={'mu': 0.0, 'sigma': 1.0})

    ## Run Simulation ##
    print("Starting for ", simExperiment2.simulation_signature)
    Y21, X2, Y22, Y23 = simExperiment2.runAlgorithms(algorithms2)


    # plot the results
    f, axa = plt.subplots(2, 1)
    C=['b','r','g','c']
    for i, alg_name in enumerate(algorithms.keys()):
        axa[0].plot(X1, Y2[alg_name], C[i]+'--', label=alg_name)
        print('%s: %.2f' % (alg_name, Y2[alg_name][-1]))
    for i, alg_name in enumerate(algorithms2.keys()):
        axa[0].plot(X2, Y22[alg_name], C[i], label=alg_name)
        axa[0].color = C[i]
        print('%s: %.2f' % (alg_name, Y22[alg_name][-1]))
    axa[0].legend(loc='upper left', prop={'size': 9})
    axa[0].set_xlabel("Iteration")
    axa[0].set_ylabel("Regret")
    axa[0].set_title("Accumulated Regret")
    # plt.show()

    # plot the estimation error of theta
    # f, axa = plt.subplots(2)
    time = range(testing_iterations)
    for i, (alg_name, alg) in enumerate(algorithms.items()):
        if alg.CanEstimateUserPreference:
            axa[1].plot(time, Y3[alg_name], C[i]+'--', label=alg_name + '_Theta')
            print("estimation error of", alg_name, "is", Y3[alg_name][-1])
    for i, (alg_name, alg) in enumerate(algorithms2.items()):
        if alg.CanEstimateUserPreference:
            axa[1].plot(time, Y23[alg_name], C[i], label=alg_name + '_Theta')
            print("estimation error of", alg_name, "is", Y23[alg_name][-1])
    axa[1].legend(loc='upper right', prop={'size': 6})
    axa[1].set_xlabel("Iteration")
    axa[1].set_ylabel("L2 Diff")
    axa[1].set_yscale('log')
    axa[1].set_title("Parameter estimation error")
    plt.show()