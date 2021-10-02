"""
Linear Bandit algorithms: e-greedy, LinearCUB, LinearTS
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

        if (self.plot == True):  # only plot
            # plot the results
            f, axa = plt.subplots(2,1)
            for alg_name in algorithms.keys():
                axa[0].plot(tim_, BatchCumlateRegret[alg_name], label=alg_name)
                print('%s: %.2f' % (alg_name, BatchCumlateRegret[alg_name][-1]))
            axa[0].legend(loc='upper left', prop={'size': 9})
            axa[0].set_xlabel("Iteration")
            axa[0].set_ylabel("Regret")
            axa[0].set_title("Accumulated Regret")
            # plt.show()

            # plot the estimation error of theta
            # f, axa = plt.subplots(2)
            time = range(self.testing_iterations)
            for alg_name, alg in algorithms.items():
                if alg.CanEstimateUserPreference:
                    axa[1].plot(time, ThetaDiffList[alg_name], label=alg_name + '_Theta')
                    print("estimation error of", alg_name, "is",ThetaDiffList[alg_name][-1])

            axa[1].legend(loc='upper right', prop={'size': 6})
            axa[1].set_xlabel("Iteration")
            axa[1].set_ylabel("L2 Diff")
            axa[1].set_yscale('log')
            axa[1].set_title("Parameter estimation error")
            plt.show()

        finalRegret = {}
        for alg_name in algorithms.keys():
            finalRegret[alg_name] = BatchCumlateRegret[alg_name][:-1]
        return finalRegret


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--contextdim', type=int, help='Set dimension of context features.')
    parser.add_argument('--subsetsize', type=int, help='Set dimension of context features.')
    parser.add_argument('--actionset', type=str, help='Set dimension of context features.')
    parser.add_argument('--noise', type=float, help='Set noise.')
    args = parser.parse_args()

    ## Environment Settings ##
    if args.contextdim:
        context_dimension = args.contextdim
    else:
        context_dimension = 15  # feature vector / theta dimension

    if args.actionset:
        actionset = args.actionset
    else:
        actionset = "basis_vector"  # "basis_vector" or "random"

    actionset = "random"

    testing_iterations = 140000  # 200000
    NoiseScale = 0.1  # standard deviation of Gaussian noise
    if args.noise:
        NoiseScale = args.noise
    n_articles = 25  # fullset size
    n_users = 10  # 10
    poolArticleSize = None
    if args.subsetsize:
        poolArticleSize = args.subsetsize

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

    algorithms['EpsilonGreedyLinearBandit'] = EpsilonGreedyLinearBandit(dimension=context_dimension, lambda_=0.1, epsilon=None)
    algorithms['LinearUpperConfidenceBound'] = LinUCB(dimension=context_dimension, lambda_=0.1, noise=NoiseScale)
    algorithms['LinearThompsonSampling'] = LinTS(dimension=context_dimension, prior_lambda_=1, noise=NoiseScale)

    ## Run Simulation ##
    print("Starting for ", simExperiment.simulation_signature)
    simExperiment.runAlgorithms(algorithms)