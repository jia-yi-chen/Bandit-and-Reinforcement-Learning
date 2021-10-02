import numpy as np
from util_functions import featureUniform, gaussianFeature, fileOverWriteWarning
from random import sample, randint
import json

class Article():	
	def __init__(self, aid, FV=None):
		self.id = aid
		self.featureVector = FV
		

class ArticleManager():
	def __init__(self, dimension, n_articles, argv ):
		self.signature = "Article manager for simulation study"
		self.dimension = dimension
		self.n_articles = n_articles
		self.argv = argv
		self.signature = "A-"+str(self.n_articles)+"+AG"

	def saveArticles(self, Articles, filename, force = False):
		with open(filename, 'w') as f:
			for i in range(len(Articles)):
				f.write(json.dumps((Articles[i].id, Articles[i].featureVector.tolist())) + '\n')

	def loadArticles(self, filename):
		articles = []
		with open(filename, 'r') as f:
			for line in f:
				aid, featureVector = json.loads(line)
				articles.append(Article(aid, np.array(featureVector)))
		return articles

	def simulateArticlePool(self, actionset="random"):
		articles = []

		if actionset == "random":
			# for key in range(self.n_articles):
			# 	featureVector = self.FeatureFunc(self.dimension, argv=self.argv)
			# 	l2_norm = np.linalg.norm(featureVector, ord=2)
			# 	articles.append(Article(key, featureVector/l2_norm))

			feature_matrix = np.empty([self.n_articles, self.dimension])
			for i in range(self.dimension):
				feature_matrix[:, i] = np.random.normal(0, np.sqrt(1.0*(self.dimension-i)/self.dimension), self.n_articles)

			for key in range(self.n_articles):
				featureVector = feature_matrix[key]
				l2_norm = np.linalg.norm(featureVector, ord =2)
				articles.append(Article(key, featureVector/l2_norm ))

		elif actionset == "basis_vector":
			# This will generate a set of basis vectors to simulate MAB env
			assert self.n_articles == self.dimension
			feature_matrix = np.identity(self.n_articles)
			for key in range(self.n_articles):
				featureVector = feature_matrix[key]
				articles.append(Article(key, featureVector))

		return articles

