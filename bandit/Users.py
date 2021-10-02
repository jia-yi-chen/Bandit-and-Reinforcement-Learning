import numpy as np 
from util_functions import featureUniform, gaussianFeature, fileOverWriteWarning
import json
from random import choice, randint

class User():
	def __init__(self, id, theta = None):
		self.id = id
		self.theta = theta

class UserManager():
	def __init__(self, dimension, userNum, thetaFunc, argv = None):
		self.dimension = dimension
		self.thetaFunc = thetaFunc
		self.userNum = userNum
		self.argv = argv
		self.signature = "A-"+"+PA"+"+TF-"+self.thetaFunc.__name__

	def saveUsers(self, users, filename, force = False):
		fileOverWriteWarning(filename, force)
		with open(filename, 'w') as f:
			for i in range(len(users)):
				print(users[i].theta)
				f.write(json.dumps((users[i].id, users[i].theta.tolist())) + '\n')
				
	def loadUsers(self, filename):
		users = []
		with open(filename, 'r') as f:
			for line in f:
				id, theta = json.loads(line)
				users.append(User(id, np.array(theta)))
		return users

	def simulateThetafromUsers(self):
		users = []

		for key in range(self.userNum):
			thetaVector = self.thetaFunc(self.dimension, argv=self.argv)
			l2_norm = np.linalg.norm(thetaVector, ord=2)
			users.append(User(key, thetaVector/l2_norm))

		return users

