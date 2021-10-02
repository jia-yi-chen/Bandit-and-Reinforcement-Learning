from MDP import build_mazeMDP, print_policy
import numpy as np
import matplotlib.pyplot as plt

class ReinforcementLearning:
	def __init__(self, mdp, sampleReward):
		"""
		Constructor for the RL class

		:param mdp: Markov decision process (T, R, discount)
		:param sampleReward: Function to sample rewards (e.g., bernoulli, Gaussian). This function takes one argument:
		the mean of the distribution and returns a sample from the distribution.
		"""

		self.mdp = mdp
		self.sampleReward = sampleReward

	def sampleRewardAndNextState(self,state,action):
		'''Procedure to sample a reward and the next state
		reward ~ Pr(r)
		nextState ~ Pr(s'|s,a)

		Inputs:
		state -- current state
		action -- action to be executed

		Outputs:
		reward -- sampled reward
		nextState -- sampled next state
		'''

		reward = self.sampleReward(self.mdp.R[action,state])
		cumProb = np.cumsum(self.mdp.T[action,state,:])
		nextState = np.where(cumProb >= np.random.rand(1))[0][0]
		return reward,nextState

	def OffPolicyTD(self, nEpisodes, epsilon=0.0):
		'''
		Off-policy TD (Q-learning) algorithm

		Inputs:
		nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
		epsilon -- probability with which an action is chosen at random

		Outputs:
		Q -- final Q function (|A|x|S| array)
		policy -- final policy
		'''
		alpha = 0.1
		Q = np.zeros([self.mdp.nActions, self.mdp.nStates])
		cum_rewards = np.zeros(nEpisodes)
		for episode in range(nEpisodes):
			state = np.random.randint(self.mdp.nStates-1) # initial state
			step = 0
			done = False
			while not done:

				# Choose action At from St using e-greedy policy dirived from Q
				if np.random.uniform(0, 1) < epsilon:
					action = np.random.randint(self.mdp.nActions)  # Explore
				else:
					action = np.argmax(Q[:,state])  # Exploit

				# Take action At, Observe Rt+1, St+1
				reward, next_state = self.sampleRewardAndNextState(state, action)

				# Update Q[St, At] <- Q[St, At] + alpha[Rt+1 + gamma * max_a Q[St+1, a] - Q[St, At]]
				td_target = reward + self.mdp.discount * np.max(Q[:, next_state])
				td_delta = td_target - Q[action, state]
				Q[action, state] += alpha * td_delta

				# St <- St+1
				state = next_state
				step += 1
				cum_rewards[episode] += reward # Calculate cumulative reward for plotting

				# done if reach terminal state
				if state == 16:
					done = True

		policy = np.zeros(self.mdp.nStates, int)
		for s in range(self.mdp.nStates):
			policy[s] = np.argmax(Q[:, s])

		return [Q, policy, cum_rewards]

	def OffPolicyMC(self, nEpisodes, epsilon=0.0):
		'''
		Off-policy MC algorithm with epsilon-soft behavior policy

		Inputs:
		nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
		epsilon -- probability with which an action is chosen at random

		Outputs:
		Q -- final Q function (|A|x|S| array)
		policy -- final policy
		'''

		def epsilon_soft(epsilon): # Choose At from St using e-greedy policy dirived from Q
			def func(Q, state):
				if np.random.uniform(0, 1) < epsilon:
					action = np.random.randint(self.mdp.nActions)  # Explore
				else:
					action = np.argmax(Q[:, state])  # Exploit
				if np.argmax(Q[:,state]) != action:  b_AtSt = epsilon / self.mdp.nActions
				else:   b_AtSt = 1 - epsilon + epsilon/self.mdp.nActions
				return action, b_AtSt
			return func

		Q = np.zeros([self.mdp.nActions,self.mdp.nStates])
		C = np.zeros([self.mdp.nActions, self.mdp.nStates])
		target_policy   = lambda q: np.argmax(q)
		behavior_policy = epsilon_soft(epsilon)
		cum_rewards = np.zeros(nEpisodes)

		for episode in range(nEpisodes):

			#### Generate an episode using behavior policy ###
			trajectory = []
			state = np.random.randint(self.mdp.nStates-1)
			for t in range(100):
				action, b_AtSt = behavior_policy(Q, state) # using the epsilon_soft policy wrt Q
				reward, next_state = self.sampleRewardAndNextState(state, action)
				trajectory.append((state, action, b_AtSt, reward))
				state = next_state
				if state == 16:
					break

			#### For each step in the episode, backward ###
			G = 0
			W = 1
			for t in range(len(trajectory))[::-1]:
				state, action, b_AtSt, reward = trajectory[t]

				# Update the total reward from step t to the end
				G = self.mdp.discount * G + reward

				# Update weighted importance sampling formula denominator
				C[action, state] += W

				# Update the action-value function
				Q[action, state] += ( W / C[action, state]) * (G - Q[action, state])

				# If At is not the action taken by target policy, we can break
				if action != np.argmax(Q[:, state]):
					break

				# action probability b(At|St) was already stored in the trajectory
				W = W * 1. / b_AtSt

			#### Cumulative reward #####
			for t in range(len(trajectory)):
				_, _, _, reward = trajectory[t]
				cum_rewards[episode] += reward

		policy = np.zeros(self.mdp.nStates, int)
		for s in range(self.mdp.nStates):
			policy[s] = np.argmax(Q[:, s])

		return [Q,policy,cum_rewards]



if __name__ == '__main__':
	mdp = build_mazeMDP()
	rl = ReinforcementLearning(mdp, np.random.normal)
	f, axa = plt.subplots(2, 1)

	# Test Q-learning
	run_times = 200
	num_episodes = 1500
	x_axis = range(num_episodes)
	avg_rewards = {}
	for epsilon in [0.05, 0.1, 0.3, 0.5]:
		avg_rewards[epsilon] = np.zeros(num_episodes)
		for run in range(run_times):
			[Q, policy, y_axis] = rl.OffPolicyTD(nEpisodes=num_episodes, epsilon=epsilon)
			# print_policy(policy)
			avg_rewards[epsilon] += y_axis
		avg_rewards[epsilon] /= run_times
		axa[1].plot(x_axis[:num_episodes], avg_rewards[epsilon][:num_episodes], label="epsilon="+str(epsilon))
	axa[1].legend(loc='lower right', prop={'size': 9})
	axa[1].set_xlabel("Episodes")
	axa[1].set_ylabel("Avg Cumulative rewards")
	axa[1].set_title("Off-policy TD Control")


	# Test Off-Policy MC
	run_times = 1
	num_episodes = 200000
	x_axis = range(num_episodes)
	avg_rewards = np.zeros(num_episodes)
	for run in range(run_times):
		[Q, policy, y_axis] = rl.OffPolicyMC(nEpisodes=num_episodes, epsilon=1)
		print_policy(policy)
		avg_rewards += y_axis
	avg_rewards /= run_times
	axa[0].plot(x_axis[:num_episodes], avg_rewards[:num_episodes])
	axa[0].set_xlabel("Episodes")
	axa[0].set_ylabel("Avg Cumulative rewards")
	axa[0].set_title("Off-policy MC Control")


	plt.show()