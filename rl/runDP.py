from MDP import build_mazeMDP, print_policy, print_state_value
import numpy as np

class DynamicProgramming:
	def __init__(self, MDP):
		self.R = MDP.R
		self.T = MDP.T
		self.discount = MDP.discount
		self.nStates = MDP.nStates
		self.nActions = MDP.nActions







	def valueIteration(self, initialV, nIterations=np.inf, tolerance=0.01):
		'''Value iteration procedure
		V <-- max_a R^a + gamma T^a V

		Inputs:
		initialV -- Initial value function: array of |S| entries
		nIterations -- limit on the # of iterations: scalar (default: infinity)
		tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

		Outputs:
		policy -- Policy: array of |S| entries
		V -- Value function: array of |S| entries
		iterId -- # of iterations performed: scalar
		epsilon -- ||V^n-V^n+1||_inf: scalar'''

		V = np.zeros(self.nStates)
		iterId = 0
		epsilon = tolerance + 1e-9

		while iterId < nIterations and epsilon > tolerance:
			iterId += 1
			epsilon = 0
			for s in range(self.nStates):
				old_value_s = V[s]
				action_values_s = self.Q_helper(V, s) # evaluate Q(s,a) for all a
				V[s] = np.max(action_values_s)
				if epsilon < np.abs(V[s] - old_value_s):
					epsilon = np.abs(V[s] - old_value_s)

		policy = np.zeros(self.nStates, dtype=int)
		for s in range(self.nStates):
			action_values_s = self.Q_helper(V, s)
			policy[s] = np.argmax(action_values_s)

		return [policy, V, iterId, epsilon]

	def Q_helper(self, V, s):
		'''
		Input:  s -- state
		Output: action_values -- Q(s, a) for all actions a
		'''
		action_values = np.zeros(self.nActions)
		for a in range(self.nActions):
			for next_state in range(self.nStates):
				action_values[a] += self.T[a, s, next_state] * V[next_state]
			action_values[a] = self.discount * action_values[a] + self.R[a, s]
		return action_values










	def policyIteration_v1(self, initialPolicy, nIterations=np.inf):
		'''Policy iteration procedure: alternate between policy
		evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
		improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

		Inputs:
		initialPolicy -- Initial policy: array of |S| entries
		nIterations -- limit on # of iterations: scalar (default: inf)
		# tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

		Outputs:
		policy -- Policy: array of |S| entries
		V -- Value function: array of |S| entries
		iterId -- # of iterations peformed by modified policy iteration: scalar'''

		policy = np.zeros(dp.nStates, dtype=int)
		iterId = 0

		while iterId < nIterations:
			iterId += 1

			# policy evaluation
			V = self.evaluatePolicy_SolvingSystemOfLinearEqs(policy)

			# policy improvement
			policy_prev = np.copy(policy)
			policy = self.extractPolicy(V)

			# until pi is stable
			if (np.all(np.equal(policy, policy_prev))):
				break

		return [policy, V, iterId]

	def evaluatePolicy_SolvingSystemOfLinearEqs(self, policy):
		'''Evaluate a policy by solving a system of linear equations
		V^pi = R^pi + gamma T^pi V^pi

		Input:
		policy -- Policy: array of |S| entries

		Ouput:
		V -- Value function: array of |S| entries'''

		# Calculate (I-A), b
		I = np.identity(self.nStates)
		A = np.zeros((self.nStates, self.nStates))
		b = np.zeros(self.nStates)
		for s in range(self.nStates):
			a = policy[s]
			b[s] = self.R[a,s]
			A[s, :] = self.T[a, s, :]
		A = I - self.discount * A

		# Solving (I-A)x = b
		V = np.linalg.solve(A, b)

		return V

	def extractPolicy(self, V):
		'''Procedure to extract a policy from a value function
		pi <-- argmax_a R^a + gamma T^a V

		Inputs:
		V -- Value function: array of |S| entries

		Output:
		policy -- Policy: array of |S| entries'''

		policy = np.zeros(self.nStates, dtype=int)
		for s in range(self.nStates):
			action_values_s = self.Q_helper(V, s)
			policy[s] = np.argmax(action_values_s)

		return policy

	print("before converge, policy evaluation requires", " steps")


	def policyIteration_v2(self, initialPolicy, initialV, nPolicyEvalIterations=2,
						   nIterations=np.inf, tolerance=0.01):
		'''Modified policy iteration procedure: alternate between
		partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
		and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

		Inputs:
		initialPolicy -- Initial policy: array of |S| entries
		initialV -- Initial value function: array of |S| entries
		nPolicyEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
		nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
		tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

		Outputs:
		policy -- Policy: array of |S| entries
		V -- Value function: array of |S| entries
		iterId -- # of iterations peformed by modified policy iteration: scalar
		epsilon -- ||V^n-V^n+1||_inf: scalar'''
		policy = np.zeros(dp.nStates, dtype=int)
		V = np.zeros(dp.nStates)
		iterId = 0
		epsilon = 0

		while iterId < nIterations:
			iterId += 1

			# policy evaluation
			V, eval_iter, epsilon = self.evaluatePolicy_IterativeUpdate(policy,V,
																		nPolicyEvalIterations,
																		tolerance)
			# policy improvement
			policy_prev = np.copy(policy)
			policy = self.extractPolicy(V)

			# policy stable
			if (np.all(np.equal(policy, policy_prev))):
				break

		return [policy, V, iterId, epsilon]

	def evaluatePolicy_IterativeUpdate(self, policy, initialV, nIterations, tolerance):
		'''Partial policy evaluation:
		Repeat V^pi <-- R^pi + gamma T^pi V^pi

		Inputs:
		policy -- Policy: array of |S| entries
		initialV -- Initial value function: array of |S| entries
		nIterations -- limit on the # of iterations: scalar (default: infinity)

		Outputs:
		V -- Value function: array of |S| entries
		iterId -- # of iterations performed: scalar
		epsilon -- ||V^n-V^n+1||_inf: scalar'''
		V = initialV
		iterId = 0
		epsilon = tolerance + 1e-9
		while iterId < nIterations and epsilon > tolerance:
			iterId += 1
			epsilon = 0
			for s in range(self.nStates):
				old_value_s = V[s]

				# V(s) = Q(s, pi(s))
				a = policy[s]
				new_value_s = 0
				for next_state in range(self.nStates):
					new_value_s += self.T[a, s, next_state] * V[next_state]
				V[s] = self.R[a, s] + self.discount * new_value_s

				if epsilon < np.abs(V[s] - old_value_s):
					epsilon = np.abs(V[s] - old_value_s)

		return V, iterId, epsilon







if __name__ == '__main__':
	mdp = build_mazeMDP()
	dp = DynamicProgramming(mdp)
	# Test value iteration
	print("##### DP1: value iteration #####\n")
	[policy, V, nIterations, epsilon] = dp.valueIteration(initialV=np.zeros(dp.nStates), tolerance=0.01)
	print_policy(policy)
	print("\n")
	print('policy converged at iteration', nIterations)
	print('V(s) epsilon after converge', epsilon)
	print("\n")
	print_state_value(V)
	print('\n')
	# Test policy iteration v1
	print("##### DP2: policy iteration v1 #####\n")
	[policy, V, nIterations] = dp.policyIteration_v1(np.zeros(dp.nStates, dtype=int))
	print_policy(policy)
	print("\n")
	print('policy converged at iteration', nIterations)
	print("\n")
	print_state_value(V)
	print('\n')
	# Test policy iteration v2
	print("##### DP3: policy iteration v2 #####\n")
	for p in range(1, 11):
		[policy, V, nIterations, epsilon] = dp.policyIteration_v2(np.zeros(dp.nStates, dtype=int), np.zeros(dp.nStates), nPolicyEvalIterations=p, tolerance=0.01)
		print("nIteration=",p)
		print_policy(policy)
		print("\n")
		print('policy converged at iteration', nIterations)
		print('V(s) epsilon after converge', epsilon)
		print("\n")
		print_state_value(V)