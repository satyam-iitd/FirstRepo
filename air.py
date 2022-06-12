import copy
import random
import numpy as np
import matplotlib.pyplot as plt


MAP = [
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
]

class MDP:

	def __init__(self, dest_loc, gamma):

		self.dest_loc = dest_loc
		self.map = MAP
		self.num_rows = 5
		self.max_row = self.num_rows-1
		self.num_columns = 5
		self.max_column = self.num_columns-1
		self.gamma = gamma
		self.num_actions = 6
		self.possible_actions = [0,1,2,3,4,5]
		self.P = {}
		self.terminal_state = self.encode(self.dest_loc, self.dest_loc, 0)

		for tr in range(self.num_rows):
			for tc in range(self.num_columns):
				for pr in range(self.num_rows):
					for pc in range(self.num_columns):
						if (pr,pc) != self.dest_loc:
							state = self.encode((tr,tc), (pr,pc), 0)
							self.P[state] = {action: [] for action in range(self.num_actions)}

		for tr in range(self.num_rows):
			for tc in range(self.num_columns):
				state = self.encode((tr,tc), (tr,tc), 1)
				self.P[state] = {action : [] for action in range(self.num_actions)}

		#case for non-picked states
		for state in self.P:
			if state[-1] == '0':
				(tr,tc), (pr,pc), picked = self.decode(state)

				for action in range(self.num_actions):
					#navigation
					if action < 4:
						transition = self.T((tr,tc), (pr,pc), picked, action)
						for n_state in transition:
							prob = transition[n_state]
							self.P[state][action].append((prob, n_state, -1))

					#pickup
					elif action == 4:

						if (tr,tc) == (pr,pc):
							n_state = self.encode((tr,tc), (pr,pc), 1)
							self.P[state][action].append((1.0, n_state, -1))
						else:
							n_state = self.encode((tr,tc), (pr,pc), 0)
							self.P[state][action].append((1.0, n_state, -10))

					#putdown
					elif action == 5:
						n_state = self.encode((tr,tc), (pr,pc), 0)

						if (tr,tc) == (pr,pc):
							self.P[state][action].append((1.0, n_state, -1))
						else:
							self.P[state][action].append((1.0, n_state, -10))

		#case for picked states
		for state in self.P:
			if state[-1] == '1':
				(tr,tc), (pr,pc), picked = self.decode(state)

				for action in range(self.num_actions):

					#navigation
					if action < 4:
						transition = self.T((tr,tc), (pr,pc), picked, action)
						for n_state in transition:
							prob = transition[n_state]
							self.P[state][action].append((prob, n_state, -1))

					#pickup
					elif action == 4:
						n_state = self.encode((tr,tc), (pr,pc), 1)
						self.P[state][action].append((1.0, n_state, -1))

					#putdown
					elif action == 5:
						n_state = self.encode((tr,tc), (pr,pc), 0)

						if (tr,tc) == self.dest_loc:
							self.P[state][action].append((1.0, n_state, +20))
						else:
							self.P[state][action].append((1.0, n_state, -1))
						

	def encode(self, taxi_loc, pass_loc, picked):
		return str(taxi_loc[0]) + str(taxi_loc[1]) + str(pass_loc[0]) + str(pass_loc[1]) + str(picked)

	def decode(self, cipher):
		tr = int(cipher[0])
		tc = int(cipher[1])
		pr = int(cipher[2])
		pc = int(cipher[3])
		picked = int(cipher[4])
		return ((tr,tc), (pr,pc), picked)

	def get_next_state(self, state, action):
		taxi_loc, pass_loc, picked = self.decode(state)

		if action < 4:
			transition = self.T(taxi_loc, pass_loc, picked, action)
			next_state = np.random.choice(list(transition.keys()), p = [transition[s] for s in transition])
			return (next_state, -1)

		elif action == 4: #PickUp
			if picked == 0:
				if taxi_loc == pass_loc:
					next_state = self.encode(taxi_loc, pass_loc, 1)
					reward = -1
				else:
					next_state = self.encode(taxi_loc, pass_loc, 0)
					reward = -10
			else:
				next_state = self.encode(taxi_loc, pass_loc, 1)
				reward = -1
			return (next_state, reward)

		elif action == 5: #PutDown
			if picked == 0:
				next_state = self.encode(taxi_loc, pass_loc, 0)
				if taxi_loc == pass_loc:
					reward = -1
				else:
					reward = -10
			else:
				next_state = self.encode(taxi_loc, pass_loc, 0)
				if taxi_loc == self.dest_loc:
					reward = +20
				else:
					reward = -1
			return (next_state, reward)

	def select_action(self, state, epsilon, Q):

		#random action
		r_action = np.random.choice(range(self.num_actions))
		#optimal action
		p_action = max(self.possible_actions, key = lambda action: Q[state][action])

		action = np.random.choice([r_action, p_action], p = [epsilon, 1-epsilon])
		return action

	def evaluate_over_episodes(self, pi):
		num_episode = 200
		
		d_rewards_list = []
		for i in range(num_episode):
			state = np.random.choice(list(self.P.keys()))
			iters = 1
			d_reward = 0
			while state != self.terminal_state and iters < 501:
				# action = max(self.possible_actions, key = lambda action: Q[state][action])
				action = pi[state]
				next_state, reward = self.get_next_state(state, action)
				d_reward += (self.gamma ** (iters-1)) * reward
				state = next_state
				iters += 1
			
			d_rewards_list.append(d_reward)
		dr = sum(d_rewards_list)/num_episode
		print(dr)
		return dr

	def q_learning(self, alpha, epsilon):
		
		Q = {}
		for state in self.P:
			Q[state] = {}
			for action in self.possible_actions:
				Q[state][action] = 0

		Q[self.terminal_state] = {}
		for action in self.possible_actions:
			Q[self.terminal_state][action] = 0

		episodes = 2000
		
		discounted_rewards = []
		iterations = []
		pi = {}
		for i in range(episodes):
			print("Episode ",i)
			state = np.random.choice(list(self.P.keys()))
			iters = 1
			d_reward = 0
			while state != self.terminal_state and iters < 501:
				action = self.select_action(state, epsilon, Q)
				next_state, reward = self.get_next_state(state, action)
				d_reward += (self.gamma ** (iters-1)) * reward
				Q[state][action] = (1-alpha)*Q[state][action] + alpha*(reward + self.gamma*max(Q[next_state].values()))
				state = next_state
				iters += 1

			# if i%50 == 0:
			# 	for state in self.P:
			# 		pi[state] = max(self.P[state], key = lambda action: Q[state][action])
				
			# 	d_reward = self.evaluate_over_episodes(pi)
			discounted_rewards.append(d_reward)
			iterations.append(i)

		for state in self.P:
			pi[state] = max(self.possible_actions, key = lambda action: Q[state][action])

		return (pi,iterations, discounted_rewards)
		
	def q_learning_decay(self, alpha, epsilon):
		
		Q = {}
		for state in self.P:
			Q[state] = {}
			for action in range(self.num_actions):
				Q[state][action] = 0

		terminal_state = self.encode(self.dest_loc, self.dest_loc, 0)
		Q[terminal_state] = {}
		for action in range(self.num_actions):
			Q[terminal_state][action] = 0

		episodes = 2000
		pi = {}
		discounted_rewards = []
		iterations = []
		for i in range(episodes):
			print("Episode ",i)
			state = np.random.choice(list(self.P.keys()))
			iters = 1
			while state != terminal_state and iters < 501:
				action = self.select_action(state, epsilon/iters, Q)
				next_state, reward = self.get_next_state(state, action)

				Q[state][action] = (1-alpha)*Q[state][action] + alpha*(reward + self.gamma*max(Q[next_state].values()))
				state = next_state
				iters += 1

			if i%20 == 0:
				for state in self.P:
					pi[state] = max(self.P[state], key = lambda action: Q[state][action])
				
				d_reward = self.evaluate_over_episodes(pi)
				discounted_rewards.append(d_reward)
				iterations.append(i)

		return (pi, iterations, discounted_rewards)

	def sarsa(self, alpha,epsilon):
		
		Q = {}
		for state in self.P:
			Q[state] = {}
			for action in range(self.num_actions):
				Q[state][action] = 0

		terminal_state = self.encode(self.dest_loc, self.dest_loc, 0)
		Q[terminal_state] = {}
		for action in range(self.num_actions):
			Q[terminal_state][action] = 0
		
		episodes = 3000
		pi = {}
		discounted_rewards = []
		iterations = []
		for i in range(episodes):
			print("Episode ",i)
			state = np.random.choice(list(self.P.keys()))
			action = self.select_action(state, epsilon, Q)
			iters = 1
			while state != terminal_state and iters < 501:
				
				next_state, reward = self.get_next_state(state, action)
				if next_state != terminal_state:
					next_action = self.select_action(next_state, epsilon, Q)
					Q[state][action] = (1-alpha)*Q[state][action] + alpha*(reward + self.gamma*Q[next_state][next_action])
				else:
					next_action = 0
					Q[state][action] = (1-alpha)*Q[state][action] + alpha*reward
				state, action = next_state, next_action
				iters += 1
			
			if i%20 == 0:
				for state in self.P:
					pi[state] = max(self.P[state], key = lambda action: Q[state][action])
				
				d_reward = self.evaluate_over_episodes(pi)
				discounted_rewards.append(d_reward)
				iterations.append(i)

		return (pi, iterations, discounted_rewards)

	def sarsa_decay(self, alpha, epsilon):
		
		Q = {}
		for state in self.P:
			Q[state] = {}
			for action in range(self.num_actions):
				Q[state][action] = 0

		terminal_state = self.encode(self.dest_loc, self.dest_loc, 0)
		Q[terminal_state] = {}
		for action in range(self.num_actions):
			Q[terminal_state][action] = 0
		
		episodes = 2000
		pi = {}
		discounted_rewards = []
		iterations = []
		for i in range(episodes):
			print("Episode ",i)
			iters = 1
			state = np.random.choice(list(self.P.keys()))
			action = self.select_action(state, epsilon, Q)
			while state != terminal_state and iters < 501:
				
				next_state, reward = self.get_next_state(state, action)
				if next_state != terminal_state:
					next_action = self.select_action(next_state, epsilon/iters, Q)
					Q[state][action] = (1-alpha)*Q[state][action] + alpha*(reward + self.gamma*Q[next_state][next_action])
				else:
					next_action = 0
					Q[state][action] = (1-alpha)*Q[state][action] + alpha*reward
				state, action = next_state, next_action
				iters += 1

			if i%20 == 0:
				for state in self.P:
					pi[state] = max(self.P[state], key = lambda action: Q[state][action])
				
				d_reward = self.evaluate_over_episodes(pi)
				discounted_rewards.append(d_reward)
				iterations.append(i)

		return (pi, iterations, discounted_rewards)

	def T(self, taxi_loc, pass_loc, picked, curr_action):

		d = {}
		for action in range(4):
			new_taxi_loc = self.go(taxi_loc, action)
			if picked == 0:
				n_state = self.encode(new_taxi_loc, pass_loc, picked)
			else:
				n_state = self.encode(new_taxi_loc, new_taxi_loc, picked)
			d[n_state] = 0

		for action in range(4):
			new_taxi_loc = self.go(taxi_loc, action)
			if picked == 0:
				n_state = self.encode(new_taxi_loc, pass_loc, picked)
			else:
				n_state = self.encode(new_taxi_loc, new_taxi_loc, picked)

			if action == curr_action:
				d[n_state] += 0.85
			else:
				d[n_state] += 0.05

		return d

	def go(self, taxi_loc, action):
		(row, col) = taxi_loc
		(new_row, new_col) = taxi_loc

		#East
		if action == 0 and self.map[row][2*col+2] == ':':
			new_col = min(col+1, self.max_column)

		#West
		elif action == 1 and self.map[row][2*col] == ':':
			new_col = max(col-1,0)

		#North
		elif action == 2:
			new_row = max(row-1, 0)

		#South
		elif action == 3:
			new_row = min(row+1, self.max_row)

		return (new_row, new_col)

	def value_iteration(self, epsilon = 0.001):

		V = {state: 0 for state in self.P}
		terminal_state = self.encode(self.dest_loc, self.dest_loc, 0)
		V[terminal_state] = 0

		while True:
			V1 = copy.deepcopy(V)
			delta = 0
			for state in self.P:
				q_values = []
				for action in self.P[state]:
					
					q_sa = 0
					for (prob, n_state, reward) in self.P[state][action]:
						q_sa += prob*(reward + self.gamma*V1[n_state])
					q_values.append(q_sa)

				V[state] = max(q_values)
				delta = max(delta, abs(V1[state]-V[state]))

			if delta < (epsilon*(1-self.gamma))/self.gamma:
				return V1
	
	def expected_utlity(self, action, state, V):
		s = 0
		for (prob, n_state, reward) in self.P[state][action]:
			s = s + prob*(reward + self.gamma*V[n_state])
		return s

	def policy_iteration(self, epsilon = 0.001):
		V = {state: 0 for state in self.P}
		terminal_state = self.encode(self.dest_loc, self.dest_loc, 0)
		V[terminal_state] = 0

		pi = {state: random.choice(self.possible_actions) for state in self.P}
		
		iters = 0
		U = []

		while True:
			V = self.policy_evaluation_matrix(pi, V, epsilon)
			U.append(V)
			unchanged = True

			for state in self.P:
				action = max(self.P[state], key = lambda action: self.expected_utlity(action, state, V))
				if action != pi[state]:
					pi[state] = action
					unchanged = False

			iters += 1
			print(iters)
			if unchanged:
				# U_star = U[-1]
				# policy_loss = []
				# Iteration = []
				# for i in range(iters):
				# 	delta = 0
				# 	for state in self.P:
				# 		delta = max(delta, abs(U_star[state]-U[i][state]))
				# 	policy_loss.append(delta)
				# 	Iteration.append(i)

				# return (policy_loss, Iteration)
				return pi

	def policy_evaluation(self, pi, V, epsilon):

		while True:
			V1 = copy.deepcopy(V)
			delta = 0
			for state in self.P:
				val = 0
				action = pi[state]

				for (prob, n_state, reward) in self.P[state][action]:
					val += prob*(reward + self.gamma*V1[n_state])
				V[state] = val
				delta = max(delta, abs(V1[state]-V[state]))
			
			if delta < (epsilon*(1-self.gamma))/self.gamma:
				return V1


	def policy_evaluation_matrix(self, pi, V, epsilon):
		state_map = {}
		i = 0
		for state in V:
			state_map[state] = i
			i = i + 1
			
		size = 1 + (self.num_rows * self.num_columns)**2
		T = np.zeros((size, size))
		R = np.zeros((size, size))
		for state in self.P:
			idx1 = state_map[state]
			(tr,tc), (pr,pc), picked = self.decode(state)
			action = pi[state]
			if action < 4:
				
				transition = self.T((tr,tc), (pr,pc), picked, action)
				for n_state in transition:
					idx2 = state_map[n_state]
					T[idx1][idx2] = transition[n_state]
					R[idx1][idx2] = -1
			else:
				if picked == 0:
					if action == 4:
						if (tr,tc) == (pr,pc):
							n_state = self.encode((tr,tc), (pr,pc), 1)
							idx2 = state_map[n_state]
							T[idx1][idx2] = 1.0
							R[idx1][idx2] = -1
						else:
							n_state = self.encode((tr,tc), (pr,pc), 0)
							idx2 = state_map[n_state]
							T[idx1][idx2] = 1.0
							R[idx1][idx2] = -10
					else:
						n_state = self.encode((tr,tc), (pr,pc), 0)
						idx2 = state_map[n_state]
						T[idx1][idx2] = 1.0
						if (tr,tc) == (pr,pc):
							R[idx1][idx2] = -1
						else:
							R[idx1][idx2] = -10
				else:
					if action == 4:
						n_state = self.encode((tr,tc), (pr,pc), 1)
						idx2 = state_map[n_state]
						T[idx1][idx2] = 1.0
						R[idx1][idx2] = -1
					else:
						n_state = self.encode((tr,tc), (pr,pc), 0)
						idx2 = state_map[n_state]
						T[idx1][idx2] = 1.0
						R[idx1][idx2] = -1

		Y = np.reshape(np.sum(np.multiply(T,R), axis = 1), (-1,1))
		I = np.identity(size, dtype = float)
		U1 = np.dot(np.linalg.inv(I - self.gamma*T), Y)
		U = list(U1.T[0])
		for state in V:
			V[state] = U[state_map[state]]
		return V

	def optimal_policy(self):
		pi = {}
		V = self.value_iteration()
		for state in self.P:
			pi[state] = max(self.P[state], key = lambda action: self.expected_utlity(action, state, V))
		return pi

	
	def get_path(self, taxi_loc, pass_loc, dest_loc, policy):
		state = self.encode(taxi_loc, pass_loc, 0)
		path = [taxi_loc]
		i = 0
		while i < 30:

			taxi_loc, pass_loc, picked = self.decode(state)
			if i >= 20:
				return path

			if (taxi_loc == dest_loc and pass_loc == dest_loc and picked == 0):
				return path

			action = policy[state]

			if action < 4:
				new_taxi_loc = self.go(taxi_loc, action)
				if picked == 0:
					state = self.encode(new_taxi_loc, pass_loc, 0)
				else:
					state = self.encode(new_taxi_loc, new_taxi_loc, 1)
				#print(new_taxi_loc)
				path.append(new_taxi_loc)

			else:
				if action == 4:
					if picked == 0:
						if taxi_loc == pass_loc:
							state = self.encode(taxi_loc, pass_loc, 1)
						else:
							state = self.encode(taxi_loc, pass_loc, 0)
					else:
						state = self.encode(taxi_loc, pass_loc, 1)
					path.append(taxi_loc)
						
				elif action == 5:
					if picked == 0:
						state = self.encode(taxi_loc, pass_loc, 0)
					else:
						state = self.encode(taxi_loc, taxi_loc, 0)
					path.append(taxi_loc)
				#print(taxi_loc)
			i = i + 1


dest_loc = (4,3)
locs = [[(4,0), (0,4)], [(0,0), (0,4)], [(4,0), (0,0)], [(0,4), (0,0)], [(0,0), (4,0)]]

gamma = 0.99
mdp = MDP(dest_loc, gamma)
policy = mdp.policy_iteration(0.001)
# policy_loss, Iteration = mdp.policy_iteration()
# alpha = 0.5
# epsilon = 0.1
# policy, iterations, avg_reward = mdp.q_learning(alpha, epsilon)
# for state in policy:
# 	n_taxi_loc, n_pass_loc, picked = mdp.decode(state)
# 	print(n_taxi_loc, n_pass_loc, picked, policy[state])

for i in range(len(locs)):
	taxi_loc, pass_loc = locs[i][0], locs[i][1]
	print("Taxi_loc ", taxi_loc, " pass_loc ", pass_loc)
	print(mdp.get_path(taxi_loc, pass_loc, dest_loc, policy))


# print(mdp.get_path(taxi_loc, pass_loc, dest_loc, policy))

# mdp = MDP(dest_loc, gamma)
# iterations, avg_reward = mdp.q_learning(0.25)

# plt.plot(iterations, avg_reward, label = "learning-rate = {alpha}".format(alpha = alpha))
# plt.xlabel("Episode Index")
# plt.ylabel("Sum of discounted rewards in each episode")
# plt.title("Variation of Discounted reward with episodes")
# plt.legend()
# plt.show()
