from state import State
import random
import timeit
import matplotlib.pyplot as plt
import statistics as stats
import numpy as np

# If the disks are in different pins, we name the state first with where the big one is
statesString = ["b1s1", "b1s2", "b1s3", "s2b2", "s3b3", "b3s2", "b2s3", "b3s3", "b2s2", "b3s1", "b2s1", "s1b1"]
obeyProb = 0.9
moves = ["s1","s2","s3", "b1", "b2","b3"]
GAMMA = 0.9
# Creation of every state object
states = []
statesTable = {}
q_table = {}
q_table_all_values = {}
times_action_executed = {}
for state in statesString:
	new_state = State(state, obeyProb)
	states.append(new_state)
	statesTable[new_state.name] = new_state
	q_table[state] = [-9999 for x in range(0, len(moves))]
	q_table_all_values[state] = [[] for x in range(0, len(moves))]
	times_action_executed[state] = [1 for x in range(0, len(moves))]

state_to_action = {
    states[0].name: ["s2", "s3"],
    states[1].name: ["s1", "s3", "b2", "b3"],
    states[2].name: ["s1", "s2", "b2", "b3"],
    states[3].name: ["b1", "b3"],
    states[4].name: ["b1", "b2"],
    states[5].name: ["s1", "s3", "b1", "b2"],
    states[6].name: ["s1", "s2", "b1", "b3"],
    states[7].name: ["s3"],
    states[8].name: ["s1", "s3"],
    states[9].name: ["s2", "s3", "b1", "b2"],
    states[10].name: ["s2", "s3", "b1", "b3"],
    states[11].name: ["b2", "b3"]
}

print(state_to_action)
for key, value in q_table.items():
	# print("for key: " +key+ " we have values: " +str(value))
	for pos, action in enumerate(moves):
		# print("action: " +str(action)+ " in: " +str(state_to_action[key]))
		if action in state_to_action[key]:
			q_table[key][pos] = 0

# for key,val in q_table.items():
	# print (key, "=>", val)

# At the beggining we use the epsilon greedy strategy
def q_learning(episodes):
	# In the beginning, this rate must be at its highest value, because we don’t know anything about the values in Q-table.
	# Therefore we set it to 1 so that it is only exploration and we choose a random state
	epsilon = 1
	current_state = random.choice(states)
	print(current_state.name)
	step = 1/episodes
	epsilons = []
	epsilons.append(epsilon)

		
	for x in range(1, episodes):

		if(random.random() > epsilon): # Exploitation, choose the best action
			best_q_value = max(q_table[current_state.name])
			pos = 0
			for position, value in enumerate(q_table[current_state.name]):
				if value == best_q_value:
					pos = position
					break
			action = moves[pos]
		else: # Exploration, we choose a random action
			action = random.choice(state_to_action[current_state.name])

		reward, new_state = current_state.make_move(action)

		current_q_value = q_table[current_state.name][moves.index(action)]

		# λ^n = n^−α
		learning_rate = times_action_executed[current_state.name][moves.index(action)] ** -0.9

		value = current_q_value + learning_rate * (reward + GAMMA * max(q_table[new_state]) - current_q_value)

		# Update the q values table
		q_table[current_state.name][moves.index(action)] = value

		# Add the value to the table for the convergence plot
		q_table_all_values[current_state.name][moves.index(action)].append(value)

		# Update the counters table
		times_action_executed[current_state.name][moves.index(action)] += 1

		current_state = statesTable[new_state]

		# If the the new state is the end state then go to another one
		while current_state.name == "b3s3":
			current_state = random.choice(states)
		epsilon = epsilon - step
		epsilons.append(epsilon)

	# for key, value in q_table.items():
	# 	# print("for key: " +key+ " we have values: " +str(value))
	# 	for pos, action in enumerate(moves):
	# 		# print("action: " +str(action)+ " in: " +str(state_to_action[key]))
	# 		if action not in state_to_action[key]:
	# 			q_table[key][pos] = None


	return epsilon



# print(q_table)

print("Epsilon ends: " +str(q_learning(10000)))
result = {}

for key,val in q_table.items():
	best_pos = 0
	best_value = -999
	for pos, value in enumerate(val):
		if value > best_value:
			best_pos = pos
			best_value = value
	print("For " +key+ " the best position is: " +str(best_pos));
	result[key] = [moves[best_pos], round(best_value, 2)]
	plt.plot(q_table_all_values[key][best_pos], label = "State " +key+ " action " +moves[best_pos])
	plt.xscale('log')

print()
for key,val in result.items():
	print (key, "=>", val)
print()
print(" ------------------- ")
print()
for key,val in times_action_executed.items():
	print (key, "=>", val)
plt.legend()
plt.show()