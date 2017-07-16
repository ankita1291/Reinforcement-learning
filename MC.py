import random
import numpy as np 

from RL_David import *

def montecarlo(value, counter):
	state = State()
	state.dealercard = random.randint(1, 10)
	state.player_sum = random.randint(1,10)

	total_reward = 0
	visits = []

	while  state != "terminal":
		#epsilon greedy approach
		action = None
		e = 100/100 + np.sum(counter[:, state.dealercard, state.player_sum])
		if(random.random()<e):
			action = random.randint(0,1)
		else:
			action = np.argmax(value[:, state.dealercard, state.player_sum])
		counter[action, state.dealercard, state.player_sum] += 1
		visits.append((action, state.dealercard, state.player_sum))
		state, reward = step(state, action)
		total_reward += reward

	for action, dealercard, player_sum in visits:
		a = 1/counter[action, dealercard, player_sum]
		g = total_reward
		value[action, dealercard, player_sum] = value[action, dealercard, player_sum]+a*(g-value[action, dealercard, player_sum])
	return value, counter





