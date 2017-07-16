import numpy as np 
import random

class State():
	dealer_card = random.randint(1, 10)
	playersum = random.randint(1, 10)

def drawcard(current):
	if random.randint(1, 3) < 3:
		current += random.randint(1, 10)
	else:
		current -= random.randint(1, 10)
	return current


#1 = hit, 0 = stick
def step(state, action):
	if action == 1:
		state.playersum = drawcard(state.playersum)
		if state.playersum < 1 or state.playersum > 21:
			return "terminal", -1
		else:
			return state, 0

	elif action == 0:
		while(state.dealer_card <17):
			state.dealer_card = drawcard(state.dealer_card)
			if state.dealercard < 1 or state.dealercard > 21:
				return "terminal", 1.0
		if state.dealercard > state.playersum:
			return "terminal", -1.0
		elif state.dealercard < state.playersum:
			return "terminal", 1.0
		else:
			return "terminal", 0.0



