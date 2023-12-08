import numpy as np
import mdptoolbox

# Define the POMDP model
P = np.array([[[0.7, 0.3], [0.3, 0.7]], [[0.4, 0.6], [0.6, 0.4]]])  # Transition probabilities
O = np.array([[[0.9, 0.1], [0.2, 0.8]], [[0.3, 0.7], [0.6, 0.4]]])  # Observation probabilities
R = np.array([[-1, 10], [5, -2]])  # Reward function
discount = 0.9  # Discount factor

# Create the POMDP object and solve it
pomdp = mdptoolbox.mdp.POMDP(P, O, R, discount)
pomdp.run()

# Print the optimal policy and the value function
print(pomdp.policy)
print(pomdp.V)