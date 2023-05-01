import numpy as np
import scipy as sp
from scipy.stats import norm
import matplotlib.pyplot as plt


np.random.seed(0)
# Hidden markov model
# See your APMA 1690/1740 notes
# full state encoding model


# Fixed ratio rewards
# interval_reward_hi = 4
# interval_reward_lo = 16
# reward_block_ix = np.arange(interval_reward_hi, npress+1, interval_reward_hi)-1
# reward_block[reward_block_ix] = 1
# nonreward_block_ix = np.arange(interval_reward_lo, npress+1, interval_reward_lo)-1
# nonreward_block[nonreward_block_ix] = 1

# Random delivery rewards
npress = 60
std_dev = .5
means = [1,.25]
def create_block(blocktype, mean=means):
    assert blocktype in ['A', 'B', 'C1', 'C2']
    if blocktype == 'A':
        # rewards = np.random.choice([0, 1], npress, p=[1 - p, p])
        rewards = np.random.normal(mean[0], std_dev, npress)
        stimuli = np.zeros(npress)

    elif blocktype == 'B':
        # rewards = np.random.choice([0, 1], npress, p=[p, 1-p])
        rewards = np.random.normal(mean[1], std_dev, npress)
        stimuli = np.ones(npress)

    elif blocktype == 'C1':
        rewards = np.random.normal(mean[0], std_dev, npress)
        stimuli = np.ones(npress) * 2

    elif blocktype == 'C2':
        rewards = np.random.normal(mean[1], std_dev, npress)
        stimuli = np.ones(npress) * 2

    else:
        raise NotImplementedError("This block type does not exist!!")

    rewards[rewards<0] = 0
    outcome = np.stack((rewards, stimuli), axis=1)
    return outcome


def block_probabilities(outcome, mean=means, std=std_dev, include_stimulus=True):
    pA = norm.pdf(outcome[:,0], mean[0], std)
    pB = norm.pdf(outcome[:,0], mean[1], std)
    pC1 = norm.pdf(outcome[:,0], mean[0], std)
    pC2 = norm.pdf(outcome[:,0], mean[1], std)

    if include_stimulus:
        pA[outcome[:, 1] != 0] = 0
        pB[outcome[:, 1] != 1] = 0
        pC1[outcome[:, 1] != 2] = 0
        pC2[outcome[:, 1] != 2] = 0

    return np.stack((pA, pB, pC1, pC2), axis=1)

p_reward = 3/4
Y = np.concatenate([create_block('A'), create_block('B'),
                    create_block('C1'), create_block('C2'),
                    create_block('A'), create_block('C2')])
pStates = block_probabilities(Y, include_stimulus=True)


Ysize = np.shape(Y)[0]
print(Ysize)
ix = np.arange(Ysize)


# probability of each data point Yi given a high reward context or low reward context
p_switch = 1/60
# from-to, ABC1C2
transition_matrix = np.array([[1-3*p_switch, p_switch, p_switch, p_switch],
                                   [p_switch, 1-3*p_switch, p_switch, p_switch],
                                   [p_switch, p_switch, 1-3*p_switch, p_switch],
                                   [p_switch, p_switch, p_switch, 1-3*p_switch]])


# the lookup table
# table = np.zeros((Ysize, 3))  # row 1 for x = 0, 1 for x = 1
table = np.zeros((Ysize, 4))  # row 1 for x = 0, 1 for x = 1
forward_state = np.zeros(Ysize)
reverse_state = np.zeros_like(forward_state).astype(int)
forward_max = np.zeros_like(forward_state)  # I think max probability

# initialize by initial probability of each state
# table[0] = pStates[0] # times uniform random probability, if desired
table[0] = pStates[0] * np.ones(4)/4 # times uniform random probability, if desired
table[0] /= np.sum(table[0])

forward_state[0] = np.argmax(table[0])
forward_max[0] = np.amax(table[0])

table_backwards = np.ones((Ysize, 4))

# table[0,0] = pA0[0]
# table[0,1] = pA1[0]

# p0 = np.array([1-p_switch, p_switch])  # probability current state is 0
# p1 = np.flip(p0)  # probability current state is 1
# pStates = np.stack((pA0, pB), axis=1)
# pStates = np.stack((pA0, pA1, pB), axis=1)

# FORWARD
for i in ix[1:]:
    # prob Xi = 0
    # table[i,0] = pA0[i] * np.sum(table[i-1] * p0)
    # prob Xi = 1
    # table[i,1] = pB[i] * np.sum(table[i-1] * p1)

    # Calculate and normalize the probabilites; they become vanishingly small otherwise
    table[i] = pStates[i] * np.dot(transition_matrix, table[i-1])
    table[i] /= np.sum(table[i])

    forward_state[i] = np.argmax(table[i])
    forward_max[i] = np.amax(table[i])

# BACKWARDS: Optimal config

reverse_state[-1] = np.argmax(table[-1])
for i in np.flip(ix[:-1]):
    # Use state at timestep ahead to calculate current state probabilities
    P = transition_matrix[reverse_state[i+1]] * table[i]
    P /= np.sum(P)
    # reverse_state[i] = np.argmax(P)

    table_backwards[i] = np.dot(transition_matrix, table_backwards[i+1] * pStates[i+1])
    table_backwards[i] /= np.sum(table_backwards[i])
    reverse_state[i] = np.argmax(table_backwards[i])

full_model = table * table_backwards
full_state = np.argmax(full_model, axis=1)

# find indices where state changes from 1 to 0
flips_forward = forward_state[1:] != forward_state[:-1]
# n_flips = np.sum(flips)
ix_flips = np.nonzero(flips_forward)[0] + 1  # add 1 since flips can only occur starting from ix 1
print('forward', ix_flips)
flips_reverse = reverse_state[1:] != reverse_state[:-1]
# n_flips = np.sum(flips)
ix_flips = np.nonzero(flips_reverse)[0] + 1  # add 1 since flips can only occur starting from ix 1
print('reverse', ix_flips)
flips_full = full_state[1:] != full_state[:-1]
ix_flips = np.nonzero(flips_full)[0] + 1  # add 1 since flips can only occur starting from ix 1
print('full', ix_flips)


# plotting
# f, ax = plt.subplots()
f, ax = plt.subplots(2,1, figsize=(6,6))
plt.sca(ax[0])
plt.title('Hidden State', fontsize=14)
plt.scatter(ix, forward_state, label='forward', s=5)
plt.scatter(ix, reverse_state, label='reverse', s=5)
plt.scatter(ix, full_state, label='full', s=5)
# plt.scatter(ix, reverse_state, s=5)
ax[0].set_yticks([0,1,2,3])
ax[0].set_yticklabels(['A','B','C1','C2'])
plt.ylabel('Inferred State', fontsize=14)
ax[0].legend()

plt.sca(ax[1])
plt.title('Observed outcome', fontsize=14)
plt.scatter(ix, Y[:,0], label='reward', s=10)
plt.plot(ix, Y[:,1], label='stimuli', c='g')
plt.ylim(-.1, 2.1)
plt.xlabel('Trial', fontsize=14)
plt.ylabel('Reward size', fontsize=14)
# plt.scatter(ix, Y[:,1], label='stimuli', s=10)
ax2 = ax[1].twinx()
plt.sca(ax2)
ax2.set_yticks([0,1, 2])
ax2.set_yticklabels(['A','B', 'C'])
plt.ylim(-.1, 2,1)
plt.ylabel('Context stimuli', fontsize=14)
ax[1].legend(fancybox=False)
# ax2.legend()
# plt.xlabel('Index (i)')
# plt.ylabel('State + Observed')
plt.suptitle('Hidden Markov Model Inference', fontsize=16)
# plt.show()

format = 'png'
f.tight_layout()
plt.savefig('../figures/HMM_behavior.' + format, format=format)
