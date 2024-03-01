import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

import pickle as pkl
from typing import Tuple, Protocol, Union
from itertools import count
from collections import deque

device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU
eps = np.finfo(np.float32).eps.item()
N_ACTIONS = 2


class Config(Protocol):
    rnn_size: int
    EI: bool
    proportion_excitatory: float

    noise: bool
    decay: float  # parameter from 0 to 1 - 1 has complete decay/complete replacement of prior state with new activity
    activation_fn: str

    train_input_connections: bool
    output_probabilities: bool

    weight_loss: float
    activity_loss: float

    time_loss_start: Union[int, None]
    time_loss_end: Union[int, None]

    batch_size: int
    learning_rate: float
    baseline: float


class RNN_reinforce(torch.nn.Module):
    """REINFORCE policy gradient model."""
    def __init__(self, data_dims: Tuple[int, int, int], opts: Config):
        super().__init__()
        self.opts = opts
        self._build(data_dims)

    def _build(self, data_dims: Tuple[int, int, int]) -> None:
        dim_ipt, dim_opt, rnn_size = data_dims
        # requires_grad defaults to False
        Wxh = torch.empty([dim_ipt, rnn_size], device=device, requires_grad=self.opts.train_input_connections)
        Whh = torch.empty([rnn_size, rnn_size], device=device, requires_grad=True)
        Wout = torch.empty([rnn_size, N_ACTIONS], device=device, requires_grad=True)

        # init = torch.nn.init.xavier_normal_
        init = torch.nn.init.xavier_uniform_
        self.Wxh = init(Wxh)
        self.Whh = init(Whh)
        self.Wout = init(Wout)

        self.Whh_mask = 1 - torch.eye(rnn_size, device=device)
        self.Wh_bias = torch.zeros([1, rnn_size], device=device, requires_grad=True)
        self.Wout_bias = torch.zeros([1, 2], device=device, requires_grad=True)

        self.optimizer = torch.optim.Adam([self.Wxh, self.Whh, self.Wh_bias, self.Wout, self.Wout_bias],
                                          lr=self.opts.learning_rate)

        self.rewards = []
        self.saved_log_probs = []
        self.discount = .99

        init_state = torch.zeros([1, rnn_size])
        self.state_series = [init_state]
        self.prediction_series = []

    def forward(self, inputs: torch.Tensor, noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # move one step forward in time
        next_state, predictions = self.step(self.state_series[-1], inputs, noise)
        m = Categorical(predictions)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        self.state_series.append(next_state)
        self.prediction_series.append(predictions)
        return next_state, predictions, action.item()

    def backward(self, apply_gradient=True) \
            -> Tuple[float, float, float, float]:

        self.state_series.pop(0)
        states = torch.stack(self.state_series, dim=1)
        policy_loss = []

        # rolling_rewards = 0
        # returns = deque()
        # for r in self.rewards[::-1]:
        #     rolling_rewards = r + self.discount * rolling_rewards
        #     returns.appendleft(rolling_rewards)
        # returns = torch.tensor(returns)


        returns = torch.tensor(self.rewards).float()
        # if self.opts.baseline:
        #     returns = (returns - self.opts.baseline) / (returns.std() + eps)
        # else:
        returns = (returns - returns.mean()) / (returns.std() + eps)  # reward standardization for backprop/variance reduction
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        policy_loss = torch.cat(policy_loss).sum()
        activity_loss = self.opts.activity_loss * torch.mean(states)
        weight_loss = self.opts.weight_loss * torch.mean(torch.pow(self.Whh, 2))
        total_loss = policy_loss + activity_loss + weight_loss

        if apply_gradient:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        # reset counters
        self.reset()
        return total_loss.item(), policy_loss.item(), activity_loss.item(), weight_loss.item()

    def step(self, prev_state: torch.Tensor, inputs: torch.Tensor, noise: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        Whh = self.Whh * self.Whh_mask
        hidden_activation = torch.matmul(prev_state, Whh) + torch.matmul(inputs, self.Wxh) + self.Wh_bias
        if self.opts.noise:
            hidden_activation += noise

        state = (1. - self.opts.decay) * prev_state + self.opts.decay * F.relu(hidden_activation)
        action_scores = torch.matmul(state, self.Wout) + self.Wout_bias
        policy = F.softmax(action_scores, dim=1)
        return state, policy

    def reset(self):
        init_state = torch.zeros([1, self.opts.rnn_size])
        self.rewards = []
        self.saved_log_probs = []
        self.state_series = [init_state]
        self.prediction_series = []


class RNN_A2C(torch.nn.Module):
    """Actor-critic model for policy gradient learning."""
    def __init__(self, data_dims: Tuple[int, int, int], opts: Config):
        super().__init__()
        self.opts = opts
        self._build(data_dims)

    def _build(self, data_dims: Tuple[int, int, int]) -> None:
        dim_ipt, dim_opt, rnn_size = data_dims
        # requires_grad defaults to False
        Wxh = torch.empty([dim_ipt, rnn_size], device=device, requires_grad=self.opts.train_input_connections)
        Whh = torch.empty([rnn_size, rnn_size], device=device, requires_grad=True)
        Wout = torch.empty([rnn_size, N_ACTIONS], device=device, requires_grad=True)

        # init = torch.nn.init.xavier_normal_
        init = torch.nn.init.xavier_uniform_
        self.Wxh = init(Wxh)
        self.Whh = init(Whh)
        self.Wout = init(Wout)

        self.Whh_mask = 1 - torch.eye(rnn_size, device=device)
        self.Wh_bias = torch.zeros([1, rnn_size], device=device, requires_grad=True)
        self.Wout_bias = torch.zeros([1, 2], device=device, requires_grad=True)

        self.optimizer = torch.optim.Adam([self.Wxh, self.Whh, self.Wh_bias, self.Wout, self.Wout_bias],
                                          lr=self.opts.learning_rate)

        self.rewards = []
        self.saved_log_probs = []
        self.discount = .99

        init_state = torch.zeros([1, rnn_size])
        self.state_series = [init_state]
        self.prediction_series = []

    def forward(self, inputs: torch.Tensor, noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # move one step forward in time
        next_state, predictions = self.step(self.state_series[-1], inputs, noise)
        m = Categorical(predictions)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        self.state_series.append(next_state)
        self.prediction_series.append(predictions)
        return next_state, predictions, action.item()

    def backward(self, apply_gradient=True) \
            -> Tuple[float, float, float, float]:

        self.state_series.pop(0)
        states = torch.stack(self.state_series, dim=1)
        policy_loss = []

        # rolling_rewards = 0
        # returns = deque()
        # for r in self.rewards[::-1]:
        #     rolling_rewards = r + self.discount * rolling_rewards
        #     returns.appendleft(rolling_rewards)
        # returns = torch.tensor(returns)

        returns = torch.tensor(self.rewards).float()
        # if self.opts.baseline:
        #     returns = (returns - self.opts.baseline) / (returns.std() + eps)
        # else:
        returns = (returns - returns.mean()) / (returns.std() + eps)  # reward standardization for backprop/variance reduction
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        policy_loss = torch.cat(policy_loss).sum()
        activity_loss = self.opts.activity_loss * torch.mean(states)
        weight_loss = self.opts.weight_loss * torch.mean(torch.pow(self.Whh, 2))
        total_loss = policy_loss + activity_loss + weight_loss

        if apply_gradient:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        # reset counters
        self.reset()
        return total_loss.item(), policy_loss.item(), activity_loss.item(), weight_loss.item()

    def step(self, prev_state: torch.Tensor, inputs: torch.Tensor, noise: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        Whh = self.Whh * self.Whh_mask
        hidden_activation = torch.matmul(prev_state, Whh) + torch.matmul(inputs, self.Wxh) + self.Wh_bias
        if self.opts.noise:
            hidden_activation += noise

        state = (1. - self.opts.decay) * prev_state + self.opts.decay * F.relu(hidden_activation)
        action_scores = torch.matmul(state, self.Wout) + self.Wout_bias
        policy = F.softmax(action_scores, dim=1)
        return state, policy

    def reset(self):
        init_state = torch.zeros([1, self.opts.rnn_size])
        self.rewards = []
        self.saved_log_probs = []
        self.state_series = [init_state]
        self.prediction_series = []


