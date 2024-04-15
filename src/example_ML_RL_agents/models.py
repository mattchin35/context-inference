import os
import torch
import torch.nn.functional as F
import pickle as pkl
from typing import Tuple, Protocol, Union

device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU


class Model(Protocol):
    def get_connections(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...

    def forward(self, prev_state: torch.Tensor, input: torch.Tensor, noise: torch.Tensor):
        ...


class Config(Protocol):
    rnn_size: int
    EI: bool
    proportion_excitatory: float

    noise: bool
    decay: float  # parameter from 0 to 1 - 1 has complete decay/complete replacement of prior state with new activity
    activation_fn: str

    train_input_connections: bool
    output_probabilities: bool

    weight_loss: float  # lol
    activity_loss: float

    time_loss_start: Union[int, None]
    time_loss_end: Union[int, None]

    batch_size: int
    learning_rate: float


def save_torch_model(model: torch.nn.Module, path: str):
    if not os.path.exists(path):
        os.makedirs(path)

    save_path = os.path.join(path, 'model.ckpt')
    torch.save(model, save_path)
    print("[***] Model saved in path: %s" % save_path)


def load_model(model: torch.nn.Module, path: str):
    save_path = os.path.join(path, 'model.ckpt')
    with open(save_path, 'rb') as f:
        load_dict = pkl.load(f)

    for k, v in load_dict.items():
        setattr(model, k, v)

    print("[***] Model restored from path: %s" % save_path)
    return model


def save_weights(model: Model, path: str):
    if not os.path.exists(path):
        os.makedirs(path)

    Wxh, Whh, Wout = model.get_connections()
    weight_dict = dict(Wxh=Wxh, Whh=Whh, Wout=Wout)

    fname = os.path.join(path, 'weights')
    with open(fname + ".pkl", 'wb') as f:
        pkl.dump(weight_dict, f)
    print("[***] Model weights saved in path: {}".format(fname))


class RNN(torch.nn.Module):
    """
    A class for a simple excitatory-inhibitory rnn to perform biological modeling.
    Input connections are constrained to be positive. Recurrent and output weights are
    constrained to excitatory or inhibitory connections determined from the start.

    """

    def __init__(self, data_dims: Tuple[int, int, int], opts: Config):
        super().__init__()
        self.opts = opts
        self._build(data_dims)

    def _build(self, data_dims: Tuple[int, int, int]) -> None:
        assert self.opts.activation_fn in ['relu', 'tanh', 'retanh'], "Invalid nonlinearity"
        dim_ipt, dim_opt, _ = data_dims

        # requires_grad defaults to False
        Wxh = torch.empty([dim_ipt, self.opts.rnn_size], device=device, requires_grad=self.opts.train_input_connections)
        Whh = torch.empty([self.opts.rnn_size, self.opts.rnn_size], device=device, requires_grad=True)
        Wout = torch.empty([self.opts.rnn_size, dim_opt], device=device, requires_grad=True)

        # init = torch.nn.init.xavier_normal_
        init = torch.nn.init.xavier_uniform_
        self.Wxh = init(Wxh)
        self.Whh = init(Whh)
        self.Wout = init(Wout)
        self.Whh_mask = 1 - torch.eye(self.opts.rnn_size, device=device)

        self.Wh_bias = torch.zeros([1, self.opts.rnn_size], device=device, requires_grad=True)
        self.Wout_bias = torch.zeros([1, dim_opt], device=device, requires_grad=True)
        self.fn = F.relu

        if self.opts.output_probabilities:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fn = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam([self.Wxh, self.Whh, self.Wh_bias, self.Wout, self.Wout_bias],
                                          lr=self.opts.learning_rate)

    def get_connections(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Whh = self.Whh * self.Whh_mask
        return self.Wxh, Whh, self.Wout

    def forward(self, inputs: torch.Tensor, noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # input the whole timeline at once, of any length T
        batch_size, timesteps, dim_input = inputs.shape  # N x T x D

        init_state = torch.zeros([batch_size, self.opts.rnn_size])
        state_series = [init_state]
        prediction_series = []
        for t in range(timesteps):
            _state, _prediction = self.step(state_series[-1], inputs[:, t, :], noise[:, t, :])
            state_series.append(_state)
            prediction_series.append(_prediction)

        state_series.pop(0)
        states = torch.stack(state_series, dim=1)
        predictions = torch.stack(prediction_series, dim=1)
        return states, predictions

    def backward(self, states: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, apply_gradient=True) \
            -> Tuple[float, float, float, float]:
        logits = logits[:, self.opts.time_loss_start:self.opts.time_loss_end]
        labels = labels[:, self.opts.time_loss_start:self.opts.time_loss_end]

        batch_size, timesteps, dim_output = logits.shape
        if self.opts.output_probabilities:
            # the torch cross-entropy loss requires labels as an index, not a one-hot vector
            logits_loss = torch.reshape(logits, (batch_size * timesteps, dim_output))
            labels = torch.argmax(labels, dim=2)  # for Cross Entropy loss, assuming one-hot labels...
            labels_loss = torch.reshape(labels, (batch_size * timesteps,))
        else:  # MSE
            logits_loss = logits
            labels_loss = labels

        error_loss = self.loss_fn(logits_loss, labels_loss)
        activity_loss = self.opts.activity_loss * torch.mean(states)
        weight_loss = self.opts.weight_loss * torch.mean(torch.pow(self.Whh, 2))
        total_loss = error_loss + activity_loss + weight_loss

        if apply_gradient:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return total_loss.item(), error_loss.item(), activity_loss.item(), weight_loss.item()

    def step(self, prev_state: torch.Tensor, inputs: torch.Tensor, noise: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        Wxh, Whh, Wout = self.get_connections()
        hidden_activation = torch.matmul(prev_state, Whh) + torch.matmul(inputs, Wxh) + self.Wh_bias
        if self.opts.noise:
            hidden_activation += noise

        state = (1. - self.opts.decay) * prev_state + self.opts.decay * self.fn(hidden_activation)
        prediction = torch.matmul(state, Wout) + self.Wout_bias
        return state, prediction

