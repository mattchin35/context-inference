import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.behavior_modeling.task import config
from src.ML_agents.models import RNN

import os, time
import pickle as pkl
from typing import Tuple
import logging
logging.basicConfig(filename='../logs/train.log', level=logging.DEBUG)

np.random.seed(0)
torch.manual_seed(0)


class genericDataset(Dataset):
    """Create a generic dataset for numpy data."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, N: np.ndarray):
        assert X.shape[0] == Y.shape[0], "Inputs and labels are not the same size"
        self.X = X
        self.Y = Y
        self.N = N

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, ix):
        return self.X[ix], self.Y[ix], self.N[ix]

    def get_dims(self) -> Tuple[int, int, int]:
        return self.X.shape[-1], self.Y.shape[-1], self.N.shape[-1]


def save_activity(model, dataloader, path=None, save_name=None):
    # run some test activity - TODO
    if path is not None:
        save_path = path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = model.save_path

    if save_name is None:
        save_name = model.opts.activity_name

    f_name = os.path.join(save_path, save_name)
    for i, batch in enumerate(dataloader):
        x, y, noise = batch
        states, logits, predictions = model.forward(x, noise)
        _, error_loss, _, _ = model.backward(states, logits, y, apply_gradient=False)

    states, predictions = states.detach().numpy(), predictions.detach().numpy()
    # states, predictions = np.stack(states, axis=1), np.stack(predictions, axis=1)
    data = {'X': x, 'Y': y, 'N': noise, 'states': states, 'predictions': predictions, 'loss': error_loss}
    with open(f_name + ".pkl", 'wb') as f:
        pkl.dump(data, f)


def create_dataloader(inputs: np.ndarray, labels: np.ndarray, batch_size: int, rnn_size: int, shuffle: bool = True, noise: bool = True) -> \
        Tuple[DataLoader, Tuple[int, int, int]]:

    n_inputs, timesteps = inputs.shape[0], inputs.shape[1]
    if noise:
        N = np.random.randn(n_inputs, timesteps, rnn_size) / rnn_size
    else:
        N = np.zeros((n_inputs, timesteps, rnn_size))

    inputs = inputs.astype(np.float32)
    labels = labels.astype(np.float32)
    N = N.astype(np.float32)

    dataset = genericDataset(inputs, labels, N)
    dataloader = DataLoader(genericDataset(inputs, labels, N), batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset.get_dims()


def train(model: torch.nn.Module, dataloader: DataLoader, n_epoch: int, save_path: str = '../saved_models/model.ckpt'):
    t = time.perf_counter()
    for ep in range(n_epoch):
        for i, (xi, yi, ni) in enumerate(dataloader):
            states, predictions = model.forward(xi, ni)
            total_loss, error_loss, activity_loss, weight_loss = model.backward(states, predictions, yi)
            assert not np.isnan(error_loss), f"Error is NaN, retry; ep {ep}, minibatch {i}"

        logging.info(f'Epoch {ep};  total_loss={total_loss:.2f};  error_loss={error_loss:.2f};  a_loss={activity_loss:.2f};  w_loss={weight_loss:.2f}')

        if ep > 0 and (ep+1) % 20 == 0:  # display in terminal
            print('[*] Epoch %d  total_loss=%.2f mse_loss=%.2f a_loss=%.2f, w_loss=%.2f'
                  % (ep+1, total_loss, error_loss, activity_loss, weight_loss))
            tnew = time.perf_counter()
            print(f'{tnew - t} seconds elapsed')
            t = tnew

    # save latest
    torch.save(model, save_path)


def eval(model: torch.nn.Module, dataloader: DataLoader, save_path: str):
    # Generate and evaluate a test set for network analysis.
    print('[*] Testing')
    for i, (xi, yi, ni) in enumerate(dataloader):  # this is only one iteration
        states, predictions = model.forward(xi, ni)
        total_loss, error_loss, activity_loss, weight_loss = model.backward(states, predictions, yi,
                                                                            apply_gradient=False)

    states, predictions = states.detach().numpy(), predictions.detach().numpy()
    data = {'inputs': xi, 'outputs': yi, 'noise': ni, 'states': states, 'predictions': predictions, 'loss': error_loss}
    with open(save_path + ".pkl", 'wb') as f:
        pkl.dump(data, f)


def main():
    # set training hyperparameters
    n_epochs = 200
    batch_size = 64
    load_checkpoint = ''  # a path to load a model from

    inputs_path = '../datasets/1d_inputs_bounded.pkl'
    model_opts = config.AgentConfig()

    # prepare data
    with open(inputs_path, 'rb') as f:
        data_dict = pkl.load(f)

    X = data_dict['inputs']
    # Y = data_dict['labels_gaussian']  # choose label type - gaussian (nonneg) or hat (surround suppression)
    Y = data_dict['position']  # choose label type - gaussian (nonneg) or hat (surround suppression)
    if len(Y.shape) == 2:
        Y = Y[:, :, None]

    # setup model
    dataloader, dims = create_dataloader(inputs=X, labels=Y, batch_size=batch_size, rnn_size=model_opts.rnn_size)
    model = RNN(data_dims=dims, opts=model_opts)
    if load_checkpoint:
        model.load(load_checkpoint)

    train(model, dataloader, n_epochs)

    # would put code to load a model here #
    # activity_path = '../saved_activity/test_activity'
    # dataloader, _ = create_dataloader(inputs=X, labels=Y, batch_size=X.shape[0], rnn_size=model_opts.rnn_size,
    #                                   shuffle=False, noise=False)
    # eval(model, dataloader, activity_path)


if __name__ == '__main__':
    main()
