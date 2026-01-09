from dataclasses import dataclass

class WeightNames(object):
    def __init__(self):
        l = ['W_in', 'W_h', 'W_h_mask', 'W_h_bias',
             'W_out', 'W_out_bias', 'ei_mask']
        for k in l:
            setattr(self, k, k)


@dataclass
class AgentConfig:  # RNN options
    rnn_size = 100

    noise = True
    noise_intensity = .1
    noise_density = .5
    decay = 1  # parameter from 0 to 1 - 1 has complete decay/complete replacement of prior state with new activity

    train_input_connections = True
    output_probabilities = False  # use softmax/cross-entropy loss if true, otherwise use MSE

    weight_loss = .1
    activity_loss = .1
    activation_fn = 'relu'  # 'tanh', 'relu', 'retanh'

    time_loss_start = 0
    time_loss_end = None  # this works for grabbing the whole array apparently

    learning_rate = .0001
    baseline = 0

    epoch = 401
    load_checkpoint = False

    # model_name = 'model'
    # weight_name = 'weight'
    # activity_name = 'activity'
    # parameter_name = 'parameters'
    # image_folder = 'images'
    # log_name = 'log'
    # save_path = '../saved'

