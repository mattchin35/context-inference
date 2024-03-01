class WeightNames(object):
    def __init__(self):
        l = ['W_in', 'W_h', 'W_h_mask', 'W_h_bias',
             'W_out', 'W_out_bias', 'ei_mask']
        for k in l:
            setattr(self, k, k)


class AgentConfig(object):  # RNN options
    def __init__(self):
        self.rnn_size = 100

        self.noise = True
        self.noise_intensity = .1
        self.noise_density = .5
        self.decay = 1  # parameter from 0 to 1 - 1 has complete decay/complete replacement of prior state with new activity

        self.train_input_connections = True
        self.output_probabilities = False  # use softmax/cross-entropy loss if true, otherwise use MSE

        # loss weights; set these to zero to turn them off
        self.weight_loss = .1
        self.activity_loss = .1

        self.time_loss_start = 0
        self.time_loss_end = None  # this works for grabbing the whole array apparently

        self.learning_rate = .001
        self.baseline: float = 0

        self.epoch = 401
        self.load_checkpoint = False

        # self.model_name = 'model'
        # self.weight_name = 'weight'
        # self.activity_name = 'activity'
        # self.parameter_name = 'parameters'
        # self.image_folder = 'images'
        # self.log_name = 'log'
        # self.save_path = '../saved'

