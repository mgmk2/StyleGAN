class Params(object):
    def __init__(self):
        self.z_dim = 100
        self.image_shape = (32, 32, 3)

        self.batch_size = 64 # バッチサイズ

        self.start_epoch = 0
        self.epochs = 100 # 学習回数

        self.use_sn_in_disc = True

        self.learning_rate = 1.0e-4
        self.lr_beta1 = 0
        self.lr_beta2 = 0.99

        self.gp_weight = 10.0

        self.dataset_train = 'cifar10'
        self.dataset_eval = 'cifar10'

        #=======================================================================
        # Style GAN params

        self.num_layers = 8
        self.use_wscale = True
        self.lr_mul = {'gen_mapping': 0.01, 'gen_synthesis': 1.0, 'disc': 1.0}
        self.mixing_prob = 0.9
        self.latent_avg_beta = 0.995
        self.truncation_psi = 0.7
        self.truncation_cutoff = 4
        self.distribution = 'untruncated_normal'

        self.epochs_at_lod_period = 10
        self.epochs_for_progressive = 10
        self.epochs_at_lod_max = 50
        self.lr_mul_shedule = {0: 1.0, 1: 0.5, 2: 0.25, 3: 0.125, 4: 0.0625}

        #=======================================================================

    def save(self, filename):
        d = self.__dict__
        with open(filename + '.json', 'w') as f:
            json.dump(d, f, indent=4)

    def load(self, filename):
        with open(filename + '.json', 'r') as f:
            d = json.load(f)
        for key, value in d.items():
            setattr(self, key, value)
