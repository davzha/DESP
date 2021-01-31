from pathlib import Path

class Config(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError

config = Config()
config.rnd_seed = 0
config.description = f"DESP set mnist {config.rnd_seed}"
config.num_epoch = 100
config.anomaly_detection = False
config.n_process_pool = 0
config.model = "desp_auto_encoder"
config.early_stopping = 7
config.eval_loss = "chamfer"
config.val_compare = "<"

config.log = Config()
config.log.dir = Path("log_set_mnist")
config.log.file = "log.txt"
config.log.freq = 10
config.log.plt_freq = 100
config.log.overwrite_ckpt = True

config.train = Config()
config.train.batch_size = 100
config.train.num_workers = 6
config.train.train_val_split = 0.95

config.test = Config()
config.test.batch_size = 100
config.test.num_workers = 6

config.optim = Config()
config.optim.lr = 1e-4

config.data = Config()
config.data.name = "MNISTSet"
config.data.n_points = 342
config.data.mem_feat = True

config.sample = Config()
config.sample.T = 500
config.sample.S = 500
config.sample.eps = 0.005

config.predict = Config()
config.predict.T = 500
config.predict.S = 400
config.predict.eps = 0.005

config.set_enc = Config()
config.set_enc.d_in = 3
config.set_enc.d_hid = 256
config.set_enc.d_out = 64
config.set_enc.n_layers = 3

config.estim = Config()
config.estim.data_noise = True
config.estim.eps_d = 0.015