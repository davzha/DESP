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
config.description = f"DESP digits {config.rnd_seed}"
config.num_epoch = 1
config.anomaly_detection = False
config.n_process_pool = 0
config.model = "desp_digits_and_polygons"
config.eval_loss = "chamfer"

config.log = Config()
config.log.dir = Path("log_digits")
config.log.file = "log.txt"
config.log.freq = 10
config.log.plt_freq = 100
config.log.overwrite_ckpt = True

config.train = Config()
config.train.batch_size = 100
config.train.num_workers = 6
config.train.kwargs = dict(length=400000)

config.test = Config()
config.test.batch_size = 100
config.test.num_workers = 6
config.test.kwargs = dict(length=4000)

config.optim = Config()
config.optim.lr = 1e-4

config.data = Config()
config.data.name = "Digits"
config.data.n_points = 64

config.sample = Config()
config.sample.T = 100
config.sample.S = 100
config.sample.eps = 0.01

config.predict = Config()
config.predict.T = 100
config.predict.S = 80
config.predict.eps = 0.01

config.energy_fn = Config()
config.energy_fn.d_x = 2
config.energy_fn.d_y = 2
config.energy_fn.d_hid = 256
config.energy_fn.n_equiv = 3
config.energy_fn.n_inv = 3
config.energy_fn.normalize = True

config.estim = Config()
config.estim.data_noise = True
config.estim.eps_d = 0.015