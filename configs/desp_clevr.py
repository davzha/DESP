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
config.description = f"DESP clevr object detection {config.rnd_seed}"
config.num_epoch = 100
config.anomaly_detection = False
config.n_process_pool = 0
config.model = "desp_object_detection"
config.early_stopping = 7
config.eval_loss = "chamfer"

config.log = Config()
config.log.dir = Path("log_clevr")
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
config.data.name = "CLEVRSet"
config.data.base_path = Path.home() / "data" / "clevr"
config.data.n_points = 10
config.data.mem_feat = True
config.data.full = True
config.data.box = True

config.sample = Config()
config.sample.T = 200
config.sample.S = 200
config.sample.eps = 0.0025

config.predict = Config()
config.predict.T = 200
config.predict.S = 160
config.predict.eps = 0.0025

config.img_enc = Config()
config.img_enc.latent = 512

config.set_enc = Config()
config.set_enc.d_in = 5
config.set_enc.d_hid = 512
config.set_enc.d_out = 512
config.set_enc.n_layers = 2

config.estim = Config()
config.estim.data_noise = True
config.estim.eps_d = 0.005