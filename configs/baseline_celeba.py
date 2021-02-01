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
config.description = f"DESP subset outlier {config.rnd_seed}"
config.device = "cuda"
config.num_epoch = 30
config.anomaly_detection = False
config.n_process_pool = 0
config.model = "baseline_subset_outlier"
config.early_stopping = 3
config.val_compare = ">"

config.log = Config()
config.log.dir = Path("log_celeba")
config.log.file = "log.txt"
config.log.freq = 10
config.log.plt_freq = 100
config.log.overwrite_ckpt = True

config.train = Config()
config.train.batch_size = 64
config.train.num_workers = 6
config.train.kwargs = dict(length=32000)

config.val = Config()
config.val.batch_size = 64
config.val.num_workers = 6
config.val.kwargs = dict(length=3200)

config.test = Config()
config.test.batch_size = 64
config.test.num_workers = 6
config.test.kwargs = dict(length=19200)

config.optim = Config()
config.optim.lr = 1e-4

config.data = Config()
config.data.name = "CelebASet"
config.data.root = Path.home() / "data" / "CelebA"
config.data.n_attr = 2
config.data.set_size = 5
config.data.n_outliers = [0,1,2]
config.data.min_n_target = 1
config.data.p_outliers = None
config.data.download = False

config.img_enc = Config()
config.img_enc.d_out = 128
config.img_enc.model_file = Path("pretrained") / "resnet34_celeba_attributes.pth.tar"
config.img_enc.freeze_resnet = True

config.ds = Config()
config.ds.d_in = 128
config.ds.d_hid = 256
config.ds.d_out = 1
config.ds.n_layers = 4


