import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import chamfer_distance, LAP_loss
from modules import ConvEncoder, FSEncoder, L1Energy
from plot import plot_rect
from gradient_iterators import langevin_sample
from utils import time_print


class Model(nn.Module):
    def __init__(self, cfg, pool=None):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.global_step = 0
        self.pool = pool
        self.cfg = cfg

        self.img_enc = ConvEncoder(**self.cfg.img_enc)
        # this set encoder differs from the paper and does not apply RN,
        # leading to much faster training times at almost the same performance
        self.set_enc = FSEncoder(**self.cfg.set_enc)
        self.energy_fn = L1Energy(self.set_enc)
        self.optimizers = self.configure_optimizer()
        self._init_datasets()

    def forward(self, batch, cfg_sample):
        img, set = batch

        enc_x = self.img_enc(img.to(self.device))
        y_T, report = langevin_sample(
            self.energy_fn, 
            enc_x,
            set_size=set.size(1),
            d_y=self.cfg.set_enc.d_in,
            **cfg_sample,
            device=self.device)

        return y_T, enc_x, report

    def loss_and_metrics(self, batch, y_T, enc_x):
        _, set = batch
        set = set.to(self.device)

        noise_target = set.clone().detach()
        if self.cfg.estim.data_noise:
            noise_target += self.cfg.estim.eps_d * torch.randn_like(set)
 
        e_p = self.energy_fn(noise_target, enc_x).mean()
        e_n = self.energy_fn(y_T, enc_x).mean()

        card_p = (y_T[...,:3] > 0.015).all(2).sum(1).float()
        card_t = (set[...,:3] > 0.015).all(2).sum(1).float()
        rmse = ((card_p - card_t) ** 2).mean(0).sqrt()

        loss = e_p - e_n

        return dict(loss=loss,
                    e_n=e_n,
                    e_p=e_p,
                    rmse=rmse)

    def compose_report(self, loss_report, sample_report):
        report = {**loss_report, "gradnorm": sample_report["gradnorm"]}
        report["first_energy"] = sample_report["energies"][0].mean()
        report["last_energy"] = sample_report["energies"][-1].mean()

        return {k: v.item() if hasattr(v, "item") else v for k,v in report.items()}


    def training_step(self, batch, batch_nb):
        y_T, enc_x, sample_report = self.forward(batch, self.cfg.sample)
        loss_report = self.loss_and_metrics(batch, y_T, enc_x)

        report = self.compose_report(loss_report, sample_report)

        loss = loss_report['loss']

        for opt in self.optimizers.values():
            opt.zero_grad()

        loss.backward()

        for opt in self.optimizers.values():
            opt.step()

        if self.global_step % self.cfg.log.plt_freq == 0:
            try:
                plot_rect(batch[1], batch[0], n_sets=16,
                    filepath=self.cfg.log.misc_dir / f"{self.global_step}_gt.png")
                plot_rect(y_T, batch[0], n_sets=16,
                    filepath=self.cfg.log.misc_dir / f"{self.global_step}_samples.png")
            except Exception as e:
                time_print(f"Error while plotting: {e}")

        self.global_step += 1

        return report

    def eval_step(self, batch, batch_nb, mode="test"):
        y_T, enc_x, sample_report = self.forward(batch, self.cfg.predict)
        loss_report = self.loss_and_metrics(batch, y_T, enc_x)

        report = self.compose_report(loss_report, sample_report)

        if self.cfg.eval_loss == "chamfer":
            report["performance"] = chamfer_distance(batch[1].to(self.device), y_T).mean(0)
        elif self.cfg.eval_loss == "lap":
            report["performance"] = LAP_loss(batch[1].to(self.device), y_T, pool=self.pool).mean(0)

        if batch_nb == 0:
            plot_rect(batch[1], batch[0], n_sets=16,
                    filepath=self.cfg.log.misc_dir / f"{self.global_step}_{mode}_gt.png")
            plot_rect(y_T, batch[0], n_sets=16,
                    filepath=self.cfg.log.misc_dir / f"{self.global_step}_{mode}_samples.png")
        return report

    def configure_optimizer(self):
        param = self.parameters()
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, param), lr=self.cfg.optim.lr)
        return dict(opt=opt)

    def get_dataset(self, mode, **kwargs):
        cfg = self.cfg.data.copy()
        name = cfg.pop("name")
        DataSet = getattr(importlib.import_module("datasets"), name, None)
        assert DataSet is not None, f"{name}"
        ds = DataSet(mode=mode, **cfg, **kwargs)
        return ds

    def _init_datasets(self):
        ds = self.get_dataset(mode="train", **getattr(self.cfg.train, "kwargs", {}))
        train_size = int(len(ds) * self.cfg.train.train_val_split)
        val_size = len(ds) - train_size
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(ds, [train_size, val_size])
        self.dataset_test = self.get_dataset(mode="val", **getattr(self.cfg.test, "kwargs", {}))

    def get_train_loader(self):
        return torch.utils.data.DataLoader(
                self.dataset_train,
                shuffle=True,
                batch_size=self.cfg.train.batch_size,
                pin_memory=True,
                num_workers=self.cfg.train.num_workers,
                drop_last=True,
            )

    def get_val_loader(self):
        return torch.utils.data.DataLoader(
                self.dataset_val,
                shuffle=False,
                batch_size=self.cfg.train.batch_size,
                pin_memory=True,
                num_workers=self.cfg.train.num_workers,
                drop_last=True,
            )

    def get_test_loader(self):
        return torch.utils.data.DataLoader(
                self.dataset_test,
                shuffle=False,
                batch_size=self.cfg.test.batch_size,
                pin_memory=True,
                num_workers=self.cfg.test.num_workers,
                drop_last=True,
            )

    def load_ckpt(self, model_file=None):
        if not self.cfg.load_model and model_file is None:
            time_print(f"Not loading from ckpt.")
            return self

        model_file = getattr(self.cfg, "model_file")
        time_print(f"Loading ckpt file {model_file}")

        ckpt = torch.load(model_file)
        self.load_state_dict(ckpt["state_dict"], strict=False)
        for k,v in ckpt["opt_state_dict"].items():
            self.optimizers[k].load_state_dict(v)
        return self