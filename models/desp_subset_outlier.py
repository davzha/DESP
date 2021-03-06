import importlib
import torch
import torch.nn as nn

from modules import PretrainedConvEncoder, DSEnergy
from plot import plot_imgs
from gradient_iterators import langevin_sample
from utils import time_print
from loss import f1score


class Model(nn.Module):
    def __init__(self, cfg, pool=None):
        super().__init__()
        self.device = cfg.device
        self.global_step = 0
        self.pool = pool
        self.cfg = cfg

        self.img_enc = PretrainedConvEncoder(**self.cfg.img_enc)
        self.energy_fn = DSEnergy(**self.cfg.energy_fn)
        self.optimizers = self.configure_optimizer()

    def forward(self, batch, cfg_sample):
        imgs, _, _ = batch
        imgs = imgs.to(self.device)
        batch_size, set_size = imgs.shape[:2]

        imgs = imgs.flatten(0,1)
        enc_x = self.img_enc(imgs)
        enc_x = enc_x.reshape(batch_size, set_size, -1)

        y_T, report = langevin_sample(
            self.energy_fn,
            enc_x,
            set_size=set_size,
            d_y=self.cfg.energy_fn.d_y,
            **cfg_sample,
            device=self.device)

        return y_T, enc_x, report

    def loss_and_metrics(self, batch, y_T, enc_x):
        _, targets, all_targets = batch
        report = {}

        if self.training:
            targets = targets.to(self.device)
            targets = targets.unsqueeze(2)
            targets = targets - 0.5  # shift by 0.5

            noise_targets = targets.clone().detach()
            if self.cfg.estim.data_noise:
                noise_targets += self.cfg.estim.eps_d * torch.randn_like(targets)
    
            e_p = self.energy_fn(noise_targets, enc_x).mean()
            e_n = self.energy_fn(y_T, enc_x).mean()
            loss = e_p - e_n

            report["loss"] = loss
            report["e_p"] = e_p
            report["e_n"] = e_n

        if not self.training:
            y_T = y_T.reshape(len(all_targets), self.cfg.predict.k, y_T.size(1))
            # additional performance metrics
            F1 = torch.zeros(1, device=self.device, requires_grad=False)
            P = torch.zeros(1, device=self.device, requires_grad=False)
            R = torch.zeros(1, device=self.device, requires_grad=False)
            for i, at in enumerate(all_targets):
                _f1, _p, _r = f1score(y_T[i] > 0., at.to(self.device))
                F1 += _f1
                P += _p
                R += _r

            F1 /= len(all_targets)
            P /= len(all_targets)
            R /= len(all_targets)

            report["f1score"] = F1
            report["precision"] = P
            report["recall"] = R

        return report

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

        self.global_step += 1

        return report

    def eval_step(self, batch, batch_nb, mode="test"):
        y_T, enc_x, sample_report = self.forward(batch, self.cfg.predict)
        loss_report = self.loss_and_metrics(batch, y_T, enc_x)

        report = self.compose_report(loss_report, sample_report)
        report["performance"] = report["f1score"]

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

    def get_collate_fn(self):
        def collate_fn(batch):
            imgs, targets, all_targets = zip(*batch)
            return *torch.utils.data.dataloader.default_collate(list(zip(imgs, targets))), all_targets
        return collate_fn

    def get_train_loader(self):
        return torch.utils.data.DataLoader(
                self.get_dataset(mode="train", **getattr(self.cfg.train, "kwargs", {})),
                shuffle=True,
                batch_size=self.cfg.train.batch_size,
                pin_memory=True,
                num_workers=self.cfg.train.num_workers,
                drop_last=True,
                collate_fn=self.get_collate_fn()
            )

    def get_val_loader(self):
        return torch.utils.data.DataLoader(
                self.get_dataset(mode="val", **getattr(self.cfg.val, "kwargs", {})),
                shuffle=False,
                batch_size=self.cfg.val.batch_size,
                pin_memory=True,
                num_workers=self.cfg.val.num_workers,
                drop_last=True,
                collate_fn=self.get_collate_fn()
            )

    def get_test_loader(self):
        return torch.utils.data.DataLoader(
                self.get_dataset(mode="test", **getattr(self.cfg.test, "kwargs", {})),
                shuffle=False,
                batch_size=self.cfg.test.batch_size,
                pin_memory=True,
                num_workers=self.cfg.test.num_workers,
                drop_last=True,
                collate_fn=self.get_collate_fn()
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