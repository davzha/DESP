import importlib
from random import sample

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from loss import chamfer_distance, LAP_loss, RotatedPolygonLoss, DigitLoss
from modules import DSEnergy
from plot import plot_set
from gradient_iterators import ClippedGradientDescent
from utils import time_print


class Model(nn.Module):
    def __init__(self, cfg, pool=None):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.global_step = 0
        self.pool = pool
        self.cfg = cfg

        self.energy_fn = DSEnergy(**self.cfg.energy_fn)
        self.cgd = ClippedGradientDescent(
                        self.energy_fn, 
                        n_points=self.cfg.data.n_points,
                        d_y=self.cfg.energy_fn.d_y,
                        T=self.cfg.cgd.T,
                        eps=self.cfg.cgd.eps,
                        norm_type=self.cfg.cgd.norm_type,
                        max_norm=self.cfg.cgd.max_norm)

        self.performance_metric = self.get_performance_metric(cfg.eval_loss)
        self.optimizers = self.configure_optimizer()

    def get_performance_metric(self, loss_type):
        if loss_type == "chamfer":
            set_loss_fn = chamfer_distance
        else: 
            def set_loss_fn(x,y):
                return LAP_loss(x, y, pool=self.pool)
        
        if self.cfg.data.name == "Polygons":
            loss_fn = RotatedPolygonLoss(self.cfg.data.n_poly, set_loss_fn)
        else:
            ds = self.get_dataset(mode="train", **self.cfg.train.kwargs)
            
            digit_examples = torch.stack([
                ds._build_one(False),
                ds._build_one(True),
                ds._build_seven(False),
                ds._build_seven(True),
            ])
            n = torch.Tensor([0,0,1,1]).long()
            loss_fn = DigitLoss(digit_examples, n, set_loss_fn)
        return loss_fn

    def _set_loss(self, pred, target):
        if self.cfg.train_loss == "chamfer":
            loss_set = [chamfer_distance(target, y).mean(0) for y in pred]
        else:
            loss_set = [LAP_loss(target, y, pool=self.pool).mean(0) for y in pred]

        return loss_set

    def one_hot_x(self, inputs):
        if self.cfg.data.name == "Polygons":
            inputs = inputs - self.cfg.data.n_poly[0]
            num_classes = len(self.cfg.data.n_poly)
        else:
            num_classes = 2

        x = F.one_hot(inputs.to(self.device), 
                num_classes=num_classes).float()

        return x

    def forward(self, batch, cfg_sample):
        inputs, targets = batch
        bsize = inputs.size(0)
        
        x = self.one_hot_x(inputs)

        ys, report = self.cgd(x)

        return ys, report

    def loss_and_metrics(self, batch, ys):
        _, targets = batch
        targets = targets.to(self.device)

        loss_set = self._set_loss(ys[-1:], targets)
        loss = sum(loss_set) / len(loss_set)

        card_p = (ys[-1] > 0.015).all(2).sum(1).float()
        card_t = (targets > 0.015).all(2).sum(1).float()
        rmse = ((card_p - card_t) ** 2).mean(0).sqrt()

        report = dict(
            loss=loss,
            rmse=rmse,
            # loss_set_first=loss_set[0],
            # loss_set_last=loss_set[-1]
        )

        return report

    def compose_report(self, loss_report, sample_report):
        report = {**loss_report, "gradnorm": sample_report["gradnorm"]}
        report["first_energy"] = sample_report["energies"][0].mean()
        report["last_energy"] = sample_report["energies"][-1].mean()

        return {k: v.item() if hasattr(v, "item") else v for k,v in report.items()}

    def training_step(self, batch, batch_nb):
        ys, sample_report = self.forward(batch, self.cfg.cgd)
        loss_report = self.loss_and_metrics(batch, ys)

        report = self.compose_report(loss_report, sample_report)
        
        loss = loss_report["loss"]

        for opt in self.optimizers.values():
            opt.zero_grad()

        loss.backward()

        for opt in self.optimizers.values():
            opt.step()

        if self.global_step % self.cfg.log.plt_freq == 0:
            try:
                plot_set(batch[1], labels=batch[0], filepath=self.cfg.log.misc_dir / f"{self.global_step}_gt.png", n_sets=self.cfg.train.batch_size)
                plot_set(ys[-1], labels=batch[0], filepath=self.cfg.log.misc_dir / f"{self.global_step}_samples.png", n_sets=self.cfg.train.batch_size)
            except Exception as e:
                time_print(f"Error while plotting: {e}")

        self.global_step += 1

        return report

    def test_step(self, batch, batch_nb):
        ys, sample_report = self.forward(batch, self.cfg.cgd)
        loss_report = self.loss_and_metrics(batch, ys)

        report = self.compose_report(loss_report, sample_report)
        report["performance"] = self.performance_metric(ys[-1], batch[0].to(self.device)).mean(0)

        if batch_nb == 0:
            plot_set(batch[1], filepath=self.cfg.log.misc_dir / f"{self.global_step}_test_gt.png", n_sets=self.cfg.test.batch_size)
            plot_set(ys[-1], filepath=self.cfg.log.misc_dir / f"{self.global_step}_test_samples.png", n_sets=self.cfg.test.batch_size)
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

    def get_train_loader(self):
        return torch.utils.data.DataLoader(
                self.get_dataset(mode="train", **getattr(self.cfg.train, "kwargs", {})),
                shuffle=True,
                batch_size=self.cfg.train.batch_size,
                pin_memory=True,
                num_workers=self.cfg.train.num_workers,
                drop_last=True,
            )

    def get_test_loader(self):
        return torch.utils.data.DataLoader(
                self.get_dataset(mode="test", **getattr(self.cfg.test, "kwargs", {})),
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

