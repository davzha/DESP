import argparse
import datetime
import importlib
import pprint
import time
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils import get_git_state, time_print, AverageMeter, ProgressMeter, save_checkpoint


def train(cfg, data_loader, model):
    data_time = AverageMeter("Data", ":6.3f")
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    
    progress = ProgressMeter(
        len(data_loader)-1,
        [batch_time, data_time, losses],
        prefix=f"Train:\t")
    
    model.train()
    
    end = time.time()
    for batch_nb, batch in enumerate(data_loader):
        d_time = time.time() - end
        data_time.update(d_time)

        global_step = model.global_step
        writer.add_scalar("time/data/train", d_time, global_step)

        report = model.training_step(batch, batch_nb)
        losses.update(report["loss"])

        for k, v in report.items():
            writer.add_scalar(f"{k}/train", v, global_step)

        b_time = time.time() - end
        batch_time.update(b_time)
        writer.add_scalar("time/batch/train", b_time, global_step)
        end = time.time()

        if batch_nb % cfg.log.freq == 0 or batch_nb == len(data_loader) - 1:
            progress.display(batch_nb, print_fn=lambda *x: time_print(*x, end="\r"))

def eval(cfg, data_loader, model, mode="test"):
    data_time = AverageMeter("Data", ":6.3f")
    batch_time = AverageMeter("Time", ":6.3f")
    metrics = ["performance"]
    metrics = {m: AverageMeter(m, ":.4e") for m in metrics}

    progress = ProgressMeter(
        len(data_loader)-1,
        [batch_time, data_time, *metrics.values()],
        prefix=f"{mode.capitalize()}:\t")

    model.eval()

    global_step = model.global_step

    end = time.time()
    for batch_nb, batch in enumerate(data_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            report = model.eval_step(batch, batch_nb, mode=mode)

        for k, v in report.items():
            if k not in metrics:
                metrics[k] = AverageMeter(k, ":.3f")
            metrics[k].update(v)
        
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_nb % cfg.log.freq == 0 or batch_nb == len(data_loader) - 1:
            progress.display(batch_nb, print_fn=lambda *x: time_print(*x, end="\r"))

    writer.add_scalar(f"time/batch/{mode}", batch_time.avg, global_step)
    writer.add_scalar(f"time/data/{mode}", data_time.avg, global_step)

    for k,v in metrics.items():
        writer.add_scalar(f"{k}/test", v.avg, global_step)

    progress.display(len(data_loader) - 1, time_print)

    return metrics["performance"].avg


def main(cfg, pool=None):
    model = importlib.import_module(f"models.{cfg.model}").Model(cfg, pool=pool)

    if getattr(cfg, "load_model", False):
        model.load_ckpt()

    if model.device != "cpu" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(model.device)

    train_loader = model.get_train_loader()
    val_loader = model.get_val_loader()
    test_loader = model.get_test_loader()

    best_val_score = np.inf
    early_stopping_count = 0

    val_compare = lambda c,b:c < b if cfg.val_compare == "<" else lambda c,b:c > b

    for epoch in range(cfg.num_epoch):
        time_print(f"\nEpoch {epoch} Training")
        train(cfg, train_loader, model)

        time_print(f"\nEpoch {epoch} Validation")
        val_score = eval(cfg, val_loader, model, mode="val")
            
        filename = "checkpoint.pth.tar"
        if not getattr(cfg.log, "overwrite_ckpt", True):
            filename = "_".join([str(epoch), filename])

        save_checkpoint(
            state={
                "epoch": epoch,
                "global_step": model.global_step,
                "state_dict": model.state_dict(),
                "opt_state_dict": {k: v.state_dict() for k,v in model.optimizers.items()},
                "cfg": cfg,
            },
            is_best=val_score < best_val_score,
            directory=cfg.log.misc_dir,
            filename=filename)
        
        if val_compare(val_score, best_val_score):
            best_val_score = val_score
            early_stopping_count = 0
        elif early_stopping_count < cfg.early_stopping:
            early_stopping_count += 1
        else:
            time_print(f"\nEarly stopping at epoch [{epoch}] due to no improvement for {early_stopping_count} epochs")
            break
    
    time_print("\nTest")
    eval(cfg, test_loader, model, mode="test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run script")
    parser.add_argument("--config", "-c",type=str, required=False, default="config")
    args = parser.parse_args()
    git_state = get_git_state()
    config = importlib.import_module(f"configs.{args.config}").config
    config.log.exp_id = git_state[1][:7] + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    config.log.misc_dir = config.log.dir / "misc" / config.log.exp_id
    config.log.tb_dir = config.log.dir / "tb" / config.log.exp_id
    config.log.misc_dir.mkdir(exist_ok=True, parents=True)
    config.log.tb_dir.mkdir(exist_ok=True, parents=True)

    torch.manual_seed(config.rnd_seed)
    np.random.seed(config.rnd_seed)
    random.seed(config.rnd_seed)

    if getattr(config, "anomaly_detection", False):
        torch.autograd.set_detect_anomaly(True)

    global writer
    writer = SummaryWriter(
        log_dir=config.log.tb_dir,
        comment=f"{config.description}, {git_state}")

    time_print(pprint.pformat(config))
    time_print(f"Git head at state: {git_state}")

    try:
        if npp:=getattr(config, "n_process_pool", 0):
            with torch.multiprocessing.Pool(npp) as pool:
                main(config, pool=pool)
        else:
            main(config)
    except KeyboardInterrupt:
        time_print(f"Keyboard interrupt")
        exit(0)