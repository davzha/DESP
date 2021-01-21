import shutil
import time
from pathlib import Path

import torch

def get_git_state():
    import git
    repo = git.Repo(search_parent_directories=True)
    try:
        branch_name = repo.active_branch.name
    except Exception:
        branch_name = "NONE"

    return branch_name, repo.head.object.hexsha

def time_print(*args, **kwargs):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *args, **kwargs)

def save_checkpoint(state, is_best=True, directory=".", filename='checkpoint.pth.tar'):
    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)

    filepath = str(directory / filename)

    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, str(directory / 'model_best.pth.tar'))



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, print_fn=print):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_fn('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
