from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, SequentialLR


def lerp(a: float, b: float, t: float) -> float:
    """Linearly interpolate between a and b."""
    return a + (b - a) * t


class WarmUpScheduler(LRScheduler):
    """Increase learning rate linearly from 0 to the target LR over the first few steps."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        start_lr: float,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                lerp(self.start_lr, base_lr, self.last_epoch / self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            return [base_lr for base_lr in self.base_lrs]


class WarmUpCosineAnnealing(SequentialLR):
    """Cosine annealing with linear warm up."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        start_lr: float,
        end_lr: float,
        last_epoch: int = -1,
    ):
        schedulers = [
            WarmUpScheduler(optimizer, warmup_steps, start_lr, last_epoch),
            CosineAnnealingLR(
                optimizer,
                T_max=max_steps - warmup_steps,
                eta_min=end_lr,
                last_epoch=max(-1, last_epoch - warmup_steps),
            ),
        ]
        milestones = [warmup_steps]

        super().__init__(optimizer, schedulers, milestones, last_epoch=last_epoch)
