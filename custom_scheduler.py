from torch.optim.lr_scheduler import LambdaLR

class ExponentialLRWarmUpRestarts(LambdaLR):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
        warmup_steps (int): number of epochs for warmup
    """
    def __init__(self, optimizer, gamma, last_epoch=-1, warmup_steps=0):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            else:
                return gamma**(step-warmup_steps)
        super(ExponentialLRWarmUpRestarts, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)
