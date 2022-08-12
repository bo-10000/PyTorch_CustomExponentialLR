# PyTorch_ExponentialLRWarmUpRestarts
extension for warmup restart of pytorch exponential learning rate scheduler

## Usage

```python
from custom_scheduler import ExponentialLRWarmUpRestarts

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ExponentialLRWarmUpRestarts(optimizer, gamma=0.97, warmup_steps=5)

for epoch in epochs:
    for x, target in dataloader:
        pred = model(x)
        loss = loss_f(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()
```

</br>

## Visualize

```python
import matplotlib.pyplot as plt

def visualize_scheduler(optimizer, scheduler, epochs):
    lrs = []
    for _ in range(epochs):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    plt.plot(lrs)
    plt.show()
    
    return lrs

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
    
import torch
import torch.optim as optim

epochs = 100
optimizer = optim.SGD([torch.tensor(1)], lr=0.001, momentum=0.9)
scheduler = ExponentialLRWarmUpRestarts(optimizer, 0.97, -1, 5)

lrs = visualize_scheduler(optimizer, scheduler, epochs)
```

![image](https://user-images.githubusercontent.com/55170796/184324061-0e02b309-a591-4851-a47c-42630cb3e187.png)
