# PyTorch_CustomExponentialLR
extension of pytorch exponential learning rate scheduler
- can use learning rate warmup
- can use periodic learning rate schedule

## Usage

```python
from custom_scheduler import CustomExponentialLR

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CustomExponentialLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_down=40, step_size_up=10)

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

from custom_scheduler import CustomExponentialLR
    
import torch
import torch.optim as optim

epochs = 100
optimizer = optim.SGD([torch.tensor(1)], lr=0.001, momentum=0.9)
scheduler = CustomExponentialLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_down=40, step_size_up=10)

lrs = visualize_scheduler(optimizer, scheduler, epochs)
```

![image](https://user-images.githubusercontent.com/55170796/184799523-1489bee7-016d-4ede-8aa5-376a70bd154f.png)
