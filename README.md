# PyTorch_CustomExponentialLR
extension of pytorch exponential learning rate scheduler
- can use learning rate warmup by setting `step_size_down` > 0
- can use periodic learning rate schedule by setting `step_size_up` and `step_size_down`
- can use decaying periodic learning rate schedule by setting `max_lr_decay' < 1.


## Usage

```python
from custom_scheduler import CustomExponentialLR

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CustomExponentialLR(optimizer, base_lr=0.0001, max_lr=0.001, max_lr_decay=0.9, step_size_down=40, step_size_up=10)

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
    
```

### 1. Exponential decay with warmup
```python
from custom_scheduler import CustomExponentialLR
    
import torch
import torch.optim as optim

epochs = 100
optimizer = optim.SGD([torch.tensor(1)], lr=0.001, momentum=0.9)
scheduler = CustomExponentialLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_down=90, step_size_up=10)

lrs = visualize_scheduler(optimizer, scheduler, epochs)
```

![image](https://user-images.githubusercontent.com/55170796/184800645-7fe21a70-edb1-45b3-9113-18db77e22eef.png)


### 2. Periodic exponential decay with warmup
```python
from custom_scheduler import CustomExponentialLR
    
import torch
import torch.optim as optim

epochs = 100
optimizer = optim.SGD([torch.tensor(1)], lr=0.001, momentum=0.9)
scheduler = CustomExponentialLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_down=40, step_size_up=10)

lrs = visualize_scheduler(optimizer, scheduler, epochs)
```

![image](https://user-images.githubusercontent.com/55170796/184800711-7d5350ac-d705-4131-9aed-3e2ec0882ebb.png)


### 3. Periodic, decaying exponential decay with warmup
```python
from custom_scheduler import CustomExponentialLR
    
import torch
import torch.optim as optim

epochs = 100
optimizer = optim.SGD([torch.tensor(1)], lr=0.001, momentum=0.9)
scheduler = CustomExponentialLR(optimizer, base_lr=0.0001, max_lr=0.001, max_lr_decay=0.8, step_size_down=40, step_size_up=10)

lrs = visualize_scheduler(optimizer, scheduler, epochs)
```

![image](https://user-images.githubusercontent.com/55170796/184800759-1206c264-723d-4fab-81fb-3c663d81be78.png)
