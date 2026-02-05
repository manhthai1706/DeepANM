# -*- coding: utf-8 -*-
"""
Synthetic Data Generator - PyTorch Implementation
High-performance generation for testing causal models.
"""

import torch
import numpy as np

def gen_each(inlist, device='cpu'):
    """
    inlist[0] - dist of x (1: Rand, 2: Normal, 3: Exp, 4: Laplace, 5: LogNormal)
    inlist[1] - f1 (0: identity, 1: exp1, 2: exp2)
    inlist[2] - dist of noise (0: None, 1: Rand, 2: Normal, 3: Exp...)
    inlist[3] - f2 (0: identity, 1: 1/(t^2+1), 2: t^2, 3: sin, 4: cos...)
    inlist[4] - number of pts
    """
    n = inlist[4]
    
    # 1. Generate X
    if inlist[0] == 1:
        x = torch.rand(n, 1, device=device)
    elif inlist[0] == 2:
        x = torch.randn(n, 1, device=device)
    elif inlist[0] == 3:
        x = torch.distributions.Exponential(2.0).sample((n, 1)).to(device)
    elif inlist[0] == 4:
        x = torch.distributions.Laplace(0, 1).sample((n, 1)).to(device)
    else:
        x = torch.randn(n, 1, device=device).exp()

    # 2. Intermediate f1
    if inlist[1] == 0:
        f1 = -x
    elif inlist[1] == 1:
        f1 = torch.exp(-(1.0 + 0.1 * torch.rand(1, device=device)) * x)
    else:
        f1 = torch.exp(-(3.0 + 0.1 * torch.rand(1, device=device)) * x)

    # 3. Add Noise
    if inlist[2] == 0:
        t = f1
    elif inlist[2] == 1:
        t = f1 + 0.2 * torch.rand(n, 1, device=device)
    elif inlist[2] == 2:
        t = f1 + 0.05 * torch.randn(n, 1, device=device)
    elif inlist[2] == 3:
        t = f1 + torch.distributions.Exponential(2.0).sample((n, 1)).to(device)
    else:
        t = f1 + torch.distributions.Laplace(0, 1).sample((n, 1)).to(device)

    # 4. Final mapping f2
    if inlist[3] == 0:
        y = t
    elif inlist[3] == 1:
        y = 1.0 / (t**2 + 1.0)
    elif inlist[3] == 2:
        y = t**2
    elif inlist[3] == 3:
        y = torch.sin(t)
    elif inlist[3] == 4:
        y = torch.cos(t)
    else:
        y = torch.pow(t, 1/3)

    return torch.cat([x, y], dim=1)

def gen_D(inlist, device='cpu'):
    """Generate Specified Mixture Dataset"""
    data_list = []
    labels = []
    
    for i, config in enumerate(inlist):
        data_part = gen_each(config, device=device)
        data_list.append(data_part)
        labels.extend([i] * config[4])
        
    data = torch.cat(data_list, dim=0)
    return data.cpu().numpy(), np.array(labels)

if __name__ == '__main__':
    # Test generation
    data, labels = gen_D([[1, 1, 2, 0, 100], [2, 0, 1, 3, 100]])
    print(f"Generated data shape: {data.shape}")
    print(f"Unique labels: {np.unique(labels)}")
	