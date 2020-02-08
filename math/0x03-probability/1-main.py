#!/usr/bin/env python3

import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
p1 = Poisson(data)
print('P(9):', p1.pmf(9))

p2 = Poisson(lambtha=5)
print('P(9):', p2.pmf(9))

np.random.seed(1)
lam = np.random.uniform(0.1, 10.0)
n = np.random.randint(100, 1000)
data = np.random.poisson(lam, n).tolist()
p = Poisson(data)
print(p.pmf(-1))
print(p.pmf(-1.5))

np.random.seed(1)
lam = np.random.uniform(0.1, 10.0)
n = np.random.randint(100, 1000)
data = np.random.poisson(lam, n).tolist()
p = Poisson(data)
k = np.random.randint(1, 11)
print(np.around(p.pmf(k), 10))
k = np.random.uniform(0.5, 10.0)
print(np.around(p.pmf(k), 10))
