#!/usr/bin/env python3

import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
p1 = Poisson(data)
print('Lambtha:', p1.lambtha)

p2 = Poisson(lambtha=5)
print('Lambtha:', p2.lambtha)

np.random.seed(1)
lam = np.random.uniform(0.1, 10.0)
n = np.random.randint(100, 1000)
data = np.random.poisson(lam, n).tolist()
p = Poisson(data)
k = np.random.randint(1, 11)
print(k)
print(np.around(p.pmf(k), 10))
k = np.random.uniform(0.5, 10.0)
print(k)
print(np.around(p.pmf(k), 10))
