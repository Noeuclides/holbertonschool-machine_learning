#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

# your code here
x = np.arange(3)
width = 0.5
p1 = plt.bar(x, np.vectorize(fruit), width)
p2 = plt.bar(x, fruit, width)
p3 = plt.bar(x, fruit, width)
p4 = plt.bar(x, fruit, width)

plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.xticks(x, ('Farrah', 'Fred', 'Felicia'))
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('apples', 'bananas', 'oranges', 'peaches'))
