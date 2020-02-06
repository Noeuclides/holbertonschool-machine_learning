#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# your code here
x = np.arange(3)
width = 0.5
p1 = plt.bar(x, fruit[0], width, color='red')
p2 = plt.bar(x, fruit[1], width, color='yellow', bottom=fruit[0])
p3 = plt.bar(x, fruit[2], width, color='orange', bottom=fruit[0] + fruit[1])
p4 = plt.bar(x, fruit[3], width, color='#ffe5b4', bottom=fruit[0] +
             fruit[1] +
             fruit[2])

plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.xticks(x, ('Farrah', 'Fred', 'Felicia'))
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1, p2, p3, p4),
           ('apples', 'bananas', 'oranges', 'peaches'))
plt.show()
