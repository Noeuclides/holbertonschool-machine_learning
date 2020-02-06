#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here
x = np.arange(101, step=10)
y = np.arange(31, step=5)
plt.hist(student_grades, bins=x, edgecolor='black', linewidth=1.2)
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')
plt.axis([0, 100, 0, 30])
plt.xticks(x)
plt.yticks(y)
plt.show()
