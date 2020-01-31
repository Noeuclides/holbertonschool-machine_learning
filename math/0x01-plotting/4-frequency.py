#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here
n, bins, patches = plt.hist(student_grades, 50, density=1, alpha=0.75)
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')
plt.show()
