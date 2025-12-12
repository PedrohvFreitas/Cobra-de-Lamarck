import numpy as np
import matplotlib.pyplot as plt
from model import Linear_QNet, QTrainer

mean = 0
std_dev = 1000

big_gaussian = np.random.normal(loc=0.0, scale=10, size=2000)
small_gaussian = np.random.normal(loc=0.0, scale=1.0, size=2000)

plt.hist(big_gaussian, bins=30, density=True, alpha=0.6, color='r')
plt.hist(small_gaussian, bins=30, density=True, alpha=0.6, color='b')
plt.title('Gaussian Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()
