from matplotlib import pyplot as plt
import numpy as np

alpha = 0.9

def f(t):
    return 1/np.sqrt(alpha) - 0.5 * (1-alpha) / np.sqrt(1-alpha**t)

t = np.arange(0, 100, 1)

plt.plot(t, f(t))
plt.show()