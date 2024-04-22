from matplotlib import pyplot as plt
import numpy as np


def cosine_schedule(num_timesteps, s=0.008):
    def f(t):
        return np.cos((t / num_timesteps + s) / (1 + s) * 0.5 * np.pi) ** 2

    x = np.linspace(0, num_timesteps, num_timesteps + 1)
    alphas_cumprod = f(x) / f(np.array([0]))
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = np.clip(betas, 0.0001, 0.999)
    return betas


betas = cosine_schedule(100)

plt.plot(betas)
plt.show()
