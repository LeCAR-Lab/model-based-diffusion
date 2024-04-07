from jax import numpy as jnp
import jax
from matplotlib import pyplot as plt

T = 60

def get_alpha_bar(t):
    # x = (t / T - 0.5) * 8.0
    x = t/T*8.0 - 4.0
    return jax.nn.sigmoid(-x)

ts = jnp.arange(T+1)
alpha_bars = get_alpha_bar(ts)
# alpha[i] = alpha_bar[i]/alpha_bar[i-1]
alphas = alpha_bars / jnp.roll(alpha_bars, 1)

var_yi = (1-alpha_bars)

plt.plot(alpha_bars[1:], label='alpha_bar')
plt.plot(alphas[1:], label='alpha')
plt.plot(var_yi[1:], label='var_yi')
plt.legend()
plt.xlabel('t')
plt.ylabel('value')
plt.show()