import torch
from denoising_diffusion_pytorch import (
    Unet1D,
    GaussianDiffusion1D,
    Trainer1D,
    Dataset1D,
)
import numpy as np
from matplotlib import pyplot as plt

# input dim = action dim
model = Unet1D(dim=256, dim_mults=(1, 2, 4), channels=8)
diffusion = GaussianDiffusion1D(
    model, seq_length=64, timesteps=100, objective="pred_noise"
)

# load data with numpy
training_seq_np = np.load("../figure/ant/positional/uss.npy")
training_seq = torch.from_numpy(training_seq_np).float().moveaxis(-1, -2)
# training_seq = torch.randn(128, 8, 64) * 0.05
dataset = Dataset1D(
    training_seq
)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

# Or using trainer

trainer = Trainer1D(
    diffusion,
    dataset=dataset,
    train_batch_size=256,
    # train_lr=1e-4,
    train_lr=8e-5,
    train_num_steps=10000,  # total training steps
    # train_num_steps=700000,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=True,  # turn on mixed precision
)
trainer.train()

# after a lot of training

sampled_seq = diffusion.sample(batch_size=4)
uss_sampled = sampled_seq.moveaxis(-1, -2).cpu().numpy()
np.save("../figure/ant/positional/uss_diffused.npy", uss_sampled)
