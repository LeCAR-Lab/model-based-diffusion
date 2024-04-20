from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import math
import scipy.interpolate
import torch
import torch.nn as nn
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import scipy

# env import
import gym
from gym import spaces
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from skvideo.io import vwrite
import gdown
import os

from huggingface_hub.utils import IGNORE_GIT_FOLDER_PATTERNS

# @markdown ### **Dataset**
# @markdown
# @markdown Defines `PushTStateDataset` and helper functions
# @markdown
# @markdown The dataset class
# @markdown - Load data (obs, action) from a zarr storage
# @markdown - Normalizes each dimension of obs and action to [-1,1]
# @markdown - Returns
# @markdown  - All possible segments with length `pred_horizon`
# @markdown  - Pads the beginning and the end of each episode with repetition
# @markdown  - key `obs`: shape (obs_horizon, obs_dim)
# @markdown  - key `action`: shape (pred_horizon, action_dim)


def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    indices = np.array(indices)
    return indices


def sample_sequence(
    train_data,
    sequence_length,
    buffer_start_idx,
    buffer_end_idx,
    sample_start_idx,
    sample_end_idx,
):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


# normalize data
def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    return stats


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data


# dataset
class PushTStateDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, pred_horizon, obs_horizon, action_horizon):

        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, "r")
        # All demonstration episodes are concatinated in the first dimension N
        train_data = {
            # (N, action_dim)
            "action": dataset_root["data"]["action"][:],
            # (N, obs_dim)
            "obs": dataset_root["data"]["state"][:],
        }
        # Marks one-past the last index for each episode
        episode_ends = dataset_root["meta"]["episode_ends"][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
            self.indices[idx]
        )

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        # discard unused observations
        nsample["obs"] = nsample["obs"][: self.obs_horizon, :]
        return nsample


# @markdown ### **Dataset Demo**

# download demonstration data from Google Drive
dataset_path = "pusht_cchi_v7_replay.zarr.zip"
if not os.path.isfile(dataset_path):
    id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
    gdown.download(id=id, output=dataset_path, quiet=False)

# parameters
pred_horizon = 16
obs_horizon = 2
action_horizon = 8
# |o|o|                             observations: 2
# | |a|a|a|a|a|a|a|a|               actions executed: 8
# |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

# create dataset from file
dataset = PushTStateDataset(
    dataset_path=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon,
)
# save training data statistics (min, max) for each dim
stats = dataset.stats

# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    num_workers=1,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True,
)

# visualize data in batch
batch = next(iter(dataloader))
print("batch['obs'].shape:", batch["obs"].shape)
print("batch['action'].shape", batch["action"].shape)

# @markdown ### **Network**
# @markdown
# @markdown Defines a 1D UNet architecture `ConditionalUnet1D`
# @markdown as the noies prediction network
# @markdown
# @markdown Components
# @markdown - `SinusoidalPosEmb` Positional encoding for the diffusion iteration k
# @markdown - `Downsample1d` Strided convolution to reduce temporal resolution
# @markdown - `Upsample1d` Transposed convolution to increase temporal resolution
# @markdown - `Conv1dBlock` Conv1d --> GroupNorm --> Mish
# @markdown - `ConditionalResidualBlock1D` Takes two inputs `x` and `cond`. \
# @markdown `x` is passed through 2 `Conv1dBlock` stacked together with residual connection.
# @markdown `cond` is applied to `x` with [FiLM](https://arxiv.org/abs/1709.07871) conditioning.


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(), nn.Linear(cond_dim, cond_channels), nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
    ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print(
            "number of parameters: {:e}".format(
                sum(p.numel() for p in self.parameters())
            )
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond=None,
    ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1, -2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1, -2)
        # (B,T,C)
        return x


# @markdown ### **Network Demo**

# observation and action dimensions corrsponding to
# the output of PushTEnv
obs_dim = 5
action_dim = 2

# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim, global_cond_dim=obs_dim * obs_horizon
)

# example inputs
noised_action = torch.randn((1, pred_horizon, action_dim))
obs = torch.zeros((1, obs_horizon, obs_dim))
diffusion_iter = torch.zeros((1,))

# the noise prediction network
# takes noisy action, diffusion iteration and observation as input
# predicts the noise added to action
noise = noise_pred_net(
    sample=noised_action, timestep=diffusion_iter, global_cond=obs.flatten(start_dim=1)
)

# illustration of removing noise
# the actual noise removal is performed by NoiseScheduler
# and is dependent on the diffusion noise schedule
denoised_action = noised_action - noise

# for this demo, we use DDPMScheduler with 100 diffusion iterations
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule="squaredcos_cap_v2",
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type="epsilon",
)

# device transfer
device = torch.device("cuda")
_ = noise_pred_net.to(device)

# @markdown ### **Training**
# @markdown
# @markdown Takes about an hour. If you don't want to wait, skip to the next cell
# @markdown to load pre-trained weights
"""
num_epochs = 30

# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(
    parameters=noise_pred_net.parameters(),
    power=0.75)

# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=noise_pred_net.parameters(),
    lr=1e-4, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    # epoch loop
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                # data normalized in dataset
                # device transfer
                nobs = nbatch['obs'].to(device)
                naction = nbatch['action'].to(device)
                B = nobs.shape[0]

                # observation as FiLM conditioning
                # (B, obs_horizon, obs_dim)
                obs_cond = nobs[:,:obs_horizon,:]
                # (B, obs_horizon * obs_dim)
                obs_cond = obs_cond.flatten(start_dim=1)

                # sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = noise_scheduler.add_noise(
                    naction, noise, timesteps)

                # predict the noise residual
                noise_pred = noise_pred_net(
                    noisy_actions, timesteps, global_cond=obs_cond)

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(noise_pred_net.parameters())

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)
        tglobal.set_postfix(loss=np.mean(epoch_loss))
        if epoch_idx % 5 == 0:
            torch.save(noise_pred_net.state_dict(), f'pusht_state_{epoch_idx}ep.ckpt')

# Weights of the EMA model
# is used for inference
ema_noise_pred_net = noise_pred_net
ema.copy_to(ema_noise_pred_net.parameters())
# save the model
"""

# @markdown ### **Loading Pretrained Checkpoint**
# @markdown Set `load_pretrained = True` to load pretrained weights.
load_pretrained = True
if load_pretrained:
    ckpt_path = "pusht_state_100ep.ckpt"
    if not os.path.isfile(ckpt_path):
        id = "1mHDr_DEZSdiGo9yecL50BBQYzR8Fjhl_&confirm=t"
        gdown.download(id=id, output=ckpt_path, quiet=False)

    state_dict = torch.load(ckpt_path, map_location="cuda")
    ema_noise_pred_net = noise_pred_net
    ema_noise_pred_net.load_state_dict(state_dict)
    print("Pretrained weights loaded.")
else:
    print("Skipped pretrained weight loading.")

# @markdown ### **Inference**
# limit enviornment interaction to 200 steps before termination
max_steps = 200
env = PushTEnv()
# use a seed >200 to avoid initial states seen in the training dataset
env.seed(100000)

# get first observation
obs, info = env.reset()

# manually set initial state
# init_state = obs
# # init_state[:2] = env.goal_pose[:2] + np.array([-130, 80])
# init_state[:2] = env.goal_pose[:2] + np.array([-150, 80])
# init_state[2:4] = env.goal_pose[:2] + np.array([-20, 20])
# init_state[4] = env.goal_pose[2]
# env._set_state(init_state)
# obs = env._get_obs()
# visualization
# img = env.render(mode="rgb_array")
# plt.imshow(img)
# plt.show()

# keep a queue of last 2 steps of observations
obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
# save visualization and rewards
imgs = [env.render(mode="rgb_array")]
rewards = list()
actions = list()
obses = [obs]
done = False
step_idx = 0


def eval_actions(env, init_state, actions, render=False):
    env._set_state(init_state)
    rewards = []
    if render:
        imgs = [env.render(mode="rgb_array")]
    for i in range(len(actions)):
        obs, reward, done, _, info = env.step(actions[i])
        rewards.append(reward)
        if render:
            imgs.append(env.render(mode="rgb_array"))
        # if done:
        #     break
    rewards = np.array(rewards)
    env._set_state(init_state)
    # save visualization
    if render:
        vwrite(f"../figure/vis_{rewards.sum():.2f}.mp4", imgs)
    return rewards


fig, ax = plt.subplots()
with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
    while not done:
        B = 1
        # stack the last obs_horizon (2) number of observations
        obs_seq = np.stack(obs_deque)
        # normalize observation
        nobs = normalize_data(obs_seq, stats=stats["obs"])
        # device transfer
        nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

        # infer action
        with torch.no_grad():
            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn((B, pred_horizon, action_dim), device=device)
            naction = noisy_action

            # init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                # predict noise
                noise_pred = ema_noise_pred_net(
                    sample=naction, timestep=k, global_cond=obs_cond
                )

                # get current noise
                noise_var = noise_scheduler._get_variance(k)

                """
                MPPI to add extra score function

                print(f"Step: {k}")
                # rollout mppi
                if k < 10:
                    alpha_prod_t = noise_scheduler.alphas_cumprod[k]
                    beta_prod_t = 1 - alpha_prod_t
                    std = np.array(beta_prod_t ** (0.5))

                    init_naction = naction.detach().to("cpu").numpy()
                    best_naction = init_naction
                    best_reward = eval_actions(env, init_state, best_naction[0]).sum()
                    for p in range(100):
                        naction_noisy = best_naction + np.random.normal(0.0, std, best_naction.shape)
                        action_pred_noisy = unnormalize_data(naction_noisy[0], stats=stats["action"])
                        rewards = eval_actions(env, init_state, action_pred_noisy)
                        if rewards.sum() > best_reward:
                            best_reward = rewards.sum()
                            best_naction = naction_noisy
                            print("New Best Score: ", rewards.sum())
                    mppi_noise_pred = (init_naction - best_naction) / std
                    # move it to device
                    mppi_noise_pred = torch.from_numpy(mppi_noise_pred).to(device, dtype=torch.float32)

                    k_mppi = 1.0 - k / 100
                    noise_pred = k_mppi * mppi_noise_pred + (1-k_mppi) * noise_pred
                
                """
                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction
                ).prev_sample

        # unnormalize action
        naction = naction.detach().to("cpu").numpy()
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = unnormalize_data(naction, stats=stats["action"])

        # rewards = eval_actions(env, init_state, action_pred, render=True)
        # print("Score: ", rewards.sum())

        # only take action_horizon number of actions
        start = obs_horizon - 1
        end = start + action_horizon
        action = action_pred[start:end, :]
        # (action_horizon, action_dim)

        # execute action_horizon number of steps
        # without replanning
        rewards = []
        for i in range(len(action)):
            # stepping env
            obs, reward, done, _, info = env.step(action[i])

            actions.append(action[i])
            obses.append(obs)

            # save observations
            obs_deque.append(obs)
            # and reward/vis
            rewards.append(reward)
            imgs.append(env.render(mode="rgb_array"))

            # update progress bar
            step_idx += 1
            pbar.update(1)
            pbar.set_postfix(reward=reward)
            if step_idx > max_steps:
                done = True
            if done:
                break

# print out the maximum target coverage
print("Score: ", max(rewards))

# Save actions and observations
actions = np.array(actions)
obses = np.array(obses)
np.savez_compressed("../figure/data/actions_obses.npz", actions=actions, obses=obses)

# visualize
vwrite("../figure/vis.mp4", imgs)


"""
### replay the trajectory
data = np.load("../figure/data/actions_obses.npz")
actions = data["actions"]
obses = data["obses"]
rewards = []
env = PushTEnv()
max_steps = 200
env.seed(100001)
obs, info = env.reset()
imgs = [env.render(mode="rgb_array")]
assert np.allclose(obs, obses[0]), "Inconsistent initial state"
for i in range(len(actions)):
    obs, reward, done, _, info = env.step(actions[i])
    rewards.append(reward)
    imgs.append(env.render(mode="rgb_array"))
    if done:
        break
vwrite("../figure/replay.mp4", imgs)
"""

### running MPPI
"""
Nmppi = 100  # sample 100 trajectories
a_mean = env.window_size // 2
a_std = env.window_size // 5
data = np.load("../figure/data/actions_obses.npz")
actions = data["actions"]
Nnode = 20
# get nodes from actions
def generate_mppi_actions(actions, Nnode):
    actions_mppi = np.zeros((Nmppi, *actions.shape))
    for i in range(Nmppi):
        actions_node_mppi = actions[:: len(actions) // Nnode].copy()  # get every Nnode actions
        if i > 0:
            actions_node_mppi += np.random.normal(0.0, a_std, actions_node_mppi.shape)
            actions_node_mppi = np.clip(actions_node_mppi, 0, env.window_size)
        # interpolate between nodes with 2nd order spline with scipy
        actions_interp = scipy.interpolate.interp1d(
            np.arange(len(actions_node_mppi)),
            actions_node_mppi,
            kind="quadratic",
            axis=0,
        )(np.linspace(0, len(actions_node_mppi) - 1, len(actions)))

        actions_mppi[i] = actions_interp  # use the same action for the first trajectory
        actions_mppi = np.clip(actions_mppi, 0, env.window_size)
    return actions_mppi
actions_mppi = generate_mppi_actions(actions, Nnode)

rewards = []
for i in range(Nmppi):
    env.seed(100000)
    obs, info = env.reset()
    imgs = [env.render(mode="rgb_array")]
    rewards_local = []
    for j in range(len(actions)):
        obs, reward, done, _, info = env.step(actions_mppi[i, j])
        rewards_local.append(reward)
        imgs.append(env.render(mode="rgb_array"))
        if done:
            break
    rewards_local = np.array(rewards_local)
    rewards.append(np.sum(rewards_local))
    if rewards_local[-1] > 0.4:
        actions_mppi = generate_mppi_actions(actions_mppi[i], Nnode)
    print(f"Trajectory {i} reward: {rewards_local.sum()}, max reward: {np.max(rewards_local)}")
    if rewards_local.sum() > 0.0:
        vwrite(f"../figure/mppi_{i}.mp4", imgs)
"""
