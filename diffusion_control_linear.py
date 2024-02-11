import torch
import torch
import torch.nn as nn
import torch.distributions as dist
import matplotlib.pyplot as plt
import numpy as np
import os

def linear_beta_schedule(timesteps):
    beta_start = 2e-5
    beta_middle = 0.7
    beta_end = 1
    beta_seq_1 = torch.linspace(beta_start,beta_middle,timesteps//2).cuda()
    beta_seq_2 = torch.linspace(beta_middle, beta_end, timesteps//2).cuda()
    beta_seq = torch.cat([beta_seq_1,beta_seq_2])
    return beta_seq

def diffusion_parameters_schedule(beta_schedule,start_noise,diffusion_dt):
    alpha_schedule = 1 - beta_schedule
    alpha_accum = torch.cumprod(alpha_schedule, 0)
    g_seq = torch.sqrt( beta_schedule / diffusion_dt)
    # noise_seq = torch.zeros(alpha_accum.shape[0]+1)
    noise_seq= start_noise * alpha_accum + (1 - alpha_accum)
    # noise_seq[0] = start_noise
    f_seq = (torch.sqrt(1 - alpha_accum) - 1) /diffusion_dt
    return alpha_schedule, alpha_accum, g_seq, noise_seq, f_seq

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class QuadraticCostLoss(torch.nn.Module):
    def __init__(self):
        super(QuadraticCostLoss, self).__init__()
        self.Q = torch.tensor([[1.0]]).cuda()
        self.R = torch.tensor([[0.8]]).cuda()

    def forward(self, x_traj, u_traj):
        cost = torch.sum(torch.matmul(x_traj,self.Q) * x_traj) + torch.sum(torch.matmul(u_traj,self.R)*u_traj)
        return cost

class FinalTargetLoss(torch.nn.Module):
    def __init__(self):
        super(FinalTargetLoss, self).__init__()
        self.Q_f = 5 * torch.eye(4).cuda()
        self.Q_o = torch.eye(4).cuda()
        self.R = torch.eye(2).cuda()
        self.obs_layer = nn.ReLU()
        self.final_layer = nn.ReLU()

    def forward(self, x_traj, u_traj):
        final_target = torch.tensor([5., 0., 0., 0.]).cuda()
        final_dist = torch.linalg.norm(x_traj[:,-1,:]-final_target,dim=1)
        final_cost = torch.mean(self.final_layer(torch.pow(final_dist,2)-1e-4))
        ## The final target is given, and the cost is the distance between the final state and the final target

        obs_center = torch.tensor([2.5,0.0]).cuda()
        traj_xy_interval = torch.cat((x_traj[:,:,0].unsqueeze(2),x_traj[:,:,2].unsqueeze(2)),dim=2)
        obs_dist = torch.pow(torch.linalg.norm(traj_xy_interval-obs_center,dim=2),1)
        obstacle_cost = torch.mean(self.obs_layer(1.5-obs_dist))
        running_cost =  torch.mean(torch.matmul(u_traj,self.R)*u_traj)
        return final_cost, obstacle_cost, running_cost


class mm_Dynamics(nn.Module):
    def __init__(self):
        self.A = torch.tensor([[0., 1, 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 1],
                  [0., 0., 0., 0.]]).cuda()
        self.B = torch.tensor([[0., 0.], [1, 0.],
                               [0., 0.], [0., 1]]).cuda()
    def delta_x(self,x ,u, dt = torch.tensor([0.1]).cuda()):
        return dt*(torch.matmul(x, self.A.T) + torch.matmul(u, self.B.T))

    def gradient_dynamics(self,x,u,dt = torch.tensor([0.1]).cuda()):
        grad_x = torch.zeros((x.shape[0], x.shape[1], x.shape[1])).cuda()
        grad_x = (grad_x + self.A)*dt
        grad_u = torch.zeros((u.shape[0], x.shape[1], u.shape[1])).cuda()
        grad_u = (grad_u + self.B)*dt
        return grad_x, grad_u

    def generate_real_traj(self, initial_state, u_traj):
        # check the batch size of initial_state and u_traj, either one of them should be 1, or they should have the same batch size
        if initial_state.shape[0] != u_traj.shape[0] and initial_state.shape[0] != 1 and u_traj.shape[0] != 1:
            raise ValueError('The batch size of initial_state and u_traj should be the same or one of them should be 1')
        with torch.no_grad():
            x_traj = torch.zeros((u_traj.shape[0],u_traj.shape[1]+1,initial_state.shape[1])).cuda()
            x_traj[:,0] = initial_state
            for i in range(1,u_traj.shape[1]+1):
                x_traj[:,i] = x_traj[:,i-1] + self.delta_x(x_traj[:,i-1],u_traj[:,i-1])
        return x_traj




class LinearDynamics(nn.Module):
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def delta_x(self, x, u, dt = torch.tensor([0.1]).cuda()):
        return dt*(torch.matmul(x, self.A) + torch.matmul(u, self.B))

    def gradient_dynamics(self, x, u, dt = torch.tensor([0.1]).cuda()):
        grad_x = torch.zeros((x.shape[0], x.shape[1], x.shape[1])).cuda()
        grad_x = (grad_x + self.A)*dt
        grad_u = torch.zeros((u.shape[0], u.shape[1], u.shape[1])).cuda()
        grad_u = (grad_u + self.B)*dt
        return grad_x, grad_u

    def generate_real_traj(self, initial_state, u_traj):
        # check the batch size of initial_state and u_traj, either one of them should be 1, or they should have the same batch size
        if initial_state.shape[0] != u_traj.shape[0] and initial_state.shape[0] != 1 and u_traj.shape[0] != 1:
            raise ValueError('The batch size of initial_state and u_traj should be the same or one of them should be 1')
        with torch.no_grad():
            x_traj = torch.zeros((u_traj.shape[0],u_traj.shape[1]+1,initial_state.shape[1])).cuda()
            x_traj[:,0] = initial_state
            for i in range(1,u_traj.shape[1]+1):
                x_traj[:,i] = x_traj[:,i-1] + self.delta_x(x_traj[:,i-1],u_traj[:,i-1])
        return x_traj

class KalmanFilterCost(torch.nn.Module):
    def __init__(self, dynamics):
        super(KalmanFilterCost, self).__init__()
        self.dynamics = dynamics

    def forward(self, x_traj_obs, u_traj_obs, cur_noise):
        log_likelihood = self.Kalman_filter(x_traj_obs, u_traj_obs, cur_noise, self.dynamics)
        return -log_likelihood.mean()

    # Kalman filter estimate the trajectory based on the noisy observation
    # x_traj_obs is the noisy observation of the state trajectory which has the shape of batch*horizon*dim_x
    # u_traj_obs is the noisy observation of the control trajectory which has the shape of batch*（horizon-1）*dim_u
    def Kalman_filter(self, x_traj_obs,u_traj_obs, noise_cur, dynamics):
        # First forward propogate the estimation
        x_traj_pred = torch.zeros_like(x_traj_obs).cuda()
        x_traj_correct = torch.zeros_like(x_traj_obs).cuda()
        x_traj_correct[:,0] = x_traj_obs[:,0]
        # Initialize the covariance matrix which has the shape of batch*horizon*dim_x*dim_x
        x_cov_pred = torch.zeros((x_traj_obs.shape[0],x_traj_obs.shape[1],x_traj_obs.shape[2],x_traj_obs.shape[2])).cuda()
        x_cov_correct = torch.zeros((x_traj_obs.shape[0],x_traj_obs.shape[1],x_traj_obs.shape[2],x_traj_obs.shape[2])).cuda()
        log_likelihood = 0
        for i in range(1,x_traj_obs.shape[1]):
            # Update prediction of mu
            x_traj_pred[:,i] = x_traj_correct[:,i-1] + dynamics.delta_x(x_traj_correct[:,i-1],u_traj_obs[:,i-1])
            # Update prediction of covariance base on the gradient of the dynamics
            # get the gradient of the dynamics on x and u
            grad_x, grad_u = dynamics.gradient_dynamics(x_traj_correct[:,i-1],u_traj_obs[:,i-1])

            # grad_x has the shape of batch*dim_x*dim_x
            # grad_u has the shape of batch*dim_x*dim_u
            x_cov_pred[:,i] = (grad_x+torch.eye(x_traj_pred.shape[2]).cuda()).matmul(x_cov_correct[:,i-1]).matmul(grad_x.transpose(1,2)+torch.eye(x_traj_pred.shape[2]).cuda()) + grad_u.matmul(grad_u.transpose(1,2)) * noise_cur
            # Update the Kalman gain
            K = x_cov_pred[:,i].matmul((x_cov_pred[:,i]+noise_cur*torch.eye(x_cov_pred.shape[2]).cuda()).inverse())
            # Update the state estimation
            x_traj_correct[:,i] = x_traj_pred[:,i] + torch.bmm(K,(x_traj_obs[:,i]-x_traj_pred[:,i]).unsqueeze(2)).squeeze(2)
            # Update the covariance estimation
            x_cov_correct[:,i] = (torch.eye(x_cov_pred.shape[2]).cuda() - K).matmul(x_cov_pred[:,i])
            log_likelihood_tmp = dist.MultivariateNormal(x_traj_pred[:,i], x_cov_pred[:,i]+torch.eye(x_traj_obs.shape[2]).cuda()*noise_cur).log_prob(x_traj_obs[:,i])
            log_likelihood += torch.mean(log_likelihood_tmp)
        return log_likelihood

if __name__ == "__main__":
    if os.path.exists('./figure'):
        os.system('rm -rf ./figure')
    os.makedirs('./figure')
    dim_x = 4
    dim_u = 2
    Horizon = 50
    batch = 1000
    TimeStep = 500
    dynamics_dt = torch.tensor([0.1]).cuda()
    diffusion_dt = torch.tensor([0.01]).cuda()
    # Uniform sample x and u from -1 to 1
    x_traj_diffuse =  2*torch.randn(batch,Horizon,dim_x).cuda()
    for i in range(Horizon):
        x_traj_diffuse[:,i,0] = 5.0 * i / Horizon
    x_traj_diffuse[:,0] = 0
    x_traj_diffuse[:,1] = 1.0
    u_traj_diffuse =  0.5 * torch.randn(batch,Horizon-1,dim_u).cuda()

    x_traj_diffuse.requires_grad = True
    u_traj_diffuse.requires_grad = True
    # Define the dynamics
    A = torch.tensor([[0.3]]).cuda()
    B = torch.tensor([[0.9]]).cuda()
    dynamics = mm_Dynamics()
    # Get the noise schedule and reverse it
    beta_seq = linear_beta_schedule(TimeStep)
    alpha_schedule,alpha_accum,g_seq,noise_seq, f_seq = diffusion_parameters_schedule(beta_seq,start_noise= 2e-6,diffusion_dt=diffusion_dt)

    # Check if the figure folder exists
    if not os.path.exists('./figure'):
        os.makedirs('./figure')
    # Define the Kalman filter cost

    noise_seq = noise_seq.flip(0)
    alpha_accum = alpha_accum.flip(0)
    f_seq = f_seq.flip(0)
    g_seq = g_seq.flip(0)
    beta_seq = beta_seq.flip(0)
    d_cost = KalmanFilterCost(dynamics)
    r_cost = FinalTargetLoss().cuda()
    x_traj_save = []
    u_traj_save = []
    final_cstr = []
    obs_cstr = []
    dyn_ll_list = []
    for i in range(TimeStep):
        ## need to adjust the order of the noise seq
        # x_traj_scale = x_traj_diffuse / torch.sqrt(alpha_accum[i])
        u_traj_diffuse.requires_grad = True
        # u_traj_scale = u_traj_diffuse / torch.sqrt(alpha_accum[i])

        x_traj_scale = x_traj_diffuse
        u_traj_scale = u_traj_diffuse
        # calculate noise_mes
        noise_mes = torch.sum(beta_seq[i:]*beta_seq[i:])*diffusion_dt + 4e-10
        # First generate the real trajectory and plot
        # get the noise from noise seq reverse
        dyn_log_l = d_cost(x_traj_scale,u_traj_scale, 2e-5)
        final_cost,obstacle_cost,running_cost = r_cost(x_traj_scale[:,1:],u_traj_scale)
        loss = -dyn_log_l #- 50000*final_cost-100*running_cost
        # 180000*obstacle_cost-
        loss.backward()
        # reward_cost.backward()
        # print epoch, dynamic loss, and reward loss
        print('beta%.3e, noise%.3e, dynamic loss:%.3e, final_loss:%.3e, obs_loss:%.3e, running_loss:%.3e'%(beta_seq[i].item(),noise_mes.item(),dyn_log_l.item(),final_cost.item(),obstacle_cost.item(),running_cost.item()))
        with torch.no_grad():
            # x_traj -= torch.randn() * x_traj.grad * torch.sqrt(noise) * 0.001
            # delta_mu_x =  -(f_seq[i] * x_traj_diffuse[:,1:] - g_seq[i] * g_seq[i] * x_traj_diffuse.grad[:,1:]) * diffusion_dt
            # delta_mu_u = -(f_seq[i] * u_traj_diffuse - g_seq[i] * g_seq[i] * u_traj_diffuse.grad) * diffusion_dt

            # delta_mu_x = -( - g_seq[i]*g_seq[i] * x_traj_diffuse.grad[:,1:]) * diffusion_dt
            # delta_mu_u = -(- g_seq[i]*g_seq[i] * u_traj_diffuse.grad) * diffusion_dt
            delta_mu_x = x_traj_diffuse.grad[:,1:] * beta_seq[i]
            delta_mu_u = u_traj_diffuse.grad * beta_seq[i]
            x_traj_diffuse[:,1:] = x_traj_diffuse[:,1:] + delta_mu_x * diffusion_dt +torch.sqrt(2*beta_seq[i]*diffusion_dt/batch)*torch.randn_like(x_traj_diffuse[:,1:])
            u_traj_diffuse = u_traj_diffuse + delta_mu_u * diffusion_dt + torch.sqrt(2*beta_seq[i]*diffusion_dt/batch)* torch.randn_like(u_traj_diffuse)
        x_traj_save.append(x_traj_diffuse.cpu().detach().numpy())
        u_traj_save.append(u_traj_diffuse.cpu().detach().numpy())
        dyn_ll_list.append(dyn_log_l.cpu().detach().numpy())
        final_cstr.append(final_cost.cpu().detach().numpy())
        obs_cstr.append(obstacle_cost.cpu().detach().numpy())
        x_traj_diffuse.grad.zero_()
        if i % 10 == 0:
            # rollout with dynamics to get gronud truth traj
            x_traj_real = dynamics.generate_real_traj(x_traj_diffuse[:,0],u_traj_diffuse)
            x_traj_real = x_traj_real.cpu().detach().numpy()
            # plot the trajectory
            plt.figure()
            plt.gca().set_aspect('equal', adjustable='box')
            # plot 2D traj
            plot_traj = 0
            xs = x_traj_diffuse[plot_traj, :, 0].cpu().detach().numpy()
            ys = x_traj_diffuse[plot_traj, :, 2].cpu().detach().numpy()
            plt.scatter(xs, ys, c=range(len(xs)), cmap='Reds')
            plt.plot(x_traj_real[plot_traj, :, 0], x_traj_real[plot_traj, :, 2], 'b--')
            # plot the obstacle
            # obs_center = [2.5, 0.0]
            # obs_radius = 1.5
            # obs = plt.Circle(obs_center, obs_radius, color='black', fill=True)
            # plt.gca().add_artist(obs)
            plt.grid()
            plt.xlim([-2, 7])
            plt.ylim([-5, 5])
            plt.savefig(f'figure/traj_{i}.png')
            plt.close()
    x_traj_save = np.array(x_traj_save)
    u_traj_save = np.array(u_traj_save)
    dyn_ll_list = np.array(dyn_ll_list)
    final_cstr = np.array(final_cstr )* 50000
    obs_cstr = np.array(obs_cstr )* 180000







