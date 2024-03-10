# Learning and control meeting - 2024 Spring Week 9

## Diffusion Planner

Major updates: considering integrate nonlinear filter (mainly Particle Filter) into the planner.

Big picture: incorperate modelling information into the diffusion model. Currently, the major issue is how to esitimate $log p_d$. 
- for linear case, we have analytical solution by using Kalman filter. 

Ways to approach the problem:
- gain idea from linear case: what does this method do in the linear case. 
- gain idea from non-parametric smoothing algorithm: see if we can get the score function directly from that. 

Motivation: 

Keynotes:
- Using smoother instead of filter for better state esitimation.
  - Smoother: given all the measurements, estimate the state at each time step, thus can get global optimal estimation. (Rauch-Tung-Striebel (RTS) smoother)
  - Filter: given the measurements up to time t, estimate the state at time t. $
p\left(x_{1: n} \mid y_{1: n}\right)=p\left(x_{1: n-1} \mid y_{1: n-1}\right) \frac{f\left(x_n \mid x_{n-1}\right) g\left(y_n \mid x_n\right)}{p\left(y_n \mid y_{1: n-1}\right)}$
- Relationship between diffusion and particle filter.
  - The way to approximate the desired distribution: 
    - Diffusion: using the score function (gradient of the log density), which is analytical. Using multiple-step update to approximate the distribution.
    - Particle filter: using the Monte-Carlo method, which is a sampling-based method. Using importance sampling to approximate the distribution.
  - What does particle filter do?
    - Esitimate the distribution $p(x_{1:T}|y_{1:T})$ with Monte-Carlo method. (Since we have no closed form solution. )
    - Methods:
      - Sequential Monte-Carlo (SMC) method: leverage the sequential structure of the problem to reduce computational complexity.
      - Importance sampling: using a proposal distribution to sample from the target distribution (which is hard to sample from).
  - Is it possible to use the idea of particle filter / sequential monte carlo to esitimate the score function directly?
    - The idea of diffusion and SMC is different: one is data-driven, i.e. approximating dataset distribution. another one is an method to approximate the distribution which hard to get analytical solution, the distribution is known.
  - Inspiration from PF:
    - Leveraging the sequential structure of the problem.
    - See the problem as a smoothing problem, instead of filtering problem.
    - using rollout to approximate the desired distribution. 

Question: can we only diffuse the state rather than the action-state pair? 
Motivation: this would greatly reduce the requirement of a good model score function. 


Questions: should we scale the dynamics when doing the fusion?

Questions: in multiple stage, the kalman filter formulation is wrong. Because we only want to go to a less noise state, not the noiselss state. -> which explain why we will have a large gradient at last if we don't tune the parameters. 

Questions: current Kalman filter is different from the standard one, seems wrong. 

Standard formulation: 

$$
p(x_{1: n} \mid y_{1: n}) = p(x_{1: n-1} \mid y_{1: n-1}) \frac{p(x_n \mid x_{n-1}) p(y_n \mid x_n)}{p(y_n \mid y_{1: n-1})}
$$

Our current formulation:

$$
p(x_{1:n}) = p(x_{1:n-1}) p(x_n \mid x_{n-1})
$$

Translate into filtering problem:

$$
p(x_{1:n} \mid y_{1:n}) = p(x_{1:n-1} \mid y_{1:n}) p(x_n \mid x_{n-1})
$$