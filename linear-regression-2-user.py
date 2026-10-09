import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import pandas as pd
import time

# We will have a 1-dimensional linear regression, where we have multiple y values at each time-slot. Then, we will find x* and compare.

def loss_fn(theta, a, y):
    # y is a vector of size (num_tasks,)
    error = a*theta - y
    return error**2/2

# Alpha-Fairness Function
def alpha_fairness(x, alpha):
    if alpha == 1:
        return np.log(x)  # Special case for alpha=1 (logarithmic fairness)
    return (x**(1 - alpha) - 1) / (1 - alpha)


alpha = 2.0
T = 500
num_tasks = 256 # Number of users.
u_min = 0.01 # hyper-parameter.
min_coeff = 0.1 # minimum value of a_tk
x = np.random.rand() # initialize x_1
u_max = 1 # hyper-parameter
loss_max = 1 # loss can be maximum 1, minimum 0.
w_min = u_max**-alpha # initialize the dual space.
w_max = u_min**-alpha
w = w_min + (w_max - w_min)*np.random.rand() # initialize w_1.
G_lips = 1
beta_smooth = 1
rho = 0
mu_strong = min_coeff**2

# Set learning rates:
# lr_update = min(0.5/beta_smooth, mu_strong/(8*rho*G_lips))
lr_update = 0.5/beta_smooth # rho = 0!
lr_dual = alpha/(u_min**(alpha+1))
lr_primal = 8*(u_max**alpha)/(mu_strong*num_tasks)
all_regrets = []
# Restricted adversary:
coeffs = np.random.randn(T)*0.1 # these are perturbations.
labels = np.random.randn(num_tasks, T)*0.1
for t in range(T):
    coeffs[t] = 1 + coeffs[t]/(t+1)
    user_labels = np.linspace(0, 1, num_tasks, endpoint=True)
    labels[:, t] = user_labels + labels[:, t]/(t+1)
coeffs = np.clip(coeffs, min_coeff, 1)
labels = np.clip(labels, 0, 1)
losses = []
fairness_stars = []
begin_time = time.time()
for t in range(T):
    if (t+1)%1000 == 0:
        print(f"Iter: {t+1}")
    a = coeffs[t]
    y = labels[:, t]
    grad_update = np.sum(a**2*x - a*y)/num_tasks # equal weights in update.
    theta = x - lr_update*grad_update # update function h().
    loss = loss_fn(theta, a, y)
    losses.append(loss)
    w -= lr_dual/(t+1)*(-w**(-1/alpha)+loss_max-loss) # dual step
    w = np.clip(w, w_min, w_max)
    x -= w@((a**2*theta-a*y)*(1-lr_update*a**2))*lr_primal/(t+1) # primal step
    x = min(max(x, 0), 1)
    benchmark = 0.5 # for this examples, x* is always 0.5.
    update = benchmark - lr_update*np.sum(benchmark - user_labels)/num_tasks
    loss_benchmark = loss_fn(update, 1, user_labels)
    improvement = loss_max - loss_benchmark
    fairness_stars.append(np.sum(alpha_fairness(improvement, alpha), -1))
end_time = time.time()
avg_time = (end_time - begin_time) / T
fairness_stars = np.array(fairness_stars)
utils = loss_max - np.stack(losses)
cum_utils = np.cumsum(utils, 0)/np.arange(1,T+1).reshape(-1, 1) # time averaged utilities
fairness = np.sum(alpha_fairness(cum_utils, alpha), -1)
regrets = fairness_stars - fairness
percent = np.abs(regrets/fairness_stars)*100
print(np.argmax(percent < 10), fairness_stars[0]*0.1, avg_time*1e6) # find the relative gap points.
# plt.plot(regrets)
# plt.show()
np.save(f'paper_results/toy_last_a{alpha}_k{num_tasks}', regrets)