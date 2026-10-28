import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# We will have a 1-dimensional linear regression, where we have multiple y values at each time-slot. Then, we will find x* and compare.

def loss_fn(theta, a, y):
    # y is a vector of size (num_tasks,)
    error = a*theta - y
    loss = error**2/2
    return loss

# Alpha-Fairness Function
def alpha_fairness(x, alpha):
    if alpha == 1:
        return np.log(x)  # Special case for alpha=1 (logarithmic fairness)
    return (x**(1 - alpha) - 1) / (1 - alpha)

# Alpha-Fairness Function
def alpha_fairness_cp(x, alpha):
    if alpha == 1:
        return cp.log(x)  # Special case for alpha=1 (logarithmic fairness)
    return (x**(1 - alpha) - 1) / (1 - alpha)

alpha = 4.0
T = 100**2
p = 0.5 # adding priorities makes it worse somehow, can be seen in desmos!
# as p goes further away from 0.5, short-term, long-term and CWS benchmarks get closer to each other.
p_low = min(p, 1-p)
p_high = max(p, 1-p)
# Following x_star provides us a pseudo regret, calculating x_star for each time-slot would require much more time.
x_star = 0.361521 # a1: 0.29284, a2: 0.32269, a3: 0.34465, a4: 0.361521, a5: 0.37491 # for different alphas
num_tasks = 2
u_min = 0.01 # hyperparameter, set a low value for weights. this prevents vanishing weights!
min_coeff = 1 # minimum of a's, here we set all a=1.
xs = np.random.rand() * np.ones(5) # initialize x_1
loss_max = 0.19140625 # Maximum loss value calculated theoretically. (using desmos)
u_max = loss_max
w_min = u_max**-alpha # initialize the dual space.
w_max = u_min**-alpha if u_min > 0 else None
print(w_min, w_max)
w = w_min + (w_max - w_min)*np.random.rand() # initialize w_1 for our long-term solution.
ws = np.vstack([w*np.ones(num_tasks), np.array([0.25, 0.75]), np.array([0.5, 0.5]), np.array([0.75, 0.25]), w*np.ones(num_tasks)]) #ours, 3 CWSs, short-term
G_lips = p_high
beta_smooth = p_high
rho = 0 # 3rd derivative is 0!
mu_strong = min_coeff**2*p_low

# lr_update = min(0.5/beta_smooth, mu_strong/(8*rho*G_lips))
lr_update = 0.5#/beta_smooth # rho = 0! #here, lr_update still respects the constraint since beta_smooth <= 1
lr_dual = alpha/(u_min**(alpha+1))
lr_primals = [8*(u_max**alpha)/(mu_strong*num_tasks), 8/(mu_strong*num_tasks*0.25), 8/(mu_strong*num_tasks*0.5), 8/(mu_strong*num_tasks*0.25), 8*(u_max**alpha)/(mu_strong*num_tasks), 0]

# Restricted adversary:
# Labels are only changed sqrt(T) times, thus, regret is of O(sqrt(T)).
# Essentially, priorities are not used since both are equal to 0.5.
coeffs = np.ones([num_tasks, T])
labels = np.zeros([num_tasks, T])
priorities =  np.empty([num_tasks, T])
priorities[0, :] = p
priorities[1, :] = 1-p
prio = np.ones([num_tasks, int(np.sqrt(T))])
prio[0, :] = 1-p
prio[1, :] = p
for t in range(int(np.ceil(np.sqrt(T)))):
    if t % 2 == 1:
        labels[0, t*int(np.sqrt(T)):(t+1)*int(np.sqrt(T))] = 1
        priorities[:, t*int(np.sqrt(T)):(t+1)*int(np.sqrt(T))] = prio
losses = [[], [], [], [], [], []]
utilities = [[], [], [], [], [], []]
weights_left = []
weights_right = []
sum_utils = 0
for t in range(T):
    # if (t+1)%100 == 0:
    #     print(f"Iter: {t+1}")
    a = coeffs[:, t]
    y = labels[:, t]
    priority = priorities[:, t]
    for i in range(6):
        lr_primal = lr_primals[i]
        if i < 5:
            x = xs[i]
            w = ws[i]
        if i == 5:
            x = x_star
            w = np.zeros(num_tasks)
        if i == 0:
            weights_left.append(w[0])
            weights_right.append(w[1])
        grad_update = (a**2*x - a*y)*priority
        grad_update = np.sum(grad_update)/num_tasks # equal weights in update.
        theta = x - lr_update*grad_update # update function h().
        loss = loss_fn(theta, a, y)*priority
        losses[i].append(loss)
        utility = loss_max - loss
        utilities[i].append(utility)
        if i == 0:
            sum_utils += utility
            # w -= lr_dual/(t+1)*(-w**(-1/alpha)+utility)
            # w = np.clip(w, w_min, w_max)
            w = np.clip((sum_utils/(t+1)), u_min, None)**(-alpha) # FTL
        elif i == 4:
            w = utility**(-alpha)
            w = np.clip(w, w_min, w_max)
        grad_meta = (a**2*theta-a*y)*priority
        x -= w@(grad_meta*(1-lr_update*a**2@priority/num_tasks))*lr_primal/(t+1)
        x = min(max(x, 0), 1)
        if i == 5:
            x = x_star
        elif i < 5:
            xs[i] = x
            ws[i] = w

fairnesses = []
regrets = []
fig, axes = plt.subplots(2, 3, figsize=(12, 4))
axes[0, 0].set_title('Loss1')
axes[0, 1].set_title('Loss2')
axes[0, 2].set_title('Fairness')
axes[1, 0].set_title('Util1')
axes[1, 1].set_title('Util2')
axes[1, 2].set_title('Regret')
res_csv = pd.DataFrame()
utils_benchmark = utilities[-1]
for i in range(5):
    utils = utilities[i]
    cum_utils = np.cumsum(utils, 0)/np.arange(1,T+1).reshape(-1, 1)
    fairness = np.sum(alpha_fairness(cum_utils, alpha), -1)
    fairness_benchmark = np.sum(alpha_fairness(np.cumsum(utils_benchmark, 0)/np.arange(1,T+1).reshape(-1, 1), alpha), -1)
    fairnesses.append(fairness)
    regret = fairness_benchmark - fairness
    regrets.append(regret)
    res_csv[f'fairness_{i}'] = fairness
    res_csv[f'regret_{i}'] = regret
    res_csv[f'util_{i}_left'] = cum_utils[:, 0]
    res_csv[f'util_{i}_right'] = cum_utils[:, 1]
    cumsum_loss = np.cumsum(np.stack(losses[i]), 0)/np.arange(1,T+1).reshape(-1, 1)
    res_csv[f'loss_{i}_left'] = cumsum_loss[:, 0]
    res_csv[f'loss_{i}_right'] = cumsum_loss[:, 1]
    axes[0, 0].plot(cumsum_loss[:, 0], label = i)
    axes[0, 1].plot(cumsum_loss[:, 1], label = i)
    axes[0, 2].plot(fairness, label = i)
    axes[1, 0].plot(cum_utils[:, 0], label = i)
    axes[1, 1].plot(cum_utils[:, 1], label = i)
    axes[1, 2].plot(regret, label=i)

res_csv['weights_left'] = weights_left
res_csv['weights_right'] = weights_right
output_file = Path('paper_results') / f'toy_compare_a{alpha}.csv'
output_file.parent.mkdir(parents=True, exist_ok=True)
res_csv[::100].to_csv(output_file, index_label='iteration')
# print(xs)
# print(fairness_benchmark[-1])
# axes[0, 2].legend()
# axes[1, 2].legend()
# plt.show()
