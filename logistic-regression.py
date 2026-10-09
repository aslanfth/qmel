import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import pandas as pd

# We will have a 10-dimensional logistic regression, and compare fairness curves.

def loss_fn(theta, a, y, lam=1e-3):
    """
    Logistic regression loss for each task.
    y: binary labels in {0, 1}
    a: list/array of feature vectors (shape: [num_tasks, d])
    theta: parameter vector (shape: [d])
    lam: L2-regularization coefficient
    """
    num_tasks, d = a.shape
    loss = np.empty(num_tasks)
    for k in range(num_tasks):
        z = a[k] @ theta                      # linear score
        # sigmoid safely
        s = 1 / (1 + np.exp(-z))
        # binary cross-entropy loss + L2 regularization
        loss[k] = - (y[k] * np.log(s + 1e-12) + (1 - y[k]) * np.log(1 - s + 1e-12)) + 0.5 * lam * np.dot(theta, theta)
    return loss


def grad_fn(theta, a, y, lam=1e-3):
    """
    Gradient of logistic regression loss for each task.
    """
    num_tasks, d = a.shape
    grads = np.empty((num_tasks, d))
    for k in range(num_tasks):
        z = a[k] @ theta
        s = 1 / (1 + np.exp(-z))
        grads[k] = (s - y[k]) * a[k] + lam * theta
    return grads

# Alpha-Fairness Function
def alpha_fairness(x, alpha):
    if alpha == 1:
        return np.log(x)  # Special case for alpha=1 (logarithmic fairness)
    return (x**(1 - alpha) - 1) / (1 - alpha)

alpha = 3.0
T = 100**2
# Following x_star provides us a pseudo regret, calculating x_star for each time-slot would require much more time.
num_tasks = 2
u_min = 0.01 # hyperparameter, set a low value for weights. this prevents vanishing weights!
d = 10
xs = np.linalg.outer(np.random.rand(d), np.ones(5)) # initialize x_1
# Restricted adversary:
# Labels are only changed sqrt(T) times, thus, regret is of O(sqrt(T)).

coeffs = np.random.rand(d, num_tasks, T) * 2 - 1
noise = np.random.randn(num_tasks, T)*0
labels = (np.sum(coeffs, axis=0) > 0).astype(int)
for t in range(int(np.ceil(np.sqrt(T)))):
    if t % 2 == 1:
        labels[1, t*int(np.sqrt(T)):(t+1)*int(np.sqrt(T))] = 1 - labels[1, t*int(np.sqrt(T)):(t+1)*int(np.sqrt(T))]
mu_strong = 1e-3
G_lips = np.sqrt(d)
beta_smooth = d/4 + mu_strong
rho = d**(3/2)/6 # 3rd derivative is 0!
loss_max = 1 # Maximum loss value.
u_max = loss_max
w_min = u_max**-alpha # initialize the dual space.
w_max = u_min**-alpha if u_min > 0 else None
print(w_min, w_max)
w = w_min + (w_max - w_min)*np.random.rand() # initialize w_1 for our long-term solution.
ws = np.vstack([w*np.ones(num_tasks), np.array([0.5, 0.5]), w*np.ones(num_tasks)]) #ours, CWS (FTML), short-term

lr_update = min(0.5/beta_smooth, mu_strong/(8*rho*G_lips))
# lr_update = 0.5/beta_smooth # rho = 0! #here, lr_update still respects the constraint since beta_smooth <= 1
lr_dual = alpha/(u_min**(alpha+1))
lr_primals = [8*(u_max**alpha)/(mu_strong*num_tasks), 8/(mu_strong*num_tasks*0.25), 8/(mu_strong*num_tasks*0.5), 8/(mu_strong*num_tasks*0.25), 8*(u_max**alpha)/(mu_strong*num_tasks), 0]

losses = [[], [], []]
utilities = [[], [], []]
weights_left = []
weights_right = []
sum_utils = 0
for t in range(T):
    # if (t+1)%100 == 0:
    #     print(f"Iter: {t+1}")
    a = coeffs[:, :, t].T
    y = labels[:, t]
    for i in range(3):
        lr_primal = lr_primals[i]
        x = xs[:, i]
        w = ws[i]
        if i == 0:
            weights_left.append(w[0])
            weights_right.append(w[1])
        grad_update = grad_fn(x, a, y)
        grad_update = np.sum(grad_update, axis=0)/num_tasks # equal weights in update.
        theta = x - lr_update*grad_update # update function h().
        loss = loss_fn(theta, a, y)
        losses[i].append(loss)
        utility = loss_max - loss
        utilities[i].append(utility)
        if i == 0:
            sum_utils += utility
            # w -= lr_dual/(t+1)*(-w**(-1/alpha)+utility)
            # w = np.clip(w, w_min, w_max)
            w = np.clip((sum_utils/(t+1)), u_min, None)**(-alpha) # FTL
        elif i == 2:
            w = utility**(-alpha)
            w = np.clip(w, w_min, w_max)
        grad_meta = grad_fn(theta, a, y)
        grad_meta = w@grad_meta
        x -= (grad_meta@(1-lr_update*(a.T@a)))*lr_primal/(t+1)
        x = np.clip(x, 0, 1)
        xs[:, i] = x
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
# utils_benchmark = utilities[-1]
for i in range(3):
    utils = utilities[i]
    cum_utils = np.cumsum(utils, 0)/np.arange(1,T+1).reshape(-1, 1)
    fairness = np.sum(alpha_fairness(cum_utils, alpha), -1)
    fairnesses.append(fairness)
    res_csv[f'fairness_{i}'] = fairness
    res_csv[f'util_{i}_left'] = cum_utils[:, 0]
    res_csv[f'util_{i}_right'] = cum_utils[:, 1]
    cumsum_loss = np.cumsum(np.stack(losses[i]), 0)/np.arange(1,T+1).reshape(-1, 1)
    res_csv[f'loss_{i}_left'] = cumsum_loss[:, 0]
    res_csv[f'loss_{i}_right'] = cumsum_loss[:, 1]
    axes[0, 0].plot(cumsum_loss[:, 0], label = i)
    axes[0, 1].plot(cumsum_loss[:, 1], label = i)
    if i < 3:
        axes[0, 2].plot(fairness, label = i)
    axes[1, 0].plot(cum_utils[:, 0], label = i)
    axes[1, 1].plot(cum_utils[:, 1], label = i)

res_csv['weights_left'] = weights_left
res_csv['weights_right'] = weights_right
res_csv[::100].to_csv(f'paper_results/aistats/app-logistic-a{alpha}.csv', index_label='iteration')
# print(xs)
# print(fairness_benchmark[-1])
axes[0, 2].legend()
axes[1, 2].legend()
plt.show()
