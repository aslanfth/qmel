import numpy as np
import re
from scipy.stats import rankdata
import torch
import os
import pandas as pd

# Compute delta m%
def compute_delta_m(STL, MTL, delta_k):
    """
    Computes the delta_m metric.

    Args:
        STL (np.array): Single-task learning performance.
        MTL (np.array): Multi-task learning performance.
        delta_k (np.array): Direction of improvement (1 for higher better).

    Returns:
        float: The computed delta_m metric.
    """
    delta_m = ((-1)**delta_k) * ((MTL - STL) / STL) * 100
    delta_m_bar = np.mean(delta_m)
    max_m = np.max(delta_m)
    return delta_m_bar, max_m

all_results = torch.load('all_results_celeba.pt')

# Load baseline data
STL = all_results['single_task']

combined_scores = {'single_task': STL}

# Initialize a result dictionary for averages
averages = {}

# Group data by method and calculate averages
methods = set(key[0] for key in all_results.keys() if key != 'single_task')  # Extract unique methods
for method in methods:
    # Collect arrays for the 3 seeds
    arrays = [all_results[(method, seed)] for seed in [10000, 20000, 30000]]
    # Compute average across the 3 seeds
    averages[method] = np.mean(arrays, axis=0)
    combined_scores[method] = (averages[method])

combined_scores['fairgrad'] = [0.673481, 0.6715126, 0.80895394, 0.59129095, 0.716146, 0.84868795, 0.5293899, 0.6024346, 0.79252076, 0.8232979, 0.49155608, 0.68601424, 0.6678187, 0.50076723, 0.5353463, 0.9543982, 0.6991702, 0.6246493, 0.87510586, 0.85260683, 0.9679586, 0.91303086, 0.53938675, 0.44097534, 0.9698319, 0.30875763, 0.5560105, 0.5475425,  0.5299987, 0.6050991, 0.72113657, 0.9100189, 0.48792267, 0.6447788, 0.7013081, 0.8596742, 0.92936283, 0.40353844, 0.7261722, 0.9126637]

delta_k = np.ones(len(STL))  # δ_k = 1 since higher F1 scores are better

# Parse log file
file_path1 = 'experiments/celeba/save/save/fairmtl_1.0_False_True_10000.stats'  # Replace with your log file path
stats1 = torch.load(file_path1)
file_path2 = 'experiments/celeba/save/save/fairmtl_1.0_False_True_20000.stats'  # Replace with your log file path
stats2 = torch.load(file_path2)
file_path3 = 'experiments/celeba/save/save/fairmtl_1.0_False_True_30000.stats'  # Replace with your log file path
stats3 = torch.load(file_path3)
avg_fairmtl = stats1['metric'][stats1['best_epoch']] + stats2['metric'][stats2['best_epoch']] + stats3['metric'][stats3['best_epoch']]
avg_fairmtl /= 3


combined_scores['fairmtl'] = avg_fairmtl
ranks = np.apply_along_axis(rankdata, 1, -np.stack(combined_scores.values()).T)
mean_ranks = np.mean(ranks, axis=0)
print(f"Methods {combined_scores.keys()} - Mean Rank: {mean_ranks}%")
delta_ms = {}
max_ms = {}
for method, score in zip(combined_scores.keys(), combined_scores.values()):
    delta_m, max_m = compute_delta_m(STL, score, delta_k)
    delta_ms[method] = delta_m
    max_ms[method] = max_m
    print(f"Method {method} - Delta_m (Test F1): {delta_m:.4f}% - Max_m  (Test F1): {max_m:.4f}%")
    # Compute mean F1 score
    mean_f1 = np.mean(score)
    # Print results
    print(f"Method {method} - Mean F1: {mean_f1:.4f}")

# Alpha-Fairness Function
def alpha_fairness(x, a=2):
    if a == 1:
        return torch.log(x)  # Special case for alpha=1 (logarithmic fairness)
    return (x**(1 - a) - 1) / (1 - a)

# Directory containing fairness statistics
logs_dir = 'experiments/celeba/save/train_fairness/'

# Prepare lists to store results
fairness_results = []
test_loss_results = []

# Iterate through all .stats files
for filename in os.listdir(logs_dir):
    if filename.endswith('.stats'):
        method_name = filename.replace('.stats', '')
        stats_file = os.path.join(logs_dir, filename)

        try:
            stats = torch.load(stats_file)

            # Fairness Computation
            losses = torch.vstack(stats['losses'])
            utils = losses[0, :] - losses
            avg_utils = utils[1:] #/ torch.arange(1, utils.shape[0]).unsqueeze(1)
            fairness_1 = torch.sum(alpha_fairness(avg_utils, a=1), dim=-1)
            fairness_2 = torch.sum(alpha_fairness(avg_utils, a=2), dim=-1)
            # fairness_5 = torch.sum(alpha_fairness(avg_utils, a=5), dim=-1)
            fairness_maxmin = torch.min(avg_utils, dim=-1).values

            fairness_results.append({
                'method': method_name,
                'fairness_1': fairness_1[-1].item(),
                'fairness_2': fairness_2[-1].item(),
                # 'fairness_5': fairness_5[-1].item(),
                'fairness_maxmin': fairness_maxmin[-1].item()
            })

            # print(f"{method_name}: Fairness@1={fairness_1[-1].item()}, "
            #       f"@2={fairness_2[-1].item()}, @5={fairness_5[-1].item()}, "
            #       f"Maxmin={fairness_maxmin[-1].item()}")

            # Test Losses at Best delta_m Index
            min_idx = stats['best_epoch']
            test_losses = stats['metric'][min_idx].tolist()
            test_loss_results.append({'method': method_name, **{f'F1_score_{i+1}': v for i, v in enumerate(test_losses)}})

        except Exception as e:
            print(f"Error processing {method_name}: {e}")

# Save Fairness Results to CSV
df_fairness = pd.DataFrame(fairness_results)
fairness_csv_path = os.path.join(logs_dir, 'fairness_summary.csv')
df_fairness.to_csv(fairness_csv_path, index=False)
print(f"\nSaved fairness summary to: {fairness_csv_path}")

# Convert combined_scores to DataFrame
df_combined = pd.DataFrame.from_dict(combined_scores, orient='index')
df_combined.columns = [f'F1_score_{i+1}' for i in range(df_combined.shape[1])]
df_combined.reset_index(inplace=True)
df_combined.rename(columns={'index': 'method'}, inplace=True)

# Append test_loss_results (which are dictionaries) as new rows with same columns
for row in test_loss_results:
    method = row['method']
    scores = [row.get(f'F1_score_{i+1}', np.nan) for i in range(40)]
    df_combined.loc[len(df_combined)] = [method] + scores

# Save to CSV
test_losses_csv_path = os.path.join(logs_dir, 'test_losses_at_best.csv')
df_combined.to_csv(test_losses_csv_path, index=False)
print(f"✅ Saved fixed CSV with 40 columns to: {test_losses_csv_path}")