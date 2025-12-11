import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

def parse():
    parser = argparse.ArgumentParser(description='Evaluate training results')
    parser.add_argument("--max_d", "-md", type=int, default=10, help="Max distance parameter")
    parser.add_argument("-distance_measure", "-dm", type=str, default="two_path_num")
    parser.add_argument("-learning_rate", "-lr", type=float, default=0.99)
    parser.add_argument("-year", "-y", type=int, default=2002)
    parser.add_argument("-window", "-w", type=int, default=3, help="Expire window size")
    parser.add_argument("-version", "-v", type=int, default=0)
    args = parser.parse_args()
    return args

def get_file_path(args):
    path = './data/'
    file_name = "chat_distance_" + str(args.year) + '.json'
    # pre_s logic from model_new.py
    # file_name[5:-5] extracts "distance_2002" from "chat_distance_2002.json"
    base_name = file_name[5:-5]
    
    idx_figure = "_year" + str(args.year) + "_"
    idx_figure += "max_d" + str(args.max_d) + "_" + args.distance_measure + "_" + "learning_rate" + str(args.learning_rate)

    version_postfix = ""
    if args.version != 0:
        version_postfix = "_" + str(args.version)
    else:
        version_postfix = ""
    
    pre_s = "window" + str(args.window) + version_postfix + "/" + base_name + "_num" + idx_figure
    
    return path + pre_s

def load_data(file_base):
    data = {}
    files = {
        'err': 'test_error',
        'err_by_date': 'test_error_by_date',
        'ndcgs': 'ndcgs',
        'ndcg_by_date': 'ndcg_by_date',
        'stat_true': 'stat_true',
        'stat_pred': 'stat_pred'
    }
    
    for key, suffix in files.items():
        try:
            with open(file_base + suffix, "rb") as fp:
                data[key] = pickle.load(fp)
        except FileNotFoundError:
            print(f"Warning: File {file_base + suffix} not found.")
            return None
            
    return data

def print_summary(data, method_names):
    print("\n" + "="*50)
    print(" TRAINING RESULTS SUMMARY ")
    print("="*50)
    
    # Calculate means
    err_means = [np.mean(m) for m in data['err']]
    ndcg_means = [np.mean(m) for m in data['ndcgs']]
    
    print(f"{'Method':<15} | {'Avg Error':<15} | {'Avg NDCG':<15}")
    print("-" * 50)
    
    for i, name in enumerate(method_names):
        print(f"{name:<15} | {err_means[i]:.4f}          | {ndcg_means[i]:.4f}")
    
    print("-" * 50)
    
    # Best method
    best_err_idx = np.argmin(err_means)
    best_ndcg_idx = np.argmax(ndcg_means)
    
    print(f"Best Error: {method_names[best_err_idx]} ({err_means[best_err_idx]:.4f})")
    print(f"Best NDCG:  {method_names[best_ndcg_idx]} ({ndcg_means[best_ndcg_idx]:.4f})")
    print("="*50 + "\n")

def plot_results(data, method_names, args, output_dir):
    sns.set_theme(style="whitegrid")
    
    # 1. Error over time (smoothed)
    plt.figure(figsize=(12, 6))
    
    # Convert err_by_date dict to sorted lists
    dates = sorted(data['err_by_date'][0].keys())
    
    for i, method in enumerate(method_names):
        # Calculate daily averages
        daily_means = []
        for d in dates:
            vals = data['err_by_date'][i][d]
            daily_means.append(sum(vals)/len(vals) if vals else 0)
            
        # Smooth with moving average
        window = 5
        smoothed = np.convolve(daily_means, np.ones(window)/window, mode='valid')
        
        plt.plot(dates[window-1:], smoothed, label=method, alpha=0.8)
        
    plt.title(f'Error Over Time (Smoothed, w={args.window})')
    plt.xlabel('Date')
    plt.ylabel('Average Rank Error')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_over_time.png'))
    plt.close()

    # 2. NDCG over time (smoothed)
    plt.figure(figsize=(12, 6))
    
    for i, method in enumerate(method_names):
        daily_means = []
        for d in dates:
            vals = data['ndcg_by_date'][i][d]
            daily_means.append(sum(vals)/len(vals) if vals else 0)
            
        window = 5
        smoothed = np.convolve(daily_means, np.ones(window)/window, mode='valid')
        
        plt.plot(dates[window-1:], smoothed, label=method, alpha=0.8)
        
    plt.title(f'NDCG Over Time (Smoothed, w={args.window})')
    plt.xlabel('Date')
    plt.ylabel('Average NDCG')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ndcg_over_time.png'))
    plt.close()
    
    # 3. Boxplot of Overall Performance - Using Log Scale (log1p)
    plt.figure(figsize=(10, 6))
    
    # Prepare data for seaborn boxplot
    plot_data = []
    for i, method in enumerate(method_names):
        # Use log1p to handle 0s while compressing large values
        # log1p(x) = log(1 + x)
        log_errors = np.log1p(data['err'][i])
        for val in log_errors:
            plot_data.append({'Method': method, 'Log(Error+1)': val})
            
    df = pd.DataFrame(plot_data)
    
    sns.boxplot(x='Method', y='Log(Error+1)', data=df, showfliers=True) 
    plt.title('Error Distribution by Method (Log Scale: log(1+x))')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_boxplot.png'))
    plt.close()

    # 4. Loss Histogram (Better than KDE for zero-inflated data)
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="white")
    
    # Filter out 0s to see the distribution of non-zero errors
    # Or plot two histograms: one for 0s vs non-0s, one for non-0 distribution
    
    for i, method in enumerate(method_names):
        errors = np.array(data['err'][i])
        # Filter only errors > 0 and < 50 (to ignore massive penalties)
        mask = (errors > 0) & (errors < 50)
        if np.any(mask):
            sns.kdeplot(errors[mask], label=method, shade=True)
        else:
            print(f"Method {method} has no errors in range (0, 50)")
            
    plt.title('Non-Zero Error Density (0 < Error < 50)')
    plt.xlim(0, 20) # Focus on small errors
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_density_nonzero.png'))
    plt.close()

    # 5. Accuracy/Error Curve (Raw values, similar to smoothed but overlayed) - Ported from model_new.py
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="white")
    
    # Plotting raw error values might be too noisy, let's plot daily averages (error_by_date_sum logic)
    # Reconstructing daily averages similar to model_new.py's logic
    dates = sorted(data['err_by_date'][0].keys())
    for i, method in enumerate(method_names):
        daily_avgs = []
        for d in dates:
            vals = data['err_by_date'][i][d]
            daily_avgs.append(sum(vals)/len(vals) if vals else 0)
        plt.plot(daily_avgs, label=method, alpha=0.4)
        
    plt.title('Daily Average Error Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_curve_daily.png'))
    plt.close()

    print(f"Plots saved to {output_dir}")

def main():
    args = parse()
    
    # Reconstruct file path base
    file_base = get_file_path(args)
    print(f"Reading results from: {file_base}...")
    
    data = load_data(file_base)
    
    if data is None:
        print("Could not load data. Please check your arguments and ensure training has finished.")
        return

    # Method names corresponding to the indices in model_new.py
    # self.learning_functions = [self.OWA1, self.OWA2, self.OWA3, self.OWA4, self.OWA5, self.timeline]
    # + our_model at the end
    
    # Check if data matches current model definition (in case using old pickle)
    num_methods = len(data['err'])
    full_method_names = ['OWA1', 'OWA2', 'OWA3', 'OWA4', 'OWA5', 'Timeline', 'MRAC (Ours)']
    
    if num_methods == 6: # Old version without OWA5
        method_names = ['OWA1', 'OWA2', 'OWA3', 'OWA4', 'Timeline', 'MRAC (Ours)']
        print("Warning: Loaded data contains 6 methods (likely missing OWA5). Using old method list.")
    elif num_methods == 7: # New version with OWA5
        method_names = full_method_names
    else:
        method_names = [f"Method {i}" for i in range(num_methods)]
        print(f"Warning: Loaded data contains {num_methods} methods. Using generic names.")
    
    # Print text summary
    print_summary(data, method_names)
    
    # Create figures directory
    output_dir = './figures_eval/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Generate plots
    plot_results(data, method_names, args, output_dir)

if __name__ == "__main__":
    main()

