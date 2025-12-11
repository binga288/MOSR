import pickle
import numpy as np
import argparse
import os

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_d", "-md", type=int, default=10)
    parser.add_argument("-distance_measure", "-dm", type=str, default="two_path_num")
    parser.add_argument("-learning_rate", "-lr", type=float, default=0.99)
    parser.add_argument("-year", "-y", type=int, default=2002)
    parser.add_argument("-window", "-w", type=int, default=3)
    return parser.parse_args()

def get_file_path(args):
    path = './data/'
    file_name = "chat_distance_" + str(args.year) + '.json'
    base_name = file_name[5:-5]
    idx_figure = "_year" + str(args.year) + "_"
    idx_figure += "max_d" + str(args.max_d) + "_" + args.distance_measure + "_" + "learning_rate" + str(args.learning_rate)
    pre_s = "window" + str(args.window) + "_ori/" + base_name + "_num" + idx_figure
    return path + pre_s + "test_error"

def main():
    args = parse()
    file_path = get_file_path(args)
    print(f"Reading from: {file_path}")
    
    try:
        with open(file_path, "rb") as fp:
            err_data = pickle.load(fp)
    except FileNotFoundError:
        print("File not found!")
        return

    num_methods = len(err_data)
    method_names = ['OWA1', 'OWA2', 'OWA3', 'OWA4', 'OWA5', 'Timeline', 'MRAC (Ours)']
    if num_methods == 6:
        method_names = ['OWA1', 'OWA2', 'OWA3', 'OWA4', 'Timeline', 'MRAC (Ours)']
    
    print("\n=== Error Data Analysis ===")
    print(f"{'Method':<15} | {'Count':<8} | {'Min':<8} | {'Max':<8} | {'Mean':<8} | {'Zeros(%)':<10} | {'>1(%)':<10}")
    print("-" * 90)
    
    for i in range(num_methods):
        errors = np.array(err_data[i])
        name = method_names[i] if i < len(method_names) else f"Method {i}"
        
        count = len(errors)
        if count == 0:
            print(f"{name:<15} | 0        | N/A      | N/A      | N/A      | N/A        | N/A")
            continue
            
        min_val = np.min(errors)
        max_val = np.max(errors)
        mean_val = np.mean(errors)
        zeros_pct = (np.sum(errors == 0) / count) * 100
        gt_one_pct = (np.sum(errors > 1) / count) * 100
        
        print(f"{name:<15} | {count:<8} | {min_val:<8.4f} | {max_val:<8.4f} | {mean_val:<8.4f} | {zeros_pct:<10.2f} | {gt_one_pct:<10.2f}")

    print("\nRaw data sample (first 20) for MRAC:")
    print(err_data[-1][:20])

if __name__ == "__main__":
    main()

