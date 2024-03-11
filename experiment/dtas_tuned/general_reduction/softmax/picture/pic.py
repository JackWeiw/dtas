import os
import json
import matplotlib.pyplot as plt
import numpy as np
def load_latency_json(folder_path):
    json_path = os.path.join(folder_path, 'latency.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
            return data
    else:
        return None

def plot_latency_comparison(root_folder):
    plt.figure(figsize=(12, 6))  # 设置图表大小

    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        # Check if the subfolder contains latency.json
        latency_data = load_latency_json(subfolder_path)

        if latency_data:
            x = list(latency_data.keys())
            y = list(latency_data.values())
            
            # Use the subfolder name as label
            label = subfolder
            plt.plot(x, y, label=label, linestyle='-', marker='o')
            
    ticks = np.arange(0, 4097, step=256)
    plt.xticks(ticks)
    plt.xticks(rotation=90) 
    plt.xlabel('col size')
    plt.ylabel('Latency (us)')
    plt.title('Latency Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig("./laytency_comparison.png")

plot_latency_comparison("/home/weitao/XIAG8XX/profile/dtas_tuned/general_reduction/softmax")