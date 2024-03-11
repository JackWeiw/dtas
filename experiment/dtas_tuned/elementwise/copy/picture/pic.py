import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_bandwidth_json(folder_path):
    json_path = os.path.join(folder_path, 'bandwidth.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
            return data
    else:
        return None

def load_latency_json(folder_path):
    json_path = os.path.join(folder_path, 'latency.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
            return data
    else:
        return None

def plot_bandwidth_comparison(root_folder):
    plt.figure(figsize=(16, 8))  # 设置图表大小

    for root, dirs, files in os.walk(root_folder):
        # Check if 'bandwidth.json' exists in the current directory
        if 'bandwidth.json' in files:
            # Load bandwidth data
            bandwidth_data = load_bandwidth_json(root)

            if bandwidth_data:
                x = list(bandwidth_data.keys())
                y = list(bandwidth_data.values())
                
                # Use the last two folder names as labels
                parent_folder = os.path.basename(os.path.dirname(root))
                current_folder = os.path.basename(root)
                label = f"{parent_folder}_{current_folder}"
                print(f"Plotting {label}, bandwidth mean: {np.mean(y)}")
                plt.plot(x, y, label=label)

    ticks = np.arange(0, 4097, step=256)
    plt.xticks(ticks)
    plt.xticks(rotation=90) 
    plt.xlabel('Key')
    plt.ylabel('Bandwidth')
    plt.title('Copy Bandwidth Comparison')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("./i8bandwidth_comparison.png")

def plot_throught_comparison(root_folder):    
    plt.figure(figsize=(16, 8))  # 设置图表大小

    for root, dirs, files in os.walk(root_folder):
        # Check if 'bandwidth.json' exists in the current directory
        if 'latency.json' in files:
            # Load bandwidth data
            latency_data = load_latency_json(root)

            if latency_data:
                x = list(latency_data.keys())
                y = list(latency_data.values())
                throughputs = [(float(i) * 2560) / float(y[int(i)-1])/1e4 for i in x]
                
                # Use the last two folder names as labels
                parent_folder = os.path.basename(os.path.dirname(root))
                current_folder = os.path.basename(root)
                label = f"{parent_folder}_{current_folder}"
                print(f"Plotting {label}, throughput mean: {np.mean(throughputs)}")
                plt.plot(x, throughputs, label=label)

    ticks = np.arange(0, 4097, step=256)
    plt.xticks(ticks)
    plt.xticks(rotation=90) 
    plt.xlabel('Key')
    plt.ylabel('throughputs (TOPs/s)')
    plt.title('Copy Throughputs Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig("./i8throughput_comparison.png")
# if __name__ == "__main__":
#     root_folder = os.path.dirname(os.path.abspath(__file__))
#     plot_bandwidth_comparison(root_folder)

plot_bandwidth_comparison('/home/weitao/XIAG8XX/profile/dtas_tuned/elementwise/copy/int8')
plot_throught_comparison('/home/weitao/XIAG8XX/profile/dtas_tuned/elementwise/copy/int8')