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
            label = os.path.basename(root_folder) + subfolder
            plt.plot(x, y, label=label, linestyle='-', marker='o')
            
    ticks = np.arange(0, 4097, step=256)
    plt.xticks(ticks)
    plt.xticks(rotation=90) 
    plt.xlabel('col size')
    plt.ylabel('Latency (us)')
    plt.title(f'Latency Comparison {os.path.basename(root_folder)} ')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./laytency_{os.path.basename(root_folder)}.png")

def plot_latency_comparison_relative(root_folder):
    dlight_folder = os.path.join(root_folder, 'dlight')

    plt.figure(figsize=(12, 6))  # 设置图表大小

    # Load baseline data (dlight)
    dlight_data = load_latency_json(dlight_folder)

    if not dlight_data:
        print("Error: 'latency.json' not found in dlight folder.")
        return

    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        # Check if the subfolder contains latency.json
        latency_data = load_latency_json(subfolder_path)

        if latency_data:
            x = list(latency_data.keys())
            y = [dlight_data[key] / latency_data[key]  for key in x]
            label = f"{os.path.basename(root_folder)}_{subfolder}"
            plt.plot(x, y, label=label, linestyle='-', marker='o')
    ticks = np.arange(0, 4097, step=256)
    plt.xticks(ticks)
    plt.xticks(rotation=90) 
    plt.xlabel('col size')
    plt.ylabel('Relative Latency')
    plt.title('Latency Comparison (Relative to dlight)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./relative_{os.path.basename(root_folder)}.png")

# plot_latency_comparison_relative("/home/weitao/XIAG8XX/profile/dtas_tuned/general_reduction/softmax/row_1000")
# plot_latency_comparison("/home/weitao/XIAG8XX/profile/dtas_tuned/general_reduction/softmax/row_12800")

def compare_json_times(dlight, dtas):
    # 读取第一个 JSON 文件
    with open(dlight, 'r') as f:
        json_data1 = json.load(f)

    # 读取第二个 JSON 文件
    with open(dtas, 'r') as f:
        json_data2 = json.load(f)

    # 提取时间数据
    times1 = [json_data1[str(i)] for i in range(1, 4097)]
    times2 = [json_data2[str(i)] for i in range(1, 4097)]
    relative_speed = [time1 / time2 for time1, time2 in zip(times1, times2)]
    # 绘制图表
    plt.figure(figsize=(10, 5))
    
    # 创建图表和子图
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制绝对时间比较
    ax1.plot(range(1, 4097), times1, label='dlight', color='blue')
    ax1.plot(range(1, 4097), times2, label='dtas', color='green')
    ax1.set_xlabel('col')
    ax1.set_ylabel('Duration(us)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left')

    # 创建共享 x 轴但不同 y 轴的双轴
    ax2 = ax1.twinx()
    ax2.plot(range(1, 4097), relative_speed, label='Relative Speed', color='red')
    ax2.set_ylabel('Relative Speed', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')
    # # 绘制第一个 JSON 文件的时间数据
    # plt.plot(range(1, 4097), times1, label='dlight')

    # # 绘制第二个 JSON 文件的时间数据
    # plt.plot(range(1, 4097), times2, label='dtas')

    # plt.xlabel('col')
    # plt.ylabel('duration(us)')
    plt.grid(True)
    plt.title('Softmax row 12800 dlight vs dtas')
    plt.legend()
    plt.savefig("./dtas.png")

# 使用示例
compare_json_times('/home/weitao/XIAG8XX/profile/dtas_tuned/general_reduction/softmax/row_12800/dlight/latency.json', '/home/weitao/XIAG8XX/profile/dtas_tuned/general_reduction/softmax/nosmem/row_12800/top10_256_10_nounroll/latency.json')