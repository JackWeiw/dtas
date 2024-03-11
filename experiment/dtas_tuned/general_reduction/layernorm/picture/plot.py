import os  
import json  
import matplotlib.pyplot as plt  
import numpy as np  
  
# 设置文件夹路径  
base_folder = "/home/weitao/XIAG8XX/profile/dtas_tuned/general_reduction/layernorm"  
  
# 获取子文件夹名列表  
subfolders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]  
  
# 初始化一个空的字典来存储均值数据  
mean_latency_data = {}  
  
# 遍历每个子文件夹  
for subfolder in subfolders:  
    latency_values = []  
    subfolder_path = os.path.join(base_folder, subfolder)  
    json_path = os.path.join(subfolder_path, 'latency.json')  
      
    # 读取json文件  
    with open(json_path, 'r') as f:  
        latency_dict = json.load(f)  
      
    # 提取latency值，并计算每256个key的均值  
    for i in range(0, 4097, 256):  
        # 使用np.mean计算当前256个key的latency均值  
        latency_slice = latency_dict.get(str(i), np.nan) if i == 0 else [latency_dict.get(str(j), np.nan) for j in range(i, min(i + 256, 4097))]  
        latency_values.append(np.nanmean(latency_slice))  # 使用np.nanmean忽略NaN值  
      
    # 将均值数据添加到mean_latency_data字典中，以子文件夹名为键  
    mean_latency_data[subfolder] = latency_values  
  
# 绘制折线图  
plt.figure(figsize=(12, 8))  
for label, values in mean_latency_data.items():  
    plt.plot(range(0, len(values) * 256, 256), values, label=label, marker='o')  
  
# 设置x轴标签（只显示开始和结束的标签）  
plt.xticks([0, len(values) * 256 - 1], ['0', '4096'])  
plt.xlabel('Key Range')  
plt.ylabel('Mean Latency')  
plt.title('Mean Latency Comparison')  
  
# 显示图例  
plt.legend()  
  
# 显示图形  
plt.grid(True)  # 添加网格线以便更好地查看数据 
  
# 显示图形  
plt.savefig("./wtm3.png")