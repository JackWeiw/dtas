import matplotlib.pyplot as plt
import numpy as np
import json
import os

num_heads = 32
head_dim = 80



def plot_duo(data_dir,cublas_data_file,dlight_data_file):
    m = []
    dlight = []
    plt.figure(figsize=(24, 13.5)) 
    with open(cublas_data_file,"r") as f1, open(dlight_data_file,"r") as f2:
        cublas_dic = json.load(f1)
        dlight_dic = json.load(f2)
        for kv1, kv2 in zip(cublas_dic.items(),dlight_dic.items()):
            m.append(kv1[0])
            dlight.append(kv1[1] / kv2[1])
        plt.plot(m, dlight, label="dlight") 
    json_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
    time_values_by_file={}
    m=[i for i in range(1,4097)]
    for file_name in json_files:
        with open(file_name, 'r') as file:
            # 使用jsonlines库处理多对象文件
            relative_speed = [cublas_dic[item[0]]/item[1] for item in json.load(file).items()]
            desired_part = file_name.split("/")[-2:]
            # os.path.basename(file_name).split(".")[0]
            desired_part = desired_part[0] +"_" + desired_part[1].split(".")[0]
            time_values_by_file[desired_part]=relative_speed
            plt.plot(m, relative_speed, label=desired_part)    
    plt.legend()
    plt.title("m_n12560_k10240 f16")
    plt.axhline(y=0.8, color="r", linestyle="--")
    plt.axhline(y=0.9, color="r", linestyle="--")
    plt.axhline(y=1.0, color="r", linestyle="--")
    ticks = np.arange(0, 4097, step=128)
    plt.xticks(ticks)
    plt.xticks(rotation=90) 
    plt.xlabel("m")
    plt.savefig("./16x16x16_duo.png")
# plot_duo("/home/weitao/XIAG8XX/profile/testIR/GEMM/m_n2560_k10240/data/16x16x16","/home/weitao/XIAG8XX/profile/testIR/GEMM/m_n2560_k10240/data/cublas.json","/home/weitao/XIAG8XX/profile/testIR/GEMM/m_n2560_k10240/data/dlight.json")


def plot_duo_abs(data_dir,cublas_data_file,dlight_data_file):
    m = []
    dlight = []
    cublas = []
    plt.figure(figsize=(24, 13.5)) 
    with open(cublas_data_file,"r") as f1, open(dlight_data_file,"r") as f2:
        cublas_dic = json.load(f1)
        dlight_dic = json.load(f2)
        for kv1, kv2 in zip(cublas_dic.items(),dlight_dic.items()):
            m.append(kv1[0])
            dlight.append(kv2[1])
            cublas.append(kv1[1])
        plt.plot(m, dlight, label="dlight") 
        plt.plot(m, cublas, label="cublas") 
    json_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
    time_values_by_file={}
    m=[i for i in range(1,4097)]
    for file_name in json_files:
        with open(file_name, 'r') as file:
            # 使用jsonlines库处理多对象文件
            relative_speed = [item[1] for item in json.load(file).items()]
            desired_part = file_name.split("/")[-2:]
            # os.path.basename(file_name).split(".")[0]
            desired_part = desired_part[0] +"_" + desired_part[1].split(".")[0]
            time_values_by_file[desired_part]=relative_speed
            plt.plot(m, relative_speed, label=desired_part)    
    plt.legend()
    plt.title("m_n12560_k10240 f16")
    # plt.axhline(y=0.8, color="r", linestyle="--")
    # plt.axhline(y=0.9, color="r", linestyle="--")
    # plt.axhline(y=1.0, color="r", linestyle="--")
    ticks = np.arange(0, 4097, step=128)
    plt.xticks(ticks)
    plt.xticks(rotation=90) 
    plt.xlabel("m")
    plt.ylabel("duration(us)")
    plt.savefig("./16x16x16_abs.png")

# plot_duo_abs("/home/weitao/XIAG8XX/profile/testIR/GEMM/m_n2560_k10240/data/16x16x16","/home/weitao/XIAG8XX/profile/testIR/GEMM/m_n2560_k10240/data/cublas.json","/home/weitao/XIAG8XX/profile/testIR/GEMM/m_n2560_k10240/data/dlight.json")

def plot_wtm(wtm_dir,cublas_data_file,dlight_data_file):
    m = []
    dlight = []
    plt.figure(figsize=(24, 13.5)) 
    with open(cublas_data_file,"r") as f1, open(dlight_data_file,"r") as f2:
        cublas_dic = json.load(f1)
        dlight_dic = json.load(f2)
        for kv1, kv2 in zip(cublas_dic.items(),dlight_dic.items()):
            m.append(kv1[0])
            dlight.append(kv1[1] / kv2[1])
        plt.plot(m, dlight, label="dlight") 

    m=[i for i in range(1,4097)]
    with open(wtm_dir, 'r') as file:
        # 使用jsonlines库处理多对象文件
        relative_speed = [cublas_dic[item[0]]/item[1] for item in json.load(file).items()]
        # os.path.basename(file_name).split(".")[0]
        plt.plot(m, relative_speed, label="wtm")    
    plt.legend()
    plt.title("m_n12560_k10240 f16")
    plt.axhline(y=0.8, color="r", linestyle="--")
    plt.axhline(y=0.9, color="r", linestyle="--")
    plt.axhline(y=1.0, color="r", linestyle="--")
    ticks = np.arange(0, 4097, step=128)
    plt.xticks(ticks)
    plt.xticks(rotation=90) 
    plt.xlabel("m")
    plt.savefig("./wtm.png")

plot_wtm("/home/weitao/XIAG8XX/profile/testIR/GEMM/m_n2560_k10240/data/top40_256.json","/home/weitao/XIAG8XX/profile/testIR/GEMM/m_n2560_k10240/data/cublas.json","/home/weitao/XIAG8XX/profile/testIR/GEMM/m_n2560_k10240/data/dlight.json")