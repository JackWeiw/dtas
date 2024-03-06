import matplotlib.pyplot as plt
import numpy as np
import json
import os

num_heads = 32
head_dim = 80
def plot_abs(data_dir, dlight_data_file):
    m = []
    dlight = []
    plt.figure(figsize=(24, 13.5)) 
    with open(dlight_data_file,"r") as f1:
        dlight_dic = json.load(f1)
        for kv1,in zip(dlight_dic.items()):
            m.append(kv1[0])
            dlight.append(kv1[1] if kv1[1]<100 else 100)
        plt.plot(m, dlight, label="dlight") 
        
    json_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
    time_values_by_file={}
    m=[i for i in range(1,4097)]
    for file_name in json_files:
        with open(file_name, 'r') as file:
            # 使用jsonlines库处理多对象文件
            relative_speed = [item[1] if item[1]<100 else 100 for item in json.load(file).items()]
            desired_part = file_name.split("/")[-1:]
            # os.path.basename(file_name).split(".")[0]
            desired_part = desired_part[0].split(".")[0]
            time_values_by_file[desired_part]=relative_speed
            plt.plot(m, relative_speed, label=desired_part)    
    plt.legend()
    plt.title("layernorm row=m col=2560 in_f32")
    # plt.axhline(y=0.8, color="r", linestyle="--")
    # plt.axhline(y=0.9, color="r", linestyle="--")
    # plt.axhline(y=1.0, color="r", linestyle="--")
    ticks = np.arange(0, 4097, step=128)
    plt.xticks(ticks)
    plt.xticks(rotation=90) 
    plt.xlabel("rows")
    plt.ylabel("duration(us)")
    plt.savefig("./2560_abs.png")
# plot_abs("/home/weitao/XIAG8XX/profile/testIR/layernorm/data/2560_hid/shared","/home/weitao/XIAG8XX/profile/testIR/layernorm/data/2560_hid/dlight.json")



def plot_duo_relative(data_dir, dlight_data_file):
    m = []
    dlight = []
    plt.figure(figsize=(24, 13.5)) 
    with open(dlight_data_file,"r") as f1:
        dlight_dic = json.load(f1)
        for kv1,in zip(dlight_dic.items()):
            m.append(kv1[0])
            dlight.append(kv1[1])
        plt.plot(m, [1 for _ in range(4096)], label="dlight") 
        
    json_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
    time_values_by_file={}
    m=[i for i in range(1,4097)]
    for file_name in json_files:
        with open(file_name, 'r') as file:
            # 使用jsonlines库处理多对象文件
            relative_speed = [dlight[int(item[0])-1] / item[1] if dlight[int(item[0])-1] / item[1] <= 2 else 2 for item in json.load(file).items()]
            desired_part = file_name.split("/")[-1:]
            # os.path.basename(file_name).split(".")[0]
            desired_part = desired_part[0].split(".")[0]
            time_values_by_file[desired_part]=relative_speed
            plt.plot(m, relative_speed, label=desired_part)    
    plt.legend()
    plt.title("layernorm row=m col=2560 in_f32")
    # plt.axhline(y=0.8, color="r", linestyle="--")
    # plt.axhline(y=0.9, color="r", linestyle="--")
    # plt.axhline(y=1.0, color="r", linestyle="--")
    ticks = np.arange(0, 4097, step=128)
    plt.xticks(ticks)
    plt.xticks(rotation=90) 
    plt.xlabel("rows")
    plt.ylabel("relative_speed")
    plt.savefig("./2560.png")


# 


def layer_shared(file1, file2):
    with open(file1,"r") as f1, open(file2, "r") as f2:
        no_shared_dic = json.load(f1)
        shared_dic = json.load(f2)
        m=[i for i in range(1,4097)]
        shared_bosst = []
        for kv1, kv2 in zip(no_shared_dic.items(), shared_dic.items()):
            shared_bosst.append(kv1[1] / kv2[1]) 
            
    plt.plot(m, shared_bosst, label="shared_boost_layernorm_len_tx224_in3")    
    plt.legend()
    plt.title("softmax num_row = col2560 in_f32")
    ticks = np.arange(0, 4097, step=128)
    plt.xticks(ticks)
    plt.xticks(rotation=90) 
    plt.xlabel("num_cols")
    plt.ylabel("relative_speed")
    plt.savefig("./shared_224_in3.png")
        
# layer_shared("/home/weitao/XIAG8XX/profile/testIR/layernorm/data/2560_hid/224.json","/home/weitao/XIAG8XX/profile/testIR/layernorm/data/2560_hid/224_shared.dyn.json")        
        