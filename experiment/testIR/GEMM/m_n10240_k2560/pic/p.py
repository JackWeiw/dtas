import matplotlib.pyplot as plt
import numpy as np
import json
import os

num_heads = 32
head_dim = 80


def plot_gflops(file1, file2, file3, file4, file5, file6):
    with open(file1, "r") as f1, open(file2, "r") as f2, open(file3, "r") as f3, open(
        file4, "r"
    ) as f4, open(file5, "r") as f5, open(file6, "r") as f6:
        dic1 = json.load(f1)
        dic2 = json.load(f2)
        dic3 = json.load(f3)
        dic4 = json.load(f4)
        dic5 = json.load(f5)
        dic6 = json.load(f6)
        m = []
        cublas = []
        dlight = []
        i4j4_x2y3_in4 = []
        i4j4_x3y2_in4 = []

        i5j4_x2y2_in4 = []
        i3j4_x3y3_in4 = []
        cmp = {}
        for kv1, kv2, kv3, kv4, kv5, kv6 in zip(
            dic1.items(),
            dic2.items(),
            dic3.items(),
            dic4.items(),
            dic5.items(),
            dic6.items(),
        ):
            # cublas/dlight
            m.append(kv1[0])
            # cublas.append()
            dlight.append(kv1[1] / kv2[1])
            i4j4_x2y3_in4.append(kv1[1] / kv3[1])
            i4j4_x3y2_in4.append(kv1[1] / kv4[1])
            i5j4_x2y2_in4.append(kv1[1] / kv5[1])
            i3j4_x3y3_in4.append(kv1[1] / kv6[1])
        plt.figure(figsize=(16, 9))
        plt.axhline(y=0.8, color="r", linestyle="--")
        plt.axhline(y=0.9, color="r", linestyle="--")
        plt.axhline(y=1.0, color="r", linestyle="--")
        plt.plot(m, dlight, label="dlight")
        plt.plot(m, i4j4_x2y3_in4, label="i4j4_x2y3_in4")
        plt.plot(m, i4j4_x3y2_in4, label="i4j4_x3y2_in4")
        plt.plot(m, i5j4_x2y2_in4, label="i5j4_x2y2_in4")
        plt.plot(m, i3j4_x3y3_in4, label="i3j4_x3y3_in4")
        plt.legend()
        plt.title("m_n10240_k2560 f16 baseline cublas")
        # plt.legend(relative,("relative speed"), loc='upper left')
        ticks = np.arange(0, 4097, step=128)
        plt.xticks(ticks)
        plt.xlabel("m")
        plt.ylabel("relative speed")
        # plt.show()
        plt.savefig("./cmp.png")


# plot_gflops(
#     "/home/weitao/XIAG8XX/profile/data/m_n10240_k2560/cublas.json",
#     "/home/weitao/XIAG8XX/profile/data/m_n10240_k2560/dlight.json",
#     "/home/weitao/XIAG8XX/profile/testIR/m_n10240_k2560/data/16x16x16_i4j4_x2y3_in4.json",
#     "/home/weitao/XIAG8XX/profile/testIR/m_n10240_k2560/data/16x16x16_i4j4_x3y2_in4.json",
#     "/home/weitao/XIAG8XX/profile/testIR/m_n10240_k2560/data/16x16x16_i5j4_x2y2_in4.json",
#     "/home/weitao/XIAG8XX/profile/testIR/m_n10240_k2560/data/16x16x16_i3j4_x3y3_in4.json",
# )

def cmp_vec_len(file1,file2,file3):
    with open(file1,"r") as f1,open(file2,"r") as f2,open(file3,"r") as f3:
        dic1 = json.load(f1)
        dic2 = json.load(f2)
        dic3 = json.load(f3)
        m=[]
        dlight = []
        i4j4_x2y2_in8 = []
        for kv1, kv2, kv3 in zip(dic1.items(),dic2.items(), dic3.items()):
            m.append(kv1[0])
            dlight.append(kv1[1] / kv2[1])
            i4j4_x2y2_in8.append(kv1[1]/kv3[1])
            
        plt.figure(figsize=(16, 9))
        plt.axhline(y=0.8, color="r", linestyle="--")
        plt.axhline(y=0.9, color="r", linestyle="--")
        plt.axhline(y=1.0, color="r", linestyle="--")
        plt.plot(m, dlight, label="dlight")
        plt.plot(m, i4j4_x2y2_in8, label="i4j4_x2y2_in8")
        plt.legend()
        plt.title("m_n10240_k2560 f16")
        # plt.legend(relative,("relative speed"), loc='upper left')
        ticks = np.arange(0, 4097, step=128)
        plt.xticks(ticks)
        plt.xlabel("m")
        plt.ylabel("relative speed")
        # plt.show()
        plt.savefig("./vec_4&8.png")
# cmp_vec_len(
#     "/home/weitao/XIAG8XX/profile/data/m_n10240_k2560/cublas.json",
#     "/home/weitao/XIAG8XX/profile/data/m_n10240_k2560/dlight.json",
#     "/home/weitao/XIAG8XX/profile/testIR/m_n10240_k2560/data/16x16x16_i4j4_x2y2_in8.json"
# )        
    

def cmp_async(file1,file2,file3,file4):
     with open(file1,"r") as f1, open(file2,"r") as f2, open(file3,"r") as f3, open(file4,"r") as f4:
        dic1 = json.load(f1)
        dic2 = json.load(f2)
        dic3 = json.load(f3)
        dic4 = json.load(f4)
        m = []
        dlight = []
        i3j4_x2y2_in4_async = []
        i4j2_x2y2_in4_async = []
        for kv1, kv2, kv3 ,kv4 in zip(dic1.items(),dic2.items(), dic3.items() ,dic4.items()):
            m.append(kv1[0])
            dlight.append(kv1[1] / kv2[1])
            i3j4_x2y2_in4_async.append(kv1[1]/kv3[1])
            i4j2_x2y2_in4_async.append(kv1[1]/kv4[1])
        plt.figure(figsize=(16, 9))
        plt.axhline(y=0.8, color="r", linestyle="--")
        plt.axhline(y=0.9, color="r", linestyle="--")
        plt.axhline(y=1.0, color="r", linestyle="--")
        plt.plot(m, dlight, label="dlight")
        plt.plot(m, i3j4_x2y2_in4_async, label="i3j4_x2y2_in4_async")
        plt.plot(m, i4j2_x2y2_in4_async, label="i4j2_x2y2_in4_async")
        plt.legend()
        plt.title("m_n10240_k2560 f16")
        # plt.legend(relative,("relative speed"), loc='upper left')
        ticks = np.arange(0, 4097, step=128)
        plt.xticks(ticks)
        plt.xlabel("m")
        plt.ylabel("relative speed")
        # plt.show()
        plt.savefig("./async.png")
        
# cmp_async(
#     "/home/weitao/XIAG8XX/profile/data/m_n10240_k2560/cublas.json",
#     "/home/weitao/XIAG8XX/profile/data/m_n10240_k2560/dlight.json",
#     "/home/weitao/XIAG8XX/profile/testIR/m_n10240_k2560/data/16x16x16_i3j4_x2y2_in4_async.json",
#     "/home/weitao/XIAG8XX/profile/testIR/m_n10240_k2560/data/16x16x16_i4j2_x2y2_in4_async.json"
# )

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
            desired_part = os.path.basename(file_name).split(".")[0]
            print(desired_part)
            time_values_by_file[desired_part]=relative_speed
            plt.plot(m, relative_speed, label=desired_part)    
    plt.legend()
    plt.title("m_n10240_k2560 f16")
    plt.axhline(y=0.8, color="r", linestyle="--")
    plt.axhline(y=0.9, color="r", linestyle="--")
    plt.axhline(y=1.0, color="r", linestyle="--")
    ticks = np.arange(0, 4097, step=128)
    plt.xticks(ticks)
    plt.xticks(rotation=90) 
    plt.xlabel("m")
    plt.savefig("./16x16x16.png")
# plot_duo("/home/weitao/XIAG8XX/profile/testIR/m_n10240_k2560/data/16x16x16","/home/weitao/XIAG8XX/profile/data/m_n10240_k2560/cublas.json","/home/weitao/XIAG8XX/profile/testIR/m_n10240_k2560/data/dlight.json")