import json
from typing import Dict, Tuple

import tvm
from tvm import tir

from ..common.config import Config, Range
from ..codegen.compile_result import CompileResult

class DataBase:
    def __init__(self, best_record_path: str, tuning_record_path: str):
        self.data = {}
        self.best_record_path = best_record_path
        self.tuning_record_path = tuning_record_path
        
    def commit_tuning_record(self, range_tuple: Tuple[Range], cr: CompileResult):
        try:
            with open(self.tuning_record_path, 'a') as f:
                f.write(f'#name: {cr.name}\n#range: {range_tuple}  latency(ms): {cr.latency:^12.4f} \n')
                f.write(f"#config: {cr.config.__str__()}\n")
                f.write(f"{cr.sch.mod}\n \n")
        except:
            return 
            
    def commit_best_record(self, range_tuple: Tuple[Range], cr:CompileResult):
        dic = {}
        key = cr.name
        for r in range_tuple:
            key += "_" + r.__str__()    
        dic[key] = {"latency(ms)":cr.latency, "config":[cr.config.to_dict()],  "mod":cr.sch.mod.script()}
        def _serialize(obj):
            if isinstance(obj, tvm.tir.IntImm):
                return obj.value
            raise TypeError(f"Type {type(obj)} not serializable")
        try:
            with open(self.best_record_path, "r") as file:
                data = json.load(file)
        except FileNotFoundError:
            # 如果文件不存在，创建一个空的数据列表
            data = []
        data.append(dic)
        with open(self.best_record_path, 'w') as f:
            f.write(json.dumps(data, default=_serialize, indent=4))

    def save_to_file(self, filename: str):
        with open(filename, 'w') as file:
            json.dump(self.data, file, indent=2)

    def load_from_file(self, filename: str):
        with open(filename, 'r') as file:
            self.data = json.load(file)