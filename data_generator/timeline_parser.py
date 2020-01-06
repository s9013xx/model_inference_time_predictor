import re
import os 
import sys
import json
import argparse
import numpy as np
import pandas as pd
import glob
from os import listdir
from pprint import pprint

class Recorders(object):
    """ "Store Data infos """
    def __init__(self, eztags, json_data):
        self.eztags    = eztags
        self.init_time = eztags.init_time
        self.json_data = json_data
        self.replica_transpose_in = RecorderBase("replica_transpose_in", eztags.init_time,
            eztags.replica_gpu.pid, eztags.transpose_in.search_pattern, json_data)
        
        self.compute_transpose_in = Recorder_CmdInHelp("compute_transpose_in", eztags.init_time,
            eztags.all_compute.pid, eztags.transpose_in.search_pattern, json_data, self.replica_transpose_in.existed)
        
        self.replica_transpose_out = RecorderBase("replica_transpose_out", eztags.init_time,
            eztags.replica_gpu.pid, eztags.transpose_out.search_pattern, json_data)
        
        self.compute_transpose_out = RecorderBase("compute_transpose_out", eztags.init_time,
            eztags.all_compute.pid, eztags.transpose_out.search_pattern, json_data)
        
        self.memcpyD2H = Recorder_MemcpyD2H("memcpyD2H", eztags.init_time,
            eztags.memcpyD2H.pid, eztags.memcpyD2H.search_pattern, json_data)

        self.retval = RecorderBase("retval", eztags.init_time,
            eztags.retval.pid, eztags.retval.search_pattern, json_data)
        
        self.first_gpu = Recorder_Frist("frist_compute", eztags.init_time,
            eztags.all_compute.pid, None, json_data)
        
        self.last_gpu = Recorder_Last("last_compute", eztags.init_time,
            eztags.all_compute.pid, None, json_data, self.memcpyD2H)

    def __str__(self):
        tmp_str  = "name:{}, s: {}, w:{}\n".format(self.replica_transpose_in.name, self.replica_transpose_in.start_time, self.replica_transpose_in.wall_time)
        tmp_str += "name:{}, s: {}, w:{}\n".format(self.compute_transpose_in.name, self.compute_transpose_in.start_time, self.compute_transpose_in.wall_time)
        tmp_str += "name:{}, s: {}, w:{}\n".format(self.replica_transpose_out.name, self.replica_transpose_out.start_time, self.replica_transpose_out.wall_time)
        tmp_str += "name:{}, s: {}, w:{}\n".format(self.compute_transpose_out.name, self.compute_transpose_out.start_time, self.compute_transpose_out.wall_time)
        tmp_str += "name:{}, s: {}, w:{}\n".format(self.memcpyD2H.name, self.memcpyD2H.start_time, self.memcpyD2H.wall_time)
        tmp_str += "name:{}, s: {}, w:{}\n".format(self.retval.name, self.retval.start_time, self.retval.wall_time)
        tmp_str += "name:{}, s: {}, w:{}\n".format(self.first_gpu.name, self.first_gpu.start_time, self.first_gpu.wall_time)
        tmp_str += "name:{}, s: {}, w:{}".format(self.last_gpu.name, self.last_gpu.start_time, self.last_gpu.wall_time)
        return tmp_str

class RecorderBase(object):
    """Store Data info"""
    def __init__(self, name, init_time, pid, pattern, json_data):
        self._name       = name
        self._init_time  = init_time
        self._pid        = pid
        self._pattern    = pattern
        self._existed    = False
        self._wall_time  = 0
        self._start_time = 0
        #self.json_data   = json_data
        self.set_time(json_data)
    
    def set_time(self, json_data):
        if self._existed:
            return
        for item in json_data['traceEvents']:
            if self._existed:
                break
            if 'pid' in item and item['pid'] == self._pid:
                if 'args' in item and re.search(self._pattern, item['args']['name'], re.M|re.I):
                    if 'ts' in item and 'dur' in item:
                        self._existed = True 
                        #print(item, self._pattern)
                        self._start_time = float(item['ts']) - self.init_time
                        self._wall_time  = item['dur']
    
    @property
    def init_time(self):
        return int(self._init_time)

    @property
    def name(self):
        return self._name

    @property
    def start_time(self):
        return int(self._start_time)
    
    @property
    def wall_time(self):
        return int(self._wall_time)
    
    @property
    def existed(self):
        return self._existed
    
class Recorder_CmdInHelp(RecorderBase):
    """Store Data info for transposeIn (Maybe not found name in pid)"""
    def __init__(self, name, init_time, pid, pattern, json_data, cmd_existed=False):
        self.cmd_existed = cmd_existed
        super().__init__(name, init_time, pid, pattern, json_data)
        
    
    def set_time(self, json_data):
        if self._existed:
            return
        if self.cmd_existed: #Frist is transpose in 
            first_exe = None 
            first_time = sys.maxsize
            for item in json_data['traceEvents']:
                if 'pid' in item and item['pid'] == self._pid:
                    if 'ts' in item and 'dur' in item and item['ts'] < first_time:
                        first_time = item['ts']
                        first_exe  = item
            if first_exe:
                self._existed = True 
                self._start_time = float(first_exe['ts']) - self.init_time
                self._wall_time  = first_exe['dur']
        else:
            for item in json_data['traceEvents']:
                if 'pid' in item and item['pid'] == self._pid:
                    if 'args' in item and re.search(self._pattern, item['args']['name'], re.M|re.I):
                        if 'ts' in item and 'dur' in item:
                            self._existed = True 
                            self._start_time = float(item['ts']) - self.init_time
                            self._wall_time  = item['dur']

class Recorder_Frist(RecorderBase):
    """Store Data info for first data"""
    def __init__(self, name, init_time, pid, pattern, json_data, cmd_existed=False):
        self.cmd_existed = cmd_existed
        super().__init__(name, init_time, pid, pattern, json_data)
        
    def set_time(self, json_data):
        if self._existed:
            return
        first_exe = None 
        first_time = sys.maxsize
        for item in json_data['traceEvents']:
            if 'pid' in item and item['pid'] == self._pid:
                if 'ts' in item and 'dur' in item and item['ts'] < first_time:
                    first_time = item['ts']
                    first_exe  = item
        if first_exe:
            self._existed = True 
            self._start_time = float(first_exe['ts']) - self.init_time
            self._wall_time  = first_exe['dur']

class Recorder_Last(RecorderBase):
    """Store Data info for Last data"""
    def __init__(self, name, init_time, pid, pattern, json_data, memcpyD2H):
        self.memcpyD2H = memcpyD2H
        super().__init__(name, init_time, pid, pattern, json_data)
    def set_time(self, json_data):
        if self._existed:
            return
        last_exe = None 
        last_time = self.init_time
        for item in json_data['traceEvents']:
            if 'pid' in item and item['pid'] == self._pid:
                if 'ts' in item and 'dur' in item and item['ts'] >= last_time:
                    if self.memcpyD2H.existed and (self.memcpyD2H.start_time + self.init_time) == item['ts']:
                        continue
                    else:
                        last_time = item['ts']
                        last_exe  = item
        if last_exe:
            self._existed = True 
            self._start_time = float(last_exe['ts']) - self.init_time
            self._wall_time  = last_exe['dur']
                        
class Recorder_MemcpyD2H(RecorderBase):
    """Store Data info for memcpyD2H"""
    def __init__(self, name, init_time, pid, pattern, json_data):
        super().__init__(name, init_time, pid, pattern, json_data)
    def set_time(self, json_data):
        if self._existed:
            return
        for item in json_data['traceEvents']:
            if self._existed:
                break
            if 'pid' in item and item['pid'] == self._pid:
                if 'args' in item and re.search(self._pattern, item['args']['name'], re.M|re.I):
                    if 'ts' in item and 'dur' in item:
                        self._existed = True 
                        self._start_time = float(item['ts']) - self.init_time
                        self._wall_time  = item['dur']
                elif 'args' in item and re.search(self._pattern, item['args']['op'], re.M|re.I):
                    if 'ts' in item and 'dur' in item:
                        self._existed = True 
                        self._start_time = float(item['ts']) - self.init_time
                        self._wall_time  = item['dur']

class EasyTags(object):
    """"Tags of all important process name"""
    def __init__(self):
        self.init_time     = sys.maxsize
        self.replica_cpu   = EasyTag('replica_cpu')
        self.replica_gpu   = EasyTag('replica_gpu')
        self.all_compute   = EasyTag('all_compute')
        self.transpose_in  = EasyTag('transpose_in')
        self.transpose_out = EasyTag('transpose_out')
        self.memcpyD2H     = EasyTag('memcpyD2H')
        self.retval        = EasyTag('retval')
        
    def __str__(self):
        tmp_str  = "[Init time] {}\n".format(self.init_time)
        tmp_str += "[{}] existed: {}, pid is {}\n".format(self.replica_cpu.name, self.replica_cpu.existed, self.replica_cpu.pid)
        tmp_str += "[{}] existed: {}, pid is {}\n".format(self.replica_gpu.name, self.replica_gpu.existed, self.replica_gpu.pid)
        tmp_str += "[{}] existed: {}, pid is {}\n".format(self.all_compute.name, self.all_compute.existed, self.all_compute.pid)
        tmp_str += "[{}] existed: {}, pid is {}\n".format(self.transpose_in.name, self.transpose_in.existed, self.transpose_in.pid)
        tmp_str += "[{}] existed: {}, pid is {}\n".format(self.transpose_out.name, self.transpose_out.existed, self.transpose_out.pid)
        tmp_str += "[{}] existed: {}, pid is {}\n".format(self.memcpyD2H.name, self.memcpyD2H.existed, self.memcpyD2H.pid)
        tmp_str += "[{}] existed: {}, pid is {}\n".format(self.retval.name, self.retval.existed, self.retval.pid)
        return tmp_str

class EasyTag(object):
    "The Tags of the important process name"
    def __init__(self, name):
        self._name = name
        self._existed = False
        self._pid  = None
        self._pattern = None

    def set_pid(self, pid):
        if not self._existed:
            self._pid = pid
            self._existed = True        

    def set_pattern(self, pattern):
        self._pattern = pattern

    @property
    def pid(self):
        return self._pid
    
    @property
    def existed(self):
        return self._existed
    
    @property
    def name(self):
        return self._name
    
    @property
    def search_pattern(self):
        return self._pattern

def read_flags():
    parser = argparse.ArgumentParser('Parser for timeline data')
    # Benchmarks parameters
    parser.add_argument('--filename', '-f', type=str, default=os.path.join(os.getcwd(), 'timeline', 'lenet_1_to_1_bs1.json'), help='input jon file')
    parser.add_argument('--all_compute', type=str, default='(GPU:0)*(all Compute)', help='search tag - all_compute')
    parser.add_argument('--replica_gpu', type=str, default='(replica:0)*(GPU:0)+ (Compute)+', help='search tag - replica_gpu')
    parser.add_argument('--replica_cpu', type=str, default='(replica:0)*(CPU:0)+ (Compute)+', help='search tag - replica_cpu')
    parser.add_argument('--memcpyD2H', type=str, default='memcpy', help='search tag - memcpy')
    parser.add_argument('--trans_in', type=str, default='TransposeNHWCToNCHW', help='search tag - trans_in')
    parser.add_argument('--trans_out', type=str, default='TransposeNCHWToNHWC', help='search tag - trans_out')
    parser.add_argument('--retval', type=str, default='retval', help='search tag - retval')
    
    # Benchmarks parameters
    parser.add_argument('--operation', '-op', type=str, default='conv', help='operation like conv, dense, pooling')
    parser.add_argument('--input_dir', '-id', type=str, default=os.path.join(os.getcwd(), 'profile_test'), help='input jon file')
    parser.add_argument('--output_dir', '-od', type=str, default=os.path.join(os.getcwd()), help='output csv file')
    parser.add_argument('--output_file_name', '-ofn', type=str, default='all_data.csv', help='output csv file name')
    parser.add_argument('--device', type=str, default='', help='Device name as appearing in logfile')

    args = parser.parse_args()

    if args.device == '':
        print('you should use --device parameter to specify collect data for which device, ex: --device 2080ti')
        exit()
    return args

def get_easytags(flags, json_data):
    eztags = EasyTags()
    eztags.replica_cpu.set_pattern(flags.replica_cpu)
    eztags.replica_gpu.set_pattern(flags.replica_gpu)
    eztags.all_compute.set_pattern(flags.all_compute)
    eztags.transpose_in.set_pattern(flags.trans_in)
    eztags.transpose_out.set_pattern(flags.trans_out)
    eztags.memcpyD2H.set_pattern(flags.memcpyD2H)
    eztags.retval.set_pattern(flags.retval)

    for item in json_data['traceEvents']:
        if 'ts' in item and item['ts'] < eztags.init_time:
            eztags.init_time = item['ts']
        if 'name' in item and item['name'] == 'process_name':
            if re.search(flags.all_compute, item['args']['name'], re.M|re.I):
                eztags.all_compute.set_pid(item['pid'])
            if re.search(flags.replica_gpu, item['args']['name'], re.M|re.I):
                eztags.replica_gpu.set_pid(item['pid'])
            if re.search(flags.replica_cpu, item['args']['name'], re.M|re.I):
                eztags.replica_cpu.set_pid(item['pid'])
            if re.search(flags.memcpyD2H, item['args']['name'], re.M|re.I):
                eztags.memcpyD2H.set_pid(item['pid'])
                
    # Second Round
    for item in json_data['traceEvents']:
        if eztags.transpose_in.existed and eztags.transpose_out.existed and eztags.retval.existed:
            break
        if eztags.replica_gpu.existed and 'args' in item:
            if 'pid' in item and eztags.replica_gpu.pid == item['pid']:
                if re.search(flags.trans_in, item['args']['name'], re.M|re.I):
                    eztags.transpose_in.set_pid(eztags.all_compute.pid)
                if re.search(flags.trans_out, item['args']['name'], re.M|re.I):
                    eztags.transpose_out.set_pid(eztags.all_compute.pid)
        if eztags.replica_cpu.existed and 'args' in item:
            if 'pid' in item and eztags.replica_cpu.pid == item['pid']:
                if re.search(flags.retval, item['args']['name'], re.M|re.I):
                    eztags.retval.set_pid(eztags.replica_cpu.pid)
    return eztags

def main():
    flags = read_flags()
    
    preprocess_time_list = []
    execution_time_list = []
    memcpy_time_list = []
    retval_time_list = []
    retval_half_time_list = []
    sess_time_list = []

    # files = listdir(flags.input_dir)
    files = list(filter(os.path.isfile, glob.glob(flags.input_dir+ "/" + "*")))
    # print(files)
    files.sort(key=lambda x: os.path.getmtime(x))
    for json_file in files:
    # for json_file in sorted(listdir(flags.input_dir)):
        # file_path = os.path.join(flags.input_dir, json_file)
        json_data = None
        print('file_path:', json_file)
        with open(json_file, 'r') as f:
            json_data = json.load(f)
            # print(json_data)
    
        eztags = get_easytags(flags, json_data)
        # print(eztags)
        recorders = Recorders(eztags, json_data)
        # print(recorders)
        transOut_time   = recorders.compute_transpose_out.wall_time + recorders.replica_transpose_out.wall_time
        last_gpu_time   = recorders.last_gpu.start_time + recorders.last_gpu.wall_time 
        memcpyD2H_time  = recorders.memcpyD2H.start_time + recorders.memcpyD2H.wall_time
        
        preprocess_time = recorders.first_gpu.start_time + recorders.compute_transpose_in.wall_time
        execution_time  = last_gpu_time - preprocess_time - transOut_time
        memcpy_time     = memcpyD2H_time - last_gpu_time + transOut_time
        retval_time     = recorders.retval.start_time + recorders.retval.wall_time - memcpyD2H_time
        retval_half_time = retval_time / 2 

        if recorders.retval.start_time:
            sess_time = recorders.retval.start_time + recorders.retval.wall_time
        elif memcpyD2H_time:
            sess_time = memcpyD2H_time
        else:
            sess_time = last_gpu_time

        preprocess_time_list.append(preprocess_time/1000)
        execution_time_list.append(execution_time/1000)
        memcpy_time_list.append(memcpy_time/1000)
        retval_time_list.append(retval_time/1000)
        retval_half_time_list.append(retval_half_time/1000)
        sess_time_list.append(sess_time/1000)

        # print("preprocess:       {} ms".format(preprocess_time/1000))
        # print("execution_time:   {} ms".format(execution_time/1000))
        # print("memcpy_time:      {} ms".format(memcpy_time/1000))
        # print("retval_time:      {} ms".format(retval_time/1000))
        # print("retval_half_time: {} ms".format(retval_half_time/1000))
        # print("session time      {} ms".format(sess_time/1000))

    if flags.operation == 'conv':
        conv_col_name = ['batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 'padding', 'activation_fct', 'use_bias', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']
        df = pd.read_csv('goldan_values/conv_goldan_values_1080ti_20191210041830.csv', usecols=conv_col_name)
    elif flags.operation == 'dense':
        fc_col_name = ['batchsize', 'dim_input', 'dim_output', 'activation_fct', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']
        df = pd.read_csv('goldan_values/fc_goldan_values_1080ti_20191213144429.csv', usecols=fc_col_name)
    elif flags.operation == 'pooling':
        pool_col_name = ['batchsize', 'matsize', 'channels_in', 'poolsize', 'padding', 'strides', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']
        df = pd.read_csv('goldan_values/pool_goldan_values_1080ti_20191216084434.csv', usecols=pool_col_name)
    else:
        print('wrong operations')
        exit()
    df['preprocess_time'] = pd.DataFrame(np.array(preprocess_time_list))
    df['execution_time'] = pd.DataFrame(np.array(execution_time_list))
    df['memcpy_time'] = pd.DataFrame(np.array(memcpy_time_list))
    df['retval_time'] = pd.DataFrame(np.array(retval_time_list))
    df['retval_half_time'] = pd.DataFrame(np.array(retval_half_time_list))
    df['sess_time'] = pd.DataFrame(np.array(sess_time_list))

    out_log_path  = os.path.join(flags.output_dir, flags.device, flags.operation)
    if not os.path.exists(out_log_path):
        os.makedirs(out_log_path)
    print('to csv ', os.path.join(out_log_path, flags.output_file_name))
    df.to_csv(os.path.join(out_log_path, flags.output_file_name))

if __name__ == '__main__':
    main()



