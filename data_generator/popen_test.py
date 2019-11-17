"""Generates benchmarks for convolutions with different, randomly determined
parameters. Saves results into a pandas dataframe (.pkl)
and a numpy array (.npy)
"""
import os
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import random
import numpy
from scipy import stats
import subprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

parser = argparse.ArgumentParser('Collect Actual Data Parser')
# Benchmarks parameters
parser.add_argument('--conv', action="store_true", default=False, help='Benchmark convolution layer')
parser.add_argument('--fc', action="store_true", default=False, help='Benchmark fully connection layer')
parser.add_argument('--pool', action="store_true", default=False, help='Benchmark pooling layer')
parser.add_argument('--test', action="store_true", default=False, help='Benchmark pooling layer')
# Convolution Operation Parameters
parser.add_argument('--batchsize', type=int, default=0, help='Batch size of convolution, fully connected, pooling operation')
parser.add_argument('--matsize', type=int, default=0, help='Matrix size of convolution, pooling operation')
parser.add_argument('--kernelsize', type=int, default=0, help='Kernel size of convolution operation')
parser.add_argument('--channels_in', type=int, default=0, help='Channel input convolution, pooling operation')
parser.add_argument('--channels_out', type=int, default=0, help='Channel output of convolution operation')
parser.add_argument('--strides', type=int, default=0, help='Strides of convolution, pooling operation')
parser.add_argument('--padding', type=int, default=0, help='Padding of convolution, pooling operation')
parser.add_argument('--activation_fct', type=int, default=0, help='Activate function of convolution operation')
parser.add_argument('--use_bias', type=int, default=0, help='Use bias of convolution operation')
# Fully Connected Operation Parameters
parser.add_argument('--dim_input', type=int, default=0, help='Dimention input of fully connected operation')
parser.add_argument('--dim_output', type=int, default=0, help='Dimention output of fully connected operation')
# Pooling Operation Parameters
parser.add_argument('--poolsize', type=int, default=0, help='Pooling size of pooling operation')
# General parameters
parser.add_argument('--log_file', type=str, default='', help='Text file to store results')
parser.add_argument('--device', type=str, default='', help='Device name as appearing in logfile')
parser.add_argument('--iter_warmup', type=int, default=5, help='Number of iterations for warm-up')
parser.add_argument('--iter_benchmark', type=int, default=10, help='Number of iterations for benchmark')

parser.add_argument('--cpu', action="store_true", default=False, help='Benchmark using CPU')

args = parser.parse_args()

if args.device == '':
    print('you should use --device parameter to specify collect data for which device, ex: --device 2080ti')
    exit()

if args.cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

def main(_):
    ########### Benchmark convolution ##########
    if args.test:
        activation_list = [
            'None',
            'tf.nn.relu']

        conv_col_name = ['batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 'padding', 'activation_fct', 'use_bias']
        df = pd.read_csv('shuffled_parameters/conv_parameters_shuffled.csv', usecols=conv_col_name)

        golden_values_col_name = ['batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 'padding', 'activation_fct', 'use_bias', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']
        golden_values_data = pd.DataFrame(columns=golden_values_col_name)

        i = 0
        print('========== conv', i , '==========')
        batchsize = df.iloc[i, 0]
        matsize = df.iloc[i, 1]
        kernelsize = df.iloc[i, 2]
        channels_in = df.iloc[i, 3]
        channels_out = df.iloc[i, 4]
        strides = df.iloc[i, 5]
        padding = df.iloc[i, 6]
        activation_fct = df.iloc[i, 7]
        use_bias = df.iloc[i, 8]
        print(batchsize, matsize, kernelsize, channels_in, channels_out, strides, padding, activation_fct, use_bias)
        time_list = []
        time_max = None
        time_min = None
        time_median = None
        time_mean = None
        time_trim_mean = None

        command = 'python operation_inference.py --conv --batchsize=62 --matsize=471 --kernelsize=7 --channels_in=19 --channels_out=18 --strides=1 --padding=0 --activation_fct=0 --use_bias=1 > temp'
        process = subprocess.Popen(command, stdout = subprocess.PIPE, shell = True)
        process.wait()

        line_count = 0
        tmp_file = open("temp", "r")
        for line in tmp_file:
            value = line.split(':')[1].strip()
            if line_count == 0:
                time_max = value
            if line_count == 1:
                time_min = value
            if line_count == 2:
                time_median = value
            if line_count == 3:
                time_mean = value
            if line_count == 4:
                time_trim_mean = value
            line_count = line_count + 1
        print('time_max:', time_max)
        print('time_min:', time_min)
        print('time_median:', time_median)
        print('time_mean:', time_mean)
        print('time_trim_mean:', time_trim_mean)

    ########### Benchmark convolution ##########
    if args.conv:
        if args.log_file == '':
            log_file = str('./goldan_values/conv_goldan_values_%s_%s.csv' %(args.device, time.strftime("%Y%m%d%H%M%S")))
        else:
            log_file = args.log_file

        activation_list = [
            'None',
            'tf.nn.relu']

        conv_col_name = ['batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 'padding', 'activation_fct', 'use_bias']
        df = pd.read_csv('shuffled_parameters/conv_parameters_shuffled.csv', usecols=conv_col_name)

        # conv_values_list = []
        golden_values_col_name = ['batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 'padding', 'activation_fct', 'use_bias', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']
        golden_values_data = pd.DataFrame(columns=golden_values_col_name)

        for i in range(len(df.index)):
        # for i in range(200):
            print('========== conv', i , '==========')
            batchsize = df.iloc[i, 0]
            matsize = df.iloc[i, 1]
            kernelsize = df.iloc[i, 2]
            channels_in = df.iloc[i, 3]
            channels_out = df.iloc[i, 4]
            strides = df.iloc[i, 5]
            padding = df.iloc[i, 6]
            activation_fct = df.iloc[i, 7]
            use_bias = df.iloc[i, 8]
            print(batchsize, matsize, kernelsize, channels_in, channels_out, strides, padding, activation_fct, use_bias)
            time_list = []
            time_max = None
            time_min = None
            time_median = None
            time_mean = None
            time_trim_mean = None

            command = 'python operation_inference.py --conv --batchsize=%d --matsize=%d --kernelsize=%d --channels_in=%d --channels_out=%d --strides=%d --padding=%d --activation_fct=%d --use_bias=%d > temp' %(batchsize, matsize, kernelsize, channels_in, channels_out, strides, padding, activation_fct, use_bias)
            process = subprocess.Popen(command, stdout = subprocess.PIPE, shell = True)
            process.wait()

            line_count = 0
            tmp_file = open("temp", "r")
            for line in tmp_file:
                value = line.split(':')[1].strip()
                if line_count == 0:
                    time_max = value
                if line_count == 1:
                    time_min = value
                if line_count == 2:
                    time_median = value
                if line_count == 3:
                    time_mean = value
                if line_count == 4:
                    time_trim_mean = value
                line_count = line_count + 1

            conv_row_data = [batchsize, matsize, kernelsize, channels_in, channels_out, strides, padding, activation_fct, use_bias, time_max, time_min, time_median, time_mean, time_trim_mean]
            golden_values_data.loc[0] = conv_row_data
            
            if i==0: 
                golden_values_data.to_csv(log_file, index=False)
            else:
                golden_values_data.to_csv(log_file, index=False, mode='a', header=False)

    ########### Benchmark fully connection ##########
    if args.fc:
        if args.log_file == '':
            log_file = str('./goldan_values/fc_goldan_values_%s_%s.csv' %(args.device, time.strftime("%Y%m%d%H%M%S")))
        else:
            log_file = args.log_file

        activation_list = [
            'None',
            'tf.nn.relu']

        fc_col_name = ['batchsize', 'dim_input', 'dim_output', 'activation_fct']
        df = pd.read_csv('shuffled_parameters/fc_parameters_shuffled.csv', usecols=fc_col_name)

        # fc_values_list = []
        golden_values_col_name = ['batchsize', 'dim_input', 'dim_output', 'activation_fct', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']
        golden_values_data = pd.DataFrame(columns=golden_values_col_name)

        for i in range(len(df.index)):
        # for i in range(200):
            print('========== fc:', i , '==========')
            batchsize = df.iloc[i, 0]
            dim_input = df.iloc[i, 1]
            dim_output = df.iloc[i, 2]
            activation_fct = df.iloc[i, 3]
            print(batchsize, dim_input, dim_output, activation_fct)
            time_list = []
            time_max = None
            time_min = None
            time_median = None
            time_mean = None
            time_trim_mean = None

            command = 'python operation_inference.py --fc --batchsize=%d --dim_input=%d --dim_output=%d --activation_fct=%d > temp' %(batchsize, dim_input, dim_output, activation_fct)
            process = subprocess.Popen(command, stdout = subprocess.PIPE, shell = True)
            process.wait()

            line_count = 0
            tmp_file = open("temp", "r")
            for line in tmp_file:
                value = line.split(':')[1].strip()
                if line_count == 0:
                    time_max = value
                if line_count == 1:
                    time_min = value
                if line_count == 2:
                    time_median = value
                if line_count == 3:
                    time_mean = value
                if line_count == 4:
                    time_trim_mean = value
                line_count = line_count + 1

            fc_row_data = [batchsize, dim_input, dim_output, activation_fct, time_max, time_min, time_median, time_mean, time_trim_mean]
            golden_values_data.loc[0] = fc_row_data
            
            if i==0: 
                golden_values_data.to_csv(log_file, index=False)
            else:
                golden_values_data.to_csv(log_file, index=False, mode='a', header=False)

    ########### Benchmark pooling ##########
    if args.pool:
        if args.log_file == '':
            log_file = str('./goldan_values/pool_goldan_values_%s_%s.csv' %(args.device, time.strftime("%Y%m%d%H%M%S")))
        else:
            log_file = args.log_file

        pool_col_name = ['batchsize', 'matsize', 'channels_in', 'poolsize', 'padding', 'strides']
        df = pd.read_csv('shuffled_parameters/pool_parameters_shuffled.csv', usecols=pool_col_name)

        # pool_values_list = []
        golden_values_col_name = ['batchsize', 'matsize', 'channels_in', 'poolsize', 'padding', 'strides', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']
        golden_values_data = pd.DataFrame(columns=golden_values_col_name)

        for i in range(len(df.index)):
        # for i in range(200):
            print('========== pool:', i , '==========')
            batchsize = df.iloc[i, 0]
            matsize = df.iloc[i, 1]
            channels_in = df.iloc[i, 2]
            poolsize = df.iloc[i, 3]
            padding = df.iloc[i, 4]
            strides = df.iloc[i, 5]
            print(batchsize, matsize, channels_in, poolsize, padding, strides)
            time_list = []
            time_max = None
            time_min = None
            time_median = None
            time_mean = None
            time_trim_mean = None

            command = 'python operation_inference.py --pool --batchsize=%d --matsize=%d --channels_in=%d --poolsize=%d --padding=%d --strides=%d> temp' %(batchsize, matsize, channels_in, poolsize, padding, strides)
            process = subprocess.Popen(command, stdout = subprocess.PIPE, shell = True)
            process.wait()

            line_count = 0
            tmp_file = open("temp", "r")
            for line in tmp_file:
                value = line.split(':')[1].strip()
                if line_count == 0:
                    time_max = value
                if line_count == 1:
                    time_min = value
                if line_count == 2:
                    time_median = value
                if line_count == 3:
                    time_mean = value
                if line_count == 4:
                    time_trim_mean = value
                line_count = line_count + 1

            pool_row_data = [batchsize, matsize, channels_in, poolsize, padding, strides, time_max, time_min, time_median, time_mean, time_trim_mean]
            golden_values_data.loc[0] = pool_row_data
            
            if i==0: 
                golden_values_data.to_csv(log_file, index=False)
            else:
                golden_values_data.to_csv(log_file, index=False, mode='a', header=False)

if __name__ == '__main__':
    tf.app.run()
