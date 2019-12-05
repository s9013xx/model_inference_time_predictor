import os
import sys
import argparse
import numpy as np
import pandas as pd
import numpy
import glob

parser = argparse.ArgumentParser('Split Data Parser')
# Benchmarks parameters
parser.add_argument('--conv', action="store_true", default=False, help='Benchmark convolution layer')
parser.add_argument('--fc', action="store_true", default=False, help='Benchmark fully connection layer')
parser.add_argument('--pool', action="store_true", default=False, help='Benchmark pooling layer')
# General parameters
parser.add_argument('--log_file', type=str, default='', help='Text file to store results')
parser.add_argument('--device', type=str, default='', help='Device name as appearing in logfile')
parser.add_argument('--iter_warmup', type=int, default=5, help='Number of iterations for warm-up')
parser.add_argument('--iter_benchmark', type=int, default=10, help='Number of iterations for benchmark')
# Check is inference or training
parser.add_argument('--inf', action="store_true", default=False, help='Benchmark convolution layer')
parser.add_argument('--tra', action="store_true", default=False, help='Benchmark fully connection layer')
args = parser.parse_args()

if args.conv:
    operation = 'conv'
    golden_values_col = ['batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 'padding', 'activation_fct', 'use_bias', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']
elif args.fc:
    operation = 'fc'
    golden_values_col = ['batchsize', 'dim_input', 'dim_output', 'activation_fct', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']
elif args.pool:
    operation = 'pool'
    golden_values_col = ['batchsize', 'matsize', 'channels_in', 'poolsize', 'padding', 'strides', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']
else:
    print('you should use specific which operation data you want to split, ex: --conv')
    exit()

if args.inf:
    inf_or_tra = 'inference'
elif args.tra:
    inf_or_tra = 'training'
else:
    print('you should use specific split inference or training data, ex: --inf or --tra')
    exit()

if args.device == '':
    print('you should use --device parameter to specify collect data for which device, ex: --device 2080ti')
    exit()

# This funcion is filter for non-zero value
def data_filter(df_ori_data):
    # Time invalid filter
    df_data = df_ori_data.dropna()
    # re-index for split data
    df_data = df_data.reset_index(drop=True)

    return df_data

# This funcion is divide total data to three parts (train, validate, test)
def data_divider(df_ori_data, start_index, end_index):
    # Divide total data to train, validation, test part
    df_data = df_ori_data.loc[start_index:end_index, :]

    return df_data

def main():
    # step1 get the dataset 
    
    read_file_path = os.path.join('goldan_values', '%s_goldan_values_%s_*.csv' % (operation, args.device))
    
    for file in glob.glob(read_file_path):
        df_ori = pd.read_csv(file, usecols=golden_values_col, index_col=None)
        
        df_ori = data_filter(df_ori)

        df_train_data_10000 = data_divider(df_ori, 0, 9999)
        df_train_data_20000 = data_divider(df_ori, 0, 19999)
        df_train_data_30000 = data_divider(df_ori, 0, 29999)
        df_train_data_40000 = data_divider(df_ori, 0, 39999)
        df_train_data_50000 = data_divider(df_ori, 0, 49999)
        df_train_data_60000 = data_divider(df_ori, 0, 59999)
        df_train_data_70000 = data_divider(df_ori, 0, 69999)
        df_train_data_80000 = data_divider(df_ori, 0, 79999)

        df_test_data_20000 = data_divider(df_ori, 80000, 99999)

        store_data_path = 'goldan_values/%s/%s/%s' % (inf_or_tra, args.device, operation)
        df_train_data_10000.to_csv(os.path.join(store_data_path, 'train_data_10000.csv'), columns=golden_values_col, index=False)
        df_train_data_20000.to_csv(os.path.join(store_data_path, 'train_data_20000.csv'), columns=golden_values_col, index=False)
        df_train_data_30000.to_csv(os.path.join(store_data_path, 'train_data_30000.csv'), columns=golden_values_col, index=False)
        df_train_data_40000.to_csv(os.path.join(store_data_path, 'train_data_40000.csv'), columns=golden_values_col, index=False)
        df_train_data_50000.to_csv(os.path.join(store_data_path, 'train_data_50000.csv'), columns=golden_values_col, index=False)
        df_train_data_60000.to_csv(os.path.join(store_data_path, 'train_data_60000.csv'), columns=golden_values_col, index=False)
        df_train_data_70000.to_csv(os.path.join(store_data_path, 'train_data_70000.csv'), columns=golden_values_col, index=False)
        df_train_data_80000.to_csv(os.path.join(store_data_path, 'train_data_80000.csv'), columns=golden_values_col, index=False)

        df_test_data_20000.to_csv(os.path.join(store_data_path, 'test_data_20000.csv'), columns=golden_values_col, index=False)

if __name__ == '__main__':
    main()
