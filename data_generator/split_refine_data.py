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
parser.add_argument('--dense', action="store_true", default=False, help='Benchmark fully connection layer')
parser.add_argument('--pooling', action="store_true", default=False, help='Benchmark pooling layer')
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
    golden_values_col = ['batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 'padding', 'activation_fct', 'use_bias', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean', 'preprocess_time', 'execution_time', 'memcpy_time', 'retval_time', 'mem_ret', 'retval_half_time', 'sess_time', 'elements_matrix', 'elements_kernel']
elif args.dense:
    operation = 'dense'
    golden_values_col = ['batchsize', 'dim_input', 'dim_output', 'activation_fct', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean', 'preprocess_time', 'execution_time', 'memcpy_time', 'retval_time', 'mem_ret', 'retval_half_time', 'sess_time']
elif args.pooling:
    operation = 'pooling'
    golden_values_col = ['batchsize', 'matsize', 'channels_in', 'poolsize', 'padding', 'strides', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean', 'preprocess_time', 'execution_time', 'memcpy_time', 'retval_time', 'mem_ret', 'retval_half_time', 'sess_time', 'elements_matrix']
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
    
    # read_file_path = os.path.join('goldan_values', '%s_goldan_values_%s_*.csv' % (operation, args.device))
    read_file_path = os.path.join('1080ti', operation, 'all_data.csv')

    for file in glob.glob(read_file_path):
        df_ori = pd.read_csv(file, index_col=None)
        
        df_ori = data_filter(df_ori)

        df_test_data_20000 = data_divider(df_ori, 0, 19999)
        df_train_data_10000 = data_divider(df_ori, 20000, 29999)
        df_train_data_20000 = data_divider(df_ori, 20000, 39999)
        df_train_data_30000 = data_divider(df_ori, 20000, 49999)
        df_train_data_40000 = data_divider(df_ori, 20000, 59999)
        df_train_data_50000 = data_divider(df_ori, 20000, 69999)
        df_train_data_60000 = data_divider(df_ori, 20000, 79999)
        df_train_data_70000 = data_divider(df_ori, 20000, 89999)
        df_train_data_80000 = data_divider(df_ori, 20000, 99999)
        
        store_data_path = 'goldan_values/%s/%s/%s' % (inf_or_tra, args.device, operation)

        df_test_data_20000.to_csv(os.path.join(store_data_path, 'test_data_20000.csv'), columns=golden_values_col, index=False)
        df_train_data_10000.to_csv(os.path.join(store_data_path, 'train_data_10000.csv'), columns=golden_values_col, index=False)
        df_train_data_20000.to_csv(os.path.join(store_data_path, 'train_data_20000.csv'), columns=golden_values_col, index=False)
        df_train_data_30000.to_csv(os.path.join(store_data_path, 'train_data_30000.csv'), columns=golden_values_col, index=False)
        df_train_data_40000.to_csv(os.path.join(store_data_path, 'train_data_40000.csv'), columns=golden_values_col, index=False)
        df_train_data_50000.to_csv(os.path.join(store_data_path, 'train_data_50000.csv'), columns=golden_values_col, index=False)
        df_train_data_60000.to_csv(os.path.join(store_data_path, 'train_data_60000.csv'), columns=golden_values_col, index=False)
        df_train_data_70000.to_csv(os.path.join(store_data_path, 'train_data_70000.csv'), columns=golden_values_col, index=False)
        df_train_data_80000.to_csv(os.path.join(store_data_path, 'train_data_80000.csv'), columns=golden_values_col, index=False)

        if args.conv:
            df_test_data_20000['elements_matrix'] = df_test_data_20000['matsize'] ** 2
            df_test_data_20000['elements_kernel'] = df_test_data_20000['kernelsize'] **2
            df_test_data_20000['mem_ret'] = df_test_data_20000['memcpy_time'] + df_test_data_20000['retval_time']
            df_test_data_20000.to_csv(os.path.join(store_data_path, 'test.csv'), columns=golden_values_col, index=False)
            df_train_data_80000['elements_matrix'] = df_train_data_80000['matsize'] ** 2
            df_train_data_80000['elements_kernel'] = df_train_data_80000['kernelsize'] **2
            df_train_data_80000['mem_ret'] = df_train_data_80000['memcpy_time'] + df_train_data_80000['retval_time']
            df_train_data_80000.to_csv(os.path.join(store_data_path, 'train.csv'), columns=golden_values_col, index=False)
        elif args.dense:
            df_test_data_20000['mem_ret'] = df_test_data_20000['memcpy_time'] + df_test_data_20000['retval_time']
            df_test_data_20000.to_csv(os.path.join(store_data_path, 'test.csv'), columns=golden_values_col, index=False)
            df_train_data_80000['mem_ret'] = df_train_data_80000['memcpy_time'] + df_train_data_80000['retval_time']
            df_train_data_80000.to_csv(os.path.join(store_data_path, 'train.csv'), columns=golden_values_col, index=False)
        elif args.pooling:
            df_test_data_20000['elements_matrix'] = df_test_data_20000['matsize'] ** 2
            df_test_data_20000['mem_ret'] = df_test_data_20000['memcpy_time'] + df_test_data_20000['retval_time']
            df_test_data_20000.to_csv(os.path.join(store_data_path, 'test.csv'), columns=golden_values_col, index=False)
            df_train_data_80000['elements_matrix'] = df_train_data_80000['matsize'] ** 2
            df_train_data_80000['mem_ret'] = df_train_data_80000['memcpy_time'] + df_train_data_80000['retval_time']
            df_train_data_80000.to_csv(os.path.join(store_data_path, 'train.csv'), columns=golden_values_col, index=False)
            # golden_values_col = ['batchsize', 'matsize', 'channels_in', 'poolsize', 'padding', 'strides', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']



if __name__ == '__main__':
    main()
