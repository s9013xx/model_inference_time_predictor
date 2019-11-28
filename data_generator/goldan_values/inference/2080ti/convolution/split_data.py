import os
import sys
import argparse
import numpy as np
import pandas as pd
import numpy

# This funcion is divide total data to three parts (train, validate, test)
def data_divider(df_ori_data, start_index, end_index):
    # 1. Divide total data to train, validation, test part
    df_data = df_ori_data.loc[start_index:end_index, :]
    # 2. Time invalid filter
    df_data = df_data[df_data['time_median']>0]

    return df_data

def main():
    # step1 get the dataset 
    golden_values_col = ['batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 'padding', 'activation_fct', 'use_bias', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']
    df_ori = pd.read_csv('conv_goldan_values_2080ti_20191118000424.csv', usecols=golden_values_col)
    
    # df_ori = pd.read_csv(file, usecols=golden_values_col)

    df_train_data_10000 = data_divider(df_ori, 0, 9999)
    df_train_data_20000 = data_divider(df_ori, 0, 19999)
    df_train_data_30000 = data_divider(df_ori, 0, 29999)
    df_train_data_40000 = data_divider(df_ori, 0, 39999)
    df_train_data_50000 = data_divider(df_ori, 0, 49999)
    df_train_data_60000 = data_divider(df_ori, 0, 59999)
    df_train_data_70000 = data_divider(df_ori, 0, 69999)
    df_train_data_80000 = data_divider(df_ori, 0, 79999)

    df_validate_data = data_divider(df_ori, 80000, 89999)
    df_test_data = data_divider(df_ori, 90000, 99999)

    df_train_data_10000.to_csv('train_data_10000.csv', columns=golden_values_col, index=False)
    df_train_data_20000.to_csv('train_data_20000.csv', columns=golden_values_col, index=False)
    df_train_data_30000.to_csv('train_data_30000.csv', columns=golden_values_col, index=False)
    df_train_data_40000.to_csv('train_data_40000.csv', columns=golden_values_col, index=False)
    df_train_data_50000.to_csv('train_data_50000.csv', columns=golden_values_col, index=False)
    df_train_data_60000.to_csv('train_data_60000.csv', columns=golden_values_col, index=False)
    df_train_data_70000.to_csv('train_data_70000.csv', columns=golden_values_col, index=False)
    df_train_data_80000.to_csv('train_data_80000.csv', columns=golden_values_col, index=False)

    df_validate_data.to_csv('validate_data.csv', columns=golden_values_col, index=False)
    df_test_data.to_csv('test_data.csv', columns=golden_values_col, index=False)

if __name__ == '__main__':
    main()
