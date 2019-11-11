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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

parser = argparse.ArgumentParser('Collect Data Paremeters Parser')
# Benchmarks parameters
parser.add_argument('--conv', action="store_true", default=False, help='Benchmark convolution layer')
parser.add_argument('--fc', action="store_true", default=False, help='Benchmark fully connection layer')
parser.add_argument('--pool', action="store_true", default=False, help='Benchmark pooling layer')
# General parameters
parser.add_argument('--num_val', type=int, default=110000, help='Number of results to compute')
parser.add_argument('--log_file', type=str, default='', help='Text file to store results')


args = parser.parse_args()


def main(_):
    ########### Benchmark convolution ##########
    if args.conv:
        if args.log_file == '':
            log_file = str('./conv_parameters.csv')
        else:
            log_file = args.log_file

        activation_list = [
            'None',
            'tf.nn.relu']

        batchsize = np.random.randint(1,65, args.num_val)
        matsize = np.random.randint(1,513, args.num_val)
        kernelsize = np.zeros(args.num_val,dtype=np.int32)
        channels_in = np.zeros(args.num_val,dtype=np.int32)
        channels_out = np.zeros(args.num_val,dtype=np.int32)
        strides = np.random.randint(1,5, args.num_val)
        padding = np.random.randint(0,2, args.num_val)
        activation_fct = np.random.randint(0,len(activation_list), args.num_val)
        use_bias = np.random.choice([True,False], args.num_val)

        for i in range(args.num_val):
            kernelsize[i] = np.random.randint(1,min(7,matsize[i])+1)
            channels_in[i] = np.random.randint(1,10000/matsize[i])
            channels_out[i] = np.random.randint(1,10000/matsize[i])

        np_array = np.array([batchsize, matsize, kernelsize, channels_in, channels_out, strides, padding, activation_fct, use_bias])
        np_array_transpose = np_array.transpose()
        unique_np_array = np.unique(np_array_transpose, axis=0)
        golden_parameters_col_name = ['batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 'padding', 'activation_fct', 'use_bias']
        golden_parameters_data = pd.DataFrame(unique_np_array, columns=golden_parameters_col_name)
        # golden_parameters_data = golden_parameters_data.reindex(np.random.permutation(golden_parameters_data.index))
        golden_parameters_data.to_csv(log_file, index=False)
    
    if args.fc:
        if args.log_file == '':
            log_file = str('./fc_parameters.csv')
        else:
            log_file = args.log_file

        # Set random parameters
        batchsize = np.random.randint(1,65,args.num_val)
        dim_input = np.random.randint(1,4096,args.num_val)
        dim_output = np.random.randint(1,4096,args.num_val)
        activation_fct = np.random.randint(0,len(activation_list), args.num_val)

        np_array = np.array([batchsize, dim_input, dim_output, activation_fct])
        np_array_transpose = np_array.transpose()
        unique_np_array = np.unique(np_array_transpose, axis=0)
        golden_parameters_col_name = ['batchsize', 'dim_input', 'dim_output', 'activation_fct']
        golden_parameters_data = pd.DataFrame(unique_np_array, columns=golden_parameters_col_name)
        # golden_parameters_data = golden_parameters_data.reindex(np.random.permutation(golden_parameters_data.index))
        golden_parameters_data.to_csv(log_file, index=False)

    if args.pool:
        if args.log_file == '':
            log_file = str('./pool_parameters.csv')
        else:
            log_file = args.log_file

        batchsize = np.random.randint(1,65, args.num_val)
        matsize = np.random.randint(1,513, args.num_val)
        channels_in = np.zeros(args.num_val,dtype=np.int32)
        poolsize = np.zeros(args.num_val,dtype=np.int32)
        padding = np.random.randint(0,2, args.num_val)
        strides = np.random.randint(1,5, args.num_val)
        
        activation_fct = np.random.randint(0,len(activation_list), args.num_val)
        use_bias = np.random.choice([True,False], args.num_val)

        for i in range(args.num_val):
            channels_in[i] = np.random.randint(1,10000/matsize[i])
            poolsize[i] = np.random.randint(1,min(7,matsize[i])+1)

        np_array = np.array([batchsize, matsize, channels_in, poolsize, padding, strides])
        np_array_transpose = np_array.transpose()
        unique_np_array = np.unique(np_array_transpose, axis=0)
        golden_parameters_col_name = ['batchsize', 'matsize', 'channels_in', 'poolsize', 'padding', 'strides']
        golden_parameters_data = pd.DataFrame(unique_np_array, columns=golden_parameters_col_name)
        # golden_parameters_data = golden_parameters_data.reindex(np.random.permutation(golden_parameters_data.index))
        golden_parameters_data.to_csv(log_file, index=False)

if __name__ == '__main__':
    tf.app.run()
