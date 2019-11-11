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
parser.add_argument('--num_val', type=int, default=100000, help='Number of results to compute')
parser.add_argument('--log_file', type=str, default='', help='Text file to store results')


args = parser.parse_args()


def main(_):
    ########### Benchmark convolution ##########
    if args.conv:
        if args.log_file == '':
            log_file = str('./conv_goldan_parameters_%s.csv' %(time.strftime("%Y%m%d%H%M%S")))
        else:
            log_file = args.log_file

        activation_list = [
            'None',
            'tf.nn.relu']

        conv_parameters_list = []
        conv_global_count = 1

        while 1 :
            print('conv_global_count:', conv_global_count)
            conv_global_count+=1
            
            batchsize = random.randint(1,64)
            matsize = random.randint(1,512)
            kernelsize = random.randint(1,min(11,matsize)+1)
            channels_in = random.randint(1,int(10000/matsize))
            channels_out = random.randint(1,int(10000/matsize))
            strides = random.randint(1,4)
            padding = random.randint(0,1)
            activation_fct = random.randint(0,1)
            use_bias = random.choice([True,False])

            conv_row_data = [batchsize, matsize, kernelsize, channels_in, channels_out, strides, padding, activation_fct, use_bias]
            conv_parameters_list.append(conv_row_data)
            np_array_parameters = np.array(conv_parameters_list)
            unique_np_array = np.unique(np_array_parameters, axis=0)
            print('len(unique_np_array): ', len(unique_np_array))
            if len(unique_np_array) >= args.num_val :
                golden_parameters_col_name = ['batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 'padding', 'activation_fct', 'use_bias']
                golden_parameters_data = pd.DataFrame(unique_np_array, columns=golden_parameters_col_name)
                golden_parameters_data.to_csv(log_file, index=False)
                break
    
    if args.fc:
        if args.log_file == '':
            log_file = str('./fc_goldan_parameters_%s.csv' %(time.strftime("%Y%m%d%H%M%S")))
        else:
            log_file = args.log_file

        fc_parameters_list = []
        fc_global_count = 1

        while 1 :
            print('fc_global_count:', fc_global_count)
            fc_global_count+=1

            # Set random parameters
            batchsize = random.randint(1,64)
            dim_input = random.randint(1,4096)
            dim_output = random.randint(1,4096)
            activation_fct = random.randint(0,1)

            fc_row_data = [batchsize, dim_input, dim_output, activation_fct]
            fc_parameters_list.append(fc_row_data)
            np_array_parameters = np.array(fc_parameters_list)
            unique_np_array = np.unique(np_array_parameters, axis=0)
            print('len(unique_np_array): ', len(unique_np_array))
            if len(unique_np_array) >= args.num_val :
                golden_parameters_col_name = ['batchsize', 'dim_input', 'dim_output', 'activation_fct']
                golden_parameters_data = pd.DataFrame(unique_np_array, columns=golden_parameters_col_name)
                golden_parameters_data.to_csv(log_file, index=False)
                break

    if args.pool:
        if args.log_file == '':
            log_file = str('./pool_goldan_parameters_%s.csv' %(time.strftime("%Y%m%d%H%M%S")))
        else:
            log_file = args.log_file

        pool_parameters_list = []
        pool_global_count = 1

        while 1 :
            print('pool_global_count:', pool_global_count)
            pool_global_count+=1

            # Set random parameters
            batchsize = random.randint(1,64)
            matsize = random.randint(1,512)
            channels_in = random.randint(1,int(10000/matsize))
            poolsize = random.randint(1,min(11,matsize)+1)
            padding = random.randint(0,1)
            strides = random.randint(1,4)

            pool_row_data = [batchsize, matsize, channels_in, poolsize, padding, strides]
            pool_parameters_list.append(pool_row_data)
            np_array_parameters = np.array(pool_parameters_list)
            unique_np_array = np.unique(np_array_parameters, axis=0)
            print('len(unique_np_array): ', len(unique_np_array))
            if len(unique_np_array) >= args.num_val :
                golden_parameters_col_name = ['batchsize', 'matsize', 'channels_in', 'poolsize', 'padding', 'strides']
                golden_parameters_data = pd.DataFrame(unique_np_array, columns=golden_parameters_col_name)
                golden_parameters_data.to_csv(log_file, index=False)
                break

        
if __name__ == '__main__':
    tf.app.run()
