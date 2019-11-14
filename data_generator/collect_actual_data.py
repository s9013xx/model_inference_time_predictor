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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

parser = argparse.ArgumentParser('Collect Actual Data Parser')
# Benchmarks parameters
parser.add_argument('--conv', action="store_true", default=False, help='Benchmark convolution layer')
parser.add_argument('--fc', action="store_true", default=False, help='Benchmark fully connection layer')
parser.add_argument('--pool', action="store_true", default=False, help='Benchmark pooling layer')
# General parameters
parser.add_argument('--num_val', type=int, default=100000, help='Number of results to compute')
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
        # for i in range(100):
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

            try:
                tf.reset_default_graph()
                image = tf.Variable(tf.random_normal([batchsize, matsize, matsize, channels_in]))
                op = tf.layers.conv2d(image, filters=channels_out, kernel_size=[kernelsize, kernelsize], strides=(strides, strides), padding=('SAME' if padding==1 else 'VALID'), activation=eval(activation_list[activation_fct]), use_bias=use_bias)
                sess = tf.Session()
                if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                    init = tf.initialize_all_variables()
                else:
                    init = tf.global_variables_initializer()
                sess.run(init)
                # Warm-up run
                for _ in range(args.iter_warmup):
                    sess.run(op)
                # Benchmark run
                for _ in range(args.iter_benchmark):
                    start_time = time.time()
                    sess.run(op)
                    time_list.append(((time.time()-start_time) * 1000))
                np_array_parameters = np.array(time_list)

                time_max = numpy.amax(np_array_parameters)
                time_min = numpy.amin(np_array_parameters)
                time_median = numpy.median(np_array_parameters)
                time_mean = numpy.mean(np_array_parameters)
                time_trim_mean = stats.trim_mean(np_array_parameters, 0.1)
            except:
                print('except')

            conv_row_data = [batchsize, matsize, kernelsize, channels_in, channels_out, strides, padding, activation_fct, use_bias, time_max, time_min, time_median, time_mean, time_trim_mean]
            golden_values_data.loc[0] = conv_row_data
            
            if i==0: 
                golden_values_data.to_csv(log_file, index=False)
            else:
                golden_values_data.to_csv(log_file, index=False, mode='a', header=False)

        #     conv_row_data = [batchsize, matsize, kernelsize, channels_in, channels_out, strides, padding, activation_fct, use_bias, time_max, time_min, time_median, time_mean, time_trim_mean]
        #     conv_values_list.append(conv_row_data)

        # np_array_values = np.array(conv_values_list)
        # golden_values_col_name = ['batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 'padding', 'activation_fct', 'use_bias', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']
        # golden_values_data = pd.DataFrame(np_array_values, columns=golden_values_col_name)
        # golden_values_data.to_csv(log_file, index=False)
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
        # for i in range(100):
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

            try:
                tf.reset_default_graph()
                vector_input = tf.Variable(tf.ones(shape=[batchsize,dim_input]))
                op = tf.layers.dense(inputs=vector_input, units=dim_output, kernel_initializer=tf.ones_initializer(), activation=eval(activation_list[activation_fct]))
                sess = tf.Session()
                if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                    init = tf.initialize_all_variables()
                else:
                    init = tf.global_variables_initializer()
                sess.run(init)
                # Warm-up run
                for _ in range(args.iter_warmup):
                    sess.run(op)
                # Benchmark run
                for _ in range(args.iter_benchmark):
                    start_time = time.time()
                    sess.run(op)
                    time_list.append(((time.time()-start_time) * 1000))
                np_array_parameters = np.array(time_list)

                time_max = numpy.amax(np_array_parameters)
                time_min = numpy.amin(np_array_parameters)
                time_median = numpy.median(np_array_parameters)
                time_mean = numpy.mean(np_array_parameters)
                time_trim_mean = stats.trim_mean(np_array_parameters, 0.1)
            except:
                print('except')

            fc_row_data = [batchsize, dim_input, dim_output, activation_fct, time_max, time_min, time_median, time_mean, time_trim_mean]
            golden_values_data.loc[0] = fc_row_data
            
            if i==0: 
                golden_values_data.to_csv(log_file, index=False)
            else:
                golden_values_data.to_csv(log_file, index=False, mode='a', header=False)

        #     fc_row_data = [batchsize, dim_input, dim_output, activation_fct, time_max, time_min, time_median, time_mean, time_trim_mean]
        #     fc_values_list.append(fc_row_data)

        # np_array_values = np.array(fc_values_list)
        # golden_values_col_name = ['batchsize', 'dim_input', 'dim_output', 'activation_fct', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']
        # golden_values_data = pd.DataFrame(np_array_values, columns=golden_values_col_name)
        # golden_values_data.to_csv(log_file, index=False)
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
        # for i in range(100):
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

            try:
                tf.reset_default_graph()
                image = tf.Variable(tf.random_normal([batchsize, matsize, matsize, channels_in]))
                op = tf.layers.max_pooling2d(image, pool_size=(poolsize, poolsize), strides=(strides, strides), padding=('SAME' if padding==1 else 'VALID'))
                sess = tf.Session()
                if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                    init = tf.initialize_all_variables()
                else:
                    init = tf.global_variables_initializer()
                sess.run(init)
                # Warm-up run
                for _ in range(args.iter_warmup):
                    sess.run(op)
                # Benchmark run
                for _ in range(args.iter_benchmark):
                    start_time = time.time()
                    sess.run(op)
                    time_list.append(((time.time()-start_time) * 1000))
                np_array_parameters = np.array(time_list)

                time_max = numpy.amax(np_array_parameters)
                time_min = numpy.amin(np_array_parameters)
                time_median = numpy.median(np_array_parameters)
                time_mean = numpy.mean(np_array_parameters)
                time_trim_mean = stats.trim_mean(np_array_parameters, 0.1)
            except:
                print('except')

            pool_row_data = [batchsize, matsize, channels_in, poolsize, padding, strides, time_max, time_min, time_median, time_mean, time_trim_mean]
            golden_values_data.loc[0] = pool_row_data
            
            if i==0: 
                golden_values_data.to_csv(log_file, index=False)
            else:
                golden_values_data.to_csv(log_file, index=False, mode='a', header=False)

        #     pool_row_data = [batchsize, matsize, channels_in, poolsize, padding, strides, time_max, time_min, time_median, time_mean, time_trim_mean]
        #     pool_values_list.append(pool_row_data)

        # np_array_values = np.array(pool_values_list)
        # golden_values_col_name = ['batchsize', 'matsize', 'channels_in', 'poolsize', 'padding', 'strides', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']
        # golden_values_data = pd.DataFrame(np_array_values, columns=golden_values_col_name)
        # golden_values_data.to_csv(log_file, index=False)

        
if __name__ == '__main__':
    tf.app.run()
