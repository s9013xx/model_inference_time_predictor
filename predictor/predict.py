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
parser.add_argument('--input_file', type=str, default='', help='Input model csv file')
parser.add_argument('--device', type=str, default='', help='Device name as appearing in logfile')
parser.add_argument('--log_file', type=str, default='', help='Text file to store results')
parser.add_argument('--iter_warmup', type=int, default=5, help='Number of iterations for warm-up')
parser.add_argument('--iter_benchmark', type=int, default=10, help='Number of iterations for benchmark')
parser.add_argument('--batch_size', type=int, default=1, help='Number of iterations for warm-up')

parser.add_argument('--cpu', action="store_true", default=False, help='Benchmark using CPU')
# Check is inference or training
parser.add_argument('--inf', action="store_true", default=False, help='Benchmark convolution layer')
parser.add_argument('--tra', action="store_true", default=False, help='Benchmark fully connection layer')
args = parser.parse_args()

if args.input_file == '':
    print('you should use --input_file parameter to specify model csv file, ex: --input_file VGG16.csv')
    exit()

if args.device == '':
    print('you should use --device parameter to specify collect data for which device, ex: --device 2080ti')
    exit()

if args.inf:
    inf_or_tra = 'inference'
elif args.tra:
    inf_or_tra = 'training'
else:
    print('you should use specific split inference or training data, ex: --inf or --tra')
    exit()

if args.cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

def main(_):

    total_col = ['layer', 'operation', 'batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 'padding', 'activation_fct', 'use_bias', 'poolsize', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean', 'pre_time']
    activation_list = ['None', 'tf.nn.relu']
    df = pd.read_csv(args.input_file, usecols=total_col)
    # print(df)
    
    for index in range(len(df)):
        time_list = []
        # convolution layer operation
        if df.values[index][total_col.index('operation')] == 'conv':
            matsize = df.values[index][total_col.index('matsize')]
            kernelsize = df.values[index][total_col.index('kernelsize')]
            channels_in = df.values[index][total_col.index('channels_in')]
            channels_out = df.values[index][total_col.index('channels_out')]
            strides = df.values[index][total_col.index('strides')]
            padding = df.values[index][total_col.index('padding')]
            activation_fct = int(df.values[index][total_col.index('activation_fct')])
            use_bias = df.values[index][total_col.index('use_bias')]

            tf.reset_default_graph()
            image = tf.Variable(tf.random_normal([args.batch_size, matsize, matsize, channels_in]))
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

            df.iloc[index, total_col.index('batchsize')] = args.batch_size
            df.iloc[index, total_col.index('time_max')] = time_max
            df.iloc[index, total_col.index('time_min')] = time_min
            df.iloc[index, total_col.index('time_median')] = time_median
            df.iloc[index, total_col.index('time_mean')] = time_mean
            df.iloc[index, total_col.index('time_trim_mean')] = time_trim_mean
        # pooling layer operation
        elif df.values[index][total_col.index('operation')] == 'pool':
            matsize = df.values[index][total_col.index('matsize')]
            channels_in = df.values[index][total_col.index('channels_in')]
            strides = df.values[index][total_col.index('strides')]
            padding = df.values[index][total_col.index('padding')]
            poolsize = df.values[index][total_col.index('poolsize')]

            tf.reset_default_graph()
            image = tf.Variable(tf.random_normal([args.batch_size, matsize, matsize, channels_in]))
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

            df.iloc[index, total_col.index('batchsize')] = args.batch_size
            df.iloc[index, total_col.index('time_max')] = time_max
            df.iloc[index, total_col.index('time_min')] = time_min
            df.iloc[index, total_col.index('time_median')] = time_median
            df.iloc[index, total_col.index('time_mean')] = time_mean
            df.iloc[index, total_col.index('time_trim_mean')] = time_trim_mean
        # fully connected layer operation
        elif df.values[index][total_col.index('operation')] == 'fc':
            matsize = df.values[index][total_col.index('matsize')]
            channels_in = df.values[index][total_col.index('channels_in')]
            channels_out = df.values[index][total_col.index('channels_out')]
            activation_fct = int(df.values[index][total_col.index('activation_fct')])

            dim_input = channels_in
            dim_output = channels_out

            tf.reset_default_graph()
            vector_input = tf.Variable(tf.ones(shape=[args.batch_size,dim_input]))
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

            df.iloc[index, total_col.index('batchsize')] = args.batch_size
            df.iloc[index, total_col.index('time_max')] = time_max
            df.iloc[index, total_col.index('time_min')] = time_min
            df.iloc[index, total_col.index('time_median')] = time_median
            df.iloc[index, total_col.index('time_mean')] = time_mean
            df.iloc[index, total_col.index('time_trim_mean')] = time_trim_mean
        # unknown operation
        else:
            print('Unknown Operation')

    store_result_file_path = 'results/%s/%s/%s' % (inf_or_tra, args.device, args.input_file)
    df.to_csv(store_result_file_path, columns=total_col, index=False)

        
if __name__ == '__main__':
    tf.app.run()
