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

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

parser = argparse.ArgumentParser('Collect Actual Data Parser')
# Benchmarks parameters
parser.add_argument('--conv', action="store_true", default=False, help='Benchmark convolution layer')
parser.add_argument('--fc', action="store_true", default=False, help='Benchmark fully connection layer')
parser.add_argument('--pool', action="store_true", default=False, help='Benchmark pooling layer')
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
parser.add_argument('--device', type=str, default='', help='Device name as appearing in logfile')
parser.add_argument('--iter_warmup', type=int, default=5, help='Number of iterations for warm-up')
parser.add_argument('--iter_benchmark', type=int, default=10, help='Number of iterations for benchmark')

parser.add_argument('--cpu', action="store_true", default=False, help='Benchmark using CPU')

args = parser.parse_args()

# if args.device == '':
#     print('you should use --device parameter to specify collect data for which device, ex: --device 2080ti')
#     exit()

if args.cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def main(_):
    ########### Benchmark convolution ##########
    if args.conv:

        activation_list = [
            'None',
            'tf.nn.relu']

        batchsize = args.batchsize
        matsize = args.matsize
        kernelsize = args.kernelsize
        channels_in = args.channels_in
        channels_out = args.channels_out
        strides = args.strides
        padding = args.padding
        activation_fct = args.activation_fct
        use_bias = args.use_bias
        time_list = []
        time_max = None
        time_min = None
        time_median = None
        time_mean = None
        time_trim_mean = None

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

        print('time_max:', time_max)
        print('time_min:', time_min)
        print('time_median:', time_median)
        print('time_mean:', time_mean)
        print('time_trim_mean:', time_trim_mean)
            
    ########### Benchmark fully connection ##########
    if args.fc:
        
        activation_list = [
            'None',
            'tf.nn.relu']

        batchsize = args.batchsize
        dim_input = args.dim_input
        dim_output = args.dim_output
        activation_fct = args.activation_fct
        time_list = []
        time_max = None
        time_min = None
        time_median = None
        time_mean = None
        time_trim_mean = None

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

        print('time_max:', time_max)
        print('time_min:', time_min)
        print('time_median:', time_median)
        print('time_mean:', time_mean)
        print('time_trim_mean:', time_trim_mean)

    if args.pool:

        batchsize = args.batchsize
        matsize = args.matsize
        channels_in = args.channels_in
        poolsize = args.poolsize
        padding = args.padding
        strides = args.strides
        time_list = []
        time_max = None
        time_min = None
        time_median = None
        time_mean = None
        time_trim_mean = None

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

        print('time_max:', time_max)
        print('time_min:', time_min)
        print('time_median:', time_median)
        print('time_mean:', time_mean)
        print('time_trim_mean:', time_trim_mean)

        
if __name__ == '__main__':
    tf.app.run()
