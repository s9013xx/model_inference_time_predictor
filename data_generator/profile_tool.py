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
from tensorflow.python.client import timeline

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

parser = argparse.ArgumentParser('Collect Actual Data Parser')
# Benchmarks parameters
parser.add_argument('--conv', action="store_true", default=False, help='Benchmark convolution layer')
parser.add_argument('--fc', action="store_true", default=False, help='Benchmark fully connection layer')
parser.add_argument('--pool', action="store_true", default=False, help='Benchmark pooling layer')
# General parameters
parser.add_argument('--device', type=str, default='', help='Device name as appearing in logfile')
parser.add_argument('--iter_warmup', type=int, default=5, help='Number of iterations for warm-up')
parser.add_argument('--iter_benchmark', type=int, default=10, help='Number of iterations for benchmark')

parser.add_argument('--cpu', action="store_true", default=False, help='Benchmark using CPU')

parser.add_argument('--profile', '-p', action="store_true", default=False, help='profiling')

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

run_metadata = tf.RunMetadata()

dict_layernum = {
    'conv2d': 0,
    'pooling': 0,
    'dense': 0,
}

activation_list = [
    'None',
    'tf.nn.relu']

def create_conv2d(op, layer, layer_type, dict_layernum, last_layer):
    print(layer['batchsize'], layer['matsize'], layer['kernelsize'], layer['channels_in']
        , layer['channels_out'], layer['strides'], layer['padding'], layer['activation_fct']
        , layer['use_bias'])
                   
    layer_name = layer_type + str(dict_layernum[layer_type] + 1)  
    if op == None:
        op = tf.Variable(tf.random_normal([layer['batchsize'].astype(int), 
            layer['matsize'].astype(int), layer['matsize'].astype(int), layer['channels_in'].astype(int)]))
    
    op = tf.layers.conv2d(op, filters=layer['channels_out'].astype(int), 
        kernel_size=[layer['kernelsize'].astype(int), layer['kernelsize'].astype(int)], 
        strides=(layer['strides'].astype(int), layer['strides'].astype(int)), 
        padding=('SAME' if layer['padding'].astype(int) ==1 else 'VALID'),
        activation=eval(activation_list[layer['activation_fct'].astype(int)]), 
        use_bias=layer['use_bias'].astype(int), 
        name=layer_name)

    dict_layernum[layer_type] += 1
    last_layer = layer_type
    return op, last_layer

def create_pooling(op, layer, layer_type, dict_layernum, last_layer):
    layer_name = layer_type + str(dict_layernum[layer_type] + 1)  
    if op == None:
        op = tf.Variable(tf.random_normal([layer['batchsize'].astype(int), 
            layer['matsize'].astype(int), layer['matsize'].astype(int), layer['channels_in'].astype(int)]))
    
    op = tf.layers.max_pooling2d(op, pool_size=(layer['poolsize'].astype(int), layer['poolsize'].astype(int)), 
        strides=(layer['strides'].astype(int), layer['strides'].astype(int)), 
        padding=('SAME' if layer['padding'].astype(int)==1 else 'VALID'), 
        name = layer_name)

    dict_layernum[layer_type] += 1
    last_layer = layer_type
    return op, last_layer


def create_dense(op, layer, layer_type, dict_layernum, last_layer):
    layer_name = layer_type + str(dict_layernum[layer_type] + 1)
    if op == None:
        op = tf.Variable(tf.random_normal([layer['batchsize'].astype(int), layer['dim_input'].astype(int)]))
    if last_layer != 'dense':
        op = tf.contrib.layers.flatten(op)


    op = tf.layers.dense(inputs=op, units=layer['dim_output'].astype(int),
        kernel_initializer=tf.ones_initializer(), 
        activation=eval(activation_list[layer['activation_fct'].astype(int)]), 
        name = layer_name)

    dict_layernum[layer_type] += 1
    last_layer = layer_type
    return op, last_layer

def profile(i, op, profile_log_path):
    time_list = []
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
    if not args.profile:
        for _ in range(args.iter_benchmark):
            start_time = time.time()
            sess.run(op)
            time_list.append(((time.time()-start_time) * 1000))
            np_array_parameters = np.array(time_list)
    else:
        # if not flags.timeline_name:
        #     pre_name1 = os.path.splitext(os.path.basename(flags.filename))[0]
        #     pre_name2 = str(flags.start_from_csv + 1) + '_to_' + str(flags.end_of_csv) if not flags.one_layer else  str(flags.start_from_csv + 1)
        #     filename = pre_name1 + '_' + pre_name2 + '_bs' + str(flags.batch_size) + '.json'
        # else:
        #     filename = flags.timeline_name
        filename = str(i) + '.json'
        filename = os.path.join(profile_log_path, filename)
        sess.run(op, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
            run_metadata=run_metadata)
        
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open(filename, 'w') as f:
            f.write(ctf)   
        return 

    time_max = numpy.amax(np_array_parameters)
    time_min = numpy.amin(np_array_parameters)
    time_median = numpy.median(np_array_parameters)
    time_mean = numpy.mean(np_array_parameters)
    time_trim_mean = stats.trim_mean(np_array_parameters, 0.1)
    print("time list: {}, time_mean: {}".format(time_list, time_mean))

def main(_):
    ########### Benchmark convolution ##########
    if args.conv:
        profile_log_path = os.path.join(os.getcwd(), 'profile_log_%s_%s' % ('conv', args.device))
        if not os.path.isdir(profile_log_path) and args.profile:
            os.makedirs(profile_log_path)

        conv_col_name = ['batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 'padding', 'activation_fct', 'use_bias']
        df = pd.read_csv('shuffled_parameters/conv_parameters_shuffled.csv', usecols=conv_col_name)

        # conv_values_list = []
        golden_values_col_name = ['batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 'padding', 'activation_fct', 'use_bias', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']
        golden_values_data = pd.DataFrame(columns=golden_values_col_name)

        # for i in range(len(df.index)):
        for i in range(5):
            print('========== conv', i , '==========')
            tf.reset_default_graph()
            layer = df.loc[i, :]
            op = None
            last_layer = None
            op, last_layer = create_conv2d(op, layer, 'conv2d', dict_layernum, last_layer)

            profile(i, op, profile_log_path)
            

    ########### Benchmark fully connection ##########
    if args.fc:
        profile_log_path = os.path.join(os.getcwd(), 'profile_log_%s_%s' % ('dense', args.device))
        if not os.path.isdir(profile_log_path) and args.profile:
            os.makedirs(profile_log_path)

        fc_col_name = ['batchsize', 'dim_input', 'dim_output', 'activation_fct']
        df = pd.read_csv('shuffled_parameters/fc_parameters_shuffled.csv', usecols=fc_col_name)

        # fc_values_list = []
        golden_values_col_name = ['batchsize', 'dim_input', 'dim_output', 'activation_fct', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']
        golden_values_data = pd.DataFrame(columns=golden_values_col_name)

        for i in range(len(df.index)):
        # for i in range(5):
            print('========== dense', i , '==========')
            tf.reset_default_graph()
            layer = df.loc[i, :]
            op = None
            last_layer = None
            op, last_layer = create_dense(op, layer, 'dense', dict_layernum, last_layer)

            profile(i, op, profile_log_path)

    ########### Benchmark pooling ##########
    if args.pool:
        profile_log_path = os.path.join(os.getcwd(), 'profile_log_%s_%s' % ('pooling', args.device))
        if not os.path.isdir(profile_log_path) and args.profile:
            os.makedirs(profile_log_path)

        pool_col_name = ['batchsize', 'matsize', 'channels_in', 'poolsize', 'padding', 'strides']
        df = pd.read_csv('shuffled_parameters/pool_parameters_shuffled.csv', usecols=pool_col_name)

        # pool_values_list = []
        golden_values_col_name = ['batchsize', 'matsize', 'channels_in', 'poolsize', 'padding', 'strides', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']
        golden_values_data = pd.DataFrame(columns=golden_values_col_name)

        for i in range(len(df.index)):
        # for i in range(5):
            print('========== pooling', i , '==========')
            tf.reset_default_graph()
            layer = df.loc[i, :]
            op = None
            last_layer = None
            op, last_layer = create_pooling(op, layer, 'pooling', dict_layernum, last_layer)

            profile(i, op, profile_log_path)

        
if __name__ == '__main__':
    tf.app.run()
