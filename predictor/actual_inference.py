import os
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import numpy
from scipy import stats

parser = argparse.ArgumentParser('Create Model Parser')
parser.add_argument('--device', type=str, default='', help='Device name as appearing in logfile')
parser.add_argument('--model', type=str, default='', help='Device name as appearing in logfile')
parser.add_argument('--cpu', action="store_true", default=False, help='Benchmark using CPU')
parser.add_argument('--iter_warmup', type=int, default=5, help='Number of iterations for warm-up')
parser.add_argument('--iter_benchmark', type=int, default=10, help='Number of iterations for benchmark')
args = parser.parse_args()

if args.device == '':
    print('you should use --device parameter to specify collect data for which device, ex: --device 2080ti')
    exit()

if args.model == '':
    print('you should use --model parameter to specify parse which neural network model, ex: --model vgg16')
    exit()

if args.cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

batches = [1, 2, 4, 8, 16, 32, 64, 128]

total_col = ['layer', 'name', 'operation', 'batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 'padding', 'activation_fct', 'use_bias', 'poolsize', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean', 'pre_time_abse', 'pre_time_re', 'pre_time_rmse']
activation_list = ['None', 'tf.nn.relu']

for batch in batches:
    df = pd.read_csv('model_csv/%s/%s_%d.csv' % (args.model, args.model, batch), usecols=total_col)

    for index in range(len(df)):
        time_list = []
        # convolution layer operation
        if "Conv2D" in str(df.values[index][total_col.index('operation')]):
        # if df.values[index][total_col.index('operation')] == 'conv':
            batchsize = df.values[index][total_col.index('batchsize')]
            matsize = df.values[index][total_col.index('matsize')]
            kernelsize = df.values[index][total_col.index('kernelsize')]
            channels_in = df.values[index][total_col.index('channels_in')]
            channels_out = df.values[index][total_col.index('channels_out')]
            strides = df.values[index][total_col.index('strides')]
            padding = df.values[index][total_col.index('padding')]
            activation_fct = int(df.values[index][total_col.index('activation_fct')])
            use_bias = df.values[index][total_col.index('use_bias')]

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

            df.iloc[index, total_col.index('time_max')] = time_max
            df.iloc[index, total_col.index('time_min')] = time_min
            df.iloc[index, total_col.index('time_median')] = time_median
            df.iloc[index, total_col.index('time_mean')] = time_mean
            df.iloc[index, total_col.index('time_trim_mean')] = time_trim_mean
        # pooling layer operation
        elif "MaxPooling2D" in str(df.values[index][total_col.index('operation')]) or "GlobalAveragePooling2D" in str(df.values[index][total_col.index('operation')]):
        # elif df.values[index][total_col.index('operation')] == 'pool':
            batchsize = df.values[index][total_col.index('batchsize')]
            matsize = df.values[index][total_col.index('matsize')]
            channels_in = df.values[index][total_col.index('channels_in')]
            strides = df.values[index][total_col.index('strides')]
            padding = df.values[index][total_col.index('padding')]
            poolsize = df.values[index][total_col.index('poolsize')]

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

            df.iloc[index, total_col.index('time_max')] = time_max
            df.iloc[index, total_col.index('time_min')] = time_min
            df.iloc[index, total_col.index('time_median')] = time_median
            df.iloc[index, total_col.index('time_mean')] = time_mean
            df.iloc[index, total_col.index('time_trim_mean')] = time_trim_mean
        # fully connected layer operation
        elif "Dense" in str(df.values[index][total_col.index('operation')]):
        # elif df.values[index][total_col.index('operation')] == 'fc':
            batchsize = df.values[index][total_col.index('batchsize')]
            matsize = df.values[index][total_col.index('matsize')]
            channels_in = df.values[index][total_col.index('channels_in')]
            channels_out = df.values[index][total_col.index('channels_out')]
            activation_fct = int(df.values[index][total_col.index('activation_fct')])

            dim_input = matsize * matsize * channels_in
            dim_output = channels_out

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

            df.iloc[index, total_col.index('time_max')] = time_max
            df.iloc[index, total_col.index('time_min')] = time_min
            df.iloc[index, total_col.index('time_median')] = time_median
            df.iloc[index, total_col.index('time_mean')] = time_mean
            df.iloc[index, total_col.index('time_trim_mean')] = time_trim_mean
        # unknown operation
        else:
            print('Unknown Operation')

    output_dir = './model_actual_result/%s/%s' % (args.device, args.model)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = '%s_%d.csv' % (args.model, batch)
    output_file = os.path.join(output_dir, file_name)    
    df.to_csv(output_file, columns=total_col, index=False)
