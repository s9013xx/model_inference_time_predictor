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

parser = argparse.ArgumentParser('Shuffle Data Paremeters Parser')
# Benchmarks parameters
parser.add_argument('--conv', action="store_true", default=False, help='Benchmark convolution layer')
parser.add_argument('--fc', action="store_true", default=False, help='Benchmark fully connection layer')
parser.add_argument('--pool', action="store_true", default=False, help='Benchmark pooling layer')

args = parser.parse_args()

def main(_):
    if args.conv:
        df_conv = pd.read_csv('conv_parameters.csv')
        df_conv = df_conv.reindex(np.random.permutation(df_conv.index))
        df_conv.to_csv('./shuffled_parameters/conv_parameters_shuffled.csv', index=False)
    if args.fc:
        df_fc = pd.read_csv('fc_parameters.csv')
        df_fc = df_fc.reindex(np.random.permutation(df_fc.index))
        df_fc.to_csv('./shuffled_parameters/fc_parameters_shuffled.csv', index=False)
    if args.pool:
        df_pool = pd.read_csv('pool_parameters.csv')
        df_pool = df_pool.reindex(np.random.permutation(df_pool.index))
        df_pool.to_csv('./shuffled_parameters/pool_parameters_shuffled.csv', index=False)

if __name__ == '__main__':
    tf.app.run()
