from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import argparse
import tensorflow as tf
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

parser = argparse.ArgumentParser('Test Parser')
parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('--devlist', type=str, default='', help='List of devices to use, overwrites num_gpu if set')
args = parser.parse_args()



# import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def generate_devlist(devlist, num_gpu):
    """Creates list with devices

    Args:
        devlist: Comma separated list of devices, overwrites num_gpu
        num_gpu: Number of GPUs to be used

    Return:
        devlist: List of devices
        use_gpu: Whether GPUs are used (boolean)
    """
    if devlist=='':
        if num_gpu==0:
            devlist = ['/cpu:0']
            use_gpu = False
        else:
            devlist = ['/gpu:%d' %i for i in range(num_gpu)]
            use_gpu = True
    else:
        use_gpu = ('gpu' in devlist.lower())
        devlist = devlist.split(',')
    return devlist, use_gpu

def main(_):

    activation_list = [
        'None',
        'tf.nn.relu']

    batchsize, matsize, kernelsize, channels_in, channels_out, strides, padding, activation_fct, use_bias = 62, 471, 7, 19, 18, 1, 0, 0, 1
    
    devlist, use_gpu = generate_devlist(args.devlist, args.num_gpu)
    print('devlist: ', devlist)
    dev = devlist[0]
    print('dev: ', dev)
    
    tf.reset_default_graph()
    image = tf.Variable(tf.random_normal([batchsize, matsize, matsize, channels_in]))
    op = tf.layers.conv2d(image, filters=channels_out, kernel_size=[kernelsize, kernelsize], strides=(strides, strides), padding=('SAME' if padding==1 else 'VALID'), activation=eval(activation_list[activation_fct]), use_bias=use_bias)
    sess = tf.Session()
    try:
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(op)
    # except tf.errors.UnknownError as e:
    #     print(e)
    # except tf.errors.AbortedError as e:
    #     print(e)
    # except Exception as e:
    #     print(e)
    # except RuntimeError:
    #     print('RuntimeError')
    except:
        print("Unexpected error:", sys.exc_info()[0])
        print('except')
    # except OSError as e:
    #     print('Exception handler error occurred: '  + str(e.errno) + os.linesep)

    print('!!!!!!!!!!!!!!')

if __name__ == '__main__':
    tf.app.run()
