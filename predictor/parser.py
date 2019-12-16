from keras.utils import plot_model
import re
import pandas as pd
import argparse

parser = argparse.ArgumentParser('Create Model Parser')
parser.add_argument('--model', type=str, default='', help='Device name as appearing in logfile')
args = parser.parse_args()
batches = [1, 2, 4, 8, 16, 32, 64, 128]

if args.model == 'vgg16':
    from keras.applications.vgg16 import VGG16
    model = VGG16()
elif args.model == 'vgg19':
    from keras.applications.vgg19 import VGG19
    model = VGG19()
elif args.model == 'resnet50': 
    from keras.applications.resnet50 import ResNet50
    model = ResNet50()
elif args.model == 'inceptionv3':
    from keras.applications.inception_v3 import InceptionV3
    model = InceptionV3()
elif args.model == 'lenet':
    from keras.models import Sequential
    from keras.layers import Dense,Flatten
    from keras.layers.convolutional import Conv2D,MaxPooling2D
    model = Sequential()
    model.add(Conv2D(32,(5,5),strides=(1,1),input_shape=(28,28,1),padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(10,activation='softmax'))
elif args.model =='alexnet':
    from keras.models import Sequential
    from keras.layers import Dense,Flatten,Dropout
    from keras.layers.convolutional import Conv2D,MaxPooling2D
    model = Sequential()
    model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(227,227,3),padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000,activation='softmax'))


# elif args.model == 'mobilenet':
#     from keras.applications.mobilenet import MobileNet
#     model = MobileNet()
else:
    print('you should use --model parameter to specify parse which neural network model, ex: --model vgg16')
    exit()

plot_model(model, to_file='model_pic/model_%s.png' % args.model)
# print('model.summary():', model.summary())
# print('layer num : ', len(model.layers))

total_col = ['layer', 'name', 'operation', 'batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 'padding', 'activation_fct', 'use_bias', 'poolsize', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean', 'pre_time']
layer_list = []
layer_count = 0
pattern = '[0-9]+'

df_layers = pd.DataFrame(columns=total_col, index=None)

for layer in model.layers:

    if "Conv2D" in str(layer.__class__):
        layer_count = layer_count + 1
        operation = str(layer.__class__)
        layer_num = layer_count
        layer_name = layer.name
        
        input_shape_re=re.findall(pattern,str(layer.input_shape))
        matsize = input_shape_re[1]
        channels_in = input_shape_re[2]

        cfg = layer.get_config()
        channels_out = cfg["filters"]
        kernelsize_re=re.findall(pattern,str(cfg["kernel_size"]))
        kernelsize = kernelsize_re[0]
        strides_re=re.findall(pattern,str(cfg["strides"]))
        strides = strides_re[0]
        padding = 1 if cfg["padding"]=='same' else 0
        use_bias = 1 if cfg["use_bias"]==True else 0
        activation_fct = 0
        # print(matsize, channels_in, channels_out, kernelsize, strides, padding, use_bias)
        row_data = [layer_count, layer_name, operation, None, matsize, kernelsize, channels_in, channels_out, strides, padding, activation_fct, use_bias, None, None, None, None, None, None, None]
        df_layers.loc[layer_count-1] = row_data
    if "Activation" in str(layer.__class__):
        if cfg["activation"] != None:
            df_layers.iloc[layer_count-1, total_col.index('activation_fct')] = 1
    if "MaxPooling2D" in str(layer.__class__):
        layer_count = layer_count + 1
        operation = str(layer.__class__)
        layer_num = layer_count
        layer_name = layer.name
        cfg = layer.get_config()

        input_shape_re=re.findall(pattern,str(layer.input_shape))
        matsize = input_shape_re[1]
        channels_in = input_shape_re[2]
        pool_size_re=re.findall(pattern,str(cfg["pool_size"]))
        pool_size = pool_size_re[0]
        strides_re=re.findall(pattern,str(cfg["strides"]))
        strides = strides_re[0]
        padding = 1 if cfg["padding"]=='same' else 0
        # print(matsize, channels_in, strides, padding, pool_size)
        row_data = [layer_count, layer_name, operation, None, matsize, None, channels_in, None, strides, padding, None, None, pool_size, None, None, None, None, None, None]
        df_layers.loc[layer_count-1] = row_data
    if "GlobalAveragePooling2D" in str(layer.__class__):
        layer_count = layer_count + 1
        operation = str(layer.__class__)
        layer_num = layer_count
        layer_name = layer.name
        input_shape_re=re.findall(pattern,str(layer.input_shape))
        matsize = input_shape_re[1]
        channels_in = input_shape_re[2]
        pool_size = matsize
        strides = 1
        padding = 0
        # print(matsize, channels_in, strides, padding, pool_size)
        row_data = [layer_count, layer_name, operation, None, matsize, None, channels_in, None, strides, padding, None, None, pool_size, None, None, None, None, None, None]
        df_layers.loc[layer_count-1] = row_data
    if "Dense" in str(layer.__class__):
        layer_count = layer_count + 1
        operation = str(layer.__class__)
        layer_num = layer_count
        layer_name = layer.name
        cfg = layer.get_config()
        if cfg["activation"] != None:
            activation_fct = 1
        matsize = 1
        input_shape_re=re.findall(pattern,str(layer.input_shape))
        channels_in = input_shape_re[0]
        channels_out = cfg["units"]
        # print(matsize, channels_in, channels_out, activation_fct)
        row_data = [layer_count, layer_name, operation, None, matsize, None, channels_in, channels_out, None, None, activation_fct, None, None, None, None, None, None, None, None]
        df_layers.loc[layer_count-1] = row_data

for batch in batches:
    df_layers['batchsize'] = batch
    df_layers.to_csv('model_csv/%s/%s_%d.csv' % (args.model, args.model, batch), columns=total_col, index=False)
    print(df_layers)

# df_layers.to_csv('model_csv/%s.csv' % args.model, columns=total_col, index=False)
# print(df_layers)