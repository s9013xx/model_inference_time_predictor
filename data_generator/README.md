# Data Generator

This folder is use to generate parameters in convolution, fully connected, pooling layers and collect inference time at each parameters. Finally, this data will store in goldan_values folder.

Step 1. Generate parameters in convolution, fully connected, pooling layers.

    python random_generate_parameters_numpy --conv --fc --pool

default is generate number of 110,000 parameter data, and you can use --num_val to specify data counts like below : 

    python random_generate_parameters_numpy --conv --fc --pool --num_val 10,000


Step 2. Collect inference time in convolution, fully connected, pooling layers.

    python collect_actual_data.py --device 2080ti --conv --fc --pool

default is collect GPU inference time, you can collect CPU inference time use --cpu parameter if you want.

    python collect_actual_data.py --device i3 --conv --fc --pool --cpu
