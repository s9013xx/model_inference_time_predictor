# Notice

1. Tensorflow timeline profile need below flow

    sudo apt-get install libcupti-dev

2. Add CPUIT in ~/.bashrc

    export PATH=/usr/local/cuda-10.0/bin:/usr/local/cuda-10.1/bin:$PATH
    
    export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.1/lib64:/usr/local/cuda-       10.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

3. Activate .bashrc file

    source ~/.bashrc
    
# Model Inference Time Predictor

Step 1. Download need software.

    sudo apt-get install python3-venv

    sudo apt-get install tmux


Step 2. Use tmux to avoid terminate when ssh connection close.

    tmux

Step 3. Download code and change to main directory.

    git clone https://github.com/s9013xx/model_inference_time_predictor.git

    cd model_inference_time_predictor

Step 4. Create python3 virtual environment and activate it.

    python3 -m venv predictor_env

    source predictor_env/bin/activate

Step 5. Install need python packet in virtual environment.

    pip install -r requirements.txt

