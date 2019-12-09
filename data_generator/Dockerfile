FROM generate_train_data_1:latest

COPY . /model
WORKDIR /model

ENTRYPOINT ["python3", \
            "/model/model_inference_time_predictor/data_generator/popen_test.py", \
            "--device=2080ti", \
            "--conv", \
            "--fc", \
            "--pool"]
