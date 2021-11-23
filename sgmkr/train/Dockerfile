FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

RUN pip3 install sagemaker-training

COPY sgmkr/src/code /opt/ml/code

ENV SAGEMAKER_PROGRAM entry_point.py