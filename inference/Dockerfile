FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

WORKDIR /opt/ml/code

COPY src/requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY src .

ENV FLASK_APP=server.py

ENTRYPOINT [ "flask", "run" ]