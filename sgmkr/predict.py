import pandas as pd

import sagemaker

from sagemaker.pytorch.model import PyTorchModel, PyTorchPredictor


sagemaker_session = sagemaker.Session(boto3.session.Session(region_name='us-west-2'))


df = pd.read_csv('data/train.csv', nrows=5)
df.drop('target', axis=1, inplace=True)
record = df.iloc[[0],:]

print('record:')
print(record)

predictor = PyTorchPredictor(
    'distilbert-disaster-tweets',
    sagemaker_session=sagemaker_session
)
print('created predictor')

response = predictor.predict(record)

print(response)

predictor.delete_endpoint()
predictor.delete_model()

print('deleted endpoint and model')
