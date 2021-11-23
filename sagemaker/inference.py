import pandas as pd

from sagemaker.pytorch.model import PyTorchModel, PyTorchPredictor


df = pd.read_csv('data/train.csv', nrows=5)
df.drop('target', axis=1, inplace=True)
record = df.iloc[[0],:]

print('record:')
print(record)

predictor = PyTorchPredictor('distilbert-disaster-tweets')
print('created predictor')

response = predictor.predict(record)

print(response)

predictor.delete_endpoint()
predictor.delete_model()

print('deleted endpoint and model')
