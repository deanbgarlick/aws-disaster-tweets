import json
import requests

from io import StringIO

import pandas as pd


data = pd.read_csv('data/data.csv', index_col=0)
data = data.iloc[[0,1],:]
print(data)

json_data = data.to_json()
headers = {'Content-type': 'application/json', 'Accept': 'application/json'}
res = requests.post('http://localhost:5000/predict', json=json_data, headers=headers)
predictions = res.json()
print(predictions)

json_data = data.to_json()
headers = {'Content-type': 'application/json', 'Accept': 'text/csv'}
res = requests.post('http://localhost:5000/predict', json=json_data, headers=headers)
byte_string = res.content.decode('UTF-8')
df = pd.read_csv(StringIO(byte_string), header=None)
print(df)

stream = StringIO()
data.drop(labels=['target'], axis=1, inplace=True)
data.to_csv(stream, header=False)
headers = {'Content-type': 'text/csv', 'Accept': 'application/json'}
res = requests.post('http://localhost:5000/predict', data=stream.getvalue(), headers=headers)
predictions = res.json()
print(predictions)

stream = StringIO()
data.drop(labels=['target'], axis=1, inplace=True)
data.to_csv(stream, header=False)
headers = {'Content-type': 'text/csv', 'Accept': 'text/csv'}
res = requests.post('http://localhost:5000/predict', data=stream.getvalue(), headers=headers)
byte_string = res.content.decode('UTF-8')
df = pd.read_csv(StringIO(byte_string), header=None)
print(df)
