### To train the model

`python -m sgmkr.estimator`


### To create, run, and test, the docker inference image locally

```
python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
aws s3 cp s3://sagemaker-us-west-2-257018485161/pytorch-inference-2021-11-20-04-02-02-410/model.tar.gz ./model
docker build -t inference-server inference
docker run -v $(pwd)/model:/opt/ml/model -p 5000:5000 inference-server --host 0.0.0.0
sleep 20
python test/test_local.py
```

###