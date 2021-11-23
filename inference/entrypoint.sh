# aws s3 cp s3://sagemaker-us-west-2-257018485161/pytorch-inference-2021-11-20-04-02-02-410/model.tar.gz ./model
docker run -v $(pwd)/model:/opt/ml/model -p 5000:5000 inference-server